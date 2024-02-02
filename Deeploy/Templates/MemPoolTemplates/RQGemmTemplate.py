# ----------------------------------------------------------------------
#
# File: RQGemmTemplate.py
#
# Last edited: 17.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

import numpy as np
from mako.template import Template

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTemplate


class _RQGemmTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()

        A = ctxt.lookup(nodeRep['A'])
        B = ctxt.lookup(nodeRep['B'])
        C = ctxt.lookup(nodeRep['C'])
        Y = ctxt.lookup(nodeRep['data_out'])
        nodeRep['A_offset'] = (A._signed == 0) * int(A.nLevels / 2)
        nodeRep['B_offset'] = (B._signed == 0) * int(B.nLevels / 2)
        nodeRep['C_offset'] = -(C._signed == 0) * int(C.nLevels / 2)
        nodeRep['Y_offset'] = -(Y._signed == 0) * int(Y.nLevels / 2)

        nodeRep['output_min'] = -(nodeRep['n_levels'] // 2)
        nodeRep['output_max'] = (nodeRep['n_levels'] // 2) - 1

        MUL = ctxt.lookup(nodeRep['mul'])
        # WIESEP: Per element and per column quantization is not supported for RQGemm

        if len(MUL.shape) == 1:
            nodeRep['perRowQuant'] = 0
        else:
            nodeRep['perRowQuant'] = int(MUL.shape[-2] != 1)

        return ctxt, nodeRep, []

    def hoistTransientBuffers(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        # Allocate buffer in L1 if original data lives in L2 to speed up the calculation,
        # by first transferring it to L2 with the DMA.

        A = ctxt.lookup(nodeRep['A'])
        B = ctxt.lookup(nodeRep['B'])
        C = ctxt.lookup(nodeRep['C'])

        names = []
        size = nodeRep['M'] * nodeRep['N'] * (A._type.referencedType.typeWidth // 8)
        name = nodeRep['nodeName'] + f"_buffer_A"
        nodeRep['ctxtBuffer_A_size'] = size
        if isinstance(A, ConstantBuffer):
            names += [name]
            ctxt.hoistTransientBuffer(name, size)
            nodeRep['ctxtBuffer_A'] = ctxt._mangle(name)
        else:
            nodeRep['ctxtBuffer_A'] = nodeRep['A']

        size = nodeRep['N'] * nodeRep['O'] * (B._type.referencedType.typeWidth // 8)
        name = nodeRep['nodeName'] + f"_buffer_B"
        nodeRep['ctxtBuffer_B_size'] = size
        if isinstance(B, ConstantBuffer):
            names += [name]
            ctxt.hoistTransientBuffer(name, size)
            nodeRep['ctxtBuffer_B'] = ctxt._mangle(name)
        else:
            nodeRep['ctxtBuffer_B'] = nodeRep['B']

        size = nodeRep['M'] * nodeRep['O'] * (C._type.referencedType.typeWidth // 8)
        name = nodeRep['nodeName'] + f"_buffer_C"
        nodeRep['ctxtBuffer_C_size'] = size
        if isinstance(C, ConstantBuffer):
            names += [name]
            ctxt.hoistTransientBuffer(name, size)
            nodeRep['ctxtBuffer_C'] = ctxt._mangle(name)
        else:
            nodeRep['ctxtBuffer_C'] = nodeRep['C']

        return ctxt, nodeRep, names


MemPoolParallelTemplate = _RQGemmTemplate("""
<%
if isinstance(log2D, int):
    log2Dstring = log2D
else:
    log2Dstring = "*"+log2D
%>

// RQGEMM Parallel (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);

%if ctxtBuffer_A != A:
// Fast copy data from L2 to L1
BEGIN_SINGLE_CORE
    #if USE_DMA
        dma_memcpy_blocking(${ctxtBuffer_A}, ${A}, ${ctxtBuffer_A_size});
    #else
        memcpy(${ctxtBuffer_A}, ${A}, ${ctxtBuffer_A_size});
    #endif
END_SINGLE_CORE
%endif

%if ctxtBuffer_B != B:
// Fast copy data from L2 to L1
BEGIN_SINGLE_CORE
    #if USE_DMA
        dma_memcpy_blocking(${ctxtBuffer_B}, ${B}, ${ctxtBuffer_B_size});
    #else
        memcpy(${ctxtBuffer_B}, ${B}, ${ctxtBuffer_B_size});
    #endif
END_SINGLE_CORE
%endif

%if ctxtBuffer_C != C:
// Fast copy data from L2 to L1
BEGIN_SINGLE_CORE
    #if USE_DMA
        dma_memcpy_blocking(${ctxtBuffer_C}, ${C}, ${ctxtBuffer_C_size});
    #else
        memcpy(${ctxtBuffer_C}, ${C}, ${ctxtBuffer_C_size});
    #endif
END_SINGLE_CORE
%endif

%if ctxtBuffer_A != A or ctxtBuffer_B != B or ctxtBuffer_C != C:
    mempool_barrier(numThreads);
%endif

${A_type.typeName} ref_${data_out}_${A} = ${ctxtBuffer_A};
${B_type.typeName} ref_${data_out}_${B} = ${ctxtBuffer_B};
${C_type.typeName} ref_${data_out}_${C} = ${ctxtBuffer_C};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for(uint32_t i=0;i<${batch};i++){
%if M%4==0 and N%4==0 and O%4==0:
    RQGemm_offset_unrolled_2x2_parallel_s${A_type.referencedType.typeWidth}(
        ref_${data_out}_${A},
        ref_${data_out}_${B},
        ref_${data_out}_${C},
        ref_${data_out}_${data_out},
        ${M},
        ${N},
        ${O},
        ${alpha},
        ${beta},
        ${transA},
        ${transB},
        ${mul},
        ${add},
        ${log2Dstring},
        1,
        ${perRowQuant},
        ${A_offset},
        ${B_offset},
        ${C_offset},
        ${Y_offset},
        core_id,
        numThreads
    );
%else:
    RQGemm_parallel_s${A_type.referencedType.typeWidth}(
        ref_${data_out}_${A},
        ref_${data_out}_${B},
        ref_${data_out}_${C},
        ref_${data_out}_${data_out},
        ${M},
        ${N},
        ${O},
        ${alpha},
        ${beta},
        ${transA},
        ${transB},
        ${mul},
        ${add},
        ${log2Dstring},
        1,
        ${perRowQuant},
        ${A_offset},
        ${B_offset},
        ${C_offset},
        ${Y_offset},
        ${output_min},
        ${output_max},
        core_id,
        numThreads
    );
%endif

    ref_${data_out}_${A} += ${M} * ${N};
    ref_${data_out}_${B} += ${N} * ${O};
    ref_${data_out}_${data_out} += ${M} * ${O};
}
mempool_barrier(numThreads);
""")
