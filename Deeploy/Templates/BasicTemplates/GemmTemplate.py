# ----------------------------------------------------------------------
#
# File: GemmTemplate.py.py
#
# Last edited: 05.01.2023
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

from typing import Dict, Tuple
from mako.template import Template

from Deeploy.DeeployTypes import NodeTemplate, NetworkContext


class _GemmTemplate(NodeTemplate):

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

        return ctxt, nodeRep


referenceTemplate = _GemmTemplate("""
// GEMM (Name: ${node_name}, Op: ${node_op})
BEGIN_SINGLE_CORE
    ${A_type._name_}* ref_${data_out}_${A} = ${A};
    ${B_type._name_}* ref_${data_out}_${B} = ${B};
    ${C_type._name_}* ref_${data_out}_${C} = ${C};
    ${data_out_type._name_}* ref_${data_out}_${data_out} = ${data_out};

    for(uint32_t i=0;i<${batch};i++){
        Gemm_s${A_type._value_}_s${B_type._value_}_s${C_type._value_}_s${data_out_type._value_}(
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
            ${A_offset}, 
            ${B_offset}, 
            ${C_offset},
            ${Y_offset}
        );

        ref_${data_out}_${A} += ${M} * ${N};
        ref_${data_out}_${B} += ${N} * ${O};
        ref_${data_out}_${C} += ${M} * ${O};
        ref_${data_out}_${data_out} += ${M} * ${O};
    }
END_SINGLE_CORE
""")
