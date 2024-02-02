# ----------------------------------------------------------------------
#
# File: GEMMTemplate.py
#
# Last edited: 10.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
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

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate


class PULPGEMMTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(nodeRep['B'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(nodeRep['A'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(nodeRep['data_out'])._type.referencedType.typeMin < 0
        nodeRep['weight_signed'] = signedW
        nodeRep['input_signed'] = signedI
        nodeRep['output_signed'] = signedO
        return ctxt, nodeRep, []


PULPGEMM_8_Template = PULPGEMMTemplate("""
// PULP NN GEMM
int8_t* ref_${data_out}_${A} = ${A};
int8_t* ref_${data_out}_${B} = ${B};
int8_t* ref_${data_out}_${data_out} = ${data_out};
for(int i=0;i<${batch};i++){
for(int j=0;j<${M};j++){
pulp_nn_linear${signatureString}(ref_${data_out}_${A}, NULL, ref_${data_out}_${data_out}, ref_${data_out}_${B}, ${mul}, ${C}, 1, ${log2D}, ${N}, ${O}, 1, 1);
ref_${data_out}_${A} += ${N};
ref_${data_out}_${data_out} += ${O};
}
ref_${data_out}_${B} += ${N} * ${O};
}
""")

PULPGEMM_8_Template = PULPGEMMTemplate("""
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
// PULP NN GEMM
int8_t* ref_${data_out}_${A} = ${A};
int8_t* ref_${data_out}_${B} = ${B};
int8_t* ref_${data_out}_${data_out} = ${data_out};
for(int i=0;i<${batch};i++){
for(int j=0;j<${M};j++){
pulp_nn_linear${signatureString}(ref_${data_out}_${A}, NULL, ref_${data_out}_${data_out}, ref_${data_out}_${B}, ${mul}, ${C}, 1, ${log2D}, ${N}, ${O}, 1, 1);
ref_${data_out}_${A} += ${N};
ref_${data_out}_${data_out} += ${O};
}
ref_${data_out}_${B} += ${N} * ${O};
}
""")


class _MatMulTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt = ctxt.copy()

        A = ctxt.lookup(nodeRep['A'])
        B = ctxt.lookup(nodeRep['B'])
        C = ctxt.lookup(nodeRep['data_out'])

        nodeRep['A_offset'] = 0
        nodeRep['B_offset'] = 0
        nodeRep['C_offset'] = 0

        if hasattr(A, "nLevels"):
            nodeRep['A_offset'] = (A._type.referencedType.typeMin == 0) * int(A.nLevels / 2)
        if hasattr(B, "nLevels"):
            nodeRep['B_offset'] = (B._type.referencedType.typeMin == 0) * int(B.nLevels / 2)
        if hasattr(C, "nLevels"):
            nodeRep['C_offset'] = -(C._type.referencedType.typeMin == 0) * int(C.nLevels / 2)

        return ctxt, nodeRep, []


PULPMM_8_Template = _MatMulTemplate("""
// MatMul (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${A_type.typeName} ref_${data_out}_${A} = ${A};
    ${B_type.typeName} ref_${data_out}_${B} = ${B};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for(uint32_t i=0;i<${batch};i++){
        MatMul_s${A_type.referencedType.typeWidth}_s${B_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}(
            ref_${data_out}_${A},
            ref_${data_out}_${B},
            ref_${data_out}_${data_out},
            ${M},
            ${N},
            ${O},
            0, 0, ${C_offset}
        );

        ref_${data_out}_${A} += ${M} * ${N};
        ref_${data_out}_${B} += ${N} * ${O};
        ref_${data_out}_${data_out} += ${M} * ${O};
    }
END_SINGLE_CORE
""")
