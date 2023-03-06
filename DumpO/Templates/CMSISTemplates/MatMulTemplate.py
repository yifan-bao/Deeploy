# ----------------------------------------------------------------------
#
# File: MatMulTemplate.py
#
# Last edited: 02.09.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
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

from typing import Dict, Tuple
from mako.template import Template

from DumpO.DumpOTypes import NodeTemplate, NetworkContext

class _MatMulTemplate(NodeTemplate):
    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()

        A = ctxt.lookup(nodeRep['A'])
        B = ctxt.lookup(nodeRep['B'])
        C = ctxt.lookup(nodeRep['data_out'])
        nodeRep['A_offset'] = (A._signed==0) * int(A.nLevels/2)
        nodeRep['B_offset'] = (B._signed==0) * int(B.nLevels/2)
        nodeRep['C_offset'] = -(C._signed==0) * int(C.nLevels/2)

        return ctxt, nodeRep

reference_8_8_32_Template = _MatMulTemplate("""
// MatMul
int8_t* ref_${data_out}_${A} = ${A};
int8_t* ref_${data_out}_${B} = ${B};
int32_t* ref_${data_out}_${data_out} = ${data_out};

for(int i=0;i<${batch};i++){
MatMul_s8_s8_s32(ref_${data_out}_${A}, ref_${data_out}_${B}, ref_${data_out}_${data_out}, ${A_offset}, ${B_offset}, ${M}, ${N}, ${O});
ref_${data_out}_${A} += ${M} * ${N};
ref_${data_out}_${B} += ${N} * ${O};
ref_${data_out}_${data_out} += ${M} * ${O};
}

""")
