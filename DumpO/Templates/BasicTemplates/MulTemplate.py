# ----------------------------------------------------------------------
#
# File: MulTemplate.py
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

class _MulTemplate(NodeTemplate):
    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()

        A = ctxt.lookup(nodeRep['A'])
        B = ctxt.lookup(nodeRep['B'])
        C = ctxt.lookup(nodeRep['C'])
        nodeRep['A_offset'] = (A._signed==0) * int(A.nLevels/2)
        nodeRep['B_offset'] = (B._signed==0) * int(B.nLevels/2)
        nodeRep['C_offset'] = -(C._signed==0) * int(C.nLevels/2)

        return ctxt, nodeRep

referenceTemplate = _MulTemplate("""
// Mul
for (uint32_t i=0;i<${size};i++){
${C}[i] = ((${A}[i] + ${A_offset}) * (*${B} + ${B_offset}) + ${C_offset});
}
""")
