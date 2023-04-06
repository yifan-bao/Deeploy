# ----------------------------------------------------------------------
#
# File: iGELUTemplate.py
#
# Last edited: 13.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
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

from Deeploy.DeeployTypes import NodeTemplate, NetworkContext


class _iGELUTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()

        data_in = ctxt.lookup(nodeRep['data_in'])
        data_out = ctxt.lookup(nodeRep['data_out'])
        nodeRep['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        nodeRep['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels / 2)

        return ctxt, nodeRep


referenceTemplate = _iGELUTemplate("""
// iGELU (Name: ${node_name}, Op: ${node_op})
SINGLE_CORE GELU_s${data_in_type._value_}_s${data_out_type._value_}(${data_in}, ${data_out}, ${size}, ${b}, ${one}, ${input_offset});
""")
