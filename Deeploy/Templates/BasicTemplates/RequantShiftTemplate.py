# ----------------------------------------------------------------------
#
# File: RequantShiftTemplate.py
#
# Last edited: 14.12.2021
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

from Deeploy.DeeployTypes import NodeTemplate

from typing import Dict, Tuple
from mako.template import Template

from Deeploy.DeeployTypes import NodeTemplate, NetworkContext


class _RequantShiftTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()

        data_in = ctxt.lookup(nodeRep['data_in'])
        data_out = ctxt.lookup(nodeRep['data_out'])
        # nodeRep['input_offset'] = (data_in._signed == 0) * nodeRep['n_levels']//2
        # nodeRep['output_offset'] = -(data_out._signed == 0) * nodeRep['n_levels']//2
        nodeRep['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        nodeRep['output_offset'] = -(data_out._signed == 0) * nodeRep['n_levels'] // 2

        nodeRep['output_min'] = -(nodeRep['n_levels'] // 2)
        nodeRep['output_max'] = (nodeRep['n_levels'] // 2) - 1

        return ctxt, nodeRep


referenceTemplate = _RequantShiftTemplate("""
<%
if isinstance(log2D, int):
    log2Dstring = log2D
else:
    log2Dstring = "*"+log2D
%>

// RequantShift (Name: ${node_name}, Op: ${node_op})
BEGIN_SINGLE_CORE
    % if channels_first:
    RequantShift_s${data_in_type._value_}_s${data_out_type._value_}_NCHW(${data_in}, ${size}, ${mul}, ${add}, ${data_out}, ${log2Dstring}, ${channel_width}, ${input_offset}, ${output_offset}, ${output_min}, ${output_max}, 1);
    % else:
    RequantShift_s${data_in_type._value_}_s${data_out_type._value_}_NHWC(${data_in}, ${size}, ${mul}, ${add}, ${data_out}, ${log2Dstring}, ${channels}, ${input_offset}, ${output_offset}, ${output_min}, ${output_max}, 1);
    %endif
END_SINGLE_CORE
""")
