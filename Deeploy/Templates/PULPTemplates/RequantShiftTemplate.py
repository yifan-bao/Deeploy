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

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate


class _RequantShiftTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt = ctxt.copy()

        data_in = ctxt.lookup(nodeRep['data_in'])
        data_out = ctxt.lookup(nodeRep['data_out'])

        nodeRep["signedI"] = data_in._type.referencedType.typeMin < 0
        nodeRep["signedO"] = data_out._type.referencedType.typeMin < 0

        nodeRep['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            nodeRep['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        nodeRep['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            nodeRep['output_offset'] = -(data_out._signed == 0) * nodeRep['n_levels'] // 2

        if nodeRep["signed"]:
            nodeRep['output_min'] = -(nodeRep['n_levels'] // 2)
            nodeRep['output_max'] = (nodeRep['n_levels'] // 2) - 1
        else:
            nodeRep['output_min'] = 0
            nodeRep['output_max'] = nodeRep['n_levels'] - 1

        return ctxt, nodeRep, []


referenceTemplate = _RequantShiftTemplate("""
<%
if isinstance(log2D, int):
    log2Dstring = log2D
else:
    log2Dstring = "*"+log2D

inSignage = "s" if signedI else "u"
outSignage = "s" if signedO else "u"
%>

// RequantShift (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    % if channels_first:
    RequantShift_${inSignage}${data_in_type.referencedType.typeWidth}_${outSignage}${data_out_type.referencedType.typeWidth}_NCHW(${data_in}, ${size}, ${mul}, ${add}, ${data_out}, ${log2Dstring}, ${channel_width}, 0, 0 , ${output_min}, ${output_max}, 1);
    % else:
    RequantShift_${inSignage}${data_in_type.referencedType.typeWidth}_${outSignage}${data_out_type.referencedType.typeWidth}_NHWC(${data_in}, ${size}, ${mul}, ${add}, ${data_out}, ${log2Dstring}, ${channels}, 0, 0, ${output_min}, ${output_max}, 1);
    %endif
END_SINGLE_CORE
""")
