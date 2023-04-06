# ----------------------------------------------------------------------
#
# File: DebugTemplate.py
#
# Last edited: 14.12.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
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
from Deeploy.DataTypes.BasicDataTypes import *


class _DebugTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()

        data_in = ctxt.lookup(nodeRep['data_in'])
        data_out = ctxt.lookup(nodeRep['data_out'])

        data_out._type = data_in._type

        nodeRep['data_in_signed'] = data_in._signed
        nodeRep['offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)

        return ctxt, nodeRep


referenceTemplate = _DebugTemplate("""
<%
tensor_type = "Input" if "input" in node_name else "Output"
tensor_name = node_name.replace("_input", "").replace("_output", "")
%>

// Debug (Name: ${node_name}, Op: ${node_op})
BEGIN_SINGLE_CORE
    ${data_out} = ${data_in};
    deeploy_log("[DEBUG] ${tensor_type} ${tensor_name} (Buffer ${data_in}, Signed: ${data_in_signed}):\\r\\n");

    %if channels_first:
    %if data_in_signed:
        PrintMatrix_s${data_in_type._value_}_NCHW(${data_in}, ${batch}, ${dim_im_in_ch}, ${dim_im_in_x}, ${dim_im_in_y}, ${offset});
    %else:
        PrintMatrix_u${data_in_type._value_}_NCHW((uint${data_in_type._value_}_t *) ${data_in}, ${batch}, ${dim_im_in_ch}, ${dim_im_in_x}, ${dim_im_in_y}, ${offset});
    %endif
    %else:
    %if data_in_signed:
        PrintMatrix_s${data_in_type._value_}_NHWC(${data_in}, ${batch}, ${dim_im_in_ch}, ${dim_im_in_x}, ${dim_im_in_y}, ${offset});
    %else:
        PrintMatrix_u${data_in_type._value_}_NHWC((uint${data_in_type._value_}_t *) ${data_in}, ${batch}, ${dim_im_in_ch}, ${dim_im_in_x}, ${dim_im_in_y}, ${offset});
    %endif
    %endif
END_SINGLE_CORE
""")
