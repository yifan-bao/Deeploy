# ----------------------------------------------------------------------
#
# File: DWConvTemplate.py
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


class _DWConv2D_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()

        data_in = ctxt.lookup(nodeRep['data_in'])
        data_out = ctxt.lookup(nodeRep['data_out'])

        nodeRep['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels // 2)
        nodeRep['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        return ctxt, nodeRep


reference1DTemplate = _DWConv2D_Template("""
<%
batchOffsetIn = ch_im_in * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_y
%>

// 1D Depth-Wise Conv (Name: ${node_name}, Op: ${node_op})
BEGIN_SINGLE_CORE
    ${data_in_type._name_}* ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type._name_}* ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        DWConv2d_s${data_in_type._value_}_s${weight_type._value_}_s${data_out_type._value_}_NCHW(
            ref_${data_out}_${data_in}, ${ch_im_in}, 1, ${dim_im_in_y}, 
            ${weight}, 1, ${dim_kernel_y}, 
            1, ${stride_y},
            ref_${data_out}_${data_out}, ${input_offset}, ${output_offset}
        );
        ref_${data_out}_${data_in} += ${batchOffsetIn};
        ref_${data_out}_${data_out} += ${batchOffsetOut};
    }
END_SINGLE_CORE
""")

reference2DTemplate = _DWConv2D_Template("""
<%
batchOffsetIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>

// 2D Depth-Wise Conv (Name: ${node_name}, Op: ${node_op})
BEGIN_SINGLE_CORE
    ${data_in_type._name_}* ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type._name_}* ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        DWConv2d_s${data_in_type._value_}_s${weight_type._value_}_s${data_out_type._value_}_NCHW(
            ref_${data_out}_${data_in}, ${ch_im_in}, ${dim_im_in_x}, ${dim_im_in_y}, 
            ${weight}, ${dim_kernel_x}, ${dim_kernel_y}, 
            ${stride_x}, ${stride_y},
            ref_${data_out}_${data_out}, ${input_offset}, ${output_offset}
        );
        ref_${data_out}_${data_in} += ${batchOffsetIn};
        ref_${data_out}_${data_out} += ${batchOffsetOut};
    }
END_SINGLE_CORE
""")
