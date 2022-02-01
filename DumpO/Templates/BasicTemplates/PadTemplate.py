# ----------------------------------------------------------------------
#
# File: PadTemplate.py
#
# Last edited: 27.12.2021        
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


from typing import Dict
from mako.template import Template

from DumpO.DumpOTypes import NodeTemplate, NetworkContext

# SCHEREMO: ASSUMES NHWC

class _Pad2DTemplate(NodeTemplate):
    def __init__(self, templateStr):
        self.template = Template(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> (NetworkContext, Dict):
        ctxt = ctxt.copy()

        # Align padding value to input signedness

        data_in = ctxt.lookup(nodeRep['data_in'])
        assert data_in._signed is not None
        if data_in._signed == False:
            nodeRep['value'] = nodeRep['value'] - int(data_in.nLevels/2)

        return ctxt, nodeRep

unrolledTemplate = _Pad2DTemplate("""
memset(${data_out}, 0, ${data_out_size}*sizeof(${data_out_type._name_}));
% for h in range(dim_im_in_y):
<%    
    y_offset_out = dim_im_out_ch*(pad_y*dim_im_out_x)
    x_offset_out = dim_im_out_ch*(dim_im_out_x * h + pad_x)
    offset_in = dim_im_in_ch*(dim_im_in_x * h)
    width = dim_im_in_ch*dim_im_in_x
%>
memcpy(${data_out}+${x_offset_out}+${y_offset_out}, ${data_in}+${offset_in}, ${width}*sizeof(${data_out_type._name_})); 
% endfor""")

referenceTemplate = _Pad2DTemplate("""
// Pad
memset(${data_out}, ${value}, ${data_out_size}*sizeof(${data_out_type._name_}));
<%    
    y_offset_out = dim_im_out_ch*(pad_y*dim_im_out_x)
    x_offset_out = dim_im_out_ch*(pad_x)
    width = dim_im_in_ch*dim_im_in_x

    addoffsetOut = dim_im_out_ch * dim_im_out_x
    addoffsetIn = dim_im_in_ch * dim_im_in_x

    startPosX = y_offset_out + x_offset_out
    startPosOffset = 0

batchOffsetIn = width * dim_im_in_y
batchOffsetOut = dim_im_out_ch * dim_im_out_y * dim_im_out_x
%>
int32_t xoffset_${data_in}; 
int32_t offset_in_${data_in};
for(int n=0; n<${batch}; n++){
xoffset_${data_in} = ${startPosX};
offset_in_${data_in} = ${startPosOffset};
for(int h=0; h<${dim_im_in_y}; h++){
memcpy(${data_out}+xoffset_${data_in}+n*${batchOffsetOut}, ${data_in}+offset_in_${data_in}+n*${batchOffsetIn}, ${width}*sizeof(${data_out_type._name_})); 
xoffset_${data_in} += ${addoffsetOut};
offset_in_${data_in} += ${addoffsetIn};
}
}
""")
