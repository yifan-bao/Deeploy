# ----------------------------------------------------------------------
#
# File: ReduceMeanTemplate.py
#
# Last edited: 05.06.2022
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

from typing import Dict
from mako.template import Template

from DumpO.DumpOTypes import NodeTemplate, NetworkContext

class _ReduceMeanTemplate(NodeTemplate):
    def __init__(self, templateStr):
        self.template = Template(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> (NetworkContext, Dict):
        ctxt = ctxt.copy()

        data_in = ctxt.lookup(nodeRep['data_in'])
        data_out = ctxt.lookup(nodeRep['data_out'])
        nodeRep['input_offset'] = (data_in._signed==0) * int(data_in.nLevels/2)
        nodeRep['output_offset'] = -(data_out._signed==0) * int(data_in.nLevels/2)

        return ctxt, nodeRep

referenceTemplate = _ReduceMeanTemplate("""
// ReduceMean
int32_t ${data_out}_accumulator = 0;
<%
reduceLength = 1
for i in axes:
    reduceLength = reduceLength * data_in_shape[i]
%>
<%
    shapeStr = ''
    accessStr = ''
%>
% for idx, i in enumerate(data_in_shape[:-1]):
<%
    shapeStr += '['+str(i)+']'
%>
% endfor
% for j in range(len(data_in_shape)):
<%
    accessStr += '[i_'+str(j)+']'
%>
% endfor
${data_in_type._name_}* dummy_${data_in} = ((${data_in_type._name_} (*)${shapeStr})${data_in});
${data_out_type._name_}* dummy_${data_out} = ${data_out};

<%
restDims = set(list(range(len(data_in_shape)))).difference(set(axes))
%>
% for i in list(restDims):
for(int i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor
${data_out}_accumulator = 0;
% for i in list(axes):
for(int i_${i} = 0; i_${i}<${data_in_shape[i]}; i_${i}++){
% endfor
${data_out}_accumulator += dummy_${data_in}${accessStr};

% for i in range(len(axes)):
}
% endfor
% if keepdims:
*$dummy_${data_out}++ = ${data_out}_accumulator / ${reduceLength};
% else:
*$dummy_${data_out}++ = ${data_out}_accumulator / ${reduceLength};
% endif
% for i in range(len(restDims)):
}
% endfor
""")
