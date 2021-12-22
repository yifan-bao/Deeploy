# ----------------------------------------------------------------------
#
# File: CMSISPasses.py
#
# Last edited: 20.12.2021        
# 
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
# Author: Georg Rutishauser, ETH Zurich
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

import copy

from DumpO.DumpOTypes import *
from DumpO.Layers.BasicLayers import *
from DumpO.OptimizationPasses.BasicPasses import *

import onnx_graphsurgeon as gs

def merge_conv_rq_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    conv = matched_nodes[0]
    rqs = matched_nodes[1]

    _inputs = conv.inputs + rqs.inputs[1:]
    _outputs = rqs.outputs

    rqsConv = gs.Node(op='RequantizedConv', name=name, attrs={**conv.attrs, **rqs.attrs})
    graph.replaceInsertNode(_inputs, _outputs, rqsConv)


class ConvRequantMergePass(ReplaceSequentialPatternPass):
    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name='input_1')
        output = graph.layer(inputs=[_input], outputs=['conv_out'], op='Conv', name='conv1')
        output = graph.layer(inputs=output, outputs=['rqs'], op='RequantShift', name='rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)
    
        name = f"_MERGE_CONVRQ_PASS"
        super().__init__(graph, merge_conv_rq_fun, name)

    
