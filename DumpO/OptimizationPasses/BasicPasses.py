# ----------------------------------------------------------------------
#
# File: BasicPasses.py
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
from typing import NamedTuple
import onnx_graphsurgeon as gs

from DumpO.DumpOTypes import *
from DumpO.Layers.BasicLayers import *
from DumpO.OptimizationPasses.PassClasses import *

def merge_transposes_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    t1 = matched_nodes[0]
    t2 = matched_nodes[1]

    #import IPython; IPython.embed()
    
    # Transpose forth and back - delete both nodes
    if (t1.inputs[0].shape == t2.outputs[0].shape):
        graph.deleteNode(t1)
        graph.deleteNode(t2)
        graph.cleanup().toposort()
        return ctxt, graph
    # Net the transpose
    else:
        p1 = t1.attrs['perm']
        p2 = t2.attrs['perm']
        newPerm = [p1[idx] for idx in p2]
        
    _inputs = list(t1.inputs)
    _outputs = list(t2.outputs)

    newTrans = gs.Node(op='Transpose', name=name, attrs={"perm": newPerm})
    graph.replaceInsertNode(_inputs, _outputs, newTrans)
    graph.cleanup().toposort()
    return ctxt, graph
    
class TransposeOptPass(ReplaceSequentialPatternPass):
    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name='input_1')
        output = graph.layer(inputs=[_input], outputs=['t1_out'], op='Transpose', name='t1')
        output = graph.layer(inputs=output, outputs=['t2_out'], op='Transpose', name='t2')
        graph.outputs.append(output)
        graph.inputs.append(_input)
    
        name = f"_MERGE_TRANSPOSES_PASS"
        super().__init__(graph, merge_transposes_fun, name)    
