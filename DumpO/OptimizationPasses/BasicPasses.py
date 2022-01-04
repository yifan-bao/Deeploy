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
    
class TransposeMergePass(ReplaceSequentialPatternPass):
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

def const_opt_transposes_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    t1 = matched_nodes[0]

    if isinstance(t1.inputs[0], gs.Constant):
        t1.inputs[0].values = np.transpose(t1.inputs[0].values, t1.attrs['perm'])
        graph.deleteNode(t1)
    
    return ctxt, graph
    
class TransposeConstOptPass(ReplaceSequentialPatternPass):
    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name='input_1')
        output = graph.layer(inputs=[_input], outputs=['t1_out'], op='Transpose', name='t1')
        graph.outputs.append(output)
        graph.inputs.append(_input)
    
        name = f"_CONST_OPT_TRANSPOSES_PASS"
        super().__init__(graph, const_opt_transposes_fun, name)    

def merge_requant_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    attrs = {}
    rqs1 = matched_nodes[0]
    rqs2 = matched_nodes[1]

    div1 = rqs1.attrs['div'].values
    div2 = rqs2.attrs['div'].values
    newDiv = max(div1,div2)
    minDiv = min(div1,div2)
    nLevels = max(rqs1.attrs['n_levels'].values,rqs2.attrs['n_levels'].values)
    signed = max(rqs1.attrs['signed'].values,rqs2.attrs['signed'].values)

    attrs['div'] = gs.Constant(name='div', values=newDiv)
    attrs['n_levels'] = gs.Constant(name='n_levels',values=nLevels)
    attrs['signed'] = gs.Constant(name='signed', values=signed)
    
    if isinstance(rqs1.inputs[1], gs.Constant) and isinstance(rqs1.inputs[2], gs.Constant) and \
       isinstance(rqs2.inputs[1], gs.Constant) and isinstance(rqs2.inputs[2], gs.Constant):
        mul1 = rqs1.inputs[1].values
        mul2 = rqs2.inputs[1].values
        add1 = rqs1.inputs[2].values
        add2 = rqs2.inputs[2].values

        newMul = (mul1*mul2)
        newAdd = (add1*mul2) + (div1*add2)

        newMul = gs.Constant(name=rqs1.name+name+'_mul',values = np.array(np.round(newMul / minDiv)))
        newAdd = gs.Constant(name=rqs1.name+name+'_add',values = np.array(np.round(newAdd / minDiv)))
        
        _inputs = [rqs1.inputs[0], newMul, newAdd]
        _outputs = rqs2.outputs
        newTrans = gs.Node(op='RequantShift', name=name, attrs=attrs)
        graph.replaceInsertNode(_inputs, _outputs, newTrans)
        return ctxt, graph
    else:    
        return ctxt, graph
    
class MergeRequantPass(ReplaceSequentialPatternPass):
    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name='input_1')
        output = graph.layer(inputs=[_input], outputs=['r1_out'], op='RequantShift', name='r1')
        output = graph.layer(inputs=output, outputs=['r2_out'], op='RequantShift', name='r2')
        graph.outputs.append(output)
        graph.inputs.append(_input)
    
        name = f"_OPT_RQS_PASS"
        super().__init__(graph, merge_requant_fun, name)    

def propagate_requant_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):

    ctxt = ctxt.copy()
    
    matched_nodes = [m for k, m in match.nodes_map.items()]
    attrs = {}
    add = matched_nodes[0]
    rqs = matched_nodes[1]

    inputNode1 = add.inputs[0]
    inputNode2 = add.inputs[1]
    
    newAdd1 = gs.Constant(name=name+'_rqs1_add', values=rqs.inputs[2].values)
    newAdd2 = gs.Constant(name=name+'_rqs2_add', values=rqs.inputs[2].values)
    newMul1 = gs.Constant(name=name+'_rqs1_mul', values=rqs.inputs[1].values)
    newMul2 = gs.Constant(name=name+'_rqs2_mul', values=rqs.inputs[1].values)

    newAddInput1 = gs.Variable(name+'_add_in_1', dtype=np.float32, shape=inputNode1.shape)
    newAddInput2 = gs.Variable(name+'_add_in_2', dtype=np.float32, shape=inputNode2.shape)

    node1 = ctxt.VariableBuffer().fromNode(newAddInput1, rqs.attrs['n_levels'])
    node2 = ctxt.VariableBuffer().fromNode(newAddInput2, rqs.attrs['n_levels'])
    
    ctxt.add(node1,'local')
    ctxt.add(node2,'local')
    
    newRQS1 = gs.Node(op='RequantShift', name=name+'_rqs1', attrs=rqs.attrs, inputs=[inputNode1, newMul1, newAdd1], outputs= [newAddInput1])
    newRQS2 = gs.Node(op='RequantShift', name=name+'_rqs2', attrs=rqs.attrs, inputs=[inputNode2, newMul2, newAdd2], outputs= [newAddInput2])

    graph.nodes.append(newRQS1)
    graph.nodes.append(newRQS2)

    add.inputs = [newAddInput1, newAddInput2]
    graph.deleteNode(rqs)

    return ctxt, graph
    
class PropagateRequantThroughAddPass(ReplaceSequentialPatternPass):
    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name='input_1')
        _input2 = gs.Variable(name='input_2')
        output = graph.layer(inputs=[_input, _input2], outputs=['add_out'], op='Add', name='add1')
        output = graph.layer(inputs=output, outputs=['r1_out'], op='RequantShift', name='r1')
        graph.outputs.append(output)
        graph.inputs = [_input, _input2]
    
        name = f"_OPT_ADD_RQS_PASS"
        super().__init__(graph, propagate_requant_fun, name)    

def extract_padding_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):

    ctxt = ctxt.copy()
    
    matched_nodes = [m for k, m in match.nodes_map.items()]
    attrs = {}
    conv = matched_nodes[0]
    if 'pads' in conv.attrs and np.sum(conv.attrs['pads'])>1:
        pads = copy.deepcopy(conv.attrs['pads'])
        shape = copy.deepcopy(conv.inputs[0].shape)
        newPads = np.zeros(2*len(shape))
        assert len(shape)-2 == len(pads)/2, "Conv padding dims do not match!"
        newShape = shape

        beginPads = pads[0:len(pads)//2]
        endPads = pads[len(pads)//2:]
        for idx, i in enumerate(beginPads):
            newShape[2+idx] = newShape[2+idx] + i
            newPads[2+idx] = i

        for idx, i in enumerate(endPads):
            newShape[2+idx] = newShape[2+idx] + i
            newPads[len(newPads)//2+2+idx] = i
            
        newConvInput = gs.Variable(name+'_padded_input', dtype=np.float32, shape=newShape)
        ctxt.add(ctxt.VariableBuffer().fromNode(newConvInput, ctxt.lookup(conv.inputs[0].name).nLevels))
        #valConst = gs.Constant('value', np.array(0))
        conv.attrs['pads'] = [0 for pad in conv.attrs['pads']]
        newPad = gs.Node(op='Pad', name=name+'_pad', attrs={'pads': newPads, 'mode': 'constant', 'value': 0}, inputs=[conv.inputs[0]], outputs= [newConvInput])
        
        conv.inputs[0] = newConvInput
        graph.nodes.append(newPad)
        graph.cleanup().toposort()
        #import IPython; IPython.embed()
        
    return ctxt, graph

class ExtractPaddingFromConvPass(ReplaceSequentialPatternPass):
    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name='input_1')
        output = graph.layer(inputs=[_input], outputs=['conv_out'], op='Conv', name='conv1')
        graph.outputs.append(output)
        graph.inputs = [_input]
    
        name = f"_EXTRACT_CONV_PASS"
        super().__init__(graph, extract_padding_fun, name)    

class ExtractPaddingFromPoolPass(ReplaceSequentialPatternPass):
    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name='input_1')
        output = graph.layer(inputs=[_input], outputs=['pool_out'], op='MaxPool', name='maxpool1')
        graph.outputs.append(output)
        graph.inputs = [_input]
    
        name = f"_EXTRACT_POOL_PASS"
        super().__init__(graph, extract_padding_fun, name)    
        
def merge_rqs_add_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    attrs = {}
    add = matched_nodes[0]
    rqs = matched_nodes[1]

    if (isinstance(add.inputs[0], gs.Constant) or isinstance(add.inputs[1], gs.Constant)) and isinstance(rqs.inputs[2], gs.Constant):
        if isinstance(add.inputs[0], gs.Constant):
            idx = 1 # Non-constant idx
            constantTensor = add.inputs[0]
        else:
            idx = 0 # non-constant idx
            constantTensor = add.inputs[1]
        if constantTensor.values.shape != tuple(add.inputs[idx].shape):
            rqs.inputs[2].values = (rqs.inputs[1].values*constantTensor.values) + rqs.inputs[2].values
            add.inputs[(idx+1)%2].values = add.inputs[(idx+1)%2].values * 0
            rqs.inputs[0] = add.inputs[idx]
        return ctxt, graph
    else:    
        return ctxt, graph
        
class MergeConstAddAndRequantPass(ReplaceSequentialPatternPass):
    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name='input_1')
        output = graph.layer(inputs=[_input], outputs=['add_out'], op='Add', name='add1')
        output = graph.layer(inputs=output, outputs=['rqs_out'], op='RequantShift', name='rqs1')
        graph.outputs.append(output)
        graph.inputs = [_input]
    
        name = f"_MERGE_RQS_ADD_PASS"
        super().__init__(graph, merge_rqs_add_fun, name)    
