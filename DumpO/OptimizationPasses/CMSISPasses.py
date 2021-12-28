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
import numpy as np
import onnx_graphsurgeon as gs

from DumpO.DumpOTypes import *
from DumpO.Layers.BasicLayers import *
from DumpO.OptimizationPasses.PassClasses import *

def merge_conv_rq_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    conv = matched_nodes[0]
    rqs = matched_nodes[1]

    totalShift = 31-np.log2(rqs.attrs['div'].values)
    
    rqs.inputs[-1].values = np.round(rqs.inputs[-1].values / rqs.inputs[-2].values) # normalize add

    # Reweight multiplicators:
    # Get maximum:
    maxMult = rqs.inputs[1].values.max()
    # Get maximum shift possible:
    MultShift = min(totalShift, np.floor(np.log2(2**31 - rqs.inputs[1].values.max())))
    # get remaining shift:
    remainingShift = totalShift - MultShift

    # shift mult:
    rqs.inputs[1].values = rqs.inputs[1].values * 2**MultShift
    shiftNode = gs.Constant(f'{conv.name}_shift', np.array(remainingShift))
    # rqs.inputs[-1].values = np.round(rqs.inputs[-1].values / rqs.inputs[-2].values) # normalize add
    # #import IPython; IPython.embed()
    # shiftNode = gs.Constant(f'{conv.name}_shift', np.array((31-np.log2(rqs.attrs['div'].values),)))

    shiftNode = gs.Constant(f'{conv.name}_shift', np.array(remainingShift))
    _inputs = list(conv.inputs) + list(rqs.inputs[1:]) + [shiftNode]
    
    _outputs = rqs.outputs

    rqsConv = gs.Node(op='RequantizedConv', name=name, attrs={**conv.attrs, **rqs.attrs})
    graph.replaceInsertNode(_inputs, _outputs, rqsConv)

    return ctxt, graph
    
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

def merge_gemm_rq_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    gemm = matched_nodes[0]
    rqs = matched_nodes[1]

    totalShift = 31-np.log2(rqs.attrs['div'].values)
    
    rqs.inputs[-1].values = np.round(rqs.inputs[-1].values / rqs.inputs[-2].values) # normalize add

    # Reweight multiplicators:
    # Get maximum:
    maxMult = rqs.inputs[1].values.max()
    # Get maximum shift possible:
    MultShift = min(totalShift, np.floor(np.log2(2**31 - rqs.inputs[1].values.max())))
    # get remaining shift:
    remainingShift = totalShift - MultShift

    # shift mult:
    rqs.inputs[1].values = rqs.inputs[1].values * 2**MultShift
    shiftNode = gs.Constant(f'{gemm.name}_shift', np.array(remainingShift))
    # rqs.inputs[-1].values = np.round(rqs.inputs[-1].values / rqs.inputs[-2].values) # normalize add
    # #import IPython; IPython.embed()
    # shiftNode = gs.Constant(f'{gemm.name}_shift', np.array((31-np.log2(rqs.attrs['div'].values),)))

    _inputs = list(gemm.inputs) + list(rqs.inputs[-1:])
    _outputs = rqs.outputs
    attrs = {**gemm.attrs, **rqs.attrs}
    attrs['shift']=gs.Constant(name='shift', values= np.array(remainingShift))
    attrs['mul']=gs.Constant(name='mul',values = np.array(rqs.inputs[1].values))
    rqsGemm = gs.Node(op='RequantizedGemm', name=name, attrs=attrs)
    graph.replaceInsertNode(_inputs, _outputs, rqsGemm)

    return ctxt, graph
    
class GEMMRequantMergePass(ReplaceSequentialPatternPass):
    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name='input_1')
        output = graph.layer(inputs=[_input], outputs=['gemm_out'], op='Gemm', name='gemm')
        output = graph.layer(inputs=output, outputs=['rqs'], op='RequantShift', name='rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)
    
        name = f"_MERGE_GEMMRQ_PASS"
        super().__init__(graph, merge_gemm_rq_fun, name)

class MatMulRequantMergePass(ReplaceSequentialPatternPass):
    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name='input_1')
        output = graph.layer(inputs=[_input], outputs=['gemm_out'], op='MatMul', name='gemm')
        output = graph.layer(inputs=output, outputs=['rqs'], op='RequantShift', name='rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)
    
        name = f"_MERGE_GEMMRQ_PASS"
        super().__init__(graph, merge_gemm_rq_fun, name)    
