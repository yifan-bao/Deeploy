# ----------------------------------------------------------------------
#
# File: onnxLayers.py
#
# Last edited: 13.12.2021        
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

from mako.template import Template
import onnx_graphsurgeon as gs
import math
import numpy as np
from typing import List

from templates import *
from parserTypes import NodeMapper, NetworkContext, _mangleVariableName, _mangleParameterName, NetworkBuffer, GlobalBuffer

class ONNXLayer():
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        self.node = node

        # Assign the first mapper that works
        #import IPython; IPython.embed()
        
        possibleMappings = [i().checkCompat(node) for i in maps]
        mappingIdxs = [idx for idx,x in enumerate(possibleMappings) if x == True]
        
        if len(mappingIdxs) == 0:
            raise RuntimeError(f'Did not find adequate mapping for node {node.name}!')
        elif len(mappingIdxs) > 1:
            print(f'[WARNING]: More than one possible mapping found for {node.name}! Choosing first match.')
            
        self.mapper = maps[mappingIdxs[0]]()
    
    # Call this, DO NOT override! -> This should assert that all variables required are in the node!
    def parse(self, ctxt: NetworkContext) -> NetworkContext:
        ctxt = ctxt.copy()
        # SCHEREMO: REALLY UGLY HACK - BITWIDTH SHOULD BE IN THE GRAPH
        ctxt = self.parseIO(ctxt, ['int8_t'])
        ctxt = self.mapper.parse(ctxt, self.node)
        return ctxt
    
    # Don't override this, it checks for consistency (all inputs are available, no outputs are defined)
    # Also hoists inputs that are parameters
    def parseIO(self, ctxt: NetworkContext, outTypes:List[str]):
        ctxt = ctxt.copy()
        data_in_buffers = []
        data_out_buffers = []
        for inputNode in self.node.inputs:
            data_in = _mangleVariableName(inputNode.name)
            if type(inputNode) == gs.ir.tensor.Constant:
                localBuffer = NetworkBuffer.fromNode(inputNode, 'int8_t')
                globalBuffer = GlobalBuffer.fromNetworkBuffer(localBuffer, values=inputNode.values)
                ctxt.add(globalBuffer, 'global')
            else:
                localBuffer = ctxt.lookup(data_in)
                ctxt.addUser(data_in, self.node.name)
            data_in_buffers.append(localBuffer.name)
        for outputNode, outtype in zip(self.node.outputs,outTypes):
            data_out_name = _mangleVariableName(outputNode.name)
            data_out_size = outputNode.shape
            data_out_type = outtype
            if data_out_name not in ctxt.globalObjects.keys():
                localBuffer = NetworkBuffer(data_out_name, data_out_size, data_out_type)
                ctxt.add(localBuffer, 'local')
            else:
                localBuffer = ctxt.lookup(data_out_name)
            data_out_buffers.append(localBuffer.name)

        self.data_in = data_in_buffers
        self.data_out = data_out_buffers
        
        return ctxt

    # Do not override - this generates code + buffer allocation / de-allocation
    # parseIO has to be called in advance!
    def generate(self, ctxt: NetworkContext) -> (NetworkContext, List[str]):
        alloc = ctxt.allocLocal(self.node.name, self.data_out)
        call = self.mapper.generate()
        dealloc = ctxt.freeLocal(self.node.name, self.data_in)

        generated_code = [alloc, call, dealloc]
        return (ctxt, generated_code)
    
class iGELULayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)
        
class RequantShiftLayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)
        
class AddLayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)

class GEMMLayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)
    
class ConvLayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)
        
class iLayerNormLayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)

class ReshapeLayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)

class TransposeLayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)

class GatherLayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)

class MHSALayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)
