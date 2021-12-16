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
        self.maps = maps
        # Assign the first mapper that works
        #import IPython; IPython.embed()
        self.mapper = None
        
    # Call this, DO NOT override! -> This should assert that all variables required are in the node!
    def parse(self, ctxt: NetworkContext) -> (NetworkContext, bool):

        # iterate through all possible mappings and return the first that works
        for mapping in self.maps:
            newCtxt = ctxt.copy()
            try:
                mapper = mapping()
                newCtxt, ret = mapper.parse(newCtxt, self.node)
                if ret:
                    self.mapper = mapper
                    return newCtxt, True
            except Exception as e:
                pass
            
        # If none worked, throw exception
        raise RuntimeError(f'Did not find adequate mapping for node {self.node.name}!')
        
    # Do not override unless you know what you're doin - this generates code + buffer allocation / de-allocation
    # parseIO has to be called in advance!
    def generate(self, ctxt: NetworkContext) -> (NetworkContext, List[str]):

        outputs = [node for node in self.node.outputs]
        inputs = [node for node in self.node.inputs]

        outputNames = [_mangleVariableName(node.name) for node in outputs]
        inputNames = [_mangleVariableName(node.name) for node in inputs]
        
        alloc = ctxt.allocLocal(self.node.name, outputNames)
        call = self.mapper.generate()
        dealloc = ctxt.freeLocal(self.node.name, inputNames)

        generated_code = [alloc, call, dealloc]
        return (ctxt, generated_code)

class ReshapeLayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)

    def generate(self, ctxt: NetworkContext) -> (NetworkContext, List[str]):        
        outputs = [node for node in self.node.outputs]
        inputs = [node for node in self.node.inputs]
        
        outputNames = [_mangleVariableName(node.name) for node in outputs]
        inputNames = [_mangleVariableName(node.name) for node in inputs]
        
        alloc = ctxt.allocLocal(self.node.name, outputNames)
        call = self.mapper.generate()
        dealloc = ctxt.freeLocal(self.node.name, inputNames)
        
        return (ctxt, [call])
        
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

class TransposeLayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)

class GatherLayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)

class MHSALayer(ONNXLayer):
    def __init__(self, node: gs.ir.node.Node, maps : List[NodeMapper]):
        super().__init__(node, maps)
