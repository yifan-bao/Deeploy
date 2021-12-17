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
from typing import List, Callable
import copy

from templates import *
from parserTypes import NodeMapper, _mangleVariableName, _mangleParameterName, NetworkContext, NetworkBuffer, GlobalBuffer

class ONNXLayer():
    
    def __init__(self, maps : List[NodeMapper]):
        self.maps = maps
        self.mapper = None
        self.node = None

    def __call__(self, node: gs.ir.node.Node):
        _copy = copy.deepcopy(self)
        _copy.node = node
        return _copy

    # Does not copy the node, so every node in the graph is kept as reference
    # Also work around the fact that NodeMappers' templates are not deepcopyable
    def __deepcopy__(self, memo):
        _copy = type(self)([])
        memo[id(self)] = _copy
        _copy.maps = copy.deepcopy(self.maps, memo)
        _copy.mapper = copy.deepcopy(self.mapper, memo)
        _copy.node = self.node

        return _copy
    
    # Call this, DO NOT override! -> This should assert that all variables required are in the node!
    def parse(self, ctxt: NetworkContext, typeInfer: Callable) -> (NetworkContext, bool):

        # iterate through all possible mappings and return the first that works
        for mapper in self.maps:
            newCtxt = ctxt.copy()
            newCtxt, ret = mapper.parse(newCtxt, self.node, typeInfer)
            if ret:
                self.mapper = mapper
                return newCtxt, True
            
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
    
