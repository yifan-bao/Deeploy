# ----------------------------------------------------------------------
#
# File: parserTypes.py
#
# Last edited: 14.12.2021        
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

import copy
import mako
import numpy as np
from typing import List, Callable, Iterable, Union, Tuple, Dict
import onnx
import onnx_graphsurgeon as gs
import re
import math
from enum import Enum

from templates import AllocateTemplate, FreeTemplate

def _mangleVariableName(name:str) -> str:
    return '_DumpO_BUFFER__' + re.sub('\.','_',name)

def _mangleParameterName(nodeName:str, parameterName:str) -> str:
    return '_DumpO_PARAMETER__' + re.sub('.','_',nodeName) + '_' + re.sub('.','_',parameterName)

class DeploymentPlatform():
    def __init__(self, Mapping: Dict[str, object], DataTypes: Enum):
        self.Mapping = Mapping
        self.DataTypes = DataTypes

class NetworkBuffer():
    def __init__(self, name: str, shape, nLevels: int):
        self.name = name
        self.shape = shape
        self.nLevels = nLevels

        # Do not override - Should be written in the parsing passes
        self._users = []
        
        # Do not override - Should be written in the typechecking passes
        self._type = None

    # Allocation code. Choose your Template, might want to override aswell!
    def alloc(self) -> str:
        # SCHEREMO: currently signed, but can be infered from n_levels and signed flags    
        return mako.template.Template(AllocateTemplate.referenceTemplate).render(type = self._type._name_, name = self.name, size = np.prod(self.shape))
    
    # Deallocation code. Choose your Template!
    def dealloc(self) -> str:
        return mako.template.Template(FreeTemplate.referenceTemplate).render(name = self.name)
    
    def fromNode(node: gs.ir.node.Node, nLevels:int):
        return(
            NetworkBuffer(
                name = _mangleVariableName(node.name),
                shape = node.shape,
                nLevels = nLevels
            )
        )
    
class GlobalBuffer(NetworkBuffer):
    def __init__(self, name, shape, nLevels, values):
        super().__init__(name,shape,nLevels)
        self.values = values
        
    def fromNetworkBuffer(buffer, values):
        gb = GlobalBuffer(name = buffer.name,
                          nLevels = buffer.nLevels,
                          shape = buffer.shape,
                          values = values)
        return(gb)

class NetworkContext():
    def __init__(self, globalObjects = {}, localObjects = {}):
        self.globalObjects = {}
        self.localObjects = {}
        
    def add(self, obj : NetworkBuffer, ctxt = 'local'):
        if ctxt == 'local':
            if obj.name not in self.localObjects.keys():
                self.localObjects[obj.name] = obj
            else:
                raise KeyError(f'Buffername {obj.name} was already in the local context!')
        elif ctxt == 'global':
            if obj.name not in self.localObjects.keys():
                self.globalObjects[obj.name] = obj
            else:
                raise KeyError(f'Buffername {obj.name} was already in the global context!')
        else:
            raise ValueError("Expected either local or global context")

    def lookup(self, name):
        if name in self.localObjects.keys():
            return self.localObjects[name]
        elif name in self.globalObjects.keys():
            return self.globalObjects[name]
        else:
            raise KeyError(f'Expected key {name} to be in either local of global context!')

    def is_global(self, name) -> bool:
        if name in self.globalObjects.keys():
            return True
        else:
            False

    def is_local(self, name) -> bool:
        if name in self.localObjects.keys():
            return True
        else:
            False

    def hoistConstant(self, node: gs.ir.node.Node):

        assert len(node.outputs) == 1, "Constant has more than one output"

        # SCHEREMO: Constants are hoisted with nLevels = 2**8 - should be read from graph at some point
        nb = NetworkBuffer(_mangleVariableName(node.outputs[0].name), node.outputs[0].shape, 2**8)
        param = GlobalBuffer.fromNetworkBuffer(nb, values=node.attrs['value'].values)
        self.add(param, 'global')

    def hoistParameter(self, buffer: GlobalBuffer):
    
        self.add(buffer, 'global')
    
    def addUser(self, name:str, node):
        _buffer = self.lookup(name)
        if node not in _buffer._users:
            _buffer._users.append(node)
        if _buffer.name in self.localObjects.keys():
            self.localObjects[_buffer.name] = _buffer
        else:
            self.globalObjects[_buffer.name] = _buffer
            
    def allocLocal(self, nodeName: str, outBuffers: List[str]) -> List[str]:

        allocCode = []
        # We have to allocate the output buffers, unless they are global
        for buffer in outBuffers:
            if self.is_local(buffer):
                nb = self.lookup(buffer)
                allocCode.append(nb.alloc())
            elif self.is_global(buffer):
                pass
            else:
                raise KeyError(f'Expected {buffer} to be either a global or local buffer!')
            
        return allocCode
    
    def freeLocal(self, nodeName: str, inBuffers: List[str]):
        allocCode = []
        
        # We have to free the input buffers, unless they are global OR we are not the last user
        for buffer in inBuffers:
            if self.is_local(buffer):
                nb = self.lookup(buffer)
                # If we are the last user in the list, we can safely free
                if nodeName == nb._users[-1]:
                    allocCode.append(nb.dealloc())
            elif self.is_global(buffer):
                pass
            else:
                raise KeyError(f'Expected {buffer} to be either a global or local buffer!')

        return allocCode
            
    def copy(self):
        return copy.deepcopy(self)

# TYPE CHECKERS ARE ASSUMED TO BE STATELESS!
class NodeTypeChecker():
    def __init__(self):
        pass

    # Override this. This should check that the input n_levels are appropriate for the kernel
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        return True

    # Override this. This should check add the output node to the context with the correct n_levels
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        return ctxt, True, 2**32
    
class NodeParser():
    def __init__(self):
        self.parserDict = {}

    # Change this
    def nodeParse(self, node) -> bool:
        return True

    # Change this
    def nodeCtxtParse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):
        return ctxt, True
    
    # Don't override this, it checks for consistency (all inputs are available, no outputs are defined)
    # Also hoists inputs that are parameters
    def parseInputs(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):
        ctxt = ctxt.copy()
        
        data_in_buffers = []
        for inputNode in node.inputs:
            data_in = _mangleVariableName(inputNode.name)
            
            # Hoist constant inputs
            if type(inputNode) == gs.ir.tensor.Constant and not ctxt.is_global(data_in):
                # SCHEREMO: This is currently heuristic, but should be annotated in ONNX
                if inputNode.values.max() < 2**8:
                    localBuffer = NetworkBuffer.fromNode(inputNode, 2**8)
                    globalBuffer = GlobalBuffer.fromNetworkBuffer(localBuffer, values=inputNode.values)
                    globalBuffer._type = 'int8_t'
                    ctxt.add(globalBuffer, 'global')
                    
                elif inputNode.values.max() < 2**16:
                    localBuffer = NetworkBuffer.fromNode(inputNode, 2**16)
                    globalBuffer = GlobalBuffer.fromNetworkBuffer(localBuffer, values=inputNode.values)
                    globalBuffer._type = 'int16_t'
                    ctxt.add(globalBuffer, 'global')
                    
                else:
                    localBuffer = NetworkBuffer.fromNode(inputNode, 2**32)
                    globalBuffer = GlobalBuffer.fromNetworkBuffer(localBuffer, values=inputNode.values)
                    globalBuffer._type = 'int32_t'
                    ctxt.add(globalBuffer, 'global')
            else:
                localBuffer = ctxt.lookup(data_in)
                ctxt.addUser(data_in, node.name)

            data_in_buffers.append(localBuffer.name)
    
        return ctxt, True

    # Don't touch this
    def parse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):
        self.parserDict = {}
        ret1 = self.nodeParse(node)
        if ret1:
            ctxt, ret2 = self.parseInputs(ctxt, node)
            return (ctxt, ret1 and ret2)
        else:
            return ctxt, False

# Don't change anything here!
class NodeMapper():
    def __init__(self, parser: NodeParser, typeChecker: NodeTypeChecker, template: mako.template.Template):
        self.parser = parser
        self.typeChecker = typeChecker
        self.template = template
        self.parserDict = {}

    # Wrokaround because mako templates are not deepcopyable
    def __deepcopy__(self, memo):
        _copy = type(self)(None, None, None)
        memo[id(self)] = _copy
        _copy.parser = copy.deepcopy(self.parser, memo)
        _copy.typeChecker = copy.deepcopy(self.typeChecker, memo)
        _copy.parserDict = copy.deepcopy(self.parserDict, memo)
        _copy.template = type(self.template)(self.template._source)

        return _copy
    
    def parse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):
        hoistedCtxt, parseable = self.parser.parse(ctxt, node)
        if parseable:
            if( self.typeChecker.typeCheckNode(ctxt = hoistedCtxt, node = node)):
                typedCtxt = self.typeChecker.typeInferOutput(hoistedCtxt, node, **self.parser.parserDict)
                return self.parser.nodeCtxtParse(typedCtxt, node)
            else:
                raise ValueError(f'Typechecking for node {node.name} failed')
        else:
            raise ValueError(f'Could not parse node {node.name}')
    
    def generate(self) -> List[str]:
        return [self.template.render(**self.parser.parserDict)]
    
class NetworkContainer():
    def __init__(self, graph: gs.Graph, platform: DeploymentPlatform, scheduler: Callable = lambda x: x):
        self.graph = graph
        self.scheduler = scheduler
        self.ctxt = NetworkContext()
        self.layerBinding = []
        self.parsed = False
        self.Platform = platform
        self.Platform.Mapping['Constant'] = lambda x: self.ctxt.hoistConstant(x)

    def _createIOBindings(self, ctxt: NetworkContext, graph):

        ctxt = ctxt.copy()

        for node in graph.inputs:
            data_name = _mangleVariableName(node.name)
            data_size = node.shape
            # SCHEREMO: Should be parsed from graph
            data_type = 2**8
            nb = NetworkBuffer(data_name, data_size, data_type)
            nb._type = self.Platform.DataTypes.int8_t
            ctxt.add(nb, 'global')

        for node in graph.outputs:
            data_name = _mangleVariableName(node.name)
            data_size = node.shape
            # SCHEREMO: Should be parsed from graph
            data_type = 2**32
            nb = NetworkBuffer(data_name, data_size, data_type)
            nb._type = self.Platform.DataTypes.int32_t
            ctxt.add(nb, 'global')

        return ctxt
        
    def parse(self) -> bool:

        # Reset context
        self.ctxt = self._createIOBindings(NetworkContext(), self.graph)
        
        # Create schedule, binding, then parse resulting program for correctness
        # Create schedule
        
        for i in self.scheduler(self.graph):
            
            # Create binding
            assert i.op in list(self.Platform.Mapping.keys()), "Layer not in layer dict!"
            layer = self.Platform.Mapping[i.op](i)
            if layer is not None:
                self.layerBinding += [(layer.node.name, layer)]

        parseSuccess = True
        for (name, node) in self.layerBinding:
            self.ctxt, parsePass = node.parse(self.ctxt)
            parseSuccess = parseSuccess and parsePass

        if parseSuccess:
            self.parsed = True
            return True
        else:
            self.parsed = False
            return False
    
    def generateInferenceCode(self) -> str:
        if not self.parsed:
            raise ValueError('You need to parse the network before generating code!')
        
        callStack = ''
        for name, node in self.layerBinding:
            self.ctxt, code = node.generate(self.ctxt)
            for section in code:
                for substr in section:
                    callStack += substr + '\n'

        return callStack

    def getParameterSize(self) -> int:
        if not self.parsed:
            raise ValueError('You need to parse the network before getting RAM Size!')
        import IPython; IPython.embed()
