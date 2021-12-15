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
from typing import List, Callable, Iterable, Union, Tuple
import onnx
import onnx_graphsurgeon as gs

from templates import AllocateTemplate, FreeTemplate

def _mangleVariableName(name:str) -> str:
    return '__QL_BUFFER_' + name

def _mangleParameterName(nodeName:str, parameterName:str) -> str:
    return '__QL_PARAMETER_' + nodeName + '_' + parameterName

class NetworkBuffer():
    def __init__(self, name: str, shape, type: str):
        self.name = name
        self.shape = shape
        self.type = type
        self._users = []

    # Allocation code. Choose your Template! Might have to override for GlobalBuffer, too, depending on memory hierarchy
    def alloc(self) -> str:
        return AllocateTemplate.referenceTemplate.render(type = self.type, name = self.name, size = np.prod(self.shape))
        
    # Deallocation code. Choose your Template! Might have to override for GlobalBuffer, too, depending on memory hierarchy
    def dealloc(self) -> str:
        return FreeTemplate.referenceTemplate.render(name = self.name)

    def fromNode(node: gs.ir.node.Node, outType:str):
        return(
            NetworkBuffer(
                name = _mangleVariableName(node.name),
                shape = node.shape,
                type = outType
            )
        )
    
class GlobalBuffer(NetworkBuffer):
    def __init__(self, name, shape, type, values):
        super().__init__(name,shape,type)
        self.values = values

    def fromNetworkBuffer(buffer, values):
        gb = GlobalBuffer(name = buffer.name,
                          type = buffer.type,
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

        nb = NetworkBuffer(_mangleVariableName(node.outputs[0].name), node.outputs[0].shape, 'int8_t')
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

    
class NodeParser():
    def __init__(self):
        self.parserDict = {}

    # Change this
    def nodeParse(self, node) -> bool:
        return True

    # Change this
    def nodeCtxtParse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):
        return ctxt, True

    # Don't touch this
    def parse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):
        self.parserDict = {}
        ret1 = self.nodeParse(node)
        if ret1:
            ctxt, ret2 = self.nodeCtxtParse(ctxt, node)
            return (ctxt, ret1 and ret2)
        else:
            return ctxt, False

# Don't change anything here!
class NodeMapper():
    def __init__(self, parser: NodeParser, template: mako.template.Template):
        self.parser = parser()
        self.template = template
        self.parserDict = {}

    def checkCompat(self, node: gs.ir.node.Node) -> bool:
        OGdict = self.parser.parserDict
        ret = self.parser.nodeParse(node)
        self.parser.parserDict = OGdict
        return ret
    
    def parse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):
        return self.parser.parse(ctxt, node)

    def generate(self) -> List[str]:
        return self.template.render(**self.parser.parserDict)
    
class NetworkContainer():
    def __init__(self, graph: gs.Graph, layerOpMapping, scheduler: Callable = lambda x: x):
        self.graph = graph
        self.scheduler = scheduler
        self.ctxt = NetworkContext()
        self.layerBinding = []
        self.parsed = False
        self.layerOpMapping = layerOpMapping
        self.layerOpMapping['Constant'] = lambda x: self.ctxt.hoistConstant(x)

    def _createGlobalContext(self, ctxt: NetworkContext, graph):

        ctxt = ctxt.copy()

        for node in graph.inputs:
            data_name = _mangleVariableName(node.name)
            data_size = node.shape
            data_type = 'int8_t'
            ctxt.add(NetworkBuffer(data_name, data_size, data_type), 'global')

        for node in graph.outputs:
            data_name = _mangleVariableName(node.name)
            data_size = node.shape
            data_type = 'int8_t'
            ctxt.add(NetworkBuffer(data_name, data_size, data_type), 'global')

        return ctxt
        
    def parse(self) -> bool:

        # Reset context
        self.ctxt = self._createGlobalContext(NetworkContext(), self.graph)
        
        # Create schedule, binding, then parse resulting program for correctness
        # Create schedule
        
        for i in self.scheduler(self.graph.nodes):
            
            # Create binding
            assert i.op in list(self.layerOpMapping.keys()), "Layer not in layer dict!"
            layer = self.layerOpMapping[i.op](i)
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
    
    def generate(self) -> str:
        if not self.parsed:
            raise ValueError('You need to parse the network before generating code!')
        
        callStack = ''
        #import IPython; IPython.embed()
        for name, node in self.layerBinding:
            self.ctxt, code = node.generate(self.ctxt)
            for substr in code:
                print(substr)
                #callStack += substr + '\n'

        return callStack
        

    
