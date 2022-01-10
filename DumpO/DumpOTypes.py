# ----------------------------------------------------------------------
#
# File: DumpOTypes.py
#
# Last edited: 17.12.2021        
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

import re
import copy
import mako
from mako.template import Template
import numpy as np
from typing import List, Callable, Iterable, Union, Tuple, Dict, Callable
import onnx
import onnx_graphsurgeon as gs
import math
from enum import Enum
from onnx.external_data_helper import convert_model_to_external_data
from collections import OrderedDict

class VariableBuffer():
    def __init__(self, name: str = '', shape = [1], nLevels: int = 1):
        self.name = name
        self.shape = shape
        self.nLevels = nLevels

        # Do not override - Should be written in the parsing passes
        self._users = []
        
        # Do not override - Should be written in the typechecking passes
        self._type = None

        # Do not override - Should be written in the typechecking passes
        self._signed = None
        
        # Do not override - Should be written in the deployment passes
        self._live = False

    def init(self) -> str:
        return ''
        
    # Allocation code. Choose your Template, might want to override aswell!
    def alloc(self) -> str:
        return ''
    
    # Deallocation code. Choose your Template!
    def dealloc(self) -> str:
        return ''
    
    def __str__(self) -> str:
        return f'VariableBuffer: name: {self.name}, type: {self._type}, levels: {self.nLevels}'

    def __repr__(self) -> str:
        return f'VariableBuffer: name: {self.name}, type: {self._type}, levels: {self.nLevels}'
    
    def fromNode(self, node: gs.ir.node.Node, nLevels:int):
        return(
            type(self)(
                name = node.name,
                shape = node.shape if not isinstance(node, gs.Constant) else node.values.shape,
                nLevels = nLevels
            )
        )
    
class ConstantBuffer(VariableBuffer):
    def __init__(self, name : str = '', shape = [1], nLevels : int = 0, values = [0]):
        super().__init__(name, shape, nLevels)
        values = np.asarray(values)
        intArray = values.astype(int)
        assert (np.abs(values - intArray)).max() < 0.001, "Constant value {name} is NOT an integer!"
        self.values = intArray

        # Do not override - ConstantBuffers are assumed to be always live!
        self._live = True

    def init(self) -> str:
        return ''
        
    def __str__(self) -> str:
        return f'ConstantBuffer: name: {self.name}, type: {self._type}, levels: {self.nLevels}'

    def __repr__(self) -> str:
        return f'ConstantBuffer: name: {self.name}, type: {self._type}, levels: {self.nLevels}'
    
    def fromVariableBuffer(self, buffer: VariableBuffer, values):
        return type(self)(name = buffer.name,
                          nLevels = buffer.nLevels,
                          shape = buffer.shape,
                          values = values)

class StructBuffer(VariableBuffer):
    def __init__(self, name: str, structDict: Dict = None):
        super().__init__(name, None, None)
        self.structDict = structDict
        
    def __str__(self) -> str:
        return f'StructBuffer: name: {self.name}, type: {self._type}'

    def __repr__(self) -> str:
        return f'StructBuffer: name: {self.name}, type: {self._type}'
    
class NetworkContext():
    def __init__(self, variableBuffer: VariableBuffer, constantBuffer: ConstantBuffer, structBuffer: StructBuffer, globalObjects = {}, localObjects = {}, name: str = 'DumpONetwork'):
        self.globalObjects = {}
        self.localObjects = {}
        self.VariableBuffer = variableBuffer
        self.ConstantBuffer = constantBuffer
        self.StructBuffer = structBuffer
        self.name = name

    def _mangle(self, name: str) -> str:
        return re.sub('\.','_',self.name) + '_DumpO_BUFFER_'  + re.sub('\.','_',name)
    
    def add(self, obj : VariableBuffer, ctxt = 'local'):
        if ctxt == 'local':
            if obj.name not in self.localObjects.keys():
                self.localObjects[obj.name] = obj
            else:
                raise KeyError(f'Buffername {obj.name} was already in the local context!')
        elif ctxt == 'global':
            if obj.name not in self.globalObjects.keys():
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
            return False

    def is_local(self, name) -> bool:
        if name in self.localObjects.keys():
            return True
        else:
            return False

    def hoistStruct(self, struct: Dict , name: str, _type: str):

        structBuffer = self.StructBuffer(name, struct)
        structBuffer._type = _type
        self.add(structBuffer, 'global')
        
    def hoistConstant(self, node: gs.ir.node.Node, name = '', type = None):

        assert len(node.outputs) <= 1, "Constant has more than one output"

        if name == "":
            name = node.name
            
        # SCHEREMO: This is currently heuristic, but should be annotated in ONNX
        localBuffer = self.VariableBuffer().fromNode(node = node, nLevels = int(node.values.max() - node.values.min()))
        globalBuffer = self.ConstantBuffer().fromVariableBuffer(localBuffer, values=node.values)
        globalBuffer.name = name
        globalBuffer._type = type

        self.add(globalBuffer, 'global')
        
        return None

    def addUser(self, name:str, node):
        _buffer = self.lookup(name)
        if node.name not in _buffer._users:
            _buffer._users.append(node.name)
        if self.is_local(_buffer.name):
            self.localObjects[_buffer.name] = _buffer
        else:
            self.globalObjects[_buffer.name] = _buffer
            
    def annotateType(self, name: str, nLevels: int, _type: Enum, signedness: bool = True):
        obj = self.lookup(name)
        if 2**(_type._value_) < nLevels:
            raise ValueError(f'Tried to annotate {name} with {_type}, but {name} has {nLevels} nLevels!')
        if self.is_global(name):
            self.globalObjects[name]._type = _type
            self.globalObjects[name].nLevels = nLevels
            self.globalObjects[name].signed = bool(signedness)
        elif self.is_local(name):
            self.localObjects[name]._type = _type
            self.localObjects[name].nLevels = nLevels
            self.localObjects[name].signed = bool(signedness)
        else:
            raise KeyError(f'Tried to annotate {name}, but it is in no Context')
        
    def allocLocal(self, nodeName: str, outBuffers: List[str]) -> List[str]:

        allocCode = []
        # We have to allocate the output buffers, unless they are global
        for buffer in outBuffers:
            if self.is_local(buffer):
                nb = copy.deepcopy(self.lookup(buffer))
                
                assert self.localObjects[nb.name]._live == False, "Tried to allocate already live buffer {nb.name}"
                self.localObjects[nb.name]._live = True
                
                nb.name = self._mangle(nb.name)
                allocCode.append(nb.alloc())

                
            elif self.is_global(buffer):
                pass
            else:
                raise KeyError(f'Expected {buffer} to be either a global or local buffer!')
            
        return allocCode
    
    def freeLocal(self, nodeName: str, inBuffers: List[str]):
        allocCode = []
        
        # We have to free the input buffers, unless they are global OR we are not the last user
        for buffer in list(set(inBuffers)):
            if self.is_local(buffer):
                nb = copy.deepcopy(self.lookup(buffer))
                # If we are the last user in the list, we can safely free
                if nodeName == nb._users[-1]:

                    assert self.localObjects[nb.name]._live == True, f'Tried to deallocate already non-live buffer {nb.name}'
                    self.localObjects[nb.name]._live = False
                    
                    nb.name = self._mangle(nb.name)
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
    def parseNode(self, node) -> bool:
        return True

    # Change this
    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node, channels_first: bool = True) -> (NetworkContext, bool):
        return ctxt, True
    
    # Don't override this, it checks for consistency (all inputs are available, no outputs are defined)
    # Also hoists inputs that are parameters
    def parseInputs(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):
        ctxt = ctxt.copy()
        
        data_in_buffers = []
        for inputNode in node.inputs:
            data_in = inputNode.name
            
            # Hoist constant inputs
            if type(inputNode) == gs.ir.tensor.Constant and not ctxt.is_global(data_in):
                ctxt.hoistConstant(inputNode)
            else:
                localBuffer = ctxt.lookup(data_in)
                ctxt.addUser(data_in, node)
                data_in_buffers.append(localBuffer.name)
                
        return ctxt, True

    # Don't touch this 
    def parseOutputs(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):
        newCtxt = ctxt.copy()
        outputNodes = node.outputs
        outputNames = [node.name for node in outputNodes]
        
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = newCtxt.VariableBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = None
                )
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)
                
        return newCtxt, True
    
    # Don't touch this
    def parse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):
        newCtxt = ctxt.copy()
        self.parserDict = {}
        ret1 = self.parseNode(node)
        if ret1:
            newCtxt, ret2 = self.parseInputs(newCtxt, node)
            newCtxt, ret3 = self.parseOutputs(newCtxt, node)
            return (newCtxt, ret1 and ret2 and ret3)
        else:
            return ctxt, False
        
class NodeTypeChecker():
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        self.input_types = input_types
        self.output_types = output_types
        self.typeDict = {}
        
    # Override this. This should compute the nLevels of each output node of the Layer
    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**64 for type in self.output_types]
    
    # Override this. This should compute the signednes of each output node of the Layer
    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [True for type in self.output_types]
    
    # Don't override this. This should check that the input n_levels are appropriate for the kernel
    def typeCheckNodeInputs(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        inputName = [i.name for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]

        #import IPython; IPython.embed()
        
        return all(
            [node.nLevels <= 2**(input_type._value_) for node, input_type in zip(inputs, self.input_types)]
        )

    # Don't override this. This should check that the output n_levels are appropriate for the kernel
    def typeCheckNodeOutputs(self, ctxt: NetworkContext, node: gs.ir.node.Node, parserDict) -> bool:
        newCtxt = ctxt.copy()
        inputName = [i.name for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]
        nLevelsList = self.inferNumLevels(inputs, parserDict)
        return all(
            [nLevels <= 2**(output_type._value_) for nLevels, output_type in zip(nLevelsList, self.output_types)]
        )

    # Don't override this. This should annotate the output node with the correct data type
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, parserDict) -> NetworkContext:
        newCtxt = ctxt.copy()

        outputTypeDict = {}
        
        inputName = [i.name for i in node.inputs]
        inputs = [newCtxt.lookup(name) for name in inputName]

        # numLevels propagation
        nLevelsList = self.inferNumLevels(inputs, parserDict)
        # signedness propagation
        signedness = self.inferSignedness(inputs, parserDict)
        
        outputNodes = node.outputs
        outputNames = [node.name for node in outputNodes]
        
        for name, nLevels, output_type, sign in \
            zip(outputNames, nLevelsList, self.output_types, signedness):
                       
            newCtxt.annotateType(name, nLevels, output_type, sign)
            
        return newCtxt

    # Don't override this. Automated annotation of global buffer
    def typeInferGlobalCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node, typeInfer: Callable) -> NetworkContext:
        ctxt = ctxt.copy()

        for inputNode, _type in zip(node.inputs, self.input_types):
            if isinstance(ctxt.lookup(inputNode.name),ConstantBuffer):
                
                # SPECIAL CASE HANDLING: CONSTANT WAS NOT FOLDED CORRECTLY                
                if len(inputNode.inputs) == 1 and hasattr(inputNode.inputs[0], 'attrs') and 'value' in list(inputNode.inputs[0].attrs.keys()):
                    typeNode = inputNode.inputs[0].attrs['value']
                else:
                    typeNode = inputNode

                _buffer = ctxt.lookup(inputNode.name)

                # Check the actual numLevels
                nLevels = _buffer.values.max() - _buffer.values.min()
                ctxt.annotateType(inputNode.name, nLevels, _type)
        
        return ctxt

    # Don't override this.
    def annotateDict(self, ctxt: NetworkContext, node: gs.ir.node.Node, parserDict: Dict):
        env = [node.name for node in node.inputs + node.outputs]
        for key, value in parserDict.items():
            # check if the referenced buffer is in the environment
            if isinstance(value, str) and value in env:
                _buffer = ctxt.lookup(value)
                self.typeDict[key + '_type'] = _buffer._type
                    
    # Don't override this. Automated type checking
    def typeCheck(self, ctxt: NetworkContext, node: gs.ir.node.Node, typeInfer: Callable, parserDict) -> (NetworkContext, bool):
        newCtxt = ctxt.copy()
        if self.typeCheckNodeInputs(newCtxt, node):
            newCtxt = self.typeInferGlobalCtxt(newCtxt, node, typeInfer)
            if self.typeCheckNodeOutputs(newCtxt, node, parserDict):
                newCtxt = self.typeInferOutput(newCtxt, node, parserDict)
                self.annotateDict(newCtxt, node, parserDict)
                return (newCtxt, True)
            else:
                return ctxt, False
        else:
            return ctxt, False

class NodeTemplate():
    def __init__(self, templateStr):
        self.template = Template(templateStr)
        
    #Override this. Reports internal size of the template (buffer size allocated in template) to the tool
    def internalSize(self) -> int:
        return 0
    
    # Don't override this
    def __deepcopy__(self, memo):
        _copy = type(self)(self.template._source)
        memo[id(self)] = _copy

        return _copy

    # Don't override this
    def generate(self, **nodeRep) -> str:
        #print(kwargs)
        try:
            return self.template.render(**nodeRep)
        except:
            print(nodeRep)
            print(mako.exceptions.text_error_template().render())
            raise KeyError("Template failed!")
        
class NodeBinding():
    def __init__(self, typeChecker: NodeTypeChecker, template: NodeTemplate):
        self.typeChecker = typeChecker
        self.template = template

    # Don't override this. This should annotate the output node with the correct data type
    def bind(self, ctxt: NetworkContext, node:gs.ir.node.Node, typeInfer: Callable, parserDict) -> (NetworkContext, bool):
        newCtxt = ctxt.copy()
        newCtxt, ret = self.typeChecker.typeCheck(newCtxt, node, typeInfer, parserDict)
        if ret:
            return (newCtxt, True)
        else:
            return (ctxt, False)
    
    def generate(self, ctxt: NetworkContext, parserDict) -> List[str]:

        parseDict = {}
        
        for key, value in parserDict.items():
            if type(value) == str and (ctxt.is_local(value) or ctxt.is_global(value)):
                parseDict[key] = ctxt._mangle(value)
            else:
                parseDict[key] = value
                
        return [self.template.generate(**{**parseDict, **self.typeChecker.typeDict})]

        
# Don't change anything here!
class NodeMapper():
    def __init__(self, parser: NodeParser, bindings: List[NodeBinding]):
        self.parser = parser
        self.bindings = bindings
        
        self.binder = None
        self.bound = False
        
    # Don't override this. Parses the networks with the correct data type
    def parse(self, ctxt: NetworkContext, node: gs.ir.node.Node, channels_first: bool) -> (NetworkContext, bool):
        hoistedCtxt, parseable = self.parser.parse(ctxt, node)
        if parseable:
            newCtxt, ret = self.parser.parseNodeCtxt(hoistedCtxt, node, channels_first)
            return (newCtxt, ret)
        else:
            return ctxt, False

    # Don't override this. This should annotate the output node with the correct data type
    # SCHEREMO: Currently simply binds the first viable binding
    def bind(self, ctxt: NetworkContext, node:gs.ir.node.Node, typeInfer: Callable) -> (NetworkContext, bool):
        for binder in self.bindings:
            newCtxt = ctxt.copy()
            newCtxt, ret = binder.bind(newCtxt, node, typeInfer, self.parser.parserDict)
            if ret:
                self.binder = binder
                self.bound = True
                return (newCtxt, True)
            
        return (ctxt, False)
        
    def generate(self, ctxt: NetworkContext) -> List[str]:
        if not self.bound:
            raise RuntimeError("Bind layer before generating code!")
        return self.binder.generate(ctxt, self.parser.parserDict)

class ONNXLayer():
    
    def __init__(self, maps : List[NodeMapper]):
        self.maps = maps
        
        self.mapper = None
        self.node = None

    # Override this for broadcasting support
    # Returns a tuple of new, broadcasted inputShapes and outputShapes
    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict: Dict, channels_first: bool) -> (List[np.shape], List[np.shape]):
        return (inputShapes, outputShapes)
        
    def broadcast(self, ctxt: NetworkContext, channels_first: bool = True) -> (NetworkContext):
        ctxt = ctxt.copy()
        
        inputShapes = [ctxt.lookup(node.name).shape for node in self.node.inputs]
        outputShapes = [ctxt.lookup(node.name).shape for node in self.node.outputs]

        newInputShapes, newOutputShapes = self.computeShapes(inputShapes, outputShapes, self.mapper.parser.parserDict, channels_first)

        #import IPython; IPython.embed()
        
        for node, newShape, oldShape in zip(self.node.inputs + self.node.outputs, newInputShapes + newOutputShapes, inputShapes + outputShapes):
            #if newShape != oldShape:
            if True:
                if ctxt.is_local(node.name):
                    ctxt.localObjects[node.name].shape = newShape
                elif ctxt.is_global(node.name):
                    ctxt.globalObjects[node.name].shape = newShape
                    if isinstance(ctxt.globalObjects[node.name], ConstantBuffer):

                        # If the number of elements is equal, reshape
                        if np.prod(ctxt.globalObjects[node.name].values.shape) == np.prod(newShape):
                            ctxt.globalObjects[node.name].values.reshape(newShape)
                        # The number of elements SHOULD be lower, and we broadcast
                        else:
                            try:
                                ctxt.globalObjects[node.name].values = np.broadcast_to(ctxt.globalObjects[node.name].values, newShape)
                            except:
                                import IPython; IPython.embed()

                else:
                    raise KeyError(f'Expected node {node.name} to be in context!')

        return ctxt
    
    # Don't override - binds the layer to a node
    def __call__(self, node: gs.ir.node.Node):
        _copy = copy.deepcopy(self)
        _copy.node = node
        return _copy

    # Don't override
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
    def parse(self, ctxt: NetworkContext, channels_first: bool) -> (NetworkContext, bool):
        retCtxt = None
        # iterate through all possible mappings and return the first that works
        for mapper in self.maps:
            newCtxt = ctxt.copy()
            newCtxt, ret = mapper.parse(newCtxt, self.node, channels_first)
            if ret:
                self.mapper = mapper
                return newCtxt, True
            
        # If none worked, throw exception
        raise RuntimeError(f'Did not find adequate mapping for node {self.node.name}!')

    def bind(self, ctxt: NetworkContext, typeInfer: Callable):
        
        newCtxt = ctxt.copy()
        newCtxt, ret = self.mapper.bind(newCtxt, self.node, typeInfer)
        
        if ret:
            return newCtxt, True
        
        # If none worked, throw exception
        raise RuntimeError(f'Did not find adequate binding for node {self.node.name}!')
    
    # Do not override unless you know what you're doin - this generates code + buffer allocation / de-allocation
    # parseIO has to be called in advance!
    def generate(self, ctxt: NetworkContext) -> (NetworkContext, List[str]):

        outputs = [node for node in self.node.outputs]
        inputs = [node for node in self.node.inputs]

        outputNames = [node.name for node in outputs]
        inputNames = [node.name for node in inputs]
        
        alloc = ctxt.allocLocal(self.node.name, outputNames)
        call = self.mapper.generate(ctxt)
        dealloc = ctxt.freeLocal(self.node.name, inputNames)

        generated_code = [alloc, call, dealloc]
        return (ctxt, generated_code)

class NetworkOptimizationPass():
    def __init__(self):
        pass

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> (NetworkContext, gs.Graph) :
        return ctxt, graph
    
class NetworkOptimizer():
    def __init__(self, passes: List[NetworkOptimizationPass]):
        self.passes = passes
        
    def optimize(self, ctxt: NetworkContext, graph: gs.Graph) -> (NetworkContext, gs.Graph):
        for _pass in self.passes:
            ctxt, graph = _pass.apply(ctxt, graph)
            graph.cleanup().toposort()
        return ctxt, graph
    
class DeploymentPlatform():
    def __init__(self, Mapping: Dict[str, ONNXLayer], DataTypes: Enum, TypeInfer: Callable, VariableBuffer: VariableBuffer, ConstantBuffer: ConstantBuffer, StructBuffer: StructBuffer):
        self.Mapping = Mapping
        self.DataTypes = DataTypes
        self.TypeInfer = TypeInfer
        self.VariableBuffer = VariableBuffer
        self.ConstantBuffer = ConstantBuffer
        self.StructBuffer = StructBuffer
    
class NetworkContainer(): 
    def __init__(self, graph: gs.Graph, platform: DeploymentPlatform, scheduler: Callable = lambda x: x, name: str = 'DumpONetwork'):
        self.graph = graph
        self.scheduler = scheduler
        self.ctxt = None
        self.layerBinding = OrderedDict()
        self.parsed = False
        self.Platform = platform
        self.Platform.Mapping['Constant'] = lambda x: self.ctxt.hoistConstant(x.attrs['value'], x.outputs[0].name)

        self.parsed = False
        self.bound = False
        self.worstCaseBufferSize = 0
    
    # Don't override this
    def _createIOBindings(self, ctxt: NetworkContext, graph: gs.Graph):

        ctxt = ctxt.copy()

        for node in graph.inputs:
            data_name = node.name
            data_size = node.shape
            # SCHEREMO: Should be parsed from graph
            data_type = 2**8
            nb = ctxt.VariableBuffer(data_name, data_size, data_type)
            nb._type = self.Platform.DataTypes.int8_t
            ctxt.add(nb, 'global')

        for node in graph.outputs:
            data_name = node.name
            data_size = node.shape
            # SCHEREMO: Should be parsed from graph
            data_type = 2**8
            nb = ctxt.VariableBuffer(data_name, data_size, data_type)
            nb._type = self.Platform.DataTypes.int8_t
            ctxt.add(nb, 'global')

        return ctxt

    # Don't override this
    def broadcast(self, channels_first: bool = True) -> bool:
        
        ctxt = self.ctxt.copy()
        
        for name, layer in self.layerBinding.items():
            ctxt = layer.broadcast(ctxt, channels_first)

        self.ctxt = ctxt
        return True
            
    # Don't override this
    def bind(self) -> bool:
        if not self.parsed:
            raise ValueError('You need to parse the network before binding!')

        # SCHEREMO: Implement backtracking here! Currently tries the cheapest branch only!
        newCtxt = self.ctxt.copy()

        backTrackList = []
        
        NetworkBindSuccess = True
        for name, layer in self.layerBinding.items():
            
            newCtxt, LayerBindSuccess = layer.bind(newCtxt, self.Platform.TypeInfer)
            NetworkBindSuccess = NetworkBindSuccess and LayerBindSuccess
                
        if not NetworkBindSuccess:
            raise RuntimeError(f'Could not find a valid binding for the graph')
        else:
            self.bound = True
            self.ctxt = newCtxt

    def _bindLayers(self):
        # Create schedule, binding, then parse resulting program for correctness
        # Create schedule
        self.layerBinding = OrderedDict()
        for i in self.scheduler(self.graph):
            
            # Create binding
            assert i.op in list(self.Platform.Mapping.keys()), f'Layer {i.op} not in layer dict!'
            layer = self.Platform.Mapping[i.op](i)
            if layer is not None:
                self.layerBinding[layer.node.name] = layer

            
    # Don't override this        
    def parse(self, channels_first=True) -> bool:
        # Reset context
        self.ctxt = NetworkContext(self.Platform.VariableBuffer, self.Platform.ConstantBuffer, self.Platform.StructBuffer, {}, {})
        self.ctxt = self._createIOBindings(self.ctxt, self.graph)

        self._bindLayers()
        
        parseSuccess = True
        for key, node in self.layerBinding.items():
            self.ctxt, parsePass = node.parse(self.ctxt, channels_first)
            parseSuccess = parseSuccess and parsePass

        if parseSuccess:
            self.parsed = True
            return True
        else:
            raise RuntimeError(f'Could not parse the graph!')

    def getWorstCaseBufferSize(self) -> int:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')

        if self.worstCaseBufferSize == 0:
            for _buffer in self.ctxt.localObjects.values():
                assert _buffer._live == False, f'There is a memory leak in buffer {_buffer.name} before the generated forward pass!'

            callStack = ''
            for key, node in self.layerBinding.items():
                self.ctxt, code = node.generate(self.ctxt)

                currentBufferSize = 0
                for _buffer in self.ctxt.localObjects.values():
                    if _buffer._live == True:
                        currentBufferSize += np.prod(_buffer.shape) * _buffer._type._value_//8
                if currentBufferSize > self.worstCaseBufferSize:
                    self.worstCaseBufferSize = currentBufferSize

                for section in code:
                    for substr in section:
                        callStack += substr + '\n'

            for _buffer in self.ctxt.localObjects.values():
                assert _buffer._live == False, f'There is a memory leak in buffer {_buffer.name} in the generated forward pass!'

        return self.worstCaseBufferSize
        
    # Don't override this
    def generateInferenceCode(self) -> str:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')
        
        for _buffer in self.ctxt.localObjects.values():
            assert _buffer._live == False, f'There is a memory leak in buffer {_buffer.name} before the generated forward pass!'
        
        callStack = ''
        for key, node in self.layerBinding.items():
            self.ctxt, code = node.generate(self.ctxt)

            currentBufferSize = 0
            for _buffer in self.ctxt.localObjects.values():
                if _buffer._live == True:
                    currentBufferSize += np.prod(_buffer.shape) * _buffer._type._value_//8
            if currentBufferSize > self.worstCaseBufferSize:
                self.worstCaseBufferSize = currentBufferSize
            
            for section in code:
                for substr in section:
                    callStack += substr + '\n'

        for _buffer in self.ctxt.localObjects.values():
            assert _buffer._live == False, f'There is a memory leak in buffer {_buffer.name} in the generated forward pass!'

        lines = callStack.split('\n')
        lines = [line for line in lines if line.strip()]
        callStack = '\n'.join(lines)
            
        return callStack

        # Don't override this
    def generateInferenceInitializationCode(self) -> str:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')

        ctxt = self.ctxt.copy()
        
        callStack = ''
        for node in ctxt.localObjects.values():
            name = node.name
            node.name = ctxt._mangle(node.name)
            callStack += node.init()
            node.name = name

        lines = callStack.split('\n')
        lines = [line for line in lines if line.strip()]
        callStack = '\n'.join(lines)
        
        return callStack
    
    # Don't override this
    def generateIOBufferInitializationCode(self) -> str:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')

        ctxt = self.ctxt.copy()
        
        callStack = ''
        inputNum = 0
        outputNum = 0
        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, (StructBuffer, ConstantBuffer)):
                name = node.name
                node.name = ctxt._mangle(node.name)
                callStack += "extern " + node.init()
                node.name = name

        lines = callStack.split('\n')
        lines = [line for line in lines if line.strip()]
        callStack = '\n'.join(lines)
                
        return callStack


    # Don't override this
    def generateBufferInitializationCode(self) -> str:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')

        ctxt = self.ctxt.copy()
        
        callStack = ''
        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, StructBuffer):
                name = node.name
                node.name = ctxt._mangle(node.name)
                callStack += node.init()
                node.name = name

        for node in ctxt.globalObjects.values():
            if isinstance(node, StructBuffer):
                name = node.name
                node.name = ctxt._mangle(node.name)
                callStack += node.init()
                node.name = name

        lines = callStack.split('\n')
        lines = [line for line in lines if line.strip()]
        callStack = '\n'.join(lines)
                
        return callStack
    
    def generateBufferAllocationCode(self) -> str:

        ctxt = self.ctxt.copy()
        callStack = ''
        
        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, StructBuffer):
                name = node.name
                node.name = ctxt._mangle(node.name)
                callStack += node.alloc()
                node.name = name

        for node in ctxt.globalObjects.values():
            if isinstance(node, StructBuffer):
                name = node.name
                node.name = ctxt._mangle(node.name)
                callStack += node.alloc()
                node.name = name
                
        lines = callStack.split('\n')
        lines = [line for line in lines if line.strip()]
        callStack = '\n'.join(lines)
                
        return callStack
            
    # Don't override this
    def generateBufferDeAllocationCode(self) -> str:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')

        ctxt = self.ctxt.copy()
        
        callStack = ''
        for node in self.ctxt.globalObjects.values():
            node.name = ctxt._mangle(node.name)
            callStack += node.dealloc() + '\n'

        lines = callStack.split('\n')
        lines = [line for line in lines if line.strip()]
        callStack = '\n'.join(lines)
            
        self.ctxt = ctxt
        return callStack
    
    # Don't override this - Returns parameter size in bytes
    def getParameterSize(self) -> int:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before getting RAM Size!')

        size = 0
        for _buffer in self.ctxt.globalObjects.values():
            # We do not count structs for now, since they are not properly modeled
            if isinstance(_buffer, ConstantBuffer):
                size += int((np.prod(_buffer.shape) * _buffer._type._value_ // 8))

        return size

    # Don't override this - Returns worst case layer and buffering size in bytes
    def getTotalSize(self) -> (str, int):
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before getting RAM Size!')

        return self.getParameterSize() + self.getWorstCaseBufferSize()
