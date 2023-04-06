# ----------------------------------------------------------------------
#
# File: DeeployTypes.py
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
from typing import List, Callable, TypeVar, Tuple, Dict, Callable, Type, Any, Sequence
import onnx
import onnx_graphsurgeon as gs
import math
from enum import Enum
from onnx.external_data_helper import convert_model_to_external_data
from collections import OrderedDict

Shape = TypeVar("Shape", bound = Any)


class VariableBuffer():

    def __init__(self, name: str = '', shape = [1], nLevels: int = 1):
        self.name = name
        self.shape = shape
        self.nLevels = nLevels

        # Do not override - Should be written in the parsing passes
        self._users = []

        # Do not override - Should be written in the typechecking passes
        self._type: Enum = None

        # Do not override - Should be written in the typechecking passes
        self._signed: bool = None

        # Do not override - Should be written in the deployment passes
        self._live = False

        # Do not override - Set in Templates depending on platform
        self._deploy = True

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

    @classmethod
    def fromNode(cls, node: gs.Node, nLevels: int):
        return (cls(name = node.name,
                    shape = node.shape if not isinstance(node, gs.Constant) else node.values.shape,
                    nLevels = nLevels))


class TransientBuffer(VariableBuffer):

    def __init__(self, name: str = '', size = 0):
        self.name = name
        self.size = size

        # Do not override - Should be written in the parsing passes
        self._users = []

        # Do not override - Should be written in the parsing passes
        self._type: Enum = None

        # Do not override - Should be written in the deployment passes
        self._live = False

        # Do not override - Set in Templates depending on platform
        self._deploy = True

    def __str__(self) -> str:
        return f'TransientBuffer: name: {self.name}, size: {self.size}'

    def __repr__(self) -> str:
        return f'TransientBuffer: name: {self.name}, size: {self.size}'

    @classmethod
    def fromVariableBuffer(cls, buffer: VariableBuffer):
        ret = cls(name = buffer.name, size = np.prod(buffer.shape) * buffer._type._value_ // 8)


class ConstantBuffer(VariableBuffer):

    def __init__(self, name: str = '', shape = [1], nLevels: int = 0, values = [0]):
        super().__init__(name, shape, nLevels)
        values = np.asarray(values)
        intArray = values.astype(int)
        assert (np.abs(values - intArray)).max() < 0.001, "Constant value {name} is NOT an integer!"
        self.values = intArray

        # Do not override - SCHEREMO: always assume signed constants!
        self._signed = True

        # Do not override - ConstantBuffers are assumed to be always live!
        self._live = True

    def init(self) -> str:
        return ''

    def __str__(self) -> str:
        return f'ConstantBuffer: name: {self.name}, type: {self._type}, levels: {self.nLevels}'

    def __repr__(self) -> str:
        return f'ConstantBuffer: name: {self.name}, type: {self._type}, levels: {self.nLevels}'

    @classmethod
    def fromVariableBuffer(cls, buffer: VariableBuffer, values):
        ret = cls(name = buffer.name, nLevels = buffer.nLevels, shape = buffer.shape, values = values)

        return ret


class StructBuffer(VariableBuffer):

    def __init__(self, name: str, structDict: Dict = None):
        super().__init__(name, None, None)
        self.structDict = structDict

    def __str__(self) -> str:
        return f'StructBuffer: name: {self.name}, type: {self._type}'

    def __repr__(self) -> str:
        return f'StructBuffer: name: {self.name}, type: {self._type}'


class NetworkContext():

    def __init__(self,
                 variableBuffer: Type[VariableBuffer],
                 constantBuffer: Type[ConstantBuffer],
                 structBuffer: Type[StructBuffer],
                 transientBuffer: Type[ConstantBuffer],
                 globalObjects = {},
                 localObjects = {},
                 name: str = 'DeeployNetwork'):
        self.globalObjects = {}
        self.localObjects = {}
        self.internalObjects = {}
        self.VariableBuffer = variableBuffer
        self.ConstantBuffer = constantBuffer
        self.StructBuffer = structBuffer
        self.TransientBuffer = transientBuffer
        self.name = name

    def _mangle(self, name: str, repr: bool = True) -> str:
        repStr = name
        repStr = re.sub('\.', '_', repStr)
        repStr = re.sub(':', '_', repStr)
        if repr:
            repStr = re.sub('\.', '_', self.name) + '_Deeploy_BUFFER_' + repStr
        return repStr

    def add(self, obj: VariableBuffer, ctxt = 'local', _id = ""):
        if _id != "":
            obj.name = self._mangle(_id + "_" + obj.name, False)

        if ctxt == 'internal':
            if obj.name not in self.internalObjects.keys():
                self.internalObjects[obj.name] = obj
            else:
                raise KeyError(f'Buffername {obj.name} was already in the internal context!')
        elif ctxt == 'local':
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

    def lookup(self, name, _id = ""):
        if _id != "":
            name = self._mangle(_id + "_" + name, False)

        if name in self.localObjects.keys():
            return self.localObjects[name]
        elif name in self.globalObjects.keys():
            return self.globalObjects[name]
        elif name in self.internalObjects.keys():
            return self.internalObjects[name]
        else:
            raise KeyError(f'Expected key {name} to be in either internal, local of global context!')

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

    def is_internal(self, name) -> bool:
        if name in self.internalObjects.keys():
            return True
        else:
            return False

    def hoistTransientBuffer(self, name: str, size: int):

        transientBuffer = self.TransientBuffer(name, size)
        self.add(transientBuffer, 'local')

    def inputs(self) -> List[ConstantBuffer]:
        inputs = []
        for key, value in self.globalObjects.items():
            if not value._users == [] and type(value) is self.VariableBuffer:
                inputs += [value]
        return inputs

    def outputs(self) -> List[ConstantBuffer]:
        outputs = []
        for key, value in self.globalObjects.items():
            if value._users == [] and type(value) is self.VariableBuffer:
                outputs += [value]
        return outputs

    def hoistStruct(self, struct: Dict, name: str, _type: str):

        structBuffer = self.StructBuffer(name, struct)
        structBuffer._type = _type
        self.add(structBuffer, 'global')

    def hoistConstant(self, node: gs.Node, name = '', type = None):

        assert len(node.outputs) <= 1, "Constant has more than one output"

        if name == "":
            name = node.name

        # SCHEREMO: This is currently heuristic, but should be annotated in ONNX
        localBuffer = self.VariableBuffer.fromNode(node = node,
                                                   nLevels = int(node.values.max() - min(node.values.min(), 0)))
        globalBuffer = self.ConstantBuffer.fromVariableBuffer(localBuffer, values = node.values)
        globalBuffer.name = name
        globalBuffer._type = type

        self.add(globalBuffer, 'global')

        return None

    def addUser(self, name: str, node):
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
            self.globalObjects[name]._signed = bool(signedness)
        elif self.is_local(name):
            self.localObjects[name]._type = _type
            self.localObjects[name].nLevels = nLevels
            self.localObjects[name]._signed = bool(signedness)
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

                    assert self.localObjects[
                        nb.name]._live == True, f'Tried to deallocate already non-live buffer {nb.name}'
                    self.localObjects[nb.name]._live = False

                    nb.name = self._mangle(nb.name)
                    allocCode.append(nb.dealloc())

            elif self.is_global(buffer):
                pass
            else:
                raise KeyError(f'Expected {buffer} to be either a global or local buffer!')

        return allocCode

    def copy(self):
        #return copy.deepcopy(self)
        return copy.copy(self)


class NodeParser():

    def __init__(self):
        self.parserDict = {}

    # Change this
    def parseNode(self, node) -> bool:
        return True

    # Change this
    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        return ctxt, True

    # Don't override this, it checks for consistency (all inputs are available, no outputs are defined)
    # Also hoists inputs that are parameters
    def parseInputs(self, ctxt: NetworkContext, node: gs.Node) -> Tuple[NetworkContext, bool]:
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
    def parseOutputs(self, ctxt: NetworkContext, node: gs.Node) -> Tuple[NetworkContext, bool]:
        newCtxt = ctxt.copy()
        outputNodes = node.outputs
        outputNames = [node.name for node in outputNodes]

        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = newCtxt.VariableBuffer(name = name, shape = node.shape, nLevels = None)
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)

        return newCtxt, True

    # Don't touch this
    def parse(self,
              ctxt: NetworkContext,
              node: gs.Node,
              default_channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt = copy.deepcopy(ctxt)
        self.parserDict = {}

        if "channels_first" in node.attrs:
            self.parserDict['channels_first'] = node.attrs['channels_first']
        else:
            self.parserDict['channels_first'] = default_channels_first

        self.parserDict['node_name'] = node.name
        self.parserDict['node_op'] = node.op
        ret1 = self.parseNode(node)
        if ret1:
            newCtxt, ret2 = self.parseInputs(newCtxt, node)
            newCtxt, ret3 = self.parseOutputs(newCtxt, node)
            return (newCtxt, ret1 and ret2 and ret3)
        else:
            return ctxt, False


class NodeTypeChecker():

    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
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
    def typeCheckNodeInputs(self, ctxt: NetworkContext, node: gs.Node) -> bool:
        inputName = [i.name for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]

        return all([node.nLevels <= 2**(input_type._value_) for node, input_type in zip(inputs, self.input_types)])

    # Don't override this. This should check that the output n_levels are appropriate for the kernel
    def typeCheckNodeOutputs(self, ctxt: NetworkContext, node: gs.Node, parserDict) -> bool:
        newCtxt = ctxt.copy()
        inputName = [i.name for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]
        nLevelsList = self.inferNumLevels(inputs, parserDict)
        return all(
            [nLevels <= 2**(output_type._value_) for nLevels, output_type in zip(nLevelsList, self.output_types)])

    # Don't override this. This should annotate the output node with the correct data type
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.Node, parserDict) -> NetworkContext:
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
    def typeInferGlobalCtxt(self, ctxt: NetworkContext, node: gs.Node, typeInfer: Callable) -> NetworkContext:
        ctxt = ctxt.copy()

        for inputNode, _type in zip(node.inputs, self.input_types):
            if isinstance(ctxt.lookup(inputNode.name), ConstantBuffer):

                # SCHEREMO: SPECIAL CASE HANDLING: CONSTANT WAS NOT FOLDED CORRECTLY
                if len(inputNode.inputs) == 1 and hasattr(inputNode.inputs[0], 'attrs') and 'value' in list(
                        inputNode.inputs[0].attrs.keys()):
                    typeNode = inputNode.inputs[0].attrs['value']
                else:
                    typeNode = inputNode

                _buffer = ctxt.lookup(inputNode.name)

                # Check the actual numLevels
                nLevels = _buffer.values.max() - min(_buffer.values.min(), 0)
                ctxt.annotateType(inputNode.name, nLevels, _type)

        return ctxt

    # Don't override this.
    def annotateDict(self, ctxt: NetworkContext, node: gs.Node, parserDict: Dict):
        env = [node.name for node in node.inputs + node.outputs]
        for key, value in parserDict.items():
            # check if the referenced buffer is in the environment
            if isinstance(value, str) and value in env:
                _buffer = ctxt.lookup(value)
                self.typeDict[key + '_type'] = _buffer._type

    # Don't override this. Automated type checking
    def typeCheck(self, ctxt: NetworkContext, node: gs.Node, typeInfer: Callable,
                  parserDict) -> Tuple[NetworkContext, bool]:
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

    # Dict of format key: (NodeTemplate, repGenerator)
    # repGenerator is a function that returns the nodeRep of the subTemplate
    # given context and nodeRep
    def __init__(self, templateStr):
        self.template = Template(templateStr)
        self.subTemplates = {}
        self.subTemplateGenerators = {}

    #Override this. Reports internal size of the template (buffer size allocated in template) to the tool
    def internalSize(self) -> int:
        return 0

    # Override this. Used to hoist optional structs, constants and so on to the GLOBAL context for specialized kernels
    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        return ctxt, nodeRep

    def hoistTransientBuffers(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        return ctxt, nodeRep, []

    # Don't override this
    def _alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        ctxt, nodeRep = self.alignToContext(ctxt, nodeRep)
        for key, (template, repGenerator) in self.subTemplates.items():
            ctxt, subNodeRep = template.alignToContext(*(repGenerator(ctxt, copy.deepcopy(nodeRep))))
            self.subTemplateGenerators[key] = (template, copy.copy(subNodeRep))
        return ctxt, nodeRep

    # Don't override this
    def __deepcopy__(self, memo):
        _copy = type(self)(self.template._source)
        memo[id(self)] = _copy

        return _copy

    def generateStartTimer(self) -> str:
        return """StartTimer(); """

    def generateStopTimer(self) -> str:
        return """StopTimer();"""

    def generateGetCyclesTimer(self) -> str:
        return """deeploy_log("%8lu cycles\\r\\n", getCycles());"""

    # Don't override this
    def generate(self, nodeRep = {}, **kwargs) -> str:
        #print(kwargs)
        try:
            for key, (template, subNodeRep) in self.subTemplateGenerators.items():
                nodeRep[f'RENDER_{key}'] = template.generate(**subNodeRep, **kwargs)
            return self.template.render(**nodeRep, **kwargs)
        except:
            print(nodeRep)
            print(mako.exceptions.text_error_template().render())
            raise KeyError(f"Template {self} failed!")


class NodeBinding():

    def __init__(self, typeChecker: NodeTypeChecker, template: NodeTemplate):
        self.typeChecker = typeChecker
        self.template = template

    # Don't override this. This should annotate the output node with the correct data type
    def bind(self, ctxt: NetworkContext, node: gs.Node, typeInfer: Callable,
             nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str], bool]:
        newCtxt = ctxt.copy()
        newCtxt, ret = self.typeChecker.typeCheck(newCtxt, node, typeInfer, nodeRep)
        if ret:
            newCtxt, nodeRep, transientBuffers = self.template.hoistTransientBuffers(newCtxt, nodeRep)
            newCtxt, nodeRep = self.template._alignToContext(newCtxt, nodeRep)
            return (newCtxt, nodeRep, transientBuffers, True)
        else:
            return (ctxt, nodeRep, [], False)

    def generate(self, ctxt: NetworkContext, nodeRep: Dict, verbose: bool = False) -> List[str]:

        for key, (template, subNodeRep) in self.template.subTemplateGenerators.items():
            for key, value in subNodeRep.items():
                if type(value) == str and (ctxt.is_local(value) or ctxt.is_global(value)):
                    subNodeRep[key] = ctxt._mangle(value)
                else:
                    subNodeRep[key] = value

        parseDict = {}

        for key, value in nodeRep.items():
            if type(value) == str and (ctxt.is_local(value) or ctxt.is_global(value)):
                parseDict[key] = ctxt._mangle(value)
            else:
                parseDict[key] = value

        nodeCall = self.template.generate({**parseDict, **self.typeChecker.typeDict})
        if verbose:
            startTimer = self.template.generateStartTimer()
            stopTimer = self.template.generateStopTimer()
            getCycles = self.template.generateGetCyclesTimer()
            return [startTimer, nodeCall, stopTimer, getCycles]
        else:
            return [nodeCall]


# Don't change anything here!
class NodeMapper():

    def __init__(self, parser: NodeParser, bindings: List[NodeBinding]):
        self.parser = parser
        self.bindings = bindings

        self.binder: NodeBinding = None
        self.bound = False

        self.nodeRep: Dict = None

    # Don't override this. Parses the networks with the correct data type
    def parse(self,
              ctxt: NetworkContext,
              node: gs.Node,
              default_channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        hoistedCtxt, parseable = self.parser.parse(ctxt, node)
        if parseable:
            # SCHEREMO: Watch out here...
            # if "channels_first" in node.attrs and self.parser.parserDict["channels_first"] == True:
            #     print(node.op)
            newCtxt, ret = self.parser.parseNodeCtxt(hoistedCtxt, node, default_channels_first)
            self.nodeRep = self.parser.parserDict
            return (newCtxt, ret)
        else:
            return ctxt, False

    # Don't override this. This should annotate the output node with the correct data type
    # SCHEREMO: Currently simply binds the first viable binding
    def bind(self, ctxt: NetworkContext, node: gs.Node, typeInfer: Callable) -> Tuple[NetworkContext, List[str], bool]:
        for binder in self.bindings:
            newCtxt = ctxt.copy()
            newCtxt, nodeRep, transientBuffers, ret = binder.bind(newCtxt, node, typeInfer, self.nodeRep)
            if ret:
                self.nodeRep = nodeRep
                self.binder = binder
                self.bound = True
                return (newCtxt, transientBuffers, True)

        return (ctxt, False)

    def generate(self, ctxt: NetworkContext, verbose: bool = False) -> List[str]:
        if not self.bound:
            raise RuntimeError("Bind layer before generating code!")
        return self.binder.generate(ctxt, self.nodeRep, verbose = verbose)


class ONNXLayer():

    def __init__(self, maps: List[NodeMapper]):
        self.maps = maps

        self.mapper: NodeMapper = None
        self.node: gs.Node = None
        self.transientBuffers = []

    def computeOps(self):
        assert self.mapper is not None, "To compute Ops, network must first be parsed!"

        return 0

    # Override this for broadcasting support
    # Returns a tuple of new, broadcasted inputShapes and outputShapes
    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, parserDict: Dict,
                      channels_first: bool) -> Tuple[Shape, Shape]:
        return (inputShapes, outputShapes)

    def broadcast(self, ctxt: NetworkContext, default_channels_first: bool = True) -> (NetworkContext):
        ctxt = ctxt.copy()

        inputShapes = [ctxt.lookup(node.name).shape for node in self.node.inputs]
        outputShapes = [ctxt.lookup(node.name).shape for node in self.node.outputs]

        if not "channels_first" in self.mapper.parser.parserDict:
            channels_first = default_channels_first
        else:
            channels_first = self.mapper.parser.parserDict['channels_first']

        newInputShapes, newOutputShapes = self.computeShapes(inputShapes, outputShapes, self.mapper.parser.parserDict,
                                                             channels_first)
        for node, newShape, oldShape in zip(self.node.inputs + self.node.outputs, newInputShapes + newOutputShapes,
                                            inputShapes + outputShapes):
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
                                ctxt.globalObjects[node.name].values = np.broadcast_to(
                                    ctxt.globalObjects[node.name].values, newShape)
                            except:
                                import IPython
                                IPython.embed()
                                raise RuntimeError(f"Could not broadcast {node.name}")

                else:
                    raise KeyError(f'Expected node {node.name} to be in context!')

        return ctxt

    # Don't override - binds the layer to a node
    def __call__(self, node: gs.Node):
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
    def parse(self, ctxt: NetworkContext, default_channels_first: bool) -> Tuple[NetworkContext, bool]:
        retCtxt = None
        # iterate through all possible mappings and return the first that works
        for mapper in self.maps:
            newCtxt = ctxt.copy()
            newCtxt, ret = mapper.parse(newCtxt, self.node, default_channels_first)
            if ret:
                self.mapper = mapper
                mapper.parser.parserDict['nodeName'] = self.node.name
                return newCtxt, True

        # If none worked, throw exception

        raise RuntimeError(f'Did not find adequate mapping for node {self.node.name}!')

    def bind(self, ctxt: NetworkContext, typeInfer: Callable):

        newCtxt = ctxt.copy()
        newCtxt, transientBuffers, ret = self.mapper.bind(newCtxt, self.node, typeInfer)

        if ret:
            self.transientBuffers = transientBuffers
            for transBuffer in transientBuffers:
                newCtxt.addUser(transBuffer, self.node)
            return newCtxt, True

        # If none worked, throw exception
        raise RuntimeError(f'Did not find adequate binding for node {self.node.name}!')

    # Do not override unless you know what you're doin - this generates code + buffer allocation / de-allocation
    # parseIO has to be called in advance!
    def generate(self, ctxt: NetworkContext, verbose: bool = False) -> Tuple[NetworkContext, List[str]]:

        outputs = [node for node in self.node.outputs]
        inputs = [node for node in self.node.inputs]

        outputNames = [node.name for node in outputs]
        inputNames = [node.name for node in inputs]

        alloc = ctxt.allocLocal(self.node.name, outputNames + self.transientBuffers)
        call = self.mapper.generate(ctxt, verbose = verbose)
        dealloc = ctxt.freeLocal(self.node.name, inputNames + self.transientBuffers)

        generated_code = [alloc, call, dealloc]
        return (ctxt, generated_code)


class NetworkOptimizationPass():

    def __init__(self):
        pass

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        return ctxt, graph


class NetworkOptimizer():

    def __init__(self, passes: List[NetworkOptimizationPass]):
        self.passes = passes

    def optimize(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        for _pass in self.passes:
            ctxt, graph = _pass.apply(ctxt, graph)
            graph.cleanup().toposort()
        return ctxt, graph


class DeploymentPlatform():
    def __init__(self, Mapping: Dict[str, ONNXLayer], DataTypes: Enum, \
                 TypeInfer: Callable, VariableBuffer: Type[VariableBuffer], \
                 ConstantBuffer: Type[ConstantBuffer], StructBuffer: Type[StructBuffer], \
                 TransientBuffer: Type[TransientBuffer], includeList: List[str] = [""]):

        self.Mapping = Mapping
        self.DataTypes = DataTypes
        self.TypeInfer = TypeInfer
        self.VariableBuffer = VariableBuffer
        self.ConstantBuffer = ConstantBuffer
        self.StructBuffer = StructBuffer
        self.TransientBuffer = TransientBuffer
        self.includeList = includeList

    def getPlatformIncludes(self) -> str:
        includeStr = []
        for include in self.includeList:
            includeStr += ["#include \"" + include + "\""]
        return ("\n").join(includeStr)


class NetworkContainer():
    def __init__(self, graph: gs.Graph, platform: DeploymentPlatform, \
                 scheduler: Callable = lambda x: x, name: str = 'DeeployNetwork', \
                 input_n_levels : Dict[str, int] = {'input_0': 256}, input_signed : Dict[str, bool] = {'input_0': False}):
        self.graph = graph
        self.scheduler = scheduler
        self.ctxt: NetworkContext = None
        self.layerBinding = OrderedDict()
        self.parsed = False
        self.Platform = platform
        self.Platform.Mapping['Constant'] = lambda x: self.ctxt.hoistConstant(x.attrs['value'], x.outputs[0].name)

        self.input_n_levels = input_n_levels
        self.input_signed = input_signed

        self.parsed = False
        self.bound = False
        self.worstCaseBufferSize = 0

    # Don't override this
    def _createIOBindings(self, ctxt: NetworkContext, graph: gs.Graph):

        ctxt = ctxt.copy()

        for node in graph.inputs:
            data_name = node.name
            data_size = node.shape
            data_type = self.input_n_levels[node.name]
            nb = ctxt.VariableBuffer(data_name, data_size, data_type)

            # SCHEREMO: Figure out smallest assignable type here
            smallestTypeValue = 2**32
            smallestType = 8
            for _type, value in zip(self.Platform.DataTypes, map(lambda x: x._value_, self.Platform.DataTypes)):
                if (2**value >= data_type) and (2**value <= smallestTypeValue):
                    smallestType = _type
                    smallestTypeValue = 2**smallestType._value_

            nb._type = smallestType
            nb._signed = self.input_signed[node.name]
            ctxt.add(nb, 'global')

        for node in graph.outputs:
            data_name = node.name
            data_size = node.shape
            # WIESEP: The shape and type will be parsed from the graph
            nb = ctxt.VariableBuffer(data_name, data_size, 0)
            ctxt.add(nb, 'global')

        return ctxt

    # Don't override this
    def broadcast(self, default_channels_first: bool = True) -> bool:

        ctxt = self.ctxt.copy()

        for name, layer in self.layerBinding.items():
            ctxt = layer.broadcast(ctxt, default_channels_first)

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

        return True

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
    def parse(self, default_channels_first: bool = True) -> bool:
        # Reset context
        self.ctxt = NetworkContext(self.Platform.VariableBuffer, self.Platform.ConstantBuffer,
                                   self.Platform.StructBuffer, self.Platform.TransientBuffer, {}, {})
        self.ctxt = self._createIOBindings(self.ctxt, self.graph)

        self._bindLayers()

        parseSuccess = True
        for key, node in self.layerBinding.items():
            self.ctxt, parsePass = node.parse(self.ctxt, default_channels_first)
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
                        currentBufferSize += np.prod(_buffer.shape) * _buffer._type._value_ // 8
                if currentBufferSize > self.worstCaseBufferSize:
                    self.worstCaseBufferSize = currentBufferSize

                for section in code:
                    for substr in section:
                        callStack += substr + '\n'

            for _buffer in self.ctxt.localObjects.values():
                assert _buffer._live == False, f'There is a memory leak in buffer {_buffer.name} in the generated forward pass!'

        return self.worstCaseBufferSize

    # Don't override this
    def generateInferenceCode(self, verbose: bool = False) -> str:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')

        for _buffer in self.ctxt.localObjects.values():
            assert _buffer._live == False, f'There is a memory leak in buffer {_buffer.name} before the generated forward pass!'

        callStack = ''
        for key, node in self.layerBinding.items():
            self.ctxt, code = node.generate(self.ctxt, verbose = verbose)
            if verbose:
                code = [[f"""deeploy_log("Layer {node.node.name}: %8u ops\\r\\n", {node.computeOps()});"""]] + code

            currentBufferSize = 0
            for _buffer in self.ctxt.localObjects.values():
                if _buffer._live == True:
                    currentBufferSize += np.prod(_buffer.shape) * _buffer._type._value_ // 8
            if currentBufferSize > self.worstCaseBufferSize:
                self.worstCaseBufferSize = currentBufferSize

            for section in code:
                for substr in section:
                    try:
                        callStack += substr + '\n'
                    except:
                        raise RuntimeError("Could not add substr!")
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
        inputs = ctxt.inputs()
        outputs = ctxt.outputs()

        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, (StructBuffer, ConstantBuffer)):
                if node._deploy:
                    name = node.name
                    node.name = ctxt._mangle(node.name)
                    callStack += "extern " + node.init()
                    # SCHEREMO: Borderline hacky, but on the okay side of things, I think
                    callStack += "static const uint32_t " + node.name + "_len" + " = " + str(np.prod(
                        node.shape)) + ";\n"
                    node.name = name

        callStack += "static const uint32_t " + ctxt._mangle("num_inputs") + f" = {len(inputs)};\n"
        callStack += "static const uint32_t " + ctxt._mangle("num_outputs") + f" = {len(outputs)};\n"

        callStack += "extern void** " + ctxt._mangle("inputs") + f"[{len(inputs)}];\n"
        callStack += "extern void** " + ctxt._mangle("outputs") + f"[{len(outputs)}];\n"

        callStack += "static const uint32_t " + ctxt._mangle("inputs_bytes") + f"[{len(inputs)}] = " + "{"
        callStack += ", ".join([str(np.prod(node.shape) * node._type._value_ // 8) for node in inputs])
        callStack += "};\n"

        callStack += "static const uint32_t " + ctxt._mangle("outputs_bytes") + f"[{len(outputs)}] = " + "{"
        callStack += ", ".join([str(np.prod(node.shape) * node._type._value_ // 8) for node in outputs])
        callStack += "};\n"

        lines = callStack.split('\n')
        lines = [line for line in lines if line.strip()]
        callStack = '\n'.join(lines)

        return callStack

    # Don't override this
    def generateBufferInitializationCode(self) -> str:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')

        ctxt = self.ctxt.copy()

        inputs = ctxt.inputs()
        outputs = ctxt.outputs()

        callStack = ''
        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, StructBuffer):
                if node._deploy:
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

        callStack += "void** " + ctxt._mangle("inputs") + f"[{len(inputs)}];\n"
        callStack += "void** " + ctxt._mangle("outputs") + f"[{len(outputs)}];\n"

        lines = callStack.split('\n')
        lines = [line for line in lines if line.strip()]
        callStack = '\n'.join(lines)

        return callStack

    def generateBufferAllocationCode(self) -> str:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')

        ctxt = self.ctxt.copy()

        inputs = ctxt.inputs()
        outputs = ctxt.outputs()
        callStack = ''

        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, StructBuffer):
                if node._deploy:
                    name = node.name
                    node.name = ctxt._mangle(node.name)
                    callStack += node.alloc()
                    node.name = name

        for node in ctxt.globalObjects.values():
            if isinstance(node, StructBuffer):
                if node._deploy:
                    name = node.name
                    node.name = ctxt._mangle(node.name)
                    callStack += node.alloc()
                    node.name = name

        for idx, i in enumerate(inputs):
            callStack += ctxt._mangle("inputs") + f"[{idx}] = (void*) {ctxt._mangle(i.name)};\n"
        for idx, i in enumerate(outputs):
            callStack += ctxt._mangle("outputs") + f"[{idx}] = (void*) {ctxt._mangle(i.name)};\n"

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
            if node._deploy:
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
            if isinstance(_buffer, ConstantBuffer) and _buffer._deploy:
                size += int((np.prod(_buffer.shape) * _buffer._type._value_ // 8))

        return size

    # Don't override this - Returns worst case layer and buffering size in bytes
    def getTotalSize(self) -> int:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before getting RAM Size!')

        return self.getParameterSize() + self.getWorstCaseBufferSize()

    def numberOfOps(self, verbose: bool = False) -> int:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before getting number of operations!')
        totalSum = 0
        for i in self.layerBinding.values():
            totalSum += i.computeOps()
            if verbose:
                print("Layer " + str(i.node.name) + str("\nNumber of operations: \t\t") +
                      str("%12s\n" % i.computeOps()))
        return totalSum


class NetworkDeployer(NetworkContainer):
    def __init__(self, graph: gs.Graph, deploymentPlatform: DeploymentPlatform,\
                 loweringOptimizer: NetworkOptimizer, scheduler: Callable = lambda x: x,\
                 name: str = 'DeeployNetwork', input_n_levels : Dict[str, int] = {'input_0': 256}, \
                 input_signed : Dict[str, bool] = {'input_0': False}, default_channels_first: bool = True):
        super().__init__(graph, deploymentPlatform, scheduler, name)
        self.name = name
        self.prepared = False
        self.baseParser = NodeParser()
        self.optimizer = loweringOptimizer
        self.input_n_levels = input_n_levels
        self.input_signed = input_signed
        self.default_channels_first = default_channels_first

    # Don't override this
    def lower(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        return self.optimizer.optimize(ctxt, graph)

    # Don't override this
    def baseParse(self) -> Tuple[NetworkContext, bool]:
        newCtxt = NetworkContext(VariableBuffer, ConstantBuffer, StructBuffer, {}, {})
        newCtxt = self._createIOBindings(newCtxt, self.graph)

        ret = False
        for node in self.graph.nodes:
            newCtxt, ret = self.baseParser.parse(newCtxt, node, self.default_channels_first)

        return newCtxt, ret

    def postLoweringOptimization(self):
        pass

    # Don't override this
    # Duplicate constants with multiple users
    def duplicateConstants(self, graph: gs.Graph):
        idx = 0
        for node in self.graph.nodes:
            for i, inputNode in enumerate(node.inputs):
                if type(inputNode) == gs.ir.tensor.Constant and len(inputNode.outputs) > 1:
                    newConst = gs.Constant(name = f"{inputNode.name}_EXTRACT_CONST_{idx}", values = inputNode.values)
                    node.inputs[i] = newConst
                    # graph.nodes.append(newConst)
                    idx += 1

    # Don't Override this
    def middleWare(self):

        # Rename graph inputs and outputs:
        for idx, inputNode in enumerate(self.graph.inputs):
            inputNode.name = "input_" + str(idx)
        for idx, outputNode in enumerate(self.graph.outputs):
            outputNode.name = "output_" + str(idx)

        self.duplicateConstants(self.graph)
        # sanity check the graph and generate a base context for lowering/optimization
        self.ctxt, ret = self.baseParse()
        if not ret:
            raise RuntimeError("The given graph was not valid - check that it is acyclic!")

        self.ctxt, self.graph = self.lower(self.ctxt, self.graph)  # This lowers the graph to a deployable format

        self.postLoweringOptimization()

    # Don't override this
    def exportGraph(self, f):
        model = gs.export_onnx(self.graph)
        convert_model_to_external_data(model, location = "model.data")
        onnx.save(model, f)

    # Don't override this unless you know what you are doin
    def backEnd(self):
        self.parse(self.default_channels_first)  # This reparses the lowered graph
        self.broadcast(self.default_channels_first)  # This broadcasts all tensors offline
        self.bind()  # This binds the graph to the node templates
        onnx.save_model(gs.export_onnx(self.graph), "final_implementation.onnx")

    # Don't override this
    def prepare(self):
        # MIDDLE END
        self.middleWare()
        onnx.save_model(gs.export_onnx(self.graph), "preParse_implementation.onnx")
        # BACK END - Inherited from NetworkContainer
        self.backEnd()
        # FINAL TRANSFORMS
        self.prepared = True

    # Don't override this
    def generateFunction(self, verbose: bool = False) -> str:
        if not self.prepared:
            self.prepare()

        return self.generateInferenceCode(verbose = verbose)
