# ----------------------------------------------------------------------
#
# File: DeeployTypes.py
#
# Last edited: 17.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Author:
# - Moritz Scherer, ETH Zurich
# - Victor Jung, ETH Zurich
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

from __future__ import annotations

import copy
import os
import re
from collections import OrderedDict, deque, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import dill
import mako
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from mako.template import Template
from onnx.external_data_helper import convert_model_to_external_data
from ortools.constraint_solver.pywrapcp import IntVar

from .AbstractDataTypes import DataTypeCollection, HelperTypes, Pointer, PointerClass, PointerType, StructClass

Shape = TypeVar("Shape", bound = Any)
SubGraph = List[gs.Node]
Schedule = Union[List[SubGraph], SubGraph]
'''
DeeployState naming convention: <level>_<stage_prefix>_<stage>
<level>: At which level the DeeployState have been exported (e.g. Frontend, Middleware, Backend, ...)
<stage_prefix>: At which position relative to the stage the DeeployState have been exported (e.g. post, pre, ...)
<stage>: At which stage the DeeployState have been exported (e.g. Tiling, Parsing, Marco Bertuletti, ...)
'''

_middlewarePreLoweringFilename = 'middleware_pre_lowering'
_middlewarePostLoweringFilename = 'middleware_post_lowering'
_backendPostParsingFilename = 'backend_post_parsing'
_backendPostBindingFilename = 'backend_post_binding'

_ctxtExtension = '.pkl'
_graphExtension = '.onnx'
_dataExtension = '.data'


class VariableBuffer():

    initTemplate: NodeTemplate = None
    allocTemplate: NodeTemplate = None
    deallocTemplate: NodeTemplate = None

    def __init__(self, name: str = '', shape = [1]):
        self.name = name
        self.shape = shape

        # Do not override - Should be written in the parsing passes
        self._users = []

        # Do not override - Should be written in the typechecking passes
        self._type: type[PointerClass] = None
        self._instance: PointerClass = None

        # Do not override - Should be written in the deployment passes
        self._live = False

        # Do not override - Set in Templates depending on platform
        self._deploy = True

    def _nodeRep(self) -> Dict:
        return {"type": self._instance, "name": self.name, "size": int(np.prod(self.shape))}

    def init(self) -> str:
        return self.initTemplate.generate(self._nodeRep())

    def alloc(self) -> str:
        return self.allocTemplate.generate(self._nodeRep())

    def dealloc(self) -> str:
        return self.deallocTemplate.generate(self._nodeRep())

    def __str__(self) -> str:
        return f'VariableBuffer: name: {self.name}, type: {self._type}'

    def __repr__(self) -> str:
        return f'VariableBuffer: name: {self.name}, type: {self._type}'

    def __eq__(self, other):
        ret = all([self.name == other.name, self.shape == other.shape])
        return ret

    @classmethod
    def fromNode(cls, node: gs.Node):
        return (cls(name = node.name, shape = node.shape if not isinstance(node, gs.Constant) else node.values.shape))


class TransientBuffer(VariableBuffer):

    def __init__(self, name: str = '', size = 0):
        self.name = name
        self.size = size

        # Do not override - Should be written in the parsing passes
        self._users = []

        # Do not override - Should be written in the parsing passes
        self._type: type[PointerClass] = Pointer(HelperTypes.void)

        # Do not override - Should be written in the deployment passes
        self._live = False

        # Do not override - Set in Templates depending on platform
        self._deploy = True

    def __eq__(self, other):

        ret = all([self.name == other.name, self.size == other.size])
        return ret

    def _nodeRep(self) -> Dict:
        return {"type": self._type, "name": self.name, "size": int(self.size)}

    def __str__(self) -> str:
        return f'TransientBuffer: name: {self.name}, size: {self.size}'

    def __repr__(self) -> str:
        return f'TransientBuffer: name: {self.name}, size: {self.size}'

    @classmethod
    def fromVariableBuffer(cls, buffer: VariableBuffer):
        ret = cls(name = buffer.name, size = np.prod(buffer.shape) * buffer._type.typeWidth // 8)


class ConstantBuffer(VariableBuffer):

    def __init__(self, name: str = '', shape = [1], values = [0]):
        super().__init__(name, shape)
        values = np.asarray(values)
        intArray = values.astype(int)
        assert (np.abs(values - intArray)).max() < 0.001, "Constant value {name} is NOT an integer!"
        self.values = intArray

        # Do not override - ConstantBuffers are assumed to be always live!
        self._live = True

    def __eq__(self, other):
        ret = all([super().__eq__(other), np.array_equal(self.values, other.values)])
        return ret

    def _valueString(self) -> str:
        values = list(self.values.reshape(-1))
        strValues = [str(value) for value in values]
        valueString = ', '.join(strValues)
        return valueString

    def __str__(self) -> str:
        return f'ConstantBuffer: name: {self.name}, type: {self._type}'

    def __repr__(self) -> str:
        return f'ConstantBuffer: name: {self.name}, type: {self._type}'

    def _nodeRep(self) -> Dict:
        return {"type": self._type, "name": self.name, "size": int(np.prod(self.shape)), "values": self._valueString()}

    @classmethod
    def fromVariableBuffer(cls, buffer: VariableBuffer, values):
        ret = cls(name = buffer.name, shape = buffer.shape, values = values)

        return ret


class StructBuffer(VariableBuffer):

    def __init__(self, name: str, structDict: Dict):
        super().__init__(name, None)
        self.structDict = structDict

    def __eq__(self, other):
        ret = super().__eq__(other) and hasattr(other, "structDict") and self.structDict == other.structDict
        return ret

    def __str__(self) -> str:
        return f'StructBuffer: name: {self.name}, type: {self._type}'

    def __repr__(self) -> str:
        return f'StructBuffer: name: {self.name}, type: {self._type}'

    def _nodeRep(self) -> Dict:
        return {"type": self._type, "name": self.name, "size": int(self._type.typeWidth), "structDict": self.structDict}


class GlobalDefinition():

    def __init__(self, name: str, definition: str):
        self.name = name
        self.definition = definition

    def alloc(self):
        return self.definition

    def __eq__(self, other):
        ret = all([self.name == other.name, self.definition == other.definition])
        return ret


class NetworkContext():

    def __init__(self,
                 variableBuffer: Type[VariableBuffer],
                 constantBuffer: Type[ConstantBuffer],
                 structBuffer: Type[StructBuffer],
                 transientBuffer: Type[ConstantBuffer],
                 globalObjects = {},
                 localObjects = {},
                 name: str = 'DeeployNetwork'):
        self.globalObjects = OrderedDict()
        self.localObjects = OrderedDict()
        self.VariableBuffer = variableBuffer
        self.ConstantBuffer = constantBuffer
        self.StructBuffer = structBuffer
        self.TransientBuffer = transientBuffer
        self.name = name

    def exportNetworkContext(self, folderPath: str, fileName: str):
        relativePath = os.path.join(folderPath, fileName + _ctxtExtension)
        absolutePath = os.path.abspath(relativePath)

        if not os.path.isabs(absolutePath):
            raise OSError(f"Error exporting the context to: {absolutePath}")

        with open(absolutePath, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def importNetworkContext(folderPath, fileName):
        relativePath = os.path.join(folderPath, fileName + _ctxtExtension)
        absolutePath = os.path.abspath(relativePath)

        if not os.path.isabs(absolutePath) or not os.path.exists(absolutePath):
            raise OSError(f"File or path does not exist: {absolutePath}")

        with open(absolutePath, 'rb') as f:
            return dill.load(f)

    def __repr__(self):
        globalObjects = []
        localObjects = []
        for item in self.globalObjects.values():
            globalObjects.append(str(item))
        for item in self.localObjects.values():
            localObjects.append(str(item))
        _repr = "globalObjects: {\n"
        _repr += ",\n ".join(globalObjects)
        _repr += "} \n\n"
        _repr += "localObjects: {\n"
        _repr += ",\n ".join(localObjects)
        _repr += "}"
        return _repr

    def __eq__(self, other):
        if not isinstance(other, NetworkContext):
            raise TypeError(f'Cannot compare NetworkContext with {type(other)}!')

        if not other.globalObjects.keys() == self.globalObjects.keys():
            return False

        if not other.localObjects.keys() == self.localObjects.keys():
            return False

        for buffer_name in self.globalObjects.keys():
            if not self.globalObjects[buffer_name] == other.globalObjects[buffer_name]:
                return False

        for buffer_name in self.localObjects.keys():
            if not self.localObjects[buffer_name] == other.localObjects[buffer_name]:
                return False

        return True

    def _mangle(self, name: str, repr: bool = True) -> str:
        repStr = name
        repStr = re.sub('\.', '_', repStr)
        repStr = re.sub(':', '_', repStr)
        if repr:
            repStr = re.sub('\.', '_', self.name) + '_' + repStr
        return repStr

    def add(self, obj: VariableBuffer, ctxt = 'local', _id = ""):
        if _id != "":
            obj.name = self._mangle(_id + "_" + obj.name, False)

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

    def lookup(self, name, _id = ""):
        if _id != "":
            name = self._mangle(_id + "_" + name, False)

        if name in self.localObjects.keys():
            return self.localObjects[name]
        elif name in self.globalObjects.keys():
            return self.globalObjects[name]
        else:
            raise KeyError(f'Expected key {name} to be in either local or global context!')

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

    def hoistTransientBuffer(self, name: str, size: int) -> str:
        transientBuffer = self.TransientBuffer(name, size)
        self.add(transientBuffer, 'local')

        return name

    def hoistGlobalDefinition(self, name: str, definition: str) -> None:
        _definition = GlobalDefinition(name, definition)
        self.add(_definition, 'global')

    def hoistStruct(self, _struct: Dict, name: str, _type: type[StructClass]) -> str:

        assert issubclass(_type, StructClass), f"Type {_type} is not a Struct!"

        if isinstance(_struct, _type):
            struct = _struct
        else:
            struct = _type(_struct, self)

        structBuffer = self.StructBuffer(name, struct)
        structBuffer._type = _type
        structBuffer._instance = struct
        self.add(structBuffer, 'local')

        return name

    def hoistConstantAndReference(self, constBuf: ConstantBuffer, pointerType: type[PointerClass]) -> str:

        name = constBuf.name
        constBuf._type = pointerType

        self.add(constBuf, "global")

        constBuf._instance = constBuf._type(name, self)

        refName = name + "_ref"
        reference = self.hoistReference(name, refName)

        return refName

    def hoistReference(self, _reference: str, name: str) -> str:

        assert _reference != name, f"Reference name {_reference} cannot be the same as {name}"
        assert not self.is_local(name), f"{name} is already in context!"

        _object = self.lookup(_reference)
        referenceBuffer = self.VariableBuffer(name, [1])

        referenceBuffer._type = _object._type
        referenceBuffer._deploy = True
        referenceBuffer.allocTemplate = NodeTemplate(" \
        ${type.typeName} ${name} = (${type.typeName}) " + f"{str(_object._instance)};")
        referenceBuffer.deallocTemplate = NodeTemplate("""
        """)
        referenceBuffer.initTemplate = NodeTemplate("""
        """)

        self.add(referenceBuffer, 'local')
        referenceBuffer._instance = _object._type(name, ctxt = self)
        referenceBuffer._referenceName = _object.name

        return name

    def hoistConstant(self, node: gs.Node, name = '', type: type[PointerClass] = None) -> str:

        if type is not None:
            assert issubclass(type, PointerClass), "Tried to hoist a constant that's not a Pointer!"

        assert len(node.outputs) <= 1, "Constant has more than one output"

        if name == "":
            name = node.name

        # SCHEREMO: This is currently heuristic, but should be annotated in ONNX
        localBuffer = self.VariableBuffer.fromNode(node = node)
        globalBuffer = self.ConstantBuffer.fromVariableBuffer(localBuffer, values = node.values)
        globalBuffer.name = name
        globalBuffer._type = type

        self.add(globalBuffer, 'global')

        return globalBuffer.name

    def addUser(self, name: str, node):
        _buffer = self.lookup(name)
        if node.name not in _buffer._users:
            _buffer._users.append(node.name)
        if self.is_local(_buffer.name):
            self.localObjects[_buffer.name] = _buffer
        else:
            self.globalObjects[_buffer.name] = _buffer

    def annotateType(self, name: str, _type: type[PointerClass]):
        obj = self.lookup(name)
        assert issubclass(_type, PointerClass), "Cannot annotate buffer with non-Pointer type"
        obj._type = _type
        obj._instance = _type(name, ctxt = self)

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
                data_in_buffers.append(localBuffer.name)

            ctxt.addUser(data_in, node)

        return ctxt, True

    # Don't touch this
    def parseOutputs(self, ctxt: NetworkContext, node: gs.Node) -> Tuple[NetworkContext, bool]:
        newCtxt = ctxt.copy()
        outputNodes = node.outputs
        outputNames = [node.name for node in outputNodes]

        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = newCtxt.VariableBuffer(name = name, shape = node.shape)
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

        ret1 = self.parseNode(node)
        if ret1:
            newCtxt, ret2 = self.parseInputs(newCtxt, node)
            newCtxt, ret3 = self.parseOutputs(newCtxt, node)
            return (newCtxt, ret1 and ret2 and ret3)
        else:
            return ctxt, False


class NodeTypeChecker():

    def __init__(self, input_types: Sequence[PointerClass], output_types: Sequence[PointerClass]):

        for input_type in input_types:
            assert issubclass(
                input_type, PointerClass
            ), f"Class NodeTypeChecker expects pointers as inputs, but TypeChecker {self} is not expecting input pointers!"
        for output_type in output_types:
            assert issubclass(
                output_type, PointerClass
            ), f"Class NodeTypeChecker expects pointers as outputs, but TypeChecker {self} is not expecting output pointers!"

        self.input_types = input_types
        self.output_types = output_types

        self.typeDict = {}

    # Override this. This should compute the signednes of each output node of the Layer
    def checkOutputType(self, inputs: List[VariableBuffer], parserDict: Dict) -> bool:
        return True

    # Don't override this. This should annotate the output node with the correct data type
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.Node, parserDict) -> NetworkContext:
        newCtxt = ctxt.copy()

        inputs = [ctxt.lookup(inputNode.name) for inputNode in node.inputs]
        outputNames = [node.name for node in node.outputs]

        outputTypes = self.output_types

        for name, output_type in zip(outputNames, outputTypes):
            newCtxt.annotateType(name, output_type)

        return newCtxt

    # Don't override this. Automated annotation of global buffer
    def typeCheckNodeInputs(self, ctxt: NetworkContext, node: gs.Node) -> bool:
        ctxt = ctxt.copy()
        retCheck = True

        for inputNode, _type in zip(node.inputs, self.input_types):
            reference = ctxt.lookup(inputNode.name)

            if not isinstance(reference, VariableBuffer):
                return False

            if hasattr(reference, "values"):
                retCheck &= _type.referencedType._checkValue(reference.values)
            else:
                retCheck &= isinstance(reference._type, _type)
        return retCheck

    # Don't override this. Automated annotation of global buffer
    def typeInferGlobalCtxt(self, ctxt: NetworkContext, node: gs.Node) -> NetworkContext:
        ctxt = ctxt.copy()

        for inputNode, _type in zip(node.inputs, self.input_types):
            if isinstance(ctxt.lookup(inputNode.name), ConstantBuffer):
                reference = ctxt.lookup(inputNode.name)
                if not _type.referencedType._checkValue(reference.values):
                    raise Exception(f"Can't cast {reference} to {_type}!")

                ctxt.annotateType(inputNode.name, _type)

        return ctxt

    # Don't override this.
    def annotateDict(self, ctxt: NetworkContext, node: gs.Node, parserDict: Dict):
        env = [node.name for node in node.inputs + node.outputs]
        for key, value in parserDict.items():
            # check if the referenced buffer is in the environment
            if isinstance(value, str) and value in env:
                self.typeDict[key + '_type'] = ctxt.lookup(value)._type

    # Don't override this. Automated type checking
    def typeCheck(self, ctxt: NetworkContext, node: gs.Node, parserDict) -> Tuple[NetworkContext, bool]:
        newCtxt = ctxt.copy()
        if not self.typeCheckNodeInputs(newCtxt, node):
            return ctxt, False

        if not self.checkOutputType(node.inputs, parserDict):
            return ctxt, False

        newCtxt = self.typeInferGlobalCtxt(newCtxt, node)
        newCtxt = self.typeInferOutput(newCtxt, node, parserDict)
        self.annotateDict(newCtxt, node, parserDict)
        return (newCtxt, True)


# SCHEREMO: mako.Templates are not copiable, since they can use shared context.
# In Deeploy we only use them by direct call (no shared context), so we can override deepcopy and workaround the issue
class _Template(Template):

    def __deepcopy__(self, memo):
        _copy = type(self)(self._source)
        _copy._code = self._code
        _copy.module = self.module
        _copy.callable_ = self.callable_
        memo[id(self)] = _copy
        return _copy


class NodeTemplate():

    # Dict of format key: (NodeTemplate, repGenerator)
    # repGenerator is a function that returns the nodeRep of the subTemplate
    # given context and nodeRep
    def __init__(self, templateStr):
        self.template = _Template(templateStr)
        self.subTemplates = {}
        self.subTemplateGenerators = {}

    #Override this. Reports internal size of the template (buffer size allocated in template) to the tool
    def internalSize(self) -> int:
        return 0

    # Override this. Used to hoist optional structs, constants and so on to the GLOBAL context for specialized kernels
    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        return ctxt, nodeRep, []

    # Override this
    def computeTransientBuffersSize(self, ctxt: NetworkContext, nodeRep: Dict) -> List[Tuple[str, Union[int, IntVar]]]:
        return []

    # Override this
    def hoistTransientBuffers(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        return ctxt, nodeRep, []

    # Don't override this
    def _alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt, nodeRep, nameList = self.alignToContext(ctxt, nodeRep)
        for key, (template, repGenerator) in self.subTemplates.items():
            ctxt, subNodeRep, _nameList = template.alignToContext(*(repGenerator(ctxt, copy.deepcopy(nodeRep))))
            self.subTemplateGenerators[key] = (template, copy.copy(subNodeRep))
            nameList += _nameList
        return ctxt, nodeRep, nameList

    # Don't override this
    def generate(self, nodeRep = {}, **kwargs) -> str:
        callStack = ""

        try:
            for key, (template, subNodeRep) in self.subTemplateGenerators.items():
                nodeRep[f'RENDER_{key}'] = template.generate(**subNodeRep, **kwargs)
            callStack += self.template.render(**nodeRep, **kwargs)
        except:
            print(nodeRep)
            print(mako.exceptions.text_error_template().render())
            raise KeyError(f"Template {self} failed!")
        return callStack


_TemplateNode = namedtuple("TemplateNode", ("template", "nodeRep"))


class ExecutionBlock():

    def __init__(self, nodeTemplate = None):
        if nodeTemplate is not None:
            self.nodeTemplates = deque([nodeTemplate])
        else:
            self.nodeTemplates = deque([])

        self.patternMemoryConstraint: Optional = None

    def addLeft(self, template, nodeRep):
        self.nodeTemplates.appendleft(_TemplateNode(template, nodeRep))

    def addRight(self, template, nodeRep):
        self.nodeTemplates.append(_TemplateNode(template, nodeRep))

    # Hoisting is allowed to add new buffers into the context
    def hoisting(self, ctxt: NetworkContext, **kwargs) -> Tuple[NetworkContext, List[str]]:

        transientBuffers = []
        contextBuffers = []

        for idx, (template, nodeRep) in enumerate(self.nodeTemplates.copy()):

            newCtxt, nodeRep, _transientBuffers = template.hoistTransientBuffers(ctxt, {**nodeRep, **kwargs})
            newCtxt, nodeRep, _contextBuffers = template._alignToContext(newCtxt, {**nodeRep, **kwargs})

            self.nodeTemplates[idx].nodeRep.update(nodeRep)
            transientBuffers += _transientBuffers
            contextBuffers += _contextBuffers

        return newCtxt, transientBuffers + contextBuffers

    @staticmethod
    def _mangleNodeRep(ctxt: NetworkContext, nodeRep: Dict) -> Dict:
        parseDict = {}

        for key, value in nodeRep.items():
            if type(value) == str and (ctxt.is_local(value) or
                                       ctxt.is_global(value)) and not isinstance(ctxt.lookup(value), GlobalDefinition):
                parseDict[key] = ctxt._mangle(value)
            else:
                parseDict[key] = value

        return parseDict

    def generate(self, ctxt, **kwargs) -> str:

        return ("\n").join([
            template.generate(ExecutionBlock._mangleNodeRep(ctxt, {
                **nodeRep,
                **kwargs
            })) for template, nodeRep in self.nodeTemplates
        ])


class NodeBinding():

    def __init__(self, typeChecker: NodeTypeChecker, template: NodeTemplate, codeTransformer: CodeTransformation):
        self._typeChecker = typeChecker
        self.template = template
        self._executionBlock: ExecutionBlock = ExecutionBlock()
        self._nodeName: str = None
        self.buffers: List[VariableBuffer] = []
        self.codeTransformer: CodeTransformation = codeTransformer

    @property
    def typeChecker(self):
        return self._typeChecker

    @property
    def executionBlock(self):
        return self._executionBlock

    @property
    def nodeName(self):
        return self._nodeName

    def earlyBinding(self, ctxt, node, nodeRep):
        self.executionBlock.addLeft(self.template, nodeRep)
        self._nodeName = nodeRep['nodeName']
        return ctxt

    def bind(self, ctxt: NetworkContext, node: gs.Node, nodeRep: Dict) -> Tuple[NetworkContext, List[str], bool]:
        newCtxt, ret = self.typeChecker.typeCheck(ctxt, node, nodeRep)
        if ret:
            newCtxt = self.earlyBinding(newCtxt, node, nodeRep)
            newCtxt, buffers = self.executionBlock.hoisting(newCtxt, **self.typeChecker.typeDict)

            for _buffer in buffers:
                newCtxt.lookup(_buffer)._users.append(self._nodeName)

            return newCtxt, None, True
        return None, None, False

    def codeTransform(self, ctxt):
        ctxt, self._executionBlock = self.codeTransformer.transform(ctxt, self.executionBlock, self.nodeName)
        return ctxt

    def generate(self, ctxt: NetworkContext, verbose: bool = False) -> List[str]:
        nodeCall = self.executionBlock.generate(ctxt, **self.typeChecker.typeDict)
        return [nodeCall]


# Don't change anything here!
class NodeMapper():

    def __init__(self, parser: NodeParser, bindings: List[NodeBinding]):
        self.parser = parser
        self.bindings = bindings

        self.binder: NodeBinding = None
        self.bound = False

    # Don't override this. Parses the networks with the correct data type
    def _parse(self,
               ctxt: NetworkContext,
               node: gs.Node,
               default_channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt, ret = self.parser.parse(ctxt, node)
        return ctxt, ret

    def _parseCtxt(self,
                   ctxt: NetworkContext,
                   node: gs.Node,
                   default_channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = self.parser.parseNodeCtxt(ctxt, node, default_channels_first)
        return (newCtxt, ret)

    # Don't override this. This should annotate the output node with the correct data type
    # SCHEREMO: Currently simply binds the first viable binding
    def bind(self, ctxt: NetworkContext, node: gs.Node) -> Tuple[NetworkContext, List[str], bool]:
        for binder in self.bindings:
            newCtxt = ctxt.copy()
            newCtxt, transientBuffers, ret = binder.bind(newCtxt, node, self.parser.parserDict)
            if ret:
                self.binder = binder
                self.bound = True
                return (newCtxt, transientBuffers, True)

        return (ctxt, [], False)

    def generate(self, ctxt: NetworkContext, verbose: bool = False) -> List[str]:
        if not self.bound:
            raise RuntimeError("Bind layer before generating code!")
        return self.binder.generate(ctxt, verbose = verbose)


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
                                raise RuntimeError(f"Could not broadcast {node.name}")

                else:
                    raise KeyError(f'Expected node {node.name} to be in context!')

        return ctxt

    # Don't override - binds the layer to a node
    def __call__(self, node: gs.Node):
        _copy = copy.deepcopy(self)
        _copy.node = node
        return _copy

    # Call this, DO NOT override! -> This should assert that all variables required are in the node!
    def _parse(self, ctxt: NetworkContext, default_channels_first: bool) -> Tuple[NetworkContext, bool]:
        retCtxt = None
        # iterate through all possible mappings and return the first that works
        for mapper in self.maps:
            self.mapper = mapper
            newCtxt = ctxt.copy()
            newCtxt, ret = mapper._parse(newCtxt, self.node, default_channels_first)
            if ret:
                return newCtxt, True

        # If none worked, throw exception
        raise RuntimeError(
            f'Did not find adequate mapping for node {self.node.name}! Candidates: {[type(x.parser).__name__ for x in self.maps]}'
        )

    # Call this, DO NOT override! -> This should assert that all variables required are in the node!
    def _parseCtxt(self, ctxt: NetworkContext, default_channels_first: bool) -> Tuple[NetworkContext, bool]:
        newCtxt = ctxt.copy()
        newCtxt, ret = self.mapper._parseCtxt(newCtxt, self.node, default_channels_first)
        if ret:
            self.mapper.parser.parserDict['nodeOp'] = self.node.op
            self.mapper.parser.parserDict['nodeName'] = self.node.name
            return newCtxt, True

        # If none worked, throw exception
        raise RuntimeError(
            f'Failed to context-aware parse node {self.node.name} ({type(self.mapper.parser).__name__})!')

    def bind(self, ctxt: NetworkContext):

        newCtxt = ctxt.copy()
        newCtxt, transientBuffers, ret = self.mapper.bind(newCtxt, self.node)

        if ret:
            # WIESEP: Compute number of ops only after binding.
            self.mapper.parser.parserDict['nodeOps'] = int(self.computeOps())
            return newCtxt, True

        # If none worked, throw exception

        raise RuntimeError(f'Did not find adequate binding for node {self.node.name}!')

    def codeTransform(self, ctxt: NetworkContext):
        newCtxt = self.mapper.binder.codeTransform(ctxt)
        return newCtxt

    def generate(self, ctxt: NetworkContext, verbose: bool = False) -> Tuple[NetworkContext, List[str]]:

        call = self.mapper.generate(ctxt, verbose = verbose)

        generated_code = [call]
        return (ctxt, generated_code)


class TopologyOptimizationPass():

    def __init__(self):
        pass

    def apply(self, graph: gs.Graph) -> Tuple[gs.Graph]:
        return graph


class TopologyOptimizer():

    def __init__(self, passes: List[TopologyOptimizationPass]):
        self.passes = passes

    def optimize(self, graph: gs.Graph) -> Tuple[gs.Graph]:
        for _pass in self.passes:
            graph = _pass.apply(graph)
            graph.cleanup().toposort()
        return graph


class NetworkOptimizationPass(TopologyOptimizationPass):

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        return ctxt, graph


class NetworkOptimizer(TopologyOptimizer):

    def optimize(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        for _pass in self.passes:
            ctxt, graph = _pass.apply(ctxt, graph)
            graph.cleanup().toposort()
        return ctxt, graph


class CodeTransformationPass():

    def __init__(self):
        pass

    def apply(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
              name: str) -> Tuple[NetworkContext, ExecutionBlock]:
        return ctxt, executionBlock


class CodeTransformation():

    def __init__(self, passes: List[CodeTransformationPass]):
        self.passes = passes

    def transform(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                  name: str) -> Tuple[NetworkContext, ExecutionBlock]:
        for _pass in self.passes:
            ctxt, executionBlock = _pass.apply(ctxt, executionBlock, name)
        return ctxt, executionBlock


class DeploymentPlatform():

    def __init__(self,
                 Mapping: Dict[str, ONNXLayer],
                 DataTypes: DataTypeCollection,
                 VariableBuffer: Type[VariableBuffer],
                 ConstantBuffer: Type[ConstantBuffer],
                 StructBuffer: Type[StructBuffer],
                 TransientBuffer: Type[TransientBuffer],
                 includeList: List[str] = [""]):

        self.Mapping = Mapping
        self.DataTypes = DataTypes
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

    parsed = False
    bound = False
    transformed = False

    def __init__(self,
                 graph: gs.Graph,
                 platform: DeploymentPlatform,
                 inputTypes: Dict[str, PointerType],
                 scheduler: Callable[[gs.Graph], Schedule] = lambda graph: list(graph.nodes),
                 name: str = 'DeeployNetwork',
                 deeployStateDir: str = "DeeployState"):
        self.graph = graph
        self.scheduler = scheduler
        self.layerBinding: 'OrderedDict[str, ONNXLayer]' = OrderedDict()
        self.parsed = False
        self.Platform = platform
        self.Platform.Mapping['Constant'] = lambda x: \
            self.ctxt.hoistConstant(x.attrs['value'], x.outputs[0].name, None)

        self.inputTypes = inputTypes
        self.worstCaseBufferSize = 0

        self.ctxt = NetworkContext(variableBuffer = self.Platform.VariableBuffer,
                                   constantBuffer = self.Platform.ConstantBuffer,
                                   structBuffer = self.Platform.StructBuffer,
                                   transientBuffer = self.Platform.TransientBuffer)

        self.deeployStateDir = deeployStateDir

    # Don't override this
    def _createIOBindings(self, ctxt: NetworkContext, graph: gs.Graph):

        ctxt = ctxt.copy()

        for node in graph.inputs:
            data_name = node.name
            data_size = node.shape
            data_type = self.inputTypes[node.name]
            nb = ctxt.VariableBuffer(data_name, data_size)

            ctxt.add(nb, 'global')
            ctxt.annotateType(data_name, data_type)

        for node in graph.outputs:
            data_name = node.name
            data_size = node.shape
            # WIESEP: The shape and type will be parsed from the graph
            nb = ctxt.VariableBuffer(data_name, data_size)
            ctxt.add(nb, 'global')

        return ctxt

    def inputs(self) -> List[VariableBuffer]:
        inputs = []

        graphInputs = [tensor.name for tensor in self.graph.inputs]

        for key, value in self.ctxt.globalObjects.items():
            if not isinstance(value, self.ctxt.VariableBuffer) or value._users == []:
                continue
            if key not in graphInputs:
                continue

            inputs += [value]
        return inputs

    def outputs(self) -> List[VariableBuffer]:
        outputs = []

        graphOutputs = [tensor.name for tensor in self.graph.outputs]

        for key, value in self.ctxt.globalObjects.items():

            if not isinstance(value, self.ctxt.VariableBuffer) or value._users != []:
                continue
            if key not in graphOutputs:
                continue

            outputs += [value]
        return outputs

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

        NetworkBindSuccess = True
        for name, layer in self.layerBinding.items():

            newCtxt, LayerBindSuccess = layer.bind(newCtxt)
            NetworkBindSuccess = NetworkBindSuccess and LayerBindSuccess

        if not NetworkBindSuccess:
            raise RuntimeError(f'Could not find a valid binding for the graph')
        else:
            self.bound = True
            self.ctxt = newCtxt

        return True

    # Don't override this
    def codeTransform(self) -> bool:
        if not self.bound:
            raise ValueError('You need to bind the network before transforming code!')

        if self.transformed:
            return

        for name, layer in self.layerBinding.items():
            self.ctxt = layer.codeTransform(self.ctxt)
        self.transformed = True

    def _bindLayers(self):
        # Create schedule, binding, then parse resulting program for correctness
        self.layerBinding: 'OrderedDict[str, ONNXLayer]' = OrderedDict()

        schedule = self.scheduler(self.graph)
        flatSchedule = []

        for subGraph in schedule:
            if isinstance(subGraph, gs.Node):
                flatSchedule.append(subGraph)
            else:
                flatSchedule += subGraph

        for i in flatSchedule:

            # Create binding
            assert i.op in list(self.Platform.Mapping.keys()), f'Layer {i.op} not in layer dict!'
            layer = self.Platform.Mapping[i.op](i)

            if isinstance(layer, ONNXLayer):
                self.layerBinding[layer.node.name] = layer

    # Don't override this
    def parse(self, default_channels_first: bool = True) -> bool:

        self.ctxt = NetworkContext(variableBuffer = self.Platform.VariableBuffer,
                                   constantBuffer = self.Platform.ConstantBuffer,
                                   structBuffer = self.Platform.StructBuffer,
                                   transientBuffer = self.Platform.TransientBuffer)

        self.ctxt = self._createIOBindings(self.ctxt, self.graph)

        self._bindLayers()

        parseSuccess = True
        for key, node in self.layerBinding.items():
            self.ctxt, parsePass = node._parse(self.ctxt, default_channels_first)
            parseSuccess = parseSuccess and parsePass

        if parseSuccess:
            return True
        else:
            raise RuntimeError('Could not parse the graph!')

    # Don't override this
    def parseCtxt(self, default_channels_first: bool = True) -> bool:
        parseSuccess = True
        for key, node in self.layerBinding.items():
            self.ctxt, parsePass = node._parseCtxt(self.ctxt, default_channels_first)
            parseSuccess = parseSuccess and parsePass

        if parseSuccess:
            self.parsed = True
            return True
        else:
            raise RuntimeError('Could not parse the graph!')

    # Don't override this
    def generateInferenceCode(self, verbose: bool = False) -> str:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')

        callStack = ''
        for key, node in self.layerBinding.items():
            self.ctxt, code = node.generate(self.ctxt, verbose = verbose)

            currentBufferSize = 0
            for _buffer in self.ctxt.localObjects.values():
                if _buffer._live == True:
                    currentBufferSize += np.prod(_buffer.shape) * _buffer._type.typeWidth // 8

            if currentBufferSize > self.worstCaseBufferSize:
                self.worstCaseBufferSize = currentBufferSize

            for section in code:
                for substr in section:
                    callStack += substr + '\n'

        lines = callStack.split('\n')
        lines = [line for line in lines if line.strip()]
        callStack = '\n'.join(lines)

        return callStack

    # Don't override this
    def generateGlobalDefinitionCode(self, verbose: bool = False) -> str:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')

        callStack = ''
        for name, obj in self.ctxt.globalObjects.items():
            if isinstance(obj, GlobalDefinition):
                callStack += obj.definition

        return callStack

    # Don't override this
    def generateInferenceInitializationCode(self) -> str:
        if not self.parsed or not self.bound:
            raise ValueError('You need to parse and bind the network before generating code!')

        ctxt = self.ctxt.copy()

        callStack = ''
        for node in ctxt.localObjects.values():
            # WIESEP: We don't want to initialize the struct buffers as this should be handled by the ArgumentStructGeneration
            if isinstance(node, StructBuffer):
                continue

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
        inputs = self.inputs()
        outputs = self.outputs()

        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, (StructBuffer, ConstantBuffer)):
                assert issubclass(node._type, PointerClass), f"IO Buffer {node.name} is not a Pointer!"
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

        numBytes = []
        for node in inputs:
            numBytes.append(str(np.prod(node.shape) * node._type.referencedType.typeWidth // 8))
        callStack += ", ".join(numBytes)

        callStack += "};\n"

        callStack += "static const uint32_t " + ctxt._mangle("outputs_bytes") + f"[{len(outputs)}] = " + "{"

        numBytes = []
        for node in outputs:
            numBytes.append(str(np.prod(node.shape) * node._type.referencedType.typeWidth // 8))
        callStack += ", ".join(numBytes)

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

        inputs = self.inputs()
        outputs = self.outputs()

        callStack = ''
        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, StructBuffer):
                assert issubclass(node._type, PointerClass), f"Global VariableBuffer {node.name} is not a Pointer!"
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

        inputs = self.inputs()
        outputs = self.outputs()
        callStack = ''

        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, StructBuffer):
                assert issubclass(node._type, PointerClass), f"Global VariableBuffer {node.name} is not a Pointer!"
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
                size += int((np.prod(_buffer.shape) * _buffer._type.typeWidth // 8))

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
            totalSum += int(i.computeOps())
            if verbose:
                print("Layer " + str(i.node.name) + str("\nNumber of operations: \t\t") +
                      str("%12s\n" % int(i.computeOps())))
        return totalSum

        # Don't override this
    def _exportGraph(self, folderPath, fileName):
        relativeDataPath = os.path.join(folderPath, fileName + _dataExtension)
        absoluteDataPath = os.path.abspath(relativeDataPath)
        relativeOnnxPath = os.path.join(folderPath, fileName + _graphExtension)
        absoluteOnnxPath = os.path.abspath(relativeOnnxPath)

        if not os.path.isabs(absoluteOnnxPath) or not os.path.isabs(absoluteDataPath):
            raise OSError(f"Error exporting the context to: {absoluteOnnxPath}")

        model = gs.export_onnx(self.graph)
        convert_model_to_external_data(model, location = fileName + _dataExtension)
        onnx.save(model, absoluteOnnxPath)

    def exportDeeployState(self, folderPath, fileName):

        os.makedirs(os.path.abspath(folderPath), exist_ok = True)
        self._exportGraph(folderPath, fileName)
        self.ctxt.exportNetworkContext(folderPath, fileName)

    @staticmethod
    def _importONNXGraph(folderPath, fileName) -> gs.Graph:
        relativePath = os.path.join(folderPath, fileName + _graphExtension)
        absolutePath = os.path.abspath(relativePath)

        if not os.path.isabs(absolutePath) or not os.path.exists(absolutePath):
            raise OSError(f"File or path does not exist: {absolutePath}")

        onnx_graph = onnx.load_model(absolutePath)
        return gs.import_onnx(onnx_graph)

    def importDeeployState(self, folderPath, fileName) -> Tuple[gs.Graph, NetworkContext]:
        self.graph = NetworkDeployer._importONNXGraph(folderPath, f"{fileName}")
        self.ctxt = NetworkContext.importNetworkCtxt(folderPath, f"{fileName}")


class NetworkDeployer(NetworkContainer):

    prepared = False

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, PointerType],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable[[gs.Graph], Schedule] = lambda graph: list(graph.nodes),
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState"):
        super().__init__(graph, deploymentPlatform, inputTypes, scheduler, name, deeployStateDir = deeployStateDir)

        self.loweringOptimizer = loweringOptimizer
        self.default_channels_first = default_channels_first

    # Don't override this
    def lower(self, graph: gs.Graph) -> gs.Graph:
        return self.loweringOptimizer.optimize(graph)

    # Don't override this
    # Duplicate constants with multiple users
    def _duplicateConstants(self, graph: gs.Graph):
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

        self._duplicateConstants(self.graph)

        self.exportDeeployState(self.deeployStateDir, _middlewarePreLoweringFilename)

        self.graph = self.lower(self.graph)  # This lowers the graph to a deployable format

        self.exportDeeployState(self.deeployStateDir, _middlewarePostLoweringFilename)

    # Don't override this unless you know what you are doin
    def backEnd(self):
        self.parse(self.default_channels_first)  # This reparses the lowered graph

        self.broadcast(self.default_channels_first)  # This broadcasts all tensors offline

        self.parseCtxt(self.default_channels_first)  # This parses the lowered and broadcasted graph

        self.exportDeeployState(self.deeployStateDir, _backendPostParsingFilename)

        self.bind()  # This binds the graph to the node templates
        self.codeTransform()

        self.exportDeeployState(self.deeployStateDir, _backendPostBindingFilename)

    # Don't override this
    def prepare(self):
        # MIDDLE END
        self.middleWare()
        # BACK END - Inherited from NetworkContainer
        self.backEnd()
        # FINAL TRANSFORMS
        self.prepared = True

    # Don't override this
    def generateFunction(self, verbose: bool = False) -> str:
        if not self.prepared:
            self.prepare()

        return self.generateInferenceCode(verbose = verbose)
