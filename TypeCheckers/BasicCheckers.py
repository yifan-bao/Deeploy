# ----------------------------------------------------------------------
#
# File: BasicCheckers.py
#
# Last edited: 16.12.2021        
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

from parserTypes import *
from parserTypes import _mangleVariableName, _mangleParameterName
from enum import Enum

class AddChecker(NodeTypeChecker):
    def __init__(self, input_type: Enum, output_type: Enum):
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type
        
    # Checks that input is at most a 31-Bit value (avoid overflow)
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]

        #SCHEREMO: Use this once global buffers get typed correctly
        #return all([node.nLevels <= 2**31 for node in inputs]) and all([node._type == inputs[0]._type for node in inputs])
        return all([node.nLevels <= 2**(self.input_type._value_) for node in inputs]) 
        
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        newCtxt = ctxt.copy()

        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [newCtxt.lookup(name) for name in inputName]        
        
        nLevels = inputs[0].nLevels + inputs[1].nLevels
        
        outputNodes = node.outputs
        outputNames = [_mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = NetworkBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                nb._type = self.output_type
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.globalObjects[name]
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                newCtxt.globalObjects[name]._type = self.output_type
                
        return newCtxt

class GatherChecker(NodeTypeChecker):
    def __init__(self, input_type: Enum):
        super().__init__()
        self.input_type = input_type

    # Checks that input is at most an 32-Bit value
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]
        
        return all([node.nLevels <= 2**(self.input_type._value_) for node in inputs])
    
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        newCtxt = ctxt.copy()

        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [newCtxt.lookup(name) for name in inputName]        
        
        nLevels = inputs[0].nLevels
        
        outputNodes = node.outputs
        outputNames = [_mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = NetworkBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                nb._type = self.input_type
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                newCtxt.globalObjects[name]._type = self.input_type
                
        return newCtxt

class ReshapeChecker(NodeTypeChecker):
    def __init__(self, input_type: Enum):
        super().__init__()
        self.input_type = input_type

    # Checks that input is at most an 32-Bit value
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]
        
        return all([node.nLevels <= 2**(self.input_type._value_) for node in inputs])
        
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        newCtxt = ctxt.copy()

        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [newCtxt.lookup(name) for name in inputName]        
        
        nLevels = inputs[0].nLevels
        
        outputNodes = node.outputs
        outputNames = [_mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = NetworkBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                nb._type = self.input_type
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                newCtxt.globalObjects[name]._type = self.input_type
                
        return newCtxt

    
class MHSAChecker(NodeTypeChecker):
    def __init__(self, input_type: Enum, output_type: Enum):
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type

    # Checks that input is at most an 8-Bit value
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]
        
        return all([node.nLevels <= 2**(self.input_type._value_) for node in inputs[0:2]])
    
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        newCtxt = ctxt.copy()

        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [newCtxt.lookup(name) for name in inputName]        

        wo_weight = inputs[-2]
        
        nLevels = 2**16 * wo_weight.shape[-1] 
        
        outputNodes = node.outputs
        outputNames = [_mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = NetworkBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                nb._type = self.output_type
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                newCtxt.globalObjects[name]._type = self.output_type
                
        return newCtxt

class GEMMChecker(NodeTypeChecker):
    def __init__(self, input_type: Enum, output_type: Enum):
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type

    # Checks that input is at most an 8-Bit value
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]
        
        return all([node.nLevels <= 2**(self.input_type._value_) for node in inputs[0:1]]) and all([node._type == self.input_type for node in inputs[0:1]])
        
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        newCtxt = ctxt.copy()

        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [newCtxt.lookup(name) for name in inputName]        

        transA = kwargs['transA']
        transB = kwargs['transB']
        
        nLevels = 2**((self.input_type._value_)*2) * inputs[0].shape[-1 -transA]
        
        outputNodes = node.outputs
        outputNames = [_mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = NetworkBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                nb._type = self.output_type
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                newCtxt.globalObjects[name]._type = self.output_type
                
        return newCtxt


class iLayerNormChecker(NodeTypeChecker):
    def __init__(self, input_type: Enum, output_type: Enum):
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type

    # Checks that input is at most an 8-Bit value
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]
        
        return all([node.nLevels <= 2**(self.input_type._value_) for node in inputs[0:1]]) and inputs[0]._type == self.input_type
                
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        newCtxt = ctxt.copy()
        
        nLevels = 2**8
        
        outputNodes = node.outputs
        outputNames = [_mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = NetworkBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                nb._type = self.output_type
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                newCtxt.globalObjects[name]._type = self.output_type
                
        return newCtxt

class GELUChecker(NodeTypeChecker):
    def __init__(self, input_type: Enum, output_type: Enum):
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type

    # Checks that input is at most an 8-Bit value
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]

        return all([node.nLevels <= 2**(self.input_type._value_) for node in inputs[0:1]]) and inputs[0]._type == self.input_type
        
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        newCtxt = ctxt.copy()
        
        nLevels = 2**8
        
        outputNodes = node.outputs
        outputNames = [_mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = NetworkBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                nb._type = self.output_type
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                newCtxt.globalObjects[name]._type = self.output_type
                
        return newCtxt


class ConvChecker(NodeTypeChecker):
    def __init__(self, input_type: Enum, output_type: Enum):
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type

    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]
        
        return all([node.nLevels <= 2**(self.input_type._value_) for node in inputs[0:1]]) and all([node._type == self.input_type for node in inputs[0:1]])
        
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        newCtxt = ctxt.copy()
        
        # Dummy value for nLevels
        inputTensor = ctxt.lookup(_mangleVariableName(node.inputs[0].name))
        weightTensor = ctxt.lookup(_mangleVariableName(node.inputs[1].name))
        nLevels = np.prod(kwargs['kernel_shape']) * weightTensor.shape[1] * 2**16
        
        outputNodes = node.outputs
        outputNames = [_mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = NetworkBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                nb._type = self.output_type
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                newCtxt.globalObjects[name]._type = self.output_type
                
        return newCtxt


class RequantShiftChecker(NodeTypeChecker):
    def __init__(self, input_type: Enum, output_type: Enum):
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type

    # Checks that input is at most a 32-Bit value
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        inputName = [_mangleVariableName(i.name) for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]

        return all([node.nLevels <= 2**(self.input_type._value_) for node in inputs[0:1]]) and inputs[0]._type == self.input_type
        
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        newCtxt = ctxt.copy()

        # Dummy value for nLevels
        nLevels = 256
        
        outputNodes = node.outputs
        outputNames = [_mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = NetworkBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                nb._type = self.output_type
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                newCtxt.globalObjects[name]._type = self.output_type
                
        return newCtxt
    
class DummyChecker(NodeTypeChecker):
    def __init__(self):
        super().__init__()

    # Override this. This should check that the input n_levels are appropriate for the kernel
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        return True

    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        newCtxt = ctxt.copy()

        # Dummy value for nLevels
        nLevels = 256
        
        outputNodes = node.outputs
        outputNames = [_mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = NetworkBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                nb._type = 'int8_t'
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                newCtxt.globalObjects[name]._type = 'int8_t'
                
        return newCtxt
