# ----------------------------------------------------------------------
#
# File: CMSISCheckers.py
#
# Last edited: 18.12.2021        
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

from DumpO.DumpOTypes import *
from DumpO.DumpOManglers import *
from enum import Enum

class CMSISSaturatingAddChecker(NodeTypeChecker):
    def __init__(self, input_type: Enum):
        super().__init__()
        self.input_type = input_type
        
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        inputName = [mangleVariableName(i.name) for i in node.inputs]
        inputs = [ctxt.lookup(name) for name in inputName]

        #SCHEREMO: Use this once global buffers get typed correctly
        #return all([node.nLevels <= 2**31 for node in inputs]) and all([node._type == inputs[0]._type for node in inputs])
        return all([node.nLevels <= 2**(self.input_type._value_) for node in inputs]) 
        
    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> NetworkContext:
        newCtxt = ctxt.copy()

        inputName = [mangleVariableName(i.name) for i in node.inputs]
        inputs = [newCtxt.lookup(name) for name in inputName]        
        
        nLevels = min(inputs[0].nLevels + inputs[1].nLevels, 2**(self.input_type._value_))
        
        outputNodes = node.outputs
        outputNames = [mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = newCtxt.VariableBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                nb._type = self.input_type
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.globalObjects[name]
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                newCtxt.globalObjects[name]._type = self.input_type
                
        return newCtxt
