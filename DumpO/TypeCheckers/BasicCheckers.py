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

from enum import Enum

from DumpO.DumpOTypes import *

class TransposeChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels]

class PadChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels]

class AddChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels + inputs[1].nLevels]
        
class GatherChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)
        
    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels]


class ReshapeChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)
        
    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels]

#SCHEREMO: Fix this
class MHSAChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)
        
    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        wo_weight = inputs[9]
        return [2**16 * wo_weight.shape[-1]]

class GEMMChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)
        
    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**((self.input_types[0]._value_)*2) * inputs[0].shape[-1 - parserDict['transA']]]

class iLayerNormChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)
        
    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]

class SoftmaxChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)
        
    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]
    
class GELUChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)
        
    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]

class ConvChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)
        
    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        weight = inputs[1]
        return [np.prod(parserDict['kernel_shape']) * weight.shape[1] * 2**(self.input_types[0]._value_)]

class RequantShiftChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)
        
    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**8]
        
class DummyChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)
        
    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]
        
