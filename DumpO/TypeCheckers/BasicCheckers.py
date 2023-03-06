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
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]

class PadChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]

class AddChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels + inputs[1].nLevels]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed or inputs[1]._signed:
            return [True]
        else:
            return [False]

class GatherChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]

class ReshapeChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]

class MHSAChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [True]

class CLCAChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [True]

class LinearAttentionChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [True]

class GEMMChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**((self.input_types[0]._value_)*2) * inputs[0].shape[-1 - parserDict['transA']]]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [True]

class iLayerNormChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [True]

class MulChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[1]._value_)]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed or inputs[1]._signed:
            return [True]
        else:
            return [False]

class IntegerDivChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.output_types[0]._value_)]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed or inputs[1]._signed:
            return [True]
        else:
            return [False]


class RQIntegerDivChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.output_types[0]._value_)]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed or inputs[1]._signed:
            return [True]
        else:
            return [False]


class MatMulChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [np.max(inputs[0].shape)*np.max(inputs[1].shape)*2**(self.input_types[0]._value_)]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed or inputs[1]._signed:
            return [True]
        else:
            return [False]


class ReduceMeanChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]

class SoftmaxChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [False]

class GELUChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]

class ConvChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        weight = inputs[1]
        return [np.prod(parserDict['kernel_shape']) * weight.nLevels * weight.shape[1] * 2**(self.input_types[0]._value_)]
    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [True]

class RequantShiftChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [parserDict['n_levels']]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [parserDict["signed"]]

class DummyChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [2**(self.input_types[0]._value_)]
