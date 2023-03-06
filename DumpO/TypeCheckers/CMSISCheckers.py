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

from enum import Enum

from DumpO.DumpOTypes import *

class CMSISSaturatingAddChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [min(inputs[0].nLevels + inputs[1].nLevels, 2**(self.input_types[0]._value_))]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed or inputs[1]._signed:
            return [True]
        else:
            return [False]

class CMSISLinearChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [parserDict['n_levels']]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [bool(parserDict["signed"])]

class CMSISConvChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [parserDict['n_levels']]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [bool(parserDict["signed"])]

class CMSISMaxPoolChecker(NodeTypeChecker):
    def __init__(self, input_types: Sequence[Enum], output_types: Sequence[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels]

    def inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]
