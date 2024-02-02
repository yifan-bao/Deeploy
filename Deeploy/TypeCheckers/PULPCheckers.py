# ----------------------------------------------------------------------
#
# File: PULPCheckers.py
#
# Last edited: 03.10.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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

from typing import Dict, List, Sequence

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.DeeployTypes import VariableBuffer
from Deeploy.TypeCheckers.SignPropTypeChecker import SignPropTypeChecker


class PULPRQAddChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[PointerClass], output_types: Sequence[PointerClass]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [parserDict['rqsOut_n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [bool(parserDict["rqsOut_signed"])]

    # Override this. This should compute the signednes of each output node of the Layer
    def checkOutputType(self, inputs: List[VariableBuffer], parserDict: Dict) -> bool:
        outputTypeSigned = self.output_types[0].referencedType.typeMin < 0
        if parserDict['rqsOut_signed'] and outputTypeSigned:
            return True
        if (not parserDict['rqsOut_signed']) and (not outputTypeSigned):
            return True
        return False


class PULPRequantShiftChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[PointerClass], output_types: Sequence[PointerClass]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [parserDict['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [parserDict["signed"]]

    def checkOutputType(self, inputs: List[VariableBuffer], parserDict: Dict) -> bool:
        outputTypeSigned = self.output_types[0].referencedType.typeMin < 0
        if parserDict['signed'] and outputTypeSigned:
            return True
        if (not parserDict['signed']) and (not outputTypeSigned):
            return True
        return False


class PULPConvChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[PointerClass], output_types: Sequence[PointerClass]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [parserDict['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [bool(parserDict["signed"])]

    # Override this. This should compute the signednes of each output node of the Layer
    def checkOutputType(self, inputs: List[VariableBuffer], parserDict: Dict) -> bool:
        outputTypeSigned = self.output_types[0].referencedType.typeMin < 0
        if parserDict['signed'] and outputTypeSigned:
            return True
        if (not parserDict['signed']) and (not outputTypeSigned):
            return True
        return False


class PULPLinearChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[PointerClass], output_types: Sequence[PointerClass]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [parserDict['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        return [bool(parserDict["signed"])]

    # Override this. This should compute the signednes of each output node of the Layer
    def checkOutputType(self, inputs: List[VariableBuffer], parserDict: Dict) -> bool:
        outputTypeSigned = self.output_types[0].referencedType.typeMin < 0
        if parserDict['signed'] and outputTypeSigned:
            return True
        if (not parserDict['signed']) and (not outputTypeSigned):
            return True
        return False


class PULPMaxPoolChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[PointerClass], output_types: Sequence[PointerClass]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]

    # Override this. This should compute the signednes of each output node of the Layer
    def checkOutputType(self, inputs: List[VariableBuffer], parserDict: Dict) -> bool:
        return True
