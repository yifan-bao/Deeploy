# ----------------------------------------------------------------------
#
# File: PULPDMACheckers.py
#
# Last edited: 07.06.2023
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

from typing import Dict, List, Optional, Sequence

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.DeeployTypes import VariableBuffer
from Deeploy.TypeCheckers.SignPropTypeChecker import SignPropTypeChecker


class PULPDMASliceChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[PointerClass], output_types: Sequence[PointerClass]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer], parserDict: Dict) -> Optional[List[int]]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer], parserDict: Dict) -> Optional[List[bool]]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]
