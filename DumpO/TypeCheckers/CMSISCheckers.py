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
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> List[int]:
        inputs = [ctxt.lookup(inputNode.name) for inputNode in node.inputs]
        
        return [min(inputs[0].nLevels + inputs[1].nLevels, 2**(self.input_type._value_))]    
        
class CMSISLinearChecker(NodeTypeChecker):
    def __init__(self, input_types: List[Enum], output_types: List[Enum]):
        super().__init__(input_types, output_types)

    def inferNumLevels(self, ctxt: NetworkContext, node: gs.ir.node.Node, **kwargs) -> List[int]:
        return [2**(self.input_type._value_)]
    
