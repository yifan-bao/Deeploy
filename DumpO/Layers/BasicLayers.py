# ----------------------------------------------------------------------
#
# File: BasicLayers.py
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

import onnx
import onnx_graphsurgeon as gs
from typing import List

from DumpO.DumpOTypes import *
from DumpO.DumpOManglers import *

class ReshapeLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def generate(self, ctxt: NetworkContext) -> (NetworkContext, List[str]):        
        outputs = [node for node in self.node.outputs]
        inputs = [node for node in self.node.inputs]
        
        outputNames = [mangleVariableName(node.name) for node in outputs]
        inputNames = [mangleVariableName(node.name) for node in inputs]
        
        alloc = ctxt.allocLocal(self.node.name, outputNames)
        call = self.mapper.generate()
        dealloc = ctxt.freeLocal(self.node.name, inputNames)
        
        return (ctxt, [call])

class GatherLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def generate(self, ctxt: NetworkContext) -> (NetworkContext, List[str]):        
        outputs = [node for node in self.node.outputs]
        inputs = [node for node in self.node.inputs]
        
        outputNames = [mangleVariableName(node.name) for node in outputs]
        inputNames = [mangleVariableName(node.name) for node in inputs]
        
        alloc = ctxt.allocLocal(self.node.name, outputNames)
        call = self.mapper.generate()
        dealloc = ctxt.freeLocal(self.node.name, inputNames)
        
        return (ctxt, [call])
    
class iGELULayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)
        
class RequantShiftLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)
        
class AddLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

class GEMMLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)
    
class ConvLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)
        
class iLayerNormLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

class TransposeLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

class MHSALayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)
