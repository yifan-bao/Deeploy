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

class ReshapeLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def generate(self, ctxt: NetworkContext) -> (NetworkContext, List[str]):        
        outputs = [node for node in self.node.outputs]
        inputs = [node for node in self.node.inputs]
        
        outputNames = [node.name for node in outputs]
        inputNames = [node.name for node in inputs]
        
        alloc = ctxt.allocLocal(self.node.name, outputNames)
        call = self.mapper.generate(ctxt)
        dealloc = ctxt.freeLocal(self.node.name, inputNames)
        
        return (ctxt, [call])
    
class GatherLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)
    
class iGELULayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

class iSoftmaxLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)
        
class RequantShiftLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict, channels_first) -> (List[np.shape], List[np.shape]):

        channel_dim = inputShapes[0][1]
        inputShapes[2]  = [inputShapes[0][0], channel_dim] + list(inputShapes[1][1:])
        inputShapes[1] = [inputShapes[0][0], channel_dim] + list(inputShapes[1][1:])
            
        return (inputShapes, outputShapes)
        
class AddLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict, channels_first) -> (List[np.shape], List[np.shape]):
        if len(inputShapes[0]) > len(inputShapes[1]):
            inputShapes[1] = inputShapes[0]
        else:
            inputShapes[0] = inputShapes[1]

        return (inputShapes, outputShapes)
            
class GEMMLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict, channels_first) -> (List[np.shape], List[np.shape]):
        if parserDict['transA']:
            M = inputShapes[0][-1]
        else:
            M = inputShapes[0][-2]

        if parserDict['transB']:
            N = inputShapes[1][-2]
        else:
            N = inputShapes[1][-1]

        if len(inputShapes) == 3:
            inputShapes[2] = [M,N]
            
        return (inputShapes, outputShapes)
    
class ConvLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict, channels_first) -> (List[np.shape], List[np.shape]):
        if len(inputShapes) == 3:
            inputShapes[2] = inputShapes[1][-1]
        return (inputShapes, outputShapes)

class PadLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

class MaxPoolLayer(ONNXLayer):
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
    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict, channels_first) -> (List[np.shape], List[np.shape]):
        inputShapes[4] = inputShapes[3][0]
        inputShapes[6] = inputShapes[5][0]
        inputShapes[8] = inputShapes[7][0]
        inputShapes[10] = inputShapes[9][0]

        return(inputShapes, outputShapes)
