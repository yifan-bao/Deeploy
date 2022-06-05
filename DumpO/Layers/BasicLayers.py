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

    def generate(self, ctxt: NetworkContext, verbose: bool = False) -> (NetworkContext, List[str]):
        outputs = [node for node in self.node.outputs]
        inputs = [node for node in self.node.inputs]

        outputNames = [node.name for node in outputs]
        inputNames = [node.name for node in inputs]

        alloc = ctxt.allocLocal(self.node.name, outputNames)
        call = self.mapper.generate(ctxt, verbose=verbose)
        dealloc = ctxt.freeLocal(self.node.name, inputNames)

        return (ctxt, [call])

class GatherLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

class iGELULayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        compAbs = self.mapper.nodeRep['size']
        compAdd = self.mapper.nodeRep['size']
        compSqr = self.mapper.nodeRep['size']
        compMul = self.mapper.nodeRep['size']
        compAdd = self.mapper.nodeRep['size']
        compMul2 = self.mapper.nodeRep['size']
        compAdd2 = self.mapper.nodeRep['size']
        compDiv = self.mapper.nodeRep['size']
        return compAbs + compAdd + compSqr + compMul + compAdd + compMul2 + compAdd2 + compDiv


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

    def computeOps(self):
        return self.mapper.nodeRep['size'] * 3 # One add, one mul, one div

class AddLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict, channels_first) -> (List[np.shape], List[np.shape]):
        if len(inputShapes[0]) > len(inputShapes[1]):
            inputShapes[1] = inputShapes[0]
        else:
            inputShapes[0] = inputShapes[1]

        return (inputShapes, outputShapes)

    def computeOps(self):
        return self.mapper.nodeRep['size']

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

class ReduceMeanLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

class iLayerNormLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        compAverage = self.mapper.nodeRep['size']
        compNormalize = self.mapper.nodeRep['size']
        compSqr = self.mapper.nodeRep['size']
        compSum = self.mapper.nodeRep['size']
        compSqrt = self.mapper.nodeRep['size']
        compDiv = self.mapper.nodeRep['size']
        return compAverage + compNormalize + compSqr + compSum + compSqrt + compDiv

class TransposeLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

class LinearAttentionLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)
    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict, channels_first) -> (List[np.shape], List[np.shape]):
        inputShapes[4] = inputShapes[3][0]
        inputShapes[6] = inputShapes[5][0]
        inputShapes[8] = inputShapes[7][0]
        inputShapes[10] = inputShapes[9][0]

        return(inputShapes, outputShapes)

    def computeOps(self):
        # seqLen = self.mapper.nodeRep['in_C']
        # dim = self.mapper.nodeRep['dim']
        # dim_head = self.mapper.nodeRep['dim_head']
        # heads = self.mapper.nodeRep['heads']
        # QOps = seqLen * dim * dim_head * heads * 2
        # # WQ * Q (H )
        # KOps = seqLen * dim * dim_head * heads * 2
        # # WK * K
        # VOps = seqLen * dim * dim_head * heads * 2
        # # WV * V
        # KVOps = seqLen * dim_head * dim_head * heads * 2
        # # Q * KT
        # QKVOps = seqLen * dim_head * dim_head * heads * 2
        # # N H S S * N H S D -> N H S D
        # OutOps = seqLen * dim_head * heads * dim * 2
        # # WO * O
        # totOps = QOps + KOps + VOps + KVOps + QKVOps + OutOps
        # return totOps

        return 0
class MHSALayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)
    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict, channels_first) -> (List[np.shape], List[np.shape]):
        inputShapes[4] = inputShapes[3][0]
        inputShapes[6] = inputShapes[5][0]
        inputShapes[8] = inputShapes[7][0]
        inputShapes[10] = inputShapes[9][0]

        return(inputShapes, outputShapes)

    def computeOps(self):
        seqLen = self.mapper.nodeRep['in_C']
        dim = self.mapper.nodeRep['dim']
        dim_head = self.mapper.nodeRep['dim_head']
        heads = self.mapper.nodeRep['heads']
        QOps = seqLen * dim * dim_head * heads * 2
        # WQ * Q (H )
        KOps = seqLen * dim * dim_head * heads * 2
        # WK * K
        VOps = seqLen * dim * dim_head * heads * 2
        # WV * V
        QKOps = seqLen * seqLen * dim_head * heads * 2
        # Q * KT
        AVOps = seqLen * seqLen * dim_head * heads * 2
        # N H S S * N H S D -> N H S D
        OutOps = seqLen * dim_head * heads * dim * 2
        # WO * O
        totOps = QOps + KOps + VOps + QKOps + AVOps + OutOps
        return totOps
