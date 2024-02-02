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

from typing import List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeMapper, ONNXLayer, Shape


class SliceLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class ReshapeLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class GatherLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class iGELULayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        compAbs = self.mapper.parser.parserDict['size']
        compAdd = self.mapper.parser.parserDict['size']
        compSqr = self.mapper.parser.parserDict['size']
        compMul = self.mapper.parser.parserDict['size']
        compAdd = self.mapper.parser.parserDict['size']
        compMul2 = self.mapper.parser.parserDict['size']
        compAdd2 = self.mapper.parser.parserDict['size']
        compDiv = self.mapper.parser.parserDict['size']
        return compAbs + compAdd + compSqr + compMul + compAdd + compMul2 + compAdd2 + compDiv


class RQSiGELULayer(iGELULayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class iSoftmaxLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class ITAMaxLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class RequantShiftLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[Shape], outputShapes: Shape, parserDict,
                      channels_first) -> Tuple[Shape, Shape]:

        channel_dim = inputShapes[0][1]
        inputShapes[2] = [inputShapes[0][0], channel_dim] + list(inputShapes[2][1:])
        inputShapes[1] = [inputShapes[0][0], channel_dim] + list(inputShapes[1][1:])

        return (inputShapes, outputShapes)

    def computeOps(self):
        return self.mapper.parser.parserDict['size'] * 3  # One add, one mul, one div


class AddLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, parserDict, channels_first) -> Tuple[Shape, Shape]:
        if len(inputShapes[0]) > len(inputShapes[1]):
            inputShapes[1] = inputShapes[0]
        else:
            inputShapes[0] = inputShapes[1]

        return (inputShapes, outputShapes)

    def computeOps(self):
        return self.mapper.parser.parserDict['size']


class MatMulLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        return 2 * self.mapper.parser.parserDict['M'] * self.mapper.parser.parserDict[
            'N'] * self.mapper.parser.parserDict['O'] * self.mapper.parser.parserDict['batch']


class RQMatMulLayer(MatMulLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[Shape], outputShapes: Shape, parserDict,
                      channels_first) -> Tuple[Shape, Shape]:

        channel_dim = inputShapes[0][1]
        inputShapes[3] = [inputShapes[0][0]] + list(inputShapes[3][1:])
        inputShapes[2] = [inputShapes[0][0]] + list(inputShapes[2][1:])

        return (inputShapes, outputShapes)

    def computeOps(self):
        matmul = super().computeOps()
        rqs = self.mapper.parser.parserDict['size'] * 3
        return matmul + rqs


class IntegerDivLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class RQIntegerDivLayer(IntegerDivLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class GEMMLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, parserDict, channels_first) -> Tuple[Shape, Shape]:
        if parserDict['transA']:
            M = inputShapes[0][-1]
        else:
            M = inputShapes[0][-2]

        if parserDict['transB']:
            N = inputShapes[1][-2]
        else:
            N = inputShapes[1][-1]

        if len(inputShapes) == 3:
            inputShapes[2] = [M, N]

        return (inputShapes, outputShapes)

    def computeOps(self):
        matmul = 2 * self.mapper.parser.parserDict['M'] * self.mapper.parser.parserDict[
            'N'] * self.mapper.parser.parserDict['O'] * self.mapper.parser.parserDict['batch']
        gemm = matmul + 3 * self.mapper.parser.parserDict['M'] * self.mapper.parser.parserDict[
            'O'] * self.mapper.parser.parserDict['batch']

        return gemm


class RQGEMMLayer(GEMMLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[Shape], outputShapes: Shape, parserDict,
                      channels_first) -> Tuple[Shape, Shape]:
        if parserDict['transA']:
            M = inputShapes[0][-1]
        else:
            M = inputShapes[0][-2]

        if parserDict['transB']:
            N = inputShapes[1][-2]
        else:
            N = inputShapes[1][-1]

        if len(inputShapes) == 5:
            inputShapes[2] = [M, N]
            inputShapes[4] = [inputShapes[0][0]] + list(inputShapes[4][1:])
            inputShapes[3] = [inputShapes[0][0]] + list(inputShapes[3][1:])
        else:
            inputShapes[3] = [inputShapes[0][0]] + list(inputShapes[3][1:])
            inputShapes[2] = [
                inputShapes[0][0],
            ] + list(inputShapes[2][1:])

        return (inputShapes, outputShapes)

    def computeOps(self):
        gemm = super().computeOps()
        rqs = self.mapper.parser.parserDict['size'] * 3
        return gemm + rqs


class MulLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, parserDict, channels_first) -> Tuple[Shape, Shape]:
        inputShapes[1] = 1
        return (inputShapes, outputShapes)


class ConvLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, parserDict, channels_first) -> Tuple[Shape, Shape]:
        if len(inputShapes) == 3:
            inputShapes[2] = inputShapes[1][-1]
        return (inputShapes, outputShapes)

    def computeOps(self):
        if "group" in self.mapper.parser.parserDict:
            groups = self.mapper.parser.parserDict['group']
        else:
            groups = 1
        opsPerPx = int(
            np.prod(self.mapper.parser.parserDict['kernel_shape']) * self.mapper.parser.parserDict['ch_im_in'] *
            self.mapper.parser.parserDict['ch_im_out'] / groups) * 2
        if 'dim_im_out_y' in self.mapper.parser.parserDict:
            numPx = self.mapper.parser.parserDict['dim_im_out_x'] * self.mapper.parser.parserDict['dim_im_out_y']
        else:
            numPx = self.mapper.parser.parserDict['dim_im_out_x']
        return numPx * opsPerPx


class RQSConvLayer(ConvLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        conv = super().computeOps()

        if 'dim_im_out_y' in self.mapper.parser.parserDict:
            rqs = self.mapper.parser.parserDict['dim_im_out_x'] * self.mapper.parser.parserDict['dim_im_out_y'] * 3
        else:
            rqs = self.mapper.parser.parserDict['dim_im_out_x'] * 3

        return conv + rqs


class PadLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class MaxPoolLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class ReduceMeanLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class ReduceSumLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class iLayerNormLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        compAverage = self.mapper.parser.parserDict['size']
        compNormalize = self.mapper.parser.parserDict['size']
        compSqr = self.mapper.parser.parserDict['size']
        compSum = self.mapper.parser.parserDict['size']
        compSqrt = self.mapper.parser.parserDict['size']
        compDiv = self.mapper.parser.parserDict['size']
        return compAverage + compNormalize + compSqr + compSum + compSqrt + compDiv


class TransposeLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class LinearAttentionLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, parserDict, channels_first) -> Tuple[Shape, Shape]:
        inputShapes[4] = inputShapes[3][0]
        inputShapes[6] = inputShapes[5][0]
        inputShapes[8] = inputShapes[7][0]
        inputShapes[10] = inputShapes[9][0]

        return (inputShapes, outputShapes)

    def computeOps(self):
        # seqLen = self.mapper.parser.parserDict['in_C']
        # dim = self.mapper.parser.parserDict['dim']
        # dim_head = self.mapper.parser.parserDict['dim_head']
        # heads = self.mapper.parser.parserDict['heads']
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


class CLCALayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, parserDict, channels_first) -> Tuple[Shape, Shape]:
        inputShapes[3] = inputShapes[2][0]
        inputShapes[5] = inputShapes[4][0]
        inputShapes[7] = inputShapes[6][0]
        # WQ Requant
        inputShapes[8] = [parserDict['dim_head'] * parserDict['heads'], 1]
        inputShapes[9] = [parserDict['dim_head'] * parserDict['heads'], 1]
        inputShapes[10] = [parserDict['dim_head'] * parserDict['heads'], 1]
        # WK Requant
        inputShapes[11] = [1, 1]
        inputShapes[12] = [1, 1]
        inputShapes[13] = [1, 1]
        # WV Requant
        inputShapes[14] = [parserDict['dim_head'] * parserDict['heads'], 1]
        inputShapes[15] = [parserDict['dim_head'] * parserDict['heads'], 1]
        inputShapes[16] = [parserDict['dim_head'] * parserDict['heads'], 1]
        # Kdiv Requanat
        inputShapes[17] = [1, 1]
        inputShapes[18] = [1, 1]
        inputShapes[19] = [1, 1]
        # Preattn Requant
        inputShapes[20] = [1, 1]
        inputShapes[21] = [1, 1]
        inputShapes[22] = [1, 1]
        # Postattn Requant
        inputShapes[23] = [1, 1]
        inputShapes[24] = [1, 1]
        inputShapes[25] = [1, 1]
        # WO Requant
        inputShapes[26] = [parserDict['out_dim'], 1]
        inputShapes[27] = [parserDict['out_dim'], 1]
        inputShapes[28] = [parserDict['out_dim'], 1]
        return (inputShapes, outputShapes)

    def computeOps(self):

        qLen = self.mapper.parser.parserDict['q_shape'][-1]
        kLen = self.mapper.parser.parserDict['kv_shape'][-1]
        inDim = self.mapper.parser.parserDict['q_shape'][-2]
        heads = self.mapper.parser.parserDict['heads']
        dim_head = self.mapper.parser.parserDict['dim_head']
        out_dim = self.mapper.parser.parserDict['out_dim']

        # q -> Q
        QOps = qLen * 1 * inDim * heads * dim_head * 2
        # v -> V
        VOps = kLen * 1 * inDim * heads * dim_head * 2
        # V -> K
        KOps = kLen * heads * dim_head * 2
        # KOps = 0

        EOps = heads * kLen * heads * dim_head

        MMKTV = heads * dim_head * kLen * dim_head * 2
        MMQA = heads * qLen * dim_head * dim_head * 2
        MMQE = heads * qLen * dim_head * 1 * 2

        # Divs, Adds(eps), muls(delta, eps)
        DivOps = heads * qLen * dim_head + heads * qLen + 2 * heads * qLen * dim_head

        OOps = (heads * dim_head) * qLen * out_dim * 1 * 2

        return QOps + VOps + KOps + EOps + MMKTV + MMQA + MMQE + DivOps + OOps


class MHSALayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        seqLen = self.mapper.parser.parserDict['S']
        dim = self.mapper.parser.parserDict['dim']
        dim_head = self.mapper.parser.parserDict['dim_head']
        heads = self.mapper.parser.parserDict['heads']
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


class DebugPrintLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)
