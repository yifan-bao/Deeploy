# ----------------------------------------------------------------------
#
# File: CMSISLayers.py
#
# Last edited: 22.12.2021
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

from DumpO.DumpOTypes import *
from DumpO.Layers.BasicLayers import *

class RQSConvLayer(ConvLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict, channels_first) -> (List[np.shape], List[np.shape]):
        if channels_first:
            inputShapes[2] = outputShapes[0][1] # Channels out dimension of Kernel
            inputShapes[3] = outputShapes[0][1] # Channels out dimension of Kernel
            inputShapes[4] = outputShapes[0][1] # Channels out dimension of Kernel
        else:
            inputShapes[2] = outputShapes[0][-1] # Channels out dimension of Kernel
            inputShapes[3] = outputShapes[0][-1] # Channels out dimension of Kernel
            inputShapes[4] = outputShapes[0][-1] # Channels out dimension of Kernel
        return (inputShapes, outputShapes)

    def computeOps(self):
        if "group" in self.mapper.nodeRep.keys():
            groups = self.mapper.nodeRep['group']
        else:
            groups = 1
        opsPerPx = int(np.prod(self.mapper.nodeRep['kernel_shape']) * self.mapper.nodeRep['ch_im_in'] * self.mapper.nodeRep['ch_im_out'] / groups) * 2
        if 'dim_im_out_y' in self.mapper.nodeRep.keys():
            numPx = self.mapper.nodeRep['dim_im_out_x'] * self.mapper.nodeRep['dim_im_out_y']
        else:
            numPx = self.mapper.nodeRep['dim_im_out_x']
        return numPx * opsPerPx


class RQSGEMMLayer(GEMMLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict, channels_first) -> (List[np.shape], List[np.shape]):
        inputShapes[2] = inputShapes[1][-2] # Channels out dimension of Kernel
        # inputShapes[3] = inputShapes[1][-1] # Channels out dimension of Kernel
        # inputShapes[4] = inputShapes[1][-1] # Channels out dimension of Kernel
        return (inputShapes, outputShapes)

    def computeOps(self):
        ops = self.mapper.nodeRep['in_N'] * self.mapper.nodeRep['in_C'] * self.mapper.nodeRep['weight_C'] * 2
        return ops

class RQIntegerDivLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict, channels_first) -> (List[np.shape], List[np.shape]):
        return (inputShapes, outputShapes)
