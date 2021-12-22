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

class RQSConvLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict) -> (List[np.shape], List[np.shape]):
        if len(inputShapes) == 4:
            inputShapes[2] = inputShapes[1][0] # Channels out dimension of Kernel
            inputShapes[3] = inputShapes[1][0] # Channels out dimension of Kernel
        return (inputShapes, outputShapes)

class RQSGEMMLayer(ONNXLayer):
    def __init__(self, maps : List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[np.shape], outputShapes: List[np.shape], parserDict) -> (List[np.shape], List[np.shape]):
        if len(inputShapes) == 4:
            inputShapes[2] = inputShapes[1][-1] # Channels out dimension of Kernel
            inputShapes[3] = inputShapes[1][-1] # Channels out dimension of Kernel
        return (inputShapes, outputShapes)
