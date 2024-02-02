# ----------------------------------------------------------------------
#
# File: RequantShiftTileConstraintFlow.py
#
# Last edited: 05.10.2023
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

from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes, SignedIntegerDataTypes
from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Tiling.MemoryConstraints import NodeMemoryConstraint
from Deeploy.Tiling.TileConstraintFlow import TileConstraintFlow
from Deeploy.Tiling.TilerModel import TilerModel
from Deeploy.Tiling.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme, \
    computeHyperRectangleList, extractTilingTransfer


class RequantShiftTileConstraintFlow(TileConstraintFlow):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        mulBufferName = parseDict['mul']
        addBufferName = parseDict['add']
        outputBufferName = parseDict['data_out']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, mulBufferName, addBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputShape = ctxt.lookup(inputBufferName).shape

        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = 0)
        addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = 0)

        tilerModel.addConstraint(mulChannelVar == addChannelVar)

        channels_first = parseDict['channels_first']
        if not channels_first:
            inChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = len(inputShape) - 1)
        else:
            inChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)

        tilerModel.addConstraint(mulChannelVar == inChannelVar)

        for dim in range(len(inputShape)):
            inputDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = dim)
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)
            tilerModel.addConstraint(inputDimVar == outputDimVar)  # Batch

        return tilerModel

    @classmethod
    def serializeTilingSolution(cls, tilingSolution: NodeMemoryConstraint, outputCubes: List[HyperRectangle],
                                targetMemLevel: str, ctxt: NetworkContext,
                                nodeRep: Dict) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        addrNames = ['data_in', 'mul', 'add', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel, nodeRep, addrNames)

        inputCubes = outputCubes

        rqCubes = []

        replacements = {"size": [], "channel_width": [], "channels": []}
        replacementTypes = {
            "size": Pointer(IntegerDataTypes.uint16_t),
            "channel_width": Pointer(IntegerDataTypes.uint16_t),
            "channels": Pointer(IntegerDataTypes.uint16_t)
        }

        for cube in inputCubes:

            if nodeRep['channels_first']:
                rqCube = HyperRectangle((cube.offset[1],), (cube.dims[1],))
                channelDim = cube.dims[1]
            else:
                rqCube = HyperRectangle((cube.offset[-1],), (cube.dims[-1],))
                channelDim = cube.dims[-1]

            rqCubes.append(rqCube)

            size = np.prod(cube.dims[1:])
            channelWidth = size // channelDim
            channels = channelDim

            replacements['size'].append(size)
            replacements['channel_width'].append(channelWidth)
            replacements['channels'].append(channels)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, rq in zip(inputCubes, rqCubes):
            inputLoadSchedule.append({"data_in": a, "add": rq, "mul": rq})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
