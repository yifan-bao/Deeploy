# ----------------------------------------------------------------------
#
# File: AddTileConstraintFlow.py
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


class AddTileConstraintFlow(TileConstraintFlow):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBuffer1Name = parseDict['data_in_1']
        inputBuffer2Name = parseDict['data_in_2']
        outputBufferName = parseDict['data_out']

        for bufferName in [inputBuffer1Name, inputBuffer2Name, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        input1Shape = ctxt.lookup(inputBuffer1Name).shape

        for dim in range(len(input1Shape)):
            inputDim1Var = tilerModel.getTensorDimVar(tensorName = inputBuffer1Name, dimIdx = dim)
            inputDim2Var = tilerModel.getTensorDimVar(tensorName = inputBuffer2Name, dimIdx = dim)
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)

            tilerModel.addConstraint(inputDim1Var == inputDim2Var)
            tilerModel.addConstraint(inputDim1Var == outputDimVar)

        return tilerModel

    @classmethod
    def serializeTilingSolution(cls, tilingSolution: NodeMemoryConstraint, outputCubes: List[HyperRectangle],
                                targetMemLevel: str, ctxt: NetworkContext,
                                nodeRep: Dict) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        addrNames = ['data_in_1', 'data_in_2', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel, nodeRep, addrNames)

        replacements = {"size": []}

        replacementTypes = {"size": Pointer(IntegerDataTypes.uint16_t)}

        for cube in outputCubes:
            newSize = np.prod(cube.dims)
            replacements["size"].append(newSize)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            inputLoadSchedule.append({"data_in_1": cube, "data_in_2": cube})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
