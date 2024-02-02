# ----------------------------------------------------------------------
#
# File: TransposeTileConstraintFlow.py
#
# Last edited: 01.06.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes, SignedIntegerDataTypes
from Deeploy.DeeployTypes import NetworkContext
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import _invertPermutation, \
    _permuteList
from Deeploy.Tiling.MemoryConstraints import NodeMemoryConstraint
from Deeploy.Tiling.TileConstraintFlow import TileConstraintFlow
from Deeploy.Tiling.TilerModel import TilerModel
from Deeploy.Tiling.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme, \
    computeHyperRectangleList, extractTilingTransfer


class TransposeTileConstraintFlow(TileConstraintFlow):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # Map output dims to inputs dims
        for idx, perm_idx in enumerate(parseDict["perm"]):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = inputBufferName, dimIdx = perm_idx))

        return tilerModel

    @classmethod
    def serializeTilingSolution(cls, tilingSolution: NodeMemoryConstraint, outputCubes: List[HyperRectangle],
                                targetMemLevel: str, ctxt: NetworkContext,
                                nodeRep: Dict) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel, nodeRep, addrNames)

        inputInCubes = []

        replacementTypes = {}
        replacements: Dict[str, List[int]] = {}

        numDims = len(ctxt.lookup(nodeRep['data_in']).shape)

        for dim in range(numDims):
            replacementTypes[f"dimLen_{dim}"] = Pointer(IntegerDataTypes.uint16_t)
            replacements[f"dimLen_{dim}"] = []

        perm = nodeRep['perm']
        invPerm = _invertPermutation(perm)

        for cube in outputCubes:

            inCubeDims = _permuteList(cube.dims, invPerm)

            InCube = HyperRectangle(_permuteList(cube.offset, invPerm), inCubeDims)
            inputInCubes.append(InCube)

            for dim in range(numDims):
                replacements[f"dimLen_{dim}"].append(inCubeDims[dim])

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a in inputInCubes:
            inputLoadSchedule.append({"data_in": a})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
