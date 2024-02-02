# ----------------------------------------------------------------------
#
# File: MaxPoolTileConstraintFlow.py
#
# Last edited: 09.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
# - Moritz Scherer, scheremo@iis.ee.ethz.ch, ETH Zurich
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

from typing import Callable, Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes, SignedIntegerDataTypes
from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Tiling.MemoryConstraints import NodeMemoryConstraint
from Deeploy.Tiling.PULPTileConstraintFlow.ConvTileConstraintFlow import Conv2DTileConstraintFlow
from Deeploy.Tiling.TileConstraintFlow import TileConstraintFlow
from Deeploy.Tiling.TilerModel import TilerModel
from Deeploy.Tiling.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme, \
    computeHyperRectangleList, extractTilingTransfer


class MaxPoolTileConstraintFlow(TileConstraintFlow):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        strides = parseDict["strides"]
        padding = parseDict["pads"]
        kernelShape = parseDict['kernel_shape']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBuffer.name, outputBuffer.name]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 3)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)  # Batch
        tilerModel.addConstraint(outputChannelVar == inputChannelVar)  # Channel

        effectiveHeight = inputHeightVar + ((padding[0] + padding[2]) * (inputHeightVar == inputBuffer.shape[1]))
        effectiveWidth = inputWidthVar + ((padding[1] + padding[3]) * (inputWidthVar == inputBuffer.shape[2]))

        tilerModel.addConstraint(
            (outputHeightVar == (effectiveHeight - ((kernelShape[0] + 1) // 2) + strides[0] - 1) // strides[0]))
        tilerModel.addConstraint(
            (outputWidthVar == (effectiveWidth - ((kernelShape[1] + 1) // 2) + strides[1] - 1) // strides[1]))

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])

        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)

        strides = parseDict["strides"]

        # VIC: Constraint the minimum tile size such that we can apply at least one kernel on it
        tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])
        tilerModel.addConstraint(inputHeightVar >= parseDict['dim_kernel_y'])
        tilerModel.addConstraint(inputWidthVar >= parseDict['dim_kernel_x'])

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        return tilerModel

    @classmethod
    def serializeTilingSolution(cls, tilingSolution: NodeMemoryConstraint, outputCubes: List[HyperRectangle],
                                targetMemLevel: str, ctxt: NetworkContext,
                                nodeRep: Dict) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel, nodeRep, addrNames)
        varOut = nodeRep['data_out']

        inputInCubes = []
        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_in": [],
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": []
        }

        replacementTypes = {
            "dim_im_in_x": Pointer(IntegerDataTypes.uint16_t),
            "dim_im_in_y": Pointer(IntegerDataTypes.uint16_t),
            "dim_im_out_x": Pointer(IntegerDataTypes.uint16_t),
            "dim_im_out_y": Pointer(IntegerDataTypes.uint16_t),
            "ch_im_in": Pointer(IntegerDataTypes.uint16_t),
            "padding_y_top": Pointer(IntegerDataTypes.uint8_t),
            "padding_y_bottom": Pointer(IntegerDataTypes.uint8_t),
            "padding_x_left": Pointer(IntegerDataTypes.uint8_t),
            "padding_x_right": Pointer(IntegerDataTypes.uint8_t)
        }

        kernelShape = nodeRep['kernel_shape']
        pads = nodeRep['pads']
        strides = nodeRep['strides']

        for cube in outputCubes:
            (BatchOffset, HOffset, WOffset, COffset) = cube.offset
            (BatchSize, HSize, WSize, CSize) = cube.dims

            InCube, padding_tuple = Conv2DTileConstraintFlow.computeInputCube((kernelShape[0], kernelShape[1]), pads,
                                                                              strides, CSize, cube,
                                                                              ctxt.lookup(varOut).shape)
            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            replacements['dim_im_in_x'].append(InCube.dims[2])
            replacements['dim_im_in_y'].append(InCube.dims[1])
            replacements['dim_im_out_x'].append(WSize)
            replacements['dim_im_out_y'].append(HSize)
            replacements['ch_im_in'].append(CSize)

            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            inputInCubes.append(InCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a in inputInCubes:
            inputLoadSchedule.append({"data_in": a})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
