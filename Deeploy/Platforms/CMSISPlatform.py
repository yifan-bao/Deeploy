# ----------------------------------------------------------------------
#
# File: CMSISPlatform.py
#
# Last edited: 17.12.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author:
# - Moritz Scherer, ETH Zurich
# - Philip Wiese, ETH Zurich
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

from typing import List

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import HelperTypes, Pointer
from Deeploy.Bindings.BasicBindings import BasicAddBindings, BasicDebugPrintBindings, BasicGatherBindings, \
    BasicGELUBinding, BasicIntegerDivBinding, BasicLayerNormBinding, BasicMatMulBinding, BasicMulBindings, \
    BasicPad1DBindings, BasicPad2DBindings, BasicReduceMeanBindings, BasicReduceSumBindings, BasicReshapeBindings, \
    BasicRQIntegerDivBinding, BasicRQSBindings, BasicRQSGELUBinding, BasicSliceBindings, BasicSoftmaxBinding, \
    BasicTransposeBindings, DummyBinding
from Deeploy.Bindings.CMSISBindings import CMSISCLCABinding, CMSISConv1DBindings, CMSISConv2DBinding, \
    CMSISDWConv1DBindings, CMSISDWConv2DBinding, CMSISGEMMBindings, CMSISLinearAttentionBinding, \
    CMSISMaxPool2DBinding
from Deeploy.DataTypes.BasicDataTypes import SignedIntegerDataTypes
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentPlatform, NodeMapper, NodeTemplate, StructBuffer, \
    TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Layers.BasicLayers import AddLayer, CLCALayer, DebugPrintLayer, GatherLayer, IntegerDivLayer, \
    LinearAttentionLayer, MatMulLayer, MaxPoolLayer, MulLayer, PadLayer, ReduceMeanLayer, ReduceSumLayer, \
    RequantShiftLayer, ReshapeLayer, RQIntegerDivLayer, RQSiGELULayer, SliceLayer, TransposeLayer, iGELULayer, \
    iLayerNormLayer, iSoftmaxLayer
from Deeploy.Layers.CMSISLayers import CMSISRQSConvLayer, CMSISRQSGEMMLayer
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.BasicPasses import IntegerDivRequantMergePass, \
    MergeConstAddAndRequantPass, iGELURequantMergePass
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.CMSISPasses import ConvRequantMergePass, \
    GEMMRequantMergePass, LinearAttentionAlignmentPass, MatMulRequantMergePass, MHSAAlignmentPass
from Deeploy.Parsers.BasicParsers import AddParser, DebugParser, DummyParser, FlattenParser, GatherParser, \
    IntegerDivParser, MatMulParser, MulParser, Pad1DParser, Pad2DParser, ReduceMeanParser, ReduceSumParser, \
    RequantShiftParser, ReshapeParser, RQIntegerDivParser, RQSiGELUParser, SliceParser, TransposeParser, \
    UnsqueezeParser, iGELUParser, iLayerNormParser, iSoftmaxParser
from Deeploy.Parsers.CMSISParsers import CMSISCLCAParser, CMSISConv1DParser, CMSISConv2DParser, CMSISDWConv1DParser, \
    CMSISDWConv2DParser, CMSISGEMMParser, CMSISLinearAttentionParser, CMSISMaxPool2DParser
from Deeploy.Templates.BasicTemplates import AllocateTemplate, FreeTemplate

AddMapper = NodeMapper(AddParser(), BasicAddBindings)
CLCA_int8_Mapper = NodeMapper(CMSISCLCAParser(), [CMSISCLCABinding])
Conv1D_Mapper = NodeMapper(CMSISConv1DParser(), CMSISConv1DBindings)
Conv2D_int8_Mapper = NodeMapper(CMSISConv2DParser(), [CMSISConv2DBinding])
DebugPrint_Mapper = NodeMapper(DebugParser(), BasicDebugPrintBindings)
DWConv1D_Mapper = NodeMapper(CMSISDWConv1DParser(), CMSISDWConv1DBindings)
DWConv2D_int8_Mapper = NodeMapper(CMSISDWConv2DParser(), [CMSISDWConv2DBinding])
FlattenMapper = NodeMapper(FlattenParser(), BasicReshapeBindings)
GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
GELU_int8_Mapper = NodeMapper(iGELUParser(), [BasicGELUBinding])
GEMMMapper = NodeMapper(CMSISGEMMParser(), CMSISGEMMBindings)
iLayerNorm_int8_Mapper = NodeMapper(iLayerNormParser(), [BasicLayerNormBinding])
IntegerDivMapper = NodeMapper(IntegerDivParser(), [BasicIntegerDivBinding])
LinearAttention_int16_Mapper = NodeMapper(CMSISLinearAttentionParser(), [CMSISLinearAttentionBinding])
MatMulMapper = NodeMapper(MatMulParser(), [BasicMatMulBinding])
MaxPool2DMapper = NodeMapper(CMSISMaxPool2DParser(), [CMSISMaxPool2DBinding])
MulMapper = NodeMapper(MulParser(), BasicMulBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
ReduceMeanMapper = NodeMapper(ReduceMeanParser(), BasicReduceMeanBindings)
ReduceSumMapper = NodeMapper(ReduceSumParser(), BasicReduceSumBindings)
RequantShiftMapper = NodeMapper(RequantShiftParser(), BasicRQSBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
RQGELU_int8_Mapper = NodeMapper(RQSiGELUParser(), [BasicRQSGELUBinding])
RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
Softmax_int8_Mapper = NodeMapper(iSoftmaxParser(), [BasicSoftmaxBinding])
TransposeMapper = NodeMapper(TransposeParser(), BasicTransposeBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)

SliceMapper = NodeMapper(SliceParser(), BasicSliceBindings)

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

CMSISMapping = {
    'Add': AddLayer([AddMapper]),
    'CLCA': CLCALayer([CLCA_int8_Mapper]),
    'DebugPrint': DebugPrintLayer([DebugPrint_Mapper]),
    'Flatten': ReshapeLayer([FlattenMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'iGELU': iGELULayer([GELU_int8_Mapper]),
    'iLayerNorm': iLayerNormLayer([iLayerNorm_int8_Mapper]),
    'IntegerDiv': IntegerDivLayer([IntegerDivMapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMeanMapper]),
    'iSoftmax': iSoftmaxLayer([Softmax_int8_Mapper]),
    'LinearAttention': LinearAttentionLayer([LinearAttention_int16_Mapper]),
    'MatMul': MatMulLayer([MatMulMapper]),
    'MaxPool': MaxPoolLayer([MaxPool2DMapper]),
    'Mul': MulLayer([MulMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMeanMapper]),
    'ReduceSum': ReduceSumLayer([ReduceSumMapper]),
    'RequantizedConv': CMSISRQSConvLayer([Conv2D_int8_Mapper, DWConv2D_int8_Mapper, Conv1D_Mapper, DWConv1D_Mapper]),
    'RequantizedGemm': CMSISRQSGEMMLayer([GEMMMapper]),
    'RequantizediGELU': RQSiGELULayer([RQGELU_int8_Mapper]),
    'RequantShift': RequantShiftLayer([RequantShiftMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    'Slice': SliceLayer([SliceMapper])
}


class CMSISVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate


class CMSISTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate


class CMSISConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.referenceGlobalInitTemplate
    allocTemplate = AllocateTemplate.referenceGlobalAllocateTemplate
    deallocTemplate = FreeTemplate.referenceGlobalTemplate


class CMSISStructBuffer(StructBuffer):

    initTemplate = AllocateTemplate.referenceStructInitTemplate
    allocTemplate = AllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


# ExtractPaddingFromConvPass(),ExtractPaddingFromPoolPass(),
CMSISOptimizer = TopologyOptimizer([
    IntegerDivRequantMergePass(),
    iGELURequantMergePass(),
    LinearAttentionAlignmentPass(),
    MHSAAlignmentPass(),
    MergeConstAddAndRequantPass(),
    ConvRequantMergePass(),
    GEMMRequantMergePass(),
    MatMulRequantMergePass(),
    # DebugPass("Conv", position='before'),
    # DebugPass("Pad", position='after'),
])

includeList = ["arm_nnfunctions.h", "DeeployMath.h"]


class CMSISPlatform(DeploymentPlatform):

    def __init__(self,
                 Mapping = CMSISMapping,
                 DataTypes = SignedIntegerDataTypes,
                 VariableBuffer = CMSISVariableBuffer,
                 ConstantBuffer = CMSISConstantBuffer,
                 StructBuffer = CMSISStructBuffer,
                 TransientBuffer = CMSISTransientBuffer,
                 includeList: List[str] = includeList):
        super().__init__(Mapping, DataTypes, VariableBuffer, ConstantBuffer, StructBuffer, TransientBuffer, includeList)
