# ----------------------------------------------------------------------
#
# File: BasicBindings.py
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

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CodeTransformationPasses import ArgumentStructGeneration, FutureGeneration, MemoryManagementGeneration, \
    MemoryPassthroughGeneration
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes, SignedIntegerDataTypes, UnsignedIntegerDataTypes
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.Templates.BasicTemplates import AddTemplate, ConvTemplate, DebugPrintTemplate, DummyTemplate, \
    DWConvTemplate, GatherTemplate, GemmTemplate, IntegerDivTemplate, ITAMaxTemplate, MatMulTemplate, MaxPoolTemplate, \
    MulTemplate, PadTemplate, ReduceMeanTemplate, ReduceSumTemplate, RequantShiftTemplate, ReshapeTemplate, \
    RQIntegerDivTemplate, RQSiGELUTemplate, SliceTemplate, TransposeTemplate, iGELUTemplate, iLayernormTemplate, \
    iSoftmaxTemplate
from Deeploy.TypeCheckers.BasicCheckers import AddChecker, ConvChecker, DebugPrintChecker, DummyChecker, \
    GatherChecker, GELUChecker, GEMMChecker, IntegerDivChecker, MatMulChecker, MaxPoolChecker, MulChecker, PadChecker, \
    ReduceMeanChecker, ReduceSumChecker, RequantShiftChecker, ReshapeChecker, RQIntegerDivChecker, SliceChecker, \
    SoftmaxChecker, TransposeChecker, iLayerNormChecker

BasicTransformer = CodeTransformation([ArgumentStructGeneration(), MemoryManagementGeneration(), FutureGeneration()])

ReshapeSkipTransformer = CodeTransformation([ArgumentStructGeneration(), MemoryPassthroughGeneration(), FutureGeneration()])

BasicSliceBindings = [
    NodeBinding(SliceChecker([Pointer(type), Pointer(UnsignedIntegerDataTypes.uint8_t), Pointer(UnsignedIntegerDataTypes.uint8_t),
                              Pointer(UnsignedIntegerDataTypes.uint8_t), Pointer(UnsignedIntegerDataTypes.uint8_t)], [Pointer(type)]), SliceTemplate.referenceTemplate, BasicTransformer) for type in SignedIntegerDataTypes + UnsignedIntegerDataTypes
]

BasicAddBindings = [NodeBinding(AddChecker([Pointer(type), Pointer(type)], [Pointer(SignedIntegerDataTypes.int32_t)]), AddTemplate.referenceTemplate, BasicTransformer) for type in SignedIntegerDataTypes + UnsignedIntegerDataTypes]

BasicConv1DBinding = NodeBinding(ConvChecker([Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t)], [Pointer(SignedIntegerDataTypes.int32_t)]), ConvTemplate.reference1DTemplate, BasicTransformer)

BasicDWConv1DBinding = NodeBinding(ConvChecker([Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t)], [Pointer(SignedIntegerDataTypes.int32_t)]), DWConvTemplate.reference1DTemplate, BasicTransformer)

BasicConv2DBinding = NodeBinding(ConvChecker([Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t)], [Pointer(SignedIntegerDataTypes.int32_t)]), ConvTemplate.reference2DTemplate, BasicTransformer)

BasicDWConv2DBinding = NodeBinding(ConvChecker([Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t)], [Pointer(SignedIntegerDataTypes.int32_t)]), DWConvTemplate.reference2DTemplate, BasicTransformer)

BasicDebugPrintBindings = [NodeBinding(DebugPrintChecker([Pointer(type)], [Pointer(type)]), DebugPrintTemplate.referenceTemplate, ReshapeSkipTransformer) for type in SignedIntegerDataTypes]

BasicGatherBindings = [NodeBinding(GatherChecker([Pointer(type), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(type)]), GatherTemplate.referenceTemplate, BasicTransformer) for type in SignedIntegerDataTypes]

BasicGELUBinding = NodeBinding(GELUChecker([Pointer(SignedIntegerDataTypes.int8_t)], [Pointer(SignedIntegerDataTypes.int32_t)]), iGELUTemplate.referenceTemplate, BasicTransformer)

BasicGEMMBinding = NodeBinding(GEMMChecker([Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int32_t)]), GemmTemplate.referenceTemplate, BasicTransformer)

BasicIntegerDivBinding = NodeBinding(IntegerDivChecker([Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int32_t)]), IntegerDivTemplate.referenceTemplate, BasicTransformer)

BasicITASoftmaxBinding = NodeBinding(SoftmaxChecker([Pointer(SignedIntegerDataTypes.int8_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), ITAMaxTemplate.referenceTemplate, BasicTransformer)

BasicLayerNormBinding = NodeBinding(iLayerNormChecker([Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), iLayernormTemplate.referenceTemplate, BasicTransformer)

BasicMatMulBinding = NodeBinding(MatMulChecker([Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t)], [Pointer(SignedIntegerDataTypes.int32_t)]), MatMulTemplate.referenceTemplate, BasicTransformer)

BasicMaxPool2DBinding = NodeBinding(MaxPoolChecker([Pointer(SignedIntegerDataTypes.int8_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), MaxPoolTemplate.referenceTemplate, BasicTransformer)

BasicMulBindings = [NodeBinding(MulChecker([Pointer(typeA), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int32_t)]), MulTemplate.referenceTemplate, BasicTransformer) for typeA in SignedIntegerDataTypes]

BasicPad1DBindings = [NodeBinding(PadChecker([Pointer(type)], [Pointer(type)]), PadTemplate.reference1DTemplate, BasicTransformer) for type in SignedIntegerDataTypes]
BasicPad2DBindings = [NodeBinding(PadChecker([Pointer(type)], [Pointer(type)]), PadTemplate.reference2DTemplate, BasicTransformer) for type in SignedIntegerDataTypes]

BasicReduceMeanBindings = [NodeBinding(ReduceMeanChecker([Pointer(type)], [Pointer(type)]), ReduceMeanTemplate.referenceTemplate, BasicTransformer) for type in SignedIntegerDataTypes]

BasicReduceSumBindings = [NodeBinding(ReduceSumChecker([Pointer(type)], [Pointer(SignedIntegerDataTypes.int32_t)]), ReduceSumTemplate.referenceTemplate, BasicTransformer) for type in SignedIntegerDataTypes]

BasicReshapeBindings = [NodeBinding(ReshapeChecker([Pointer(type), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(type)]), ReshapeTemplate.referenceTemplate, ReshapeSkipTransformer) for type in IntegerDataTypes]

BasicRQSBindings = [NodeBinding(RequantShiftChecker([Pointer(type), Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), RequantShiftTemplate.referenceTemplate, BasicTransformer) for type in SignedIntegerDataTypes]

BasicRQSGELUBinding = NodeBinding(GELUChecker(
    [Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), RQSiGELUTemplate.referenceTemplate, BasicTransformer)

BasicRQIntegerDivBinding = NodeBinding(
    RQIntegerDivChecker(
        [Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t),
         Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), RQIntegerDivTemplate.referenceTemplate, BasicTransformer)

BasicSoftmaxBinding = NodeBinding(SoftmaxChecker([Pointer(SignedIntegerDataTypes.int8_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), iSoftmaxTemplate.referenceTemplate, BasicTransformer)

BasicTransposeBindings = [NodeBinding(TransposeChecker([Pointer(type)], [Pointer(type)]), TransposeTemplate.referenceTemplate, BasicTransformer) for type in IntegerDataTypes]

DummyBinding = NodeBinding(DummyChecker([Pointer(SignedIntegerDataTypes.int8_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), DummyTemplate.referenceTemplate, BasicTransformer)
