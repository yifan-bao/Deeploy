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

from Deeploy.DeeployTypes import *

from Deeploy.DataTypes.BasicDataTypes import DataTypes
from Deeploy.TypeCheckers.BasicCheckers import *
from Deeploy.Templates.BasicTemplates import *

BasicAddBindings = [NodeBinding(AddChecker([type, type], [DataTypes.int32_t]), AddTemplate.referenceTemplate) for type in DataTypes]

BasicConv1DBinding = NodeBinding(ConvChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), ConvTemplate.reference1DTemplate)

BasicDWConv1DBinding = NodeBinding(ConvChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), DWConvTemplate.reference1DTemplate)

BasicConv2DBinding = NodeBinding(ConvChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), ConvTemplate.reference2DTemplate)

BasicDWConv2DBinding = NodeBinding(ConvChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), DWConvTemplate.reference2DTemplate)

BasicDebugBindings = [NodeBinding(DebugChecker([type], [type]), DebugTemplate.referenceTemplate) for type in DataTypes]

BasicGatherBindings = [NodeBinding(GatherChecker([type, DataTypes.int32_t], [type]), GatherTemplate.referenceTemplate) for type in DataTypes]

BasicGELUBinding = NodeBinding(GELUChecker([DataTypes.int8_t], [DataTypes.int32_t]), iGELUTemplate.referenceTemplate)

BasicGEMMBinding = NodeBinding(GEMMChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int32_t], [DataTypes.int32_t]), GemmTemplate.referenceTemplate)

BasicIntegerDivBinding = NodeBinding(IntegerDivChecker([DataTypes.int32_t, DataTypes.int32_t], [DataTypes.int32_t]), IntegerDivTemplate.referenceTemplate)

BasicLayerNormBinding = NodeBinding(iLayerNormChecker([DataTypes.int8_t, DataTypes.int32_t, DataTypes.int32_t], [DataTypes.int8_t]), iLayernormTemplate.referenceTemplate)

BasicMatMulBinding = NodeBinding(MatMulChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), MatMulTemplate.referenceTemplate)

BasicMaxPool2DBinding = NodeBinding(MaxPoolChecker([DataTypes.int8_t], [DataTypes.int8_t]), MaxPoolTemplate.referenceTemplate)

BasicMulBindings = [NodeBinding(MulChecker([typeA, DataTypes.int32_t], [DataTypes.int32_t]), MulTemplate.referenceTemplate) for typeA in DataTypes]

BasicPad1DBindings = [NodeBinding(PadChecker([type], [type]), PadTemplate.reference1DTemplate) for type in DataTypes]
BasicPad2DBindings = [NodeBinding(PadChecker([type], [type]), PadTemplate.reference2DTemplate) for type in DataTypes]

BasicReduceMeanBindings = [NodeBinding(ReduceMeanChecker([type], [type]), ReduceMeanTemplate.referenceTemplate) for type in DataTypes]

BasicReshapeBindings = [NodeBinding(ReshapeChecker([type, DataTypes.int32_t], [type]), ReshapeTemplate.referenceTemplate) for type in DataTypes]

BasicRQSBindings = [NodeBinding(RequantShiftChecker([type, DataTypes.int32_t, DataTypes.int32_t], [DataTypes.int8_t]), RequantShiftTemplate.referenceTemplate) for type in DataTypes]

BasicRQSGELUBinding = NodeBinding(GELUChecker([DataTypes.int8_t, DataTypes.int32_t, DataTypes.int32_t, DataTypes.int32_t], [DataTypes.int8_t]), RQSiGELUTemplate.referenceTemplate)

BasicRQIntegerDivBinding = NodeBinding(RQIntegerDivChecker([DataTypes.int32_t, DataTypes.int32_t, DataTypes.int32_t, DataTypes.int32_t, DataTypes.int32_t], [DataTypes.int8_t]), RQIntegerDivTemplate.referenceTemplate)

BasicSoftmaxBinding = NodeBinding(SoftmaxChecker([DataTypes.int8_t], [DataTypes.int8_t]), iSoftmaxTemplate.referenceTemplate)

BasicTransposeBindings = [NodeBinding(TransposeChecker([type], [type]), TransposeTemplate.referenceTemplate) for type in DataTypes]

DummyBinding = NodeBinding(DummyChecker([DataTypes.int8_t], [DataTypes.int8_t]), DummyTemplate.referenceTemplate)
