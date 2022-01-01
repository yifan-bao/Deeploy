# ----------------------------------------------------------------------
#
# File: BasicBindings.py
#
# Last edited: 21.12.2021        
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
from DumpO.TypeCheckers.BasicCheckers import *
from DumpO.Templates.BasicTemplates import *

class DataTypes(Enum):
    int8_t = 8
    int16_t = 16
    int32_t = 32

BasicGELUBinding = NodeBinding(GELUChecker([DataTypes.int8_t], [DataTypes.int8_t]), iGELUTemplate.referenceTemplate)
BasicSoftmaxBinding = NodeBinding(SoftmaxChecker([DataTypes.int8_t], [DataTypes.int8_t]), iSoftmaxTemplate.referenceTemplate)
BasicLayerNormBinding = NodeBinding(iLayerNormChecker([DataTypes.int8_t,DataTypes.int32_t,DataTypes.int32_t], [DataTypes.int8_t]), iLayernormTemplate.referenceTemplate)

BasicGEMMBinding = NodeBinding(GEMMChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), GEMMTemplate.referenceTemplate)
BasicConv2DBinding = NodeBinding(ConvChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), DummyTemplate.referenceTemplate)
BasicMHSABinding = NodeBinding(MHSAChecker([DataTypes.int8_t], [DataTypes.int32_t]), MHSATemplate.referenceTemplate)

BasicGatherBindings = [NodeBinding(GatherChecker([type],[type]), GatherTemplate.referenceTemplate) for type in DataTypes]
BasicReshapeBindings = [NodeBinding(ReshapeChecker([type],[type]), SkipTemplate.referenceTemplate) for type in DataTypes]
BasicTransposeBindings = [NodeBinding(TransposeChecker([type],[type]), TransposeTemplate.referenceTemplate) for type in DataTypes]
BasicRQSBindings = [NodeBinding(RequantShiftChecker([type,DataTypes.int32_t,DataTypes.int32_t], [DataTypes.int8_t]), RequantShiftTemplate.referenceTemplate) for type in DataTypes]

BasicAddBindings = [NodeBinding(AddChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int16_t]), AddTemplate.referenceTemplate)]
BasicAddBindings += [NodeBinding(AddChecker([DataTypes.int16_t, DataTypes.int16_t], [DataTypes.int32_t]), AddTemplate.referenceTemplate) for type in DataTypes]

BasicPadBindings = [NodeBinding(PadChecker([type], [type]), PadTemplate.referenceTemplate) for type in DataTypes]

DummyBinding = NodeBinding(DummyChecker([DataTypes.int8_t],[DataTypes.int8_t]), DummyTemplate.referenceTemplate)
