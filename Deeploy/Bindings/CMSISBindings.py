# ----------------------------------------------------------------------
#
# File: CMSISBindings.py
#
# Last edited: 17.12.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
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

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CodeTransformationPasses import ArgumentStructGeneration, FutureGeneration, MemoryManagementGeneration
from Deeploy.DataTypes.BasicDataTypes import SignedIntegerDataTypes
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.Templates.CMSISTemplates import CLCATemplate, ConvTemplate, DWConvTemplate, GEMMTemplate, \
    LinearAttentionTemplate, MaxPool2DTemplate
from Deeploy.TypeCheckers.BasicCheckers import CLCAChecker, LinearAttentionChecker
from Deeploy.TypeCheckers.CMSISCheckers import CMSISConvChecker, CMSISLinearChecker, CMSISMaxPoolChecker

BasicTransformer = CodeTransformation([ArgumentStructGeneration(), MemoryManagementGeneration(), FutureGeneration()])

CMSISCLCABinding = NodeBinding(
    CLCAChecker([Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t)] + [Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int32_t)] * 3 +
                [Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t)] * 7, [Pointer(SignedIntegerDataTypes.int8_t)]), CLCATemplate.referenceTemplate, BasicTransformer)

CMSISConv1DBinding_16 = NodeBinding(
    CMSISConvChecker(
        [Pointer(SignedIntegerDataTypes.int16_t), Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int32_t),
         Pointer(SignedIntegerDataTypes.int64_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int16_t)]), ConvTemplate.cmsis1D_16_Template, BasicTransformer)
CMSISConv1DBinding_8 = NodeBinding(CMSISConvChecker(
    [Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int32_t),
     Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), ConvTemplate.cmsis1D_8_Template, BasicTransformer)
CMSISConv1DBindings = [CMSISConv1DBinding_8, CMSISConv1DBinding_16]

CMSISConv2DBinding = NodeBinding(CMSISConvChecker(
    [Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int32_t),
     Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), ConvTemplate.cmsis2D_8_Template, BasicTransformer)

CMSISDWConv1DBinding_16 = NodeBinding(
    CMSISConvChecker(
        [Pointer(SignedIntegerDataTypes.int16_t), Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int32_t),
         Pointer(SignedIntegerDataTypes.int64_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int16_t)]), DWConvTemplate.conv1D_16_Template, BasicTransformer)
CMSISDWConv1DBinding_8 = NodeBinding(
    CMSISConvChecker(
        [Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int32_t),
         Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), DWConvTemplate.conv1D_8_Template, BasicTransformer)
CMSISDWConv1DBindings = [CMSISDWConv1DBinding_8, CMSISDWConv1DBinding_16]

CMSISDWConv2DBinding = NodeBinding(CMSISConvChecker(
    [Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int32_t),
     Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), DWConvTemplate.conv2D_8_Template, BasicTransformer)

CMSISGEMM_16_Binding = NodeBinding(CMSISLinearChecker(
    [Pointer(SignedIntegerDataTypes.int16_t), Pointer(SignedIntegerDataTypes.int16_t), Pointer(SignedIntegerDataTypes.int64_t), Pointer(SignedIntegerDataTypes.int64_t)], [Pointer(SignedIntegerDataTypes.int16_t)]), GEMMTemplate.Linear_16_Template, BasicTransformer)
CMSISGEMM_8_Binding = NodeBinding(CMSISLinearChecker(
    [Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int32_t), Pointer(SignedIntegerDataTypes.int32_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), GEMMTemplate.Linear_8_Template, BasicTransformer)
CMSISGEMMBindings = [CMSISGEMM_8_Binding, CMSISGEMM_16_Binding]

CMSISLinearAttentionBinding = NodeBinding(
    LinearAttentionChecker([Pointer(SignedIntegerDataTypes.int16_t), Pointer(SignedIntegerDataTypes.int16_t), Pointer(SignedIntegerDataTypes.int16_t)] + [Pointer(SignedIntegerDataTypes.int8_t), Pointer(SignedIntegerDataTypes.int64_t)] * 4, [Pointer(SignedIntegerDataTypes.int16_t)]),
    LinearAttentionTemplate.referenceTemplate, BasicTransformer)

CMSISMaxPool2DBinding = NodeBinding(CMSISMaxPoolChecker([Pointer(SignedIntegerDataTypes.int8_t)], [Pointer(SignedIntegerDataTypes.int8_t)]), MaxPool2DTemplate.cmsisTemplate, BasicTransformer)
