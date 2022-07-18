# ----------------------------------------------------------------------
#
# File: CMSISBindings.py
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
from DumpO.Bindings.BasicBindings import DataTypes

from DumpO.TypeCheckers.CMSISCheckers import *
from DumpO.Templates.CMSISTemplates import *

from DumpO.Templates.BasicTemplates import AddTemplate as AddTemplate

CMSISConv2DBinding = NodeBinding(CMSISConvChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int32_t,DataTypes.int32_t], [DataTypes.int8_t]), ConvTemplate.cmsis2D_8_Template)

CMSISDW3x3Conv2DBinding = NodeBinding(CMSISConvChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int32_t,DataTypes.int32_t], [DataTypes.int8_t]), DWConvTemplate.conv2D_8_Template)

CMSISConv1DBinding_16 = NodeBinding(CMSISConvChecker([DataTypes.int16_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int64_t,DataTypes.int32_t], [DataTypes.int16_t]), ConvTemplate.cmsis1D_16_Template)

CMSISConv1DBinding_8 = NodeBinding(CMSISConvChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int32_t,DataTypes.int32_t], [DataTypes.int8_t]), ConvTemplate.cmsis1D_8_Template)

CMSISConv1DBindings = [CMSISConv1DBinding_16,CMSISConv1DBinding_8]

CMSISDW3x3Conv1DBinding_16 = NodeBinding(CMSISConvChecker([DataTypes.int16_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int64_t,DataTypes.int32_t], [DataTypes.int16_t]), DWConvTemplate.conv1D_16_Template)

CMSISDW3x3Conv1DBinding_8 = NodeBinding(CMSISConvChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int32_t,DataTypes.int32_t], [DataTypes.int8_t]), DWConvTemplate.conv1D_8_Template)

CMSISDW3x3Conv1DBindings = [CMSISDW3x3Conv1DBinding_16,CMSISDW3x3Conv1DBinding_8]

CMSISGEMM_8_Binding = NodeBinding(CMSISLinearChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int32_t], [DataTypes.int8_t]), GEMMTemplate.Linear_8_Template)

CMSISGEMM_16_Binding = NodeBinding(CMSISLinearChecker([DataTypes.int16_t, DataTypes.int8_t, DataTypes.int64_t], [DataTypes.int16_t]), GEMMTemplate.Linear_16_Template)

CMSISGEMMBindings = [CMSISGEMM_8_Binding,CMSISGEMM_16_Binding]

CMSISMaxPool2DBinding = NodeBinding(CMSISMaxPoolChecker([DataTypes.int8_t], [DataTypes.int8_t]), MaxPool2DTemplate.cmsisTemplate)

CMSISLayerNormBinding = NodeBinding(iLayerNormChecker([DataTypes.int8_t,DataTypes.int32_t,DataTypes.int32_t], [DataTypes.int8_t]), iLayernormTemplate.referenceTemplate)

CMSISMHSABinding = NodeBinding(MHSAChecker([DataTypes.int8_t,DataTypes.int8_t,DataTypes.int8_t]+[DataTypes.int8_t, DataTypes.int32_t]*4, [DataTypes.int8_t]), MHSATemplate.referenceTemplate)

CMSISLinearAttentionBinding = NodeBinding(LinearAttentionChecker([DataTypes.int16_t,DataTypes.int16_t,DataTypes.int16_t]+[DataTypes.int8_t, DataTypes.int64_t]*4, [DataTypes.int16_t]), LinearAttentionTemplate.referenceTemplate)

CMSISSaturatingAddBindings = [NodeBinding(CMSISSaturatingAddChecker([DataTypes.int8_t, DataTypes.int8_t],[DataTypes.int8_t]), AddTemplate.referenceTemplate)]

# CMSISSaturatingAddBindings = [NodeBinding(CMSISSaturatingAddChecker([DataTypes.int8_t],[DataTypes.int8_t]), AddTemplate.AddInt8Template),
#                               NodeBinding(CMSISSaturatingAddChecker([DataTypes.int16_t],[DataTypes.int16_t]), AddTemplate.AddInt16Template),
#                               NodeBinding(CMSISSaturatingAddChecker([DataTypes.int32_t],[DataTypes.int32_t]), AddTemplate.AddInt32Template)]
