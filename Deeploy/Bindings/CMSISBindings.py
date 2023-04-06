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

from Deeploy.DeeployTypes import *

from Deeploy.Bindings.BasicBindings import DataTypes
from Deeploy.TypeCheckers.BasicCheckers import *
from Deeploy.Templates.BasicTemplates import *

from Deeploy.TypeCheckers.CMSISCheckers import *
from Deeploy.Templates.CMSISTemplates import *

# Overwrite some templates with the basic version
from Deeploy.Templates.BasicTemplates import AddTemplate as AddTemplate

CMSISCLCABinding = NodeBinding(CLCAChecker([DataTypes.int8_t, DataTypes.int8_t] + [DataTypes.int8_t, DataTypes.int32_t] * 3 + [DataTypes.int32_t, DataTypes.int32_t, DataTypes.int32_t] * 7, [DataTypes.int8_t]), CLCATemplate.referenceTemplate)

CMSISConv1DBinding_16 = NodeBinding(CMSISConvChecker([DataTypes.int16_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int64_t, DataTypes.int32_t], [DataTypes.int16_t]), ConvTemplate.cmsis1D_16_Template)
CMSISConv1DBinding_8 = NodeBinding(CMSISConvChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int32_t, DataTypes.int32_t], [DataTypes.int8_t]), ConvTemplate.cmsis1D_8_Template)
CMSISConv1DBindings = [CMSISConv1DBinding_8, CMSISConv1DBinding_16]

CMSISConv2DBinding = NodeBinding(CMSISConvChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int32_t, DataTypes.int32_t], [DataTypes.int8_t]), ConvTemplate.cmsis2D_8_Template)

CMSISDWConv1DBinding_16 = NodeBinding(CMSISConvChecker([DataTypes.int16_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int64_t, DataTypes.int32_t], [DataTypes.int16_t]), DWConvTemplate.conv1D_16_Template)
CMSISDWConv1DBinding_8 = NodeBinding(CMSISConvChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int32_t, DataTypes.int32_t], [DataTypes.int8_t]), DWConvTemplate.conv1D_8_Template)
CMSISDWConv1DBindings = [CMSISDWConv1DBinding_8, CMSISDWConv1DBinding_16]

CMSISDWConv2DBinding = NodeBinding(CMSISConvChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int32_t, DataTypes.int32_t], [DataTypes.int8_t]), DWConvTemplate.conv2D_8_Template)

CMSISGEMM_16_Binding = NodeBinding(CMSISLinearChecker([DataTypes.int16_t, DataTypes.int16_t, DataTypes.int64_t, DataTypes.int64_t], [DataTypes.int16_t]), GEMMTemplate.Linear_16_Template)
CMSISGEMM_8_Binding = NodeBinding(CMSISLinearChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int32_t, DataTypes.int32_t], [DataTypes.int8_t]), GEMMTemplate.Linear_8_Template)
CMSISGEMMBindings = [CMSISGEMM_8_Binding, CMSISGEMM_16_Binding]

CMSISLinearAttentionBinding = NodeBinding(LinearAttentionChecker([DataTypes.int16_t, DataTypes.int16_t, DataTypes.int16_t] + [DataTypes.int8_t, DataTypes.int64_t] * 4, [DataTypes.int16_t]), LinearAttentionTemplate.referenceTemplate)

CMSISMaxPool2DBinding = NodeBinding(CMSISMaxPoolChecker([DataTypes.int8_t], [DataTypes.int8_t]), MaxPool2DTemplate.cmsisTemplate)
