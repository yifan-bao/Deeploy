# ----------------------------------------------------------------------
#
# File: MemPoolBindings.py
#
# Last edited: 13.11.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
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

from Deeploy.TypeCheckers.MemPoolCheckers import *
from Deeploy.Templates.MemPoolTemplates import *

MemPoolConv1D_8_8_32_Binding = NodeBinding(ConvChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), ConvTemplate.MemPoolParallel1DTemplate)
MemPoolConv2D_8_8_32_Binding = NodeBinding(ConvChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), ConvTemplate.MemPoolParallel2DTemplate)
MemPoolDWConv1D_8_8_32_Binding = NodeBinding(ConvChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), DWConvTemplate.MemPoolParallel1DTemplate)
MemPoolDWConv2D_8_8_32_Binding = NodeBinding(ConvChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), DWConvTemplate.MemPoolParallel2DTemplate)
MemPoolMatMul_8_8_32_Binding = NodeBinding(MatMulChecker([DataTypes.int8_t, DataTypes.int8_t], [DataTypes.int32_t]), MatMulTemplate.MemPoolParallelTemplate)
MemPoolMaxPool2D_8_8_Binding = NodeBinding(MaxPoolChecker([DataTypes.int8_t], [DataTypes.int8_t]), MaxPoolTemplate.MemPoolParallelTemplate)
MemPoolMHSA_8_8_8_8_Binding = NodeBinding(MHSAChecker([DataTypes.int8_t, DataTypes.int8_t, DataTypes.int8_t] + [DataTypes.int8_t, DataTypes.int8_t] * 4, [DataTypes.int8_t]), MHSATemplate.MemPoolParallelTemplate)