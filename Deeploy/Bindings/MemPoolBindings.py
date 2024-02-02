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

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CodeTransformationPasses import ArgumentStructGeneration, FutureGeneration, MemoryManagementGeneration
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes as DataTypes
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.Templates.MemPoolTemplates import ConvTemplate, DWConvTemplate, GemmTemplate, ITAMaxTemplate, \
    MatMulTemplate, MaxPoolTemplate, MHSATemplate, RequantShiftTemplate, RQGemmTemplate, RQMatMulTemplate
from Deeploy.TypeCheckers.BasicCheckers import ConvChecker, GEMMChecker, MatMulChecker, MaxPoolChecker, MHSAChecker, \
    RequantShiftChecker, RQGEMMChecker, RQMatMulChecker, SoftmaxChecker

BasicTransformer = CodeTransformation([MemoryManagementGeneration(), ArgumentStructGeneration(), FutureGeneration()])

MemPoolConv1D_8_8_32_Binding = NodeBinding(ConvChecker([Pointer(DataTypes.int8_t), Pointer(DataTypes.int8_t)], [Pointer(DataTypes.int32_t)]), ConvTemplate.MemPoolParallel1DTemplate, BasicTransformer)
MemPoolConv2D_8_8_32_Binding = NodeBinding(ConvChecker([Pointer(DataTypes.int8_t), Pointer(DataTypes.int8_t)], [Pointer(DataTypes.int32_t)]), ConvTemplate.MemPoolParallel2DTemplate, BasicTransformer)
MemPoolDWConv1D_8_8_32_Binding = NodeBinding(ConvChecker([Pointer(DataTypes.int8_t), Pointer(DataTypes.int8_t)], [Pointer(DataTypes.int32_t)]), DWConvTemplate.MemPoolParallel1DTemplate, BasicTransformer)
MemPoolDWConv2D_8_8_32_Binding = NodeBinding(ConvChecker([Pointer(DataTypes.int8_t), Pointer(DataTypes.int8_t)], [Pointer(DataTypes.int32_t)]), DWConvTemplate.MemPoolParallel2DTemplate, BasicTransformer)
MemPoolGEMMBinding_8_8_32_32 = NodeBinding(GEMMChecker([Pointer(DataTypes.int8_t), Pointer(DataTypes.int8_t), Pointer(DataTypes.int32_t)], [Pointer(DataTypes.int32_t)]), GemmTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolITASoftmaxBinding_8_8 = NodeBinding(SoftmaxChecker([Pointer(DataTypes.int8_t)], [Pointer(DataTypes.int8_t)]), ITAMaxTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolMatMul_8_8_32_Binding = NodeBinding(MatMulChecker([Pointer(DataTypes.int8_t), Pointer(DataTypes.int8_t)], [Pointer(DataTypes.int32_t)]), MatMulTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolMaxPool2D_8_8_Binding = NodeBinding(MaxPoolChecker([Pointer(DataTypes.int8_t)], [Pointer(DataTypes.int8_t)]), MaxPoolTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolMHSA_8_8_8_8_Binding = NodeBinding(MHSAChecker([Pointer(DataTypes.int8_t), Pointer(DataTypes.int8_t), Pointer(DataTypes.int8_t)] + [Pointer(DataTypes.int8_t), Pointer(DataTypes.int8_t)] * 4, [Pointer(DataTypes.int8_t)]), MHSATemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolRQGEMMBinding_8_8_32_32_32_8 = NodeBinding(RQGEMMChecker([Pointer(DataTypes.int8_t), Pointer(DataTypes.int8_t), Pointer(DataTypes.int32_t), Pointer(DataTypes.int32_t), Pointer(DataTypes.int32_t)], [Pointer(DataTypes.int8_t)]), RQGemmTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolRQMatMul_8_8_32_32_Binding = NodeBinding(RQMatMulChecker([Pointer(DataTypes.int8_t), Pointer(DataTypes.int8_t), Pointer(DataTypes.int32_t), Pointer(DataTypes.int32_t)], [Pointer(DataTypes.int8_t)]), RQMatMulTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolRQSBindings_x_32_32_8 = [NodeBinding(RequantShiftChecker([Pointer(type), Pointer(DataTypes.int32_t), Pointer(DataTypes.int32_t)], [Pointer(DataTypes.int8_t)]), RequantShiftTemplate.MemPoolParallelTemplate, BasicTransformer) for type in DataTypes]
