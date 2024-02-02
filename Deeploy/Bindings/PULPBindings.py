# ----------------------------------------------------------------------
#
# File: PULPBindings.py
#
# Last edited: 10.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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

from functools import partial

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.Bindings.AutoFutureBinding import AutoFutureBinding
from Deeploy.CodeTransformationPasses import ArgumentStructGeneration, ClosureGeneration, FutureGeneration, \
    MemoryAwareClosureGeneration, MemoryManagementGeneration, TilingVariableReplacement
from Deeploy.CodeTransformationPasses.PrintInputs import MemoryAwarePrintConstantGeneration, \
    MemoryAwarePrintInputGeneration, MemoryAwarePrintOutputGeneration, PrintOutputGeneration
from Deeploy.CodeTransformationPasses.PULPCodeTransformationPasses import PULPClusterTilingGeneration, \
    PULPClusterTilingGenerationDB, PULPL3TilingGeneration, PULPL3TilingGenerationDB, PULPSynchCoresPass
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes, SignedIntegerDataTypes, UnsignedIntegerDataTypes
from Deeploy.DataTypes.PULPDataTypes import PULPDMAFuture
from Deeploy.DeeployTypes import CodeTransformation, CodeTransformationPass, NodeBinding, NodeTemplate
from Deeploy.Templates.PULPTemplates import ConvTemplate, GEMMTemplate, MaxPool2DTemplate, ReduceMeanTemplate, \
    RequantShiftTemplate, RQAddTemplate, SliceTemplate, TransposeTemplate, iSoftmaxTemplate
from Deeploy.TypeCheckers.BasicCheckers import MatMulChecker, ReduceMeanChecker, RequantShiftChecker, SliceChecker, \
    SoftmaxChecker, TransposeChecker
from Deeploy.TypeCheckers.PULPCheckers import PULPConvChecker, PULPLinearChecker, PULPMaxPoolChecker, \
    PULPRequantShiftChecker, PULPRQAddChecker

_clusterEntryClosureCallTemplate = NodeTemplate("""
// ${closureName} CLOSURE CALL
static struct pi_cluster_task cluster_task;

pi_cluster_task(&cluster_task, ${closureName}, &${closureStructArgName});
cluster_task.stack_size = 5000;
cluster_task.slave_stack_size = 3800;
pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
//pi_cluster_close(&cluster_dev);
""")

_clusterForkClosureCallTemplate = NodeTemplate("""
pi_cl_team_fork(NUM_CORES, (void*)${closureName}, &${closureStructArgName});
""")

FunctionCallClosure = partial(ClosureGeneration, closureSuffix = "_closure")
ClusterClosure = partial(ClosureGeneration, closureSuffix = "_cluster_entry", closureCallTemplate = _clusterEntryClosureCallTemplate)
ForkClosure = partial(ClosureGeneration, closureSuffix = "_cluster_fork", closureCallTemplate = _clusterForkClosureCallTemplate)

TilingCallClosure = partial(ClosureGeneration, closureSuffix = "_tiling_closure")
FunctionCallClosure = partial(ClosureGeneration, closureSuffix = "_closure")
ForkClosure = partial(ClosureGeneration, closureSuffix = "_cluster_fork", closureCallTemplate = _clusterForkClosureCallTemplate)

MemoryAwareClusterClosure = partial(MemoryAwareClosureGeneration, closureSuffix = "_cluster_entry", closureCallTemplate = _clusterEntryClosureCallTemplate, startRegion = "L2", endRegion = "L1")
MemoryAwareFunctionCallClosure = partial(MemoryAwareClosureGeneration, closureSuffix = "_closure", startRegion = "L2", endRegion = "L1")

L3MemoryAwareFunctionCallClosure = partial(MemoryAwareClosureGeneration, closureSuffix = "_closure_L3", startRegion = "L2", endRegion = "L2")

MemoryAwareForkTransformer = CodeTransformation([ArgumentStructGeneration(), ForkClosure(generateStruct = False), FutureGeneration(), ArgumentStructGeneration(), MemoryManagementGeneration("L1"), FunctionCallClosure(writeback = True), MemoryManagementGeneration("L2"), MemoryManagementGeneration()])

ForkTransformer = CodeTransformation([
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False),
    PULPSynchCoresPass(),
    ForkClosure(writeback = False, generateStruct = True),
    PULPClusterTilingGeneration("L1"),
    PULPClusterTilingGenerationDB("L1"),
    MemoryManagementGeneration("L1"),
    MemoryAwareFunctionCallClosure(writeback = False),
    TilingVariableReplacement("L2"),
    PULPL3TilingGeneration("L2"),
    PULPL3TilingGenerationDB("L2"),
    L3MemoryAwareFunctionCallClosure(writeback = False),
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration(),
])

ClusterTransformer = CodeTransformation([
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False, generateStruct = True),
    PULPClusterTilingGeneration("L1"),
    PULPClusterTilingGenerationDB("L1"),
    MemoryManagementGeneration("L1"),
    MemoryAwareFunctionCallClosure(writeback = False),
    TilingVariableReplacement("L2"),
    PULPL3TilingGeneration("L2"),
    PULPL3TilingGenerationDB("L2"),
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration(),
])

SimpleTransformer = CodeTransformation([
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration(),
])

PULPDMASliceBindings = [
    AutoFutureBinding(SliceChecker([Pointer(type), Pointer(UnsignedIntegerDataTypes.uint8_t), Pointer(UnsignedIntegerDataTypes.uint8_t),
                                    Pointer(UnsignedIntegerDataTypes.uint8_t), Pointer(UnsignedIntegerDataTypes.uint8_t)], [PULPDMAFuture(pointer = Pointer(type))]), SliceTemplate.referenceTemplate, MemoryAwareForkTransformer) for type in UnsignedIntegerDataTypes + SignedIntegerDataTypes
]

PULPRQAddBindings = [NodeBinding(PULPRQAddChecker([Pointer(_type), Pointer(_type)], [Pointer(_type)]), RQAddTemplate.RQAddTemplate, ForkTransformer) for _type in [IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t]]

PULPRQSConv2DBindings = [
    NodeBinding(PULPConvChecker(
        [Pointer(type1), Pointer(IntegerDataTypes.int8_t), Pointer(IntegerDataTypes.int32_t), Pointer(IntegerDataTypes.int32_t), Pointer(IntegerDataTypes.int32_t)], [Pointer(type2)]), ConvTemplate.PULPConv2D_8_Template, ForkTransformer)
    for type1, type2 in zip([IntegerDataTypes.int8_t, IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t, IntegerDataTypes.uint8_t], [IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t, IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t])
]

PULPRQSDWConv2DBindings = [
    NodeBinding(PULPConvChecker(
        [Pointer(type1), Pointer(IntegerDataTypes.int8_t), Pointer(IntegerDataTypes.int32_t), Pointer(IntegerDataTypes.int32_t), Pointer(IntegerDataTypes.int32_t)], [Pointer(type2)]), ConvTemplate.PULPDWConv2D_8_Template, ForkTransformer)
    for type1, type2 in zip([IntegerDataTypes.int8_t, IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t, IntegerDataTypes.uint8_t], [IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t, IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t])
]

PULPRQSGEMM_8_Binding = [
    NodeBinding(PULPLinearChecker([Pointer(type1), Pointer(IntegerDataTypes.int8_t), Pointer(IntegerDataTypes.int32_t), Pointer(IntegerDataTypes.int32_t)], [Pointer(type2)]), GEMMTemplate.PULPGEMM_8_Template, ForkTransformer)
    for type1, type2 in zip([IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t, IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t], [IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t, IntegerDataTypes.uint8_t, IntegerDataTypes.int8_t])
]
PULPRQSGEMMBindings = PULPRQSGEMM_8_Binding

PULPMaxPool2DBindings = [NodeBinding(PULPMaxPoolChecker([Pointer(type)], [Pointer(type)]), MaxPool2DTemplate.PULPMaxPool2D_8_Template, ForkTransformer) for type in [IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t]]

PULPConv1DBinding = NodeBinding(PULPConvChecker([Pointer(IntegerDataTypes.int8_t), Pointer(IntegerDataTypes.int8_t), Pointer(IntegerDataTypes.int32_t), Pointer(IntegerDataTypes.int32_t)], [Pointer(IntegerDataTypes.int8_t)]), ConvTemplate.PULPConv1D_8_Template, ForkTransformer)

PULPDWConv1DBinding = NodeBinding(PULPConvChecker([Pointer(IntegerDataTypes.int8_t), Pointer(IntegerDataTypes.int8_t), Pointer(IntegerDataTypes.int32_t), Pointer(IntegerDataTypes.int32_t)], [Pointer(IntegerDataTypes.int8_t)]), ConvTemplate.PULPDWConv1D_8_Template, ForkTransformer)

PULPMatMulBinding = NodeBinding(MatMulChecker([Pointer(IntegerDataTypes.int8_t), Pointer(IntegerDataTypes.int8_t)], [Pointer(IntegerDataTypes.int32_t)]), GEMMTemplate.PULPMM_8_Template, ClusterTransformer)

PULPReduceMeanBindings = [NodeBinding(ReduceMeanChecker([Pointer(type)], [Pointer(type)]), ReduceMeanTemplate.referenceTemplate, ClusterTransformer) for type in IntegerDataTypes]

PULPRQSBindings = [NodeBinding(PULPRequantShiftChecker([Pointer(type), Pointer(IntegerDataTypes.int32_t), Pointer(IntegerDataTypes.int32_t)], [Pointer(IntegerDataTypes.int8_t)]), RequantShiftTemplate.referenceTemplate, ClusterTransformer) for type in IntegerDataTypes
                  ] + [NodeBinding(PULPRequantShiftChecker([Pointer(type), Pointer(IntegerDataTypes.int32_t), Pointer(IntegerDataTypes.int32_t)], [Pointer(IntegerDataTypes.uint8_t)]), RequantShiftTemplate.referenceTemplate, ClusterTransformer) for type in IntegerDataTypes]

PULPSoftmaxBindings = [NodeBinding(SoftmaxChecker([Pointer(_type)], [Pointer(IntegerDataTypes.uint8_t)]), iSoftmaxTemplate.referenceTemplate, ForkTransformer) for _type in [IntegerDataTypes.int8_t, IntegerDataTypes.uint8_t]]

PULPTransposeBindings = [NodeBinding(TransposeChecker([Pointer(type)], [Pointer(type)]), TransposeTemplate.referenceTemplate, ForkTransformer) for type in IntegerDataTypes]
