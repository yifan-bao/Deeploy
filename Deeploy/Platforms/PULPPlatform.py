# ----------------------------------------------------------------------
#
# File: PULPPlatform.py
#
# Last edited: 07.03.2023
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

from typing import List

import numpy as np

from Deeploy.Bindings.BasicBindings import BasicAddBindings, BasicGatherBindings, BasicMulBindings, \
    BasicPad1DBindings, BasicPad2DBindings, BasicReshapeBindings, BasicRQIntegerDivBinding, BasicRQSGELUBinding, \
    BasicTransposeBindings
from Deeploy.Bindings.PULPBindings import PULPConv1DBinding, PULPDMASliceBindings, PULPDWConv1DBinding, \
    PULPMatMulBinding, PULPMaxPool2DBindings, PULPReduceMeanBindings, PULPRQAddBindings, PULPRQSBindings, \
    PULPRQSConv2DBindings, PULPRQSGEMMBindings, PULPSoftmaxBindings
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes as DataTypes
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentPlatform, NodeMapper, NodeTemplate, StructBuffer, \
    TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Layers.BasicLayers import AddLayer, GatherLayer, MatMulLayer, MaxPoolLayer, MulLayer, PadLayer, \
    ReduceMeanLayer, RequantShiftLayer, ReshapeLayer, RQIntegerDivLayer, RQSiGELULayer, SliceLayer, TransposeLayer, \
    iGELULayer, iLayerNormLayer, iSoftmaxLayer
from Deeploy.Layers.PULPLayers import PULPRQSConvLayer, PULPRQSGEMMLayer
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.BasicPasses import IntegerDivRequantMergePass, \
    MergeConstAddAndRequantPass, iGELURequantMergePass
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.PULPPasses import PULPAddRequantMergePass, \
    PULPConvRequantMergePass, PULPGEMMRequantMergePass, PULPMatMulRequantMergePass
from Deeploy.Parsers.BasicParsers import AddParser, FlattenParser, GatherParser, MatMulParser, MulParser, Pad1DParser, \
    Pad2DParser, ReduceMeanParser, RequantShiftParser, ReshapeParser, RQIntegerDivParser, RQSiGELUParser, SliceParser, \
    TransposeParser, UnsqueezeParser, iGELUParser, iLayerNormParser, iSoftmaxParser
from Deeploy.Parsers.CMSISParsers import CMSISGEMMParser, CMSISMaxPool2DParser
from Deeploy.Parsers.PULPParsers import PULPConv1DParser, PULPConv2DParser, PULPDWConv1DParser, PULPDWConv2DParser, \
    PULPRQAddParser
from Deeploy.Templates.BasicTemplates import AllocateTemplate as BasicAllocateTemplate
from Deeploy.Templates.PULPTemplates import AllocateTemplate, FreeTemplate
from Deeploy.Tiling.PlatformTilers.BasicTiler import BasicAddTilingReadyBindings, BasicFlattenTilingReadyBindings
from Deeploy.Tiling.PlatformTilers.PULPTiler import PULPAddTilingReadyBindings, PULPFlattenTilingReadyBindings, \
    PULPiSoftmaxTilingReadyBindings, PULPMatMulTilingReadyBindings, PULPMaxPool2DTilingReadyBindings, \
    PULPRQAddTilingReadyBindings, PULPRQSConv2DTilingReadyBindings, PULPRQSDWConv2DTilingReadyBindings, \
    PULPRQSGEMMTilingReadyBindings, PULPRQSTilingReadyBindings, PULPTransposeTilingReadyBindings

RQAddMapper = NodeMapper(PULPRQAddParser(), PULPRQAddTilingReadyBindings)
AddMapper = NodeMapper(AddParser(), PULPAddTilingReadyBindings)
FlattenMapper = NodeMapper(FlattenParser(), PULPFlattenTilingReadyBindings)
GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
MulMapper = NodeMapper(MulParser(), BasicMulBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), PULPFlattenTilingReadyBindings)
TransposeMapper = NodeMapper(TransposeParser(), PULPTransposeTilingReadyBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)

RequantShiftMapper = NodeMapper(RequantShiftParser(), PULPRQSTilingReadyBindings)
ReduceMeanMapper = NodeMapper(ReduceMeanParser(), PULPReduceMeanBindings)
MatMulMapper = NodeMapper(MatMulParser(), PULPMatMulTilingReadyBindings)
RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
RQGELU_int8_Mapper = NodeMapper(RQSiGELUParser(), [BasicRQSGELUBinding])

Conv1DMapper = NodeMapper(PULPConv1DParser(), [PULPConv1DBinding])
DWConv1DMapper = NodeMapper(PULPDWConv1DParser(), [PULPDWConv1DBinding])

Conv2DMapper = NodeMapper(PULPConv2DParser(), PULPRQSConv2DTilingReadyBindings)
DWConv2DMapper = NodeMapper(PULPDWConv2DParser(), PULPRQSDWConv2DTilingReadyBindings)
GEMMMapper = NodeMapper(CMSISGEMMParser(), PULPRQSGEMMTilingReadyBindings)
MaxPool2DMapper = NodeMapper(CMSISMaxPool2DParser(), PULPMaxPool2DTilingReadyBindings)
Softmax_int8_Mapper = NodeMapper(iSoftmaxParser(), PULPiSoftmaxTilingReadyBindings)

SliceMapper = NodeMapper(SliceParser(), PULPDMASliceBindings)

PULPMapping = {
    'RequantizedConv': PULPRQSConvLayer([Conv2DMapper, DWConv2DMapper, Conv1DMapper, DWConv1DMapper]),
    'RequantizedGemm': PULPRQSGEMMLayer([GEMMMapper]),
    'MaxPool': MaxPoolLayer([MaxPool2DMapper]),
    'RequantizediGELU': RQSiGELULayer([RQGELU_int8_Mapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'MatMul': MatMulLayer([MatMulMapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMeanMapper]),
    'iSoftmax': iSoftmaxLayer([Softmax_int8_Mapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMeanMapper]),
    'RequantShift': RequantShiftLayer([RequantShiftMapper]),
    'Add': AddLayer([AddMapper]),
    'Flatten': ReshapeLayer([FlattenMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'Mul': MulLayer([MulMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    'Slice': SliceLayer([SliceMapper]),
    'RequantizedAdd': AddLayer([RQAddMapper])
}


class PULPVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.pulpL2InitTemplate
    # allocTemplate = AllocateTemplate.pulpL2AllocateTemplate
    # deallocTemplate = FreeTemplate.pulpL2LocalTemplate

    allocTemplate = AllocateTemplate.pulpGenericAllocate
    deallocTemplate = FreeTemplate.pulpGenericFree

    def _nodeRep(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {
            "type": self._instance,
            "name": self.name,
            "size": int(np.prod(self.shape)),
            "_memoryLevel": memoryLevel
        }


class PULPTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.pulpL2InitTemplate
    allocTemplate = AllocateTemplate.pulpGenericAllocate
    deallocTemplate = FreeTemplate.pulpGenericFree

    # allocTemplate = AllocateTemplate.pulpL2AllocateTemplate
    # deallocTemplate = FreeTemplate.pulpL2GlobalTemplate

    def _nodeRep(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {"type": self._type, "name": self.name, "size": self.size, "_memoryLevel": memoryLevel}


class PULPConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.pulpGenericGlobalInitTemplate
    allocTemplate = AllocateTemplate.pulpL2GlobalAllocateTemplate
    deallocTemplate = FreeTemplate.pulpL2GlobalTemplate

    def _nodeRep(self):
        nodeRep = super()._nodeRep()

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        nodeRep["_memoryLevel"] = memoryLevel

        return nodeRep


class PULPStructBuffer(StructBuffer):

    initTemplate = BasicAllocateTemplate.referenceStructInitTemplate
    allocTemplate = BasicAllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


PULPOptimizer = TopologyOptimizer([
    IntegerDivRequantMergePass(),
    iGELURequantMergePass(),
    PULPConvRequantMergePass(),
    MergeConstAddAndRequantPass(),
    PULPGEMMRequantMergePass(),
    PULPMatMulRequantMergePass(),
    PULPAddRequantMergePass()
])

# SCHEREMO: stdint is included before pulp_nn_kernels.h because it is supposed to be included in there, but isn't...
_includeList = [
    "pmsis.h", "stdint.h", "pulp_nn_kernels.h", "DeeployBasicMath.h", "dory_dma.h", "dory_mem.h", "bsp/ram.h"
]


class PULPPlatform(DeploymentPlatform):

    def __init__(self,
                 Mapping = PULPMapping,
                 DataTypes = DataTypes,
                 VariableBuffer = PULPVariableBuffer,
                 ConstantBuffer = PULPConstantBuffer,
                 StructBuffer = PULPStructBuffer,
                 TransientBuffer = PULPTransientBuffer,
                 includeList: List[str] = _includeList):
        super().__init__(Mapping,
                         DataTypes,
                         VariableBuffer,
                         ConstantBuffer,
                         StructBuffer,
                         TransientBuffer,
                         includeList = includeList)
