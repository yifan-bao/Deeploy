# ----------------------------------------------------------------------
#
# File: MemPoolPlatform.py
#
# Last edited: 17.12.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author:
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

import onnx_graphsurgeon as gs

from Deeploy.Parsers.BasicParsers import *
from Deeploy.Parsers.GenericParsers import *
from Deeploy.Parsers.MemPoolParsers import *

from Deeploy.Layers.BasicLayers import *
from Deeploy.Layers.MemPoolLayers import *

from Deeploy.Bindings.BasicBindings import *
from Deeploy.Bindings.MemPoolBindings import *

from Deeploy.OptimizationPasses.BasicPasses import *
from Deeploy.OptimizationPasses.DebugPasses import *
from Deeploy.OptimizationPasses.MemPoolPasses import *

# Fallback bindings from the generic platform
# (they support a wider range of attribute values)
GenericConv1D_Mapper = NodeMapper(GenericConv1DParser(), [BasicConv1DBinding])
GenericDWConv1D_Mapper = NodeMapper(GenericDWConv1DParser(), [BasicDWConv1DBinding])
GenericConv2D_Mapper = NodeMapper(GenericConv2DParser(), [BasicConv2DBinding])
GenericDWConv2D_Mapper = NodeMapper(GenericDWConv2DParser(), [BasicDWConv2DBinding])
GenericGEMM_Mapper = NodeMapper(GenericGEMMParser(), [BasicGEMMBinding])

GenericConv_Mappers = [GenericConv2D_Mapper, GenericDWConv2D_Mapper, GenericConv1D_Mapper, GenericDWConv1D_Mapper]

# Basic bindings
BasicAdd_Mapper = NodeMapper(AddParser(), BasicAddBindings)
Debug_Mapper = NodeMapper(DebugParser(), BasicDebugBindings)
Flatten_Mapper = NodeMapper(FlattenParser(), BasicReshapeBindings)
Gather_Mapper = NodeMapper(GatherParser(), BasicGatherBindings)
GELU_Mapper = NodeMapper(iGELUParser(), [BasicGELUBinding])
iLayerNorm_Mapper = NodeMapper(iLayerNormParser(), [BasicLayerNormBinding])
IntegerDiv_Mapper = NodeMapper(IntegerDivParser(), [BasicIntegerDivBinding])
Mul_Mapper = NodeMapper(MulParser(), BasicMulBindings)
Pad1D_Mapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2D_Mapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
ReduceMean_Mapper = NodeMapper(ReduceMeanParser(), BasicReduceMeanBindings)
RequantShift_Mapper = NodeMapper(RequantShiftParser(), BasicRQSBindings)
Reshape_Mapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
RQGELU_Mapper = NodeMapper(RQSiGELUParser(), [BasicRQSGELUBinding])
RQIntegerDiv_Mapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
Softmax_Mapper = NodeMapper(iSoftmaxParser(), [BasicSoftmaxBinding])
Transpose_Mapper = NodeMapper(TransposeParser(), BasicTransposeBindings)
Unsqueeze_Mapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)

# MemPool specific bindings
Conv1D_Mapper = NodeMapper(GenericConv1DParser(), [MemPoolConv1D_8_8_32_Binding])
Conv2D_Mapper = NodeMapper(GenericConv2DParser(), [MemPoolConv2D_8_8_32_Binding])
DWConv1D_Mapper = NodeMapper(GenericDWConv1DParser(), [MemPoolDWConv1D_8_8_32_Binding])
DWConv2D_Mapper = NodeMapper(GenericDWConv2DParser(), [MemPoolDWConv2D_8_8_32_Binding])
MatMul_Mapper = NodeMapper(MatMulParser(), [MemPoolMatMul_8_8_32_Binding])
MaxPool_Mapper = NodeMapper(GenericMaxPool2DParser(), [MemPoolMaxPool2D_8_8_Binding])
MHSA_Mapper = NodeMapper(DebugMemPoolMHSAParser(), [MemPoolMHSA_8_8_8_8_Binding])

Conv_Mappers = [Conv2D_Mapper, DWConv2D_Mapper, Conv1D_Mapper, DWConv1D_Mapper]

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

MemPoolMapping = {
    'Add': AddLayer([BasicAdd_Mapper]),
    'Conv': ConvLayer(Conv_Mappers + GenericConv_Mappers),  # Mapper with higher priority should be placed first!
    'Debug': DebugLayer([Debug_Mapper]),
    'Div': IntegerDivLayer([IntegerDiv_Mapper]),
    'Flatten': ReshapeLayer([Flatten_Mapper]),
    'Gather': GatherLayer([Gather_Mapper]),
    'Gemm': GEMMLayer([GenericGEMM_Mapper]),
    'iGELU': iGELULayer([GELU_Mapper]),
    'iLayerNorm': iLayerNormLayer([iLayerNorm_Mapper]),
    'IntegerDiv': IntegerDivLayer([IntegerDiv_Mapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMean_Mapper]),
    'iSoftmax': iSoftmaxLayer([Softmax_Mapper]),
    'MatMul': MatMulLayer([MatMul_Mapper]),
    'MatMulInteger': MatMulLayer([MatMul_Mapper]),
    'MaxPool': MaxPoolLayer([MaxPool_Mapper]),
    'MHSA': MHSALayer([MHSA_Mapper]),
    'Mul': MulLayer([Mul_Mapper]),
    'Pad': PadLayer([Pad1D_Mapper, Pad2D_Mapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMean_Mapper]),
    'RequantizediGELU': RQSiGELULayer([RQGELU_Mapper]),
    'RequantShift': RequantShiftLayer([RequantShift_Mapper]),
    'Reshape': ReshapeLayer([Reshape_Mapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDiv_Mapper]),
    'Transpose': TransposeLayer([Transpose_Mapper]),
    'Unsqueeze': ReshapeLayer([Unsqueeze_Mapper]),
}


def MemPoolTypeInfer(node: gs.Node):
    if hasattr(node, 'values'):
        assert len(node.outputs) <= 1, "Expected node for type inference to only have ONE output!"
        outNode = node
    else:
        import IPython
        IPython.embed()
        raise ValueError("TypeInfer was given a wrong type of node!")

    if hasattr(outNode, 'signed') and outNode.attrs['signed']:
        signed = True
    else:
        signed = False

    for _type in DataTypes:
        if signed and outNode.values.max() < 2**(_type._value_ - 1) and outNode.values.min() >= -2**(_type._value_ - 1):
            return _type
        # For nor we only have signed kernels
        elif not signed and outNode.values.max() < 2**(_type._value_ - 1):
            return _type

    raise TypeError(f'Could not infer type of node {node.name}')


class SimpleVariableBuffer(VariableBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        return AllocateTemplate.MemPoolInitTemplate.generate(type = self._type._name_, name = self.name)

    def alloc(self):
        return AllocateTemplate.MemPoolAllocateTemplate.generate(type = self._type._name_,
                                                                 name = self.name,
                                                                 size = int(np.prod(self.shape)))

    def dealloc(self):
        return FreeTemplate.MemPoolLocalTemplate.generate(name = self.name)


class SimpleTransientBuffer(TransientBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        return AllocateTemplate.referenceInitTemplate.generate(type = "void", name = self.name)

    def alloc(self):
        return AllocateTemplate.referenceAllocateTemplate.generate(type = "int8_t", name = self.name, size = self.size)

    def dealloc(self):
        return FreeTemplate.referenceLocalTemplate.generate(name = self.name)


class SimpleConstantBuffer(ConstantBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        values = list(self.values.reshape(-1))
        strValues = [str(value) for value in values]
        valueString = ', '.join(strValues)

        # WIESEP: Workaround for banshee simulations.
        # Due to problems wrongly copied bytes, we want array sized a multiple of 4
        bytes = np.prod(self.shape) * (self._type.value // 8)
        if bytes % 4 != 0:
            bytes = 4 * int((bytes / 4 + 1))
        size = (bytes * 8) // self._type.value

        return AllocateTemplate.MemPoolGlobalInitTemplate.generate(type = self._type._name_,
                                                                   name = self.name,
                                                                   size = int(size),
                                                                   values = valueString)

    def alloc(self):
        values = list(self.values.reshape(-1))
        strValues = [str(value) for value in values]
        valueString = ', '.join(strValues)

        return AllocateTemplate.MemPoolGlobalAllocateTemplate.generate(type = self._type._name_,
                                                                       name = self.name,
                                                                       size = int(np.prod(self.shape)),
                                                                       values = valueString)

    def dealloc(self):
        return FreeTemplate.MemPoolGlobalTemplate.generate(name = self.name)


class SimpleStructBuffer(StructBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        return AllocateTemplate.MemPoolStructInitTemplate.generate(type = self._type,
                                                                   name = self.name,
                                                                   structDict = self.structDict)

    def alloc(self) -> str:
        return AllocateTemplate.MemPoolStructAllocateTemplate.generate(type = self._type,
                                                                       name = self.name,
                                                                       structDict = self.structDict)

    def dealloc(self) -> str:
        return FreeTemplate.MemPoolLocalTemplate.generate(name = self.name)


MemPoolOptimizer = NetworkOptimizer([
    FuseMHSAPass(),
    iGELURequantMergePass(),
    MatMulAddMergePass(),
    MergeConstAddAndRequantPass(),
    ExtractPaddingFromConvPass(),
    ExtractPaddingFromPoolPass(),
    # DebugPass("Conv", position='before'),
    # DebugPass("Pad", position='after'),
])

includeList = ["DeeployMath.h", "runtime.h", "synchronization.h"]


class MemPoolPlatform(DeploymentPlatform):

    def __init__(self,
                 Mapping = MemPoolMapping,
                 DataTypes = DataTypes,
                 TypeInfer = MemPoolTypeInfer,
                 VariableBuffer = SimpleVariableBuffer,
                 ConstantBuffer = SimpleConstantBuffer,
                 StructBuffer = SimpleStructBuffer,
                 TransientBuffer = SimpleTransientBuffer,
                 includeList: List[str] = includeList):
        super().__init__(Mapping, DataTypes, TypeInfer, VariableBuffer, ConstantBuffer, StructBuffer, TransientBuffer,
                         includeList)
