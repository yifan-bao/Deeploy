# ----------------------------------------------------------------------
#
# File: GenericPlatform.py
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

import onnx_graphsurgeon as gs

from Deeploy.Parsers.BasicParsers import *
from Deeploy.Parsers.GenericParsers import *

from Deeploy.Layers.BasicLayers import *

from Deeploy.Bindings.BasicBindings import *

from Deeploy.OptimizationPasses.BasicPasses import *
from Deeploy.OptimizationPasses.DebugPasses import *

AddMapper = NodeMapper(AddParser(), BasicAddBindings)
Conv1DMapper = NodeMapper(GenericConv1DParser(), [BasicConv1DBinding])
DWConv1DMapper = NodeMapper(GenericDWConv1DParser(), [BasicDWConv1DBinding])
Conv2DMapper = NodeMapper(GenericConv2DParser(), [BasicConv2DBinding])
DWConv2DMapper = NodeMapper(GenericDWConv2DParser(), [BasicDWConv2DBinding])
DebugMapper = NodeMapper(DebugParser(), BasicDebugBindings)
FlattenMapper = NodeMapper(FlattenParser(), BasicReshapeBindings)
GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
GELUMapper = NodeMapper(iGELUParser(), [BasicGELUBinding])
GEMMMapper = NodeMapper(GenericGEMMParser(), [BasicGEMMBinding])
iLayerNormMapper = NodeMapper(iLayerNormParser(), [BasicLayerNormBinding])
IntegerDivMapper = NodeMapper(IntegerDivParser(), [BasicIntegerDivBinding])
MatMulMapper = NodeMapper(MatMulParser(), [BasicMatMulBinding])
MaxPoolMapper = NodeMapper(GenericMaxPool2DParser(), [BasicMaxPool2DBinding])
MulMapper = NodeMapper(MulParser(), BasicMulBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
ReduceMeanMapper = NodeMapper(ReduceMeanParser(), BasicReduceMeanBindings)
RequantShiftMapper = NodeMapper(RequantShiftParser(), BasicRQSBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
RQGELUMapper = NodeMapper(RQSiGELUParser(), [BasicRQSGELUBinding])
RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
SoftmaxMapper = NodeMapper(iSoftmaxParser(), [BasicSoftmaxBinding])
TransposeMapper = NodeMapper(TransposeParser(), BasicTransposeBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

GenericMapping = {
    'Add': AddLayer([AddMapper]),
    'Conv': ConvLayer([Conv2DMapper, DWConv2DMapper, Conv1DMapper, DWConv1DMapper]),
    'Debug': DebugLayer([DebugMapper]),
    'Div': IntegerDivLayer([IntegerDivMapper]),
    'Flatten': ReshapeLayer([FlattenMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'Gemm': GEMMLayer([GEMMMapper]),
    'iGELU': iGELULayer([GELUMapper]),
    'iLayerNorm': iLayerNormLayer([iLayerNormMapper]),
    'IntegerDiv': IntegerDivLayer([IntegerDivMapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMeanMapper]),
    'iSoftmax': iSoftmaxLayer([SoftmaxMapper]),
    'MatMul': GEMMLayer([MatMulMapper]),
    'MatMulInteger': MatMulLayer([MatMulMapper]),
    'MaxPool': MaxPoolLayer([MaxPoolMapper]),
    'Mul': MulLayer([MulMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMeanMapper]),
    'RequantizediGELU': RQSiGELULayer([RQGELUMapper]),
    'RequantShift': RequantShiftLayer([RequantShiftMapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),

    # # For example, you can use the DummpyMapper, in case you want to test
    # # deployment or optimizations with GlobalAveragePool nodes but did not yet
    # # implement the corresponding kernel
    # 'GlobalAveragePool': ConvLayer([DummyMapper]),
}


def GenericTypeInfer(node: gs.Node):
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
        return AllocateTemplate.referenceInitTemplate.generate(type = self._type._name_, name = self.name)

    def alloc(self):
        return AllocateTemplate.referenceAllocateTemplate.generate(type = self._type._name_,
                                                                   name = self.name,
                                                                   size = np.prod(self.shape))

    def dealloc(self):
        return FreeTemplate.referenceLocalTemplate.generate(name = self.name)


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
        return AllocateTemplate.referenceGlobalInitTemplate.generate(type = self._type._name_,
                                                                     name = self.name,
                                                                     size = int(np.prod(self.shape)),
                                                                     values = valueString)

    def alloc(self):
        values = list(self.values.reshape(-1))
        strValues = [str(value) for value in values]
        valueString = ', '.join(strValues)

        return AllocateTemplate.referenceGlobalAllocateTemplate.generate(type = self._type._name_,
                                                                         name = self.name,
                                                                         size = int(np.prod(self.shape)),
                                                                         values = valueString)

    def dealloc(self):
        return FreeTemplate.referenceGlobalTemplate.generate(name = self.name)


class SimpleStructBuffer(StructBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        return AllocateTemplate.referenceStructInitTemplate.generate(type = self._type,
                                                                     name = self.name,
                                                                     structDict = self.structDict)

    def alloc(self) -> str:
        return AllocateTemplate.referenceStructAllocateTemplate.generate(type = self._type,
                                                                         name = self.name,
                                                                         structDict = self.structDict)

    def dealloc(self) -> str:
        return FreeTemplate.referenceLocalTemplate.generate(name = self.name)


GenericOptimizer = NetworkOptimizer([
    iGELURequantMergePass(),
    MatMulAddMergePass(),
    MergeConstAddAndRequantPass(),
    ExtractPaddingFromConvPass(),
    ExtractPaddingFromPoolPass(),
    # DebugPass("Conv", position='before'),
    # DebugPass("Pad", position='after'),
])

includeList = ["DeeployBasicMath.h"]


class GenericPlatform(DeploymentPlatform):

    def __init__(self,
                 Mapping = GenericMapping,
                 DataTypes = DataTypes,
                 TypeInfer = GenericTypeInfer,
                 VariableBuffer = SimpleVariableBuffer,
                 ConstantBuffer = SimpleConstantBuffer,
                 StructBuffer = SimpleStructBuffer,
                 TransientBuffer = SimpleTransientBuffer,
                 includeList: List[str] = includeList):
        super().__init__(Mapping, DataTypes, TypeInfer, VariableBuffer, ConstantBuffer, StructBuffer, TransientBuffer,
                         includeList)
