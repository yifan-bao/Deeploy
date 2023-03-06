# ----------------------------------------------------------------------
#
# File: CMSISPlatform.py
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

from DumpO.Parsers.BasicParsers import *
from DumpO.Parsers.CMSISParsers import *

from DumpO.Layers.BasicLayers import *
from DumpO.Layers.CMSISLayers import *

from DumpO.Bindings.BasicBindings import *
from DumpO.Bindings.CMSISBindings import *

from DumpO.OptimizationPasses.BasicPasses import *
from DumpO.OptimizationPasses.CMSISPasses import *

AddMapper = NodeMapper(AddParser(), BasicAddBindings)
CLCA_int8_Mapper = NodeMapper(CMSISCLCAParser(), [CMSISCLCABinding])
Conv1D_Mapper = NodeMapper(CMSISConv1DParser(), CMSISConv1DBindings)
Conv2D_int8_Mapper = NodeMapper(CMSISConv2DParser(), [CMSISConv2DBinding])
DWConv1D_Mapper = NodeMapper(CMSISDWConv1DParser(), CMSISDWConv1DBindings)
DWConv2D_int8_Mapper = NodeMapper(CMSISDWConv2DParser(), [CMSISDWConv2DBinding])
FlattenMapper = NodeMapper(FlattenParser(), BasicReshapeBindings)
GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
GELU_int8_Mapper = NodeMapper(iGELUParser(), [BasicGELUBinding])
GEMMMapper = NodeMapper(CMSISGEMMParser(), CMSISGEMMBindings)
iLayerNorm_int8_Mapper = NodeMapper(iLayerNormParser(), [CMSISLayerNormBinding])
IntegerDivMapper = NodeMapper(IntegerDivParser(), [CMSISIntegerDivBinding])
LinearAttention_int16_Mapper = NodeMapper(CMSISLinearAttentionParser(), [CMSISLinearAttentionBinding])
MatMulMapper = NodeMapper(MatMulParser(), [CMSISMatMul_8_8_32_Binding])
MaxPool2DMapper = NodeMapper(CMSISMaxPool2DParser(), [CMSISMaxPool2DBinding])
MHSA_int8_Mapper = NodeMapper(CMSISMHSAParser(), [CMSISMHSABinding])
MulMapper = NodeMapper(MulParser(), BasicMulBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
ReduceMeanMapper = NodeMapper(ReduceMeanParser(), BasicReduceMeanBindings)
RequantShiftMapper = NodeMapper(RequantShiftParser(), BasicRQSBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
RQGELU_int8_Mapper = NodeMapper(RQSiGELUParser(), [BasicRQSGELUBinding])
RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [CMSISRQIntegerDivBinding])
Softmax_int8_Mapper = NodeMapper(iSoftmaxParser(), [BasicSoftmaxBinding])
TransposeMapper = NodeMapper(TransposeParser(), BasicTransposeBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

CMSISMapping = {
    'Add': AddLayer([AddMapper]),
    'CLCA': CLCALayer([CLCA_int8_Mapper]),
    'Flatten': ReshapeLayer([FlattenMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'iGELU': iGELULayer([GELU_int8_Mapper]),
    'iLayerNorm': iLayerNormLayer([iLayerNorm_int8_Mapper]),
    'IntegerDiv': IntegerDivLayer([IntegerDivMapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMeanMapper]),
    'iSoftmax': iSoftmaxLayer([Softmax_int8_Mapper]),
    'LinearAttention': LinearAttentionLayer([LinearAttention_int16_Mapper]),
    'MatMul': MatMulLayer([MatMulMapper]),
    'MaxPool': MaxPoolLayer([MaxPool2DMapper]),
    'Mul': MulLayer([MulMapper]),
    'MultiHeadSelfAttention': MHSALayer([MHSA_int8_Mapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMeanMapper]),
    'RequantizedConv': CMSISRQSConvLayer([Conv2D_int8_Mapper, DWConv2D_int8_Mapper, Conv1D_Mapper, DWConv1D_Mapper]),
    'RequantizedGemm': CMSISRQSGEMMLayer([GEMMMapper]),
    'RequantizediGELU': RQSiGELULayer([RQGELU_int8_Mapper]),
    'RequantShift': RequantShiftLayer([RequantShiftMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
}


def CMSISTypeInfer(node:  gs.Node):
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


class SimpleNetworkBuffer(VariableBuffer):

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


class SimpleGlobalBuffer(ConstantBuffer):

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


# ExtractPaddingFromConvPass(),ExtractPaddingFromPoolPass(),
CMSISOptimizer = NetworkOptimizer([
    IntegerDivRequantMergePass(),
    iGELURequantMergePass(),
    LinearAttentionAlignmentPass(),
    MHSAAlignmentPass(),
    MergeConstAddAndRequantPass(),
    ConvRequantMergePass(),
    GEMMRequantMergePass(),
    MatMulRequantMergePass(),
])

includeList = ["arm_math.h", "arm_nnfunctions.h", "DumpOMath.h"]


class CMSISPlatform(DeploymentPlatform):

    def __init__(self,
                 CMSISMapping = CMSISMapping,
                 DataTypes = DataTypes,
                 CMSISTypeInfer = CMSISTypeInfer,
                 SimpleNetworkBuffer = SimpleNetworkBuffer,
                 SimpleGlobalBuffer = SimpleGlobalBuffer,
                 SimpleStructBuffer = SimpleStructBuffer,
                 includeList: List[str] = includeList):
        super().__init__(CMSISMapping, DataTypes, CMSISTypeInfer, SimpleNetworkBuffer, SimpleGlobalBuffer,
                         SimpleStructBuffer, includeList)
