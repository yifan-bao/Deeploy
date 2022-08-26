# ----------------------------------------------------------------------
#
# File: CMSISPlatform.py
#
# Last edited: 18.12.2021
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

from enum import Enum
import mako
import onnx_graphsurgeon as gs

from DumpO.Parsers.BasicParsers import *
from DumpO.Parsers.CMSISParsers import *

from DumpO.Layers.BasicLayers import *
from DumpO.Layers.CMSISLayers import *

from DumpO.Bindings.BasicBindings import *
from DumpO.Bindings.CMSISBindings import *

from DumpO.OptimizationPasses.CMSISPasses import *
from DumpO.OptimizationPasses.BasicPasses import *

GELU_int8_Mapper = NodeMapper(iGELUParser(), [BasicGELUBinding])
RQGELU_int8_Mapper = NodeMapper(RQSiGELUParser(), [RQSGELUBinding])
Softmax_int8_Mapper = NodeMapper(iSoftmaxParser(), [BasicSoftmaxBinding])
iLayerNorm_int8_Mapper = NodeMapper(iLayerNormParser(), [CMSISLayerNormBinding])
MHSA_int8_Mapper = NodeMapper(CMSISMHSAParser(), [CMSISMHSABinding])
LinearAttention_int16_Mapper = NodeMapper(CMSISLinearAttentionParser(), [CMSISLinearAttentionBinding])

CLCA_int8_Mapper = NodeMapper(CMSISCLCAParser(), [CMSISCLCABinding])

GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
FlattenMapper = NodeMapper(FlattenParser(), BasicReshapeBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)
RequantShiftMapper = NodeMapper(RequantShiftParser(), BasicRQSBindings)
ReduceMeanMapper = NodeMapper(ReduceMeanParser(), BasicReduceMeanBindings)
GEMMMapper = NodeMapper(CMSISGEMMParser(), CMSISGEMMBindings)

Conv2D_int8_Mapper = NodeMapper(CMSISConv2DParser(), [CMSISConv2DBinding])
DWConv2D_int8_Mapper = NodeMapper(CMSISDWConv2DParser(), [CMSISDWConv2DBinding])
Conv1D_Mapper = NodeMapper(CMSISConv1DParser(), CMSISConv1DBindings)
DWConv1D_Mapper = NodeMapper(CMSISDWConv1DParser(), CMSISDWConv1DBindings)

AddMapper = NodeMapper(AddParser(), BasicAddBindings)
MulMapper = NodeMapper(MulParser(), BasicMulBindings)
MatMulMapper = NodeMapper(MatMulParser(), [MatMul_8_8_32_Binding])

IntegerDivMapper = NodeMapper(IntegerDivParser(), [IntegerDivBinding])
RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [RQIntegerDivBinding])

TransposeMapper = NodeMapper(TransposeParser(), BasicTransposeBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
MaxPool2DMapper = NodeMapper(CMSISMaxPool2DParser(), [CMSISMaxPool2DBinding])

DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

CMSISMapping = {
    'RequantizedConv' : RQSConvLayer([Conv2D_int8_Mapper, DWConv2D_int8_Mapper, Conv1D_Mapper, DWConv1D_Mapper ]),
    'RequantizedGemm': RQSGEMMLayer([GEMMMapper]),
    'RequantShift': RequantShiftLayer([RequantShiftMapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMeanMapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMeanMapper]),
    'MaxPool': MaxPoolLayer([MaxPool2DMapper]),
    'IntegerDiv': IntegerDivLayer([IntegerDivMapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'Mul': MulLayer([MulMapper]),
    'MatMul': MatMulLayer([MatMulMapper]),
    'iLayerNorm': iLayerNormLayer([iLayerNorm_int8_Mapper]),
    'MultiHeadSelfAttention': MHSALayer([MHSA_int8_Mapper]),
    'LinearAttention': LinearAttentionLayer([LinearAttention_int16_Mapper]),
    'CLCA': CLCALayer([CLCA_int8_Mapper]),
    'iGELU' : iGELULayer([GELU_int8_Mapper]),
    'RequantizediGELU' : RQSiGELULayer([RQGELU_int8_Mapper]),
    'iSoftmax' : iSoftmaxLayer([Softmax_int8_Mapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'Add': AddLayer([AddMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    'Flatten': ReshapeLayer([FlattenMapper]),
}

def CMSISTypeInfer(node: gs.ir.node.Node):
    if hasattr(node, 'values'):
        assert len(node.outputs) <= 1, "Expected node for type inference to only have ONE output!"
        outNode = node
    else:
        import IPython; IPython.embed()
        raise ValueError("TypeInfer was given a wrong type of node!")

    if hasattr(outNode, 'signed') and outNode.attrs['signed']:
        signed = True
    else:
        signed = False

    for _type in DataTypes:
        if signed and outNode.values.max() < 2**(_type._value_-1) and outNode.values.min() >= -2**(_type._value_-1):
            return _type
        # For nor we only have signed kernels :(
        elif not signed and outNode.values.max() < 2**(_type._value_-1):
            return _type

    raise TypeError(f'Could not infer type of node {node.name}')

class SimpleNetworkBuffer(VariableBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        return AllocateTemplate.referenceInitTemplate.generate(type=self._type._name_, name=self.name)

    def alloc(self):
        return AllocateTemplate.referenceAllocateTemplate.generate(type = self._type._name_, name=self.name, size = np.prod(self.shape))

    def dealloc(self):
        return FreeTemplate.referenceLocalTemplate.generate(name = self.name)

class SimpleGlobalBuffer(ConstantBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        values = list(self.values.reshape(-1))
        strValues = [str(value) for value in values]
        valueString = ', '.join(strValues)
        return AllocateTemplate.referenceGlobalInitTemplate.generate(type=self._type._name_, name=self.name, size = int(np.prod(self.shape)), values = valueString)

    def alloc(self):
        values = list(self.values.reshape(-1))
        strValues = [str(value) for value in values]
        valueString = ', '.join(strValues)

        return AllocateTemplate.referenceGlobalAllocateTemplate.generate(type = self._type._name_, name=self.name, size = int(np.prod(self.shape)), values = valueString)

    def dealloc(self):
        return FreeTemplate.referenceGlobalTemplate.generate(name = self.name)

class SimpleStructBuffer(StructBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        return AllocateTemplate.referenceStructInitTemplate.generate(type=self._type, name=self.name, structDict = self.structDict)

    def alloc(self) -> str:
        return AllocateTemplate.referenceStructAllocateTemplate.generate(type=self._type, name=self.name, structDict=self.structDict)

    def dealloc(self) -> str:
        return FreeTemplate.referenceLocalTemplate.generate(name=self.name)

#ExtractPaddingFromConvPass(),ExtractPaddingFromPoolPass(),
CMSISOptimizer = NetworkOptimizer([ IntegerDivRequantMergePass(),iGELURequantMergePass(),LinearAttentionAlignmentPass(), MHSAAlignmentPass(), MergeConstAddAndRequantPass(), ConvRequantMergePass(), GEMMRequantMergePass(), MatMulRequantMergePass()])

includeList = ["arm_math.h", "arm_nnfunctions.h", "DumpOMath.h"]

class CMSISPlatform(DeploymentPlatform):
    def __init__(self, CMSISMapping = CMSISMapping, DataTypes = DataTypes, \
                 CMSISTypeInfer = CMSISTypeInfer, SimpleNetworkBuffer = SimpleNetworkBuffer,\
                 SimpleGlobalBuffer = SimpleGlobalBuffer, SimpleStructBuffer = SimpleStructBuffer,
                 includeList : List[str] = includeList):
        super().__init__(CMSISMapping, DataTypes, CMSISTypeInfer, \
                         SimpleNetworkBuffer, SimpleGlobalBuffer, SimpleStructBuffer, \
                         includeList)
