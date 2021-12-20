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

from DumpO.TypeCheckers.BasicCheckers import *
from DumpO.TypeCheckers.CMSISCheckers import *

from DumpO.Layers.BasicLayers import *
from DumpO.Templates.BasicTemplates import *

from DumpO.Templates.CMSISTemplates import ConvTemplate, AddTemplate

class CMSISDataTypes(Enum):
    int8_t = 8
    int16_t = 16
    int32_t = 32

GELU_int8_Mapper = NodeMapper(GELUParser(), GELUChecker([CMSISDataTypes.int8_t], [CMSISDataTypes.int8_t]), iGELUTemplate.referenceTemplate)
iLayerNorm_int8_Mapper = NodeMapper(iLayerNormParser(), iLayerNormChecker([CMSISDataTypes.int8_t,CMSISDataTypes.int32_t,CMSISDataTypes.int32_t], [CMSISDataTypes.int8_t]), DummyTemplate.referenceTemplate)
MatMul_int8_Mapper = NodeMapper(MatMulParser(), GEMMChecker([CMSISDataTypes.int8_t, CMSISDataTypes.int8_t], [CMSISDataTypes.int32_t]), GEMMTemplate.referenceTemplate)
GEMM_int8_Mapper = NodeMapper(GEMMParser(), GEMMChecker([CMSISDataTypes.int8_t, CMSISDataTypes.int8_t, CMSISDataTypes.int32_t], [CMSISDataTypes.int32_t]), GEMMTemplate.referenceTemplate)
Conv_int8_Mapper = NodeMapper(Conv2DParser(), ConvChecker([CMSISDataTypes.int8_t, CMSISDataTypes.int8_t], [CMSISDataTypes.int32_t]), DummyTemplate.referenceTemplate)
#Conv_int8_Mapper_testo = NodeMapper(Conv2DParser(), ConvChecker([CMSISDataTypes.int8_t, CMSISDataTypes.int8_t], [CMSISDataTypes.int16_t]), DummyTemplate.referenceTemplate)
MHSA_int8_Mapper = NodeMapper(MHSAParser(), MHSAChecker([CMSISDataTypes.int8_t], [CMSISDataTypes.int32_t]), MHSATemplate.referenceTemplate)

GatherMappers = [NodeMapper(GatherParser(), GatherChecker([type],[type]), GatherTemplate.referenceTemplate) for type in CMSISDataTypes]
ReshapeMappers = [NodeMapper(ReshapeParser(), ReshapeChecker([type],[type]), SkipTemplate.referenceTemplate) for type in CMSISDataTypes]
FlattenMappers = [NodeMapper(FlattenParser(), ReshapeChecker([type],[type]), SkipTemplate.referenceTemplate) for type in CMSISDataTypes]
RequantShiftMappers = [NodeMapper(RequantShiftParser(), RequantShiftChecker([type,CMSISDataTypes.int32_t,CMSISDataTypes.int32_t], [CMSISDataTypes.int8_t]), RequantShiftTemplate.referenceTemplate) for type in CMSISDataTypes]

Conv_int8_Mapper = NodeMapper(CMSISConv2DParser(), ConvChecker([CMSISDataTypes.int8_t,CMSISDataTypes.int8_t], [CMSISDataTypes.int32_t]), ConvTemplate.conv2DBasicTemplate)
AddMappers = [
    NodeMapper(AddParser(), CMSISSaturatingAddChecker([CMSISDataTypes.int8_t],[CMSISDataTypes.int8_t]), AddTemplate.AddInt8Template),
    NodeMapper(AddParser(), CMSISSaturatingAddChecker([CMSISDataTypes.int16_t],[CMSISDataTypes.int16_t]), AddTemplate.AddInt16Template),
    NodeMapper(AddParser(), CMSISSaturatingAddChecker([CMSISDataTypes.int32_t],[CMSISDataTypes.int32_t]), AddTemplate.AddInt32Template) ]

DummyMapper = NodeMapper(DummyParser(), DummyChecker([CMSISDataTypes.int8_t],[CMSISDataTypes.int8_t]), DummyTemplate.referenceTemplate)

CMSISMapping = {
    'Conv' : ConvLayer([Conv_int8_Mapper]),
    #'Conv' : ConvLayer([Conv_int8_Mapper_testo, Conv_int8_Mapper]),
    'iLayerNorm': iLayerNormLayer([iLayerNorm_int8_Mapper]),
    'MultiHeadSelfAttention': MHSALayer([MHSA_int8_Mapper]),
    'iGELU' : iGELULayer([GELU_int8_Mapper]),
    'MatMul': GEMMLayer([MatMul_int8_Mapper]),
    'Gemm': GEMMLayer([GEMM_int8_Mapper]),
    
    'Gather': GatherLayer(GatherMappers),
    'Add': AddLayer(AddMappers),
    'RequantShift' : RequantShiftLayer(RequantShiftMappers),
    'Reshape': ReshapeLayer(ReshapeMappers),
    'Flatten': ReshapeLayer(FlattenMappers),
#     'GlobalAveragePool': ConvLayer([DummyMapper]),
}

def CMSISTypeInfer(node):
    if type(node) == gs.ir.node.Node:
        assert len(node.outputs) == 1, "Expected node for type inference to only have ONE output!"
        outNode = node.attrs['value']
    elif hasattr(node, 'values'):
        outNode = node
    else:
        raise ValueError("TypeInfer was given a wring type of node!")

    if hasattr(outNode, 'signed') and outNode.attrs['signed']:
        signed = True
    else:
        signed = False
    
    for _type in CMSISDataTypes:
        if signed and outNode.values.max() < 2**(_type._value_-1) and outNode.values.min() >= -2**(_type._value_-1): 
            return _type
        # For nor we only have signed kernels :(
        elif not signed and outNode.values.max() < 2**(_type._value_-1): 
            return _type
            
    raise TypeError(f'Could not infer type of node {node.name}')

class SimpleNetworkBuffer(VariableBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def alloc(self):
        return AllocateTemplate.referenceLocalTemplate.generate(type = self._type._name_, name=self.name, size = np.prod(self.shape))

    def dealloc(self):
        return FreeTemplate.referenceLocalTemplate.generate(name = self.name)

class SimpleGlobalBuffer(ConstantBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def alloc(self):
        values = list(self.values.reshape(-1))
        strValues = [str(value) for value in values]
        valueString = ', '.join(strValues)
        try:
            return AllocateTemplate.referenceGlobalTemplate.generate(type = self._type._name_, name=self.name, size = np.prod(self.shape), values = valueString)
        except Exception as e:
            print(e)
            import IPython; IPython.embed()

    def dealloc(self):
        return FreeTemplate.referenceGlobalTemplate.generate(name = self.name)
    
CMSISPlatform = DeploymentPlatform(CMSISMapping, CMSISDataTypes, CMSISTypeInfer, SimpleNetworkBuffer, SimpleGlobalBuffer)
