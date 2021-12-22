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

from DumpO.Bindings.BasicBindings import *
from DumpO.Bindings.CMSISBindings import *

from DumpO.OptimizationPasses.CMSISPasses import *

GELU_int8_Mapper = NodeMapper(GELUParser(), None)
iLayerNorm_int8_Mapper = NodeMapper(iLayerNormParser(), None)
MatMul_int8_Mapper = NodeMapper(CMSISLinearParser(), [BasicGEMMBinding])
GEMM_int8_Mapper = NodeMapper(CMSISLinearParser(), [BasicGEMMBinding])
#Conv_int8_Mapper_testo = NodeMapper(Conv2DParser(), ConvChecker([CMSISDataTypes.int8_t, CMSISDataTypes.int8_t], [CMSISDataTypes.int16_t]), DummyTemplate.referenceTemplate)
MHSA_int8_Mapper = NodeMapper(MHSAParser(), None)

GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
FlattenMapper = NodeMapper(FlattenParser(), BasicReshapeBindings)
RequantShiftMapper = NodeMapper(RequantShiftParser(), None)

Conv_int8_Mapper = NodeMapper(CMSISConv2DParser(), [CMSISConv2DBinding])
AddMapper = NodeMapper(AddParser(), CMSISSaturatingAddBindings)

DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

CMSISMapping = {
    #'Conv' : ConvLayer([Conv_int8_Mapper]),
    'RequantizedConv' : ConvLayer([Conv_int8_Mapper]),
    'iLayerNorm': iLayerNormLayer([iLayerNorm_int8_Mapper]),
    'MultiHeadSelfAttention': MHSALayer([MHSA_int8_Mapper]),
    'iGELU' : iGELULayer([GELU_int8_Mapper]),
    'MatMul': GEMMLayer([MatMul_int8_Mapper]),
    'Gemm': GEMMLayer([GEMM_int8_Mapper]),
    
    'Gather': GatherLayer([GatherMapper]),
    'Add': AddLayer([AddMapper]),
    'RequantShift' : RequantShiftLayer([RequantShiftMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
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
        
        return AllocateTemplate.referenceGlobalTemplate.generate(type = self._type._name_, name=self.name, size = np.prod(self.shape), values = valueString)

    def dealloc(self):
        return FreeTemplate.referenceGlobalTemplate.generate(name = self.name)

class SimpleStructBuffer(StructBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def alloc(self) -> str:
        return AllocateTemplate.referenceStructTemplate.generate(type=self._type, name=self.name, structDict=self.structDict)

    def dealloc(self) -> str:
        return FreeTemplate.referenceLocalTemplate.generate(name=self.name)
    
    
CMSISOptimizer = NetworkOptimizer([ConvRequantMergePass()])
    
CMSISPlatform = DeploymentPlatform(CMSISMapping, CMSISDataTypes, CMSISTypeInfer, SimpleNetworkBuffer, SimpleGlobalBuffer, SimpleStructBuffer)
