# ----------------------------------------------------------------------
#
# File: BasicPlatform.py
#
# Last edited: 15.12.2021        
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

from functools import partial
from enum import Enum
import mako
import onnx_graphsurgeon as gs

from DumpO.Parsers.BasicParsers import *
from DumpO.TypeCheckers.BasicCheckers import *
from DumpO.Layers.BasicLayers import *
from DumpO.Templates.BasicTemplates import *

class DataTypes(Enum):
    int8_t = 8
    int16_t = 16
    int32_t = 32

GELU_int8_Mapper = NodeMapper(GELUParser(), GELUChecker(DataTypes.int8_t, DataTypes.int8_t), mako.template.Template(iGELUTemplate.referenceTemplate))
iLayerNorm_int8_Mapper = NodeMapper(iLayerNormParser(), iLayerNormChecker(DataTypes.int8_t, DataTypes.int8_t), mako.template.Template(DummyTemplate.referenceTemplate))
MatMul_int8_Mapper = NodeMapper(MatMulParser(), GEMMChecker(DataTypes.int8_t, DataTypes.int32_t), mako.template.Template(GEMMTemplate.referenceTemplate))
GEMM_int8_Mapper = NodeMapper(GEMMParser(), GEMMChecker(DataTypes.int8_t, DataTypes.int32_t), mako.template.Template(GEMMTemplate.referenceTemplate))
Conv_int8_Mapper = NodeMapper(Conv2DParser(), ConvChecker(DataTypes.int8_t, DataTypes.int32_t), mako.template.Template(DummyTemplate.referenceTemplate))
MHSA_int8_Mapper = NodeMapper(MHSAParser(), MHSAChecker(DataTypes.int8_t, DataTypes.int32_t), mako.template.Template(MHSATemplate.referenceTemplate))

GatherMappers = [NodeMapper(GatherParser(), GatherChecker(type), mako.template.Template(GatherTemplate.referenceTemplate)) for type in DataTypes]
ReshapeMappers = [NodeMapper(ReshapeParser(), ReshapeChecker(type), mako.template.Template(SkipTemplate.referenceTemplate)) for type in DataTypes]
RequantShiftMappers = [NodeMapper(RequantShiftParser(), RequantShiftChecker(type, DataTypes.int8_t), mako.template.Template(RequantShiftTemplate.referenceTemplate)) for type in DataTypes]
AddMappers = [NodeMapper(AddParser(), AddChecker(type, DataTypes.int32_t), mako.template.Template(AddTemplate.referenceTemplate)) for type in DataTypes]

DummyMapper = NodeMapper(DummyParser(), DummyChecker(DataTypes.int8_t), mako.template.Template(DummyTemplate.referenceTemplate))

BasicMapping = {
    'Conv' : ConvLayer([Conv_int8_Mapper]),
    'iLayerNorm': iLayerNormLayer([iLayerNorm_int8_Mapper]),
    'MultiHeadSelfAttention': MHSALayer([MHSA_int8_Mapper]),
    'iGELU' : iGELULayer([GELU_int8_Mapper]),
    'MatMul': GEMMLayer([MatMul_int8_Mapper]),
    'Gemm': GEMMLayer([GEMM_int8_Mapper]),
    
    'Gather': GatherLayer(GatherMappers),
    'Add': AddLayer(AddMappers),
    'RequantShift' : RequantShiftLayer(RequantShiftMappers),
    'Reshape': ReshapeLayer(ReshapeMappers),
    'Flatten': ConvLayer([DummyMapper]),
    'GlobalAveragePool': ConvLayer([DummyMapper]),
}

def TypeInfer(node):
    if type(node) == gs.ir.node.Node:
        assert len(node.outputs) == 1, "Expected node for type inference to only have ONE output!"
        outNode = node.attrs['value']
    elif hasattr(node, 'values'):
        outNode = node
    else:
        raise ValueError("TypeInfer was given a wring type of node!")
    
    for _type in DataTypes:
        if outNode.values.max() < 2**(_type._value_): #and outNode.values.min() >= -2**(_type._value_-1):
            return _type

    #SCHEREMO: Remove this - this should raise and error
    #return DataTypes.int32_t
    raise TypeError(f'Could not infer type of node {node.name}')

BasicPlatform = DeploymentPlatform(BasicMapping, DataTypes, TypeInfer)
