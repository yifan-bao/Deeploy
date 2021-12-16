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

from Parsers.BasicParsers import *
from TypeCheckers.BasicCheckers import *
import onnxLayers
from templates import *

GELU_int8_Mapper = lambda : NodeMapper(GELUParser, GELU_int8_Checker, iGELUTemplate.referenceTemplate)
RequantShift_int32_Mapper = lambda : NodeMapper(RequantShiftParser, RequantShift_int32_Checker, RequantShiftTemplate.int8Template)
RequantShift_int16_Mapper = lambda : NodeMapper(RequantShiftParser, RequantShift_int16_Checker, RequantShiftTemplate.int16Template)
RequantShift_int8_Mapper = lambda : NodeMapper(RequantShiftParser, RequantShift_int8_Checker, RequantShiftTemplate.int32Template)
ReshapeMapper = lambda : NodeMapper(ReshapeParser, ReshapeChecker, SkipTemplate.referenceTemplate)
Add_int8_Mapper = lambda : NodeMapper(AddParser, Add_int8_Checker, AddTemplate.int8Template)
Add_int16_Mapper = lambda : NodeMapper(AddParser, Add_int16_Checker, AddTemplate.int16Template)
Add_int32_Mapper = lambda : NodeMapper(AddParser, Add_int32_Checker, AddTemplate.int32Template)
iLayerNorm_int8_Mapper = lambda : NodeMapper(iLayerNormParser, iLayerNorm_int8_Checker, DummyTemplate.referenceTemplate)
MatMul_int8_Mapper = lambda : NodeMapper(MatMulParser, GEMM_int8_Checker, GEMMTemplate.referenceTemplate)
GEMM_int8_Mapper = lambda : NodeMapper(GEMMParser, GEMM_int8_Checker, GEMMTemplate.referenceTemplate)
Conv_int8_Mapper = lambda : NodeMapper(ConvParser, Conv_int8_Checker, DummyTemplate.referenceTemplate)
MHSA_int8_Mapper = lambda : NodeMapper(MHSAParser, MHSA_int8_Checker, MHSATemplate.referenceTemplate)
GatherMapper = lambda : NodeMapper(GatherParser, GatherChecker, GatherTemplate.referenceTemplate)

DummyMapper = lambda : NodeMapper(DummyParser, DummyChecker, DummyTemplate.referenceTemplate)

BasicMapping = {
    'Conv' : partial(onnxLayers.ConvLayer, maps=[Conv_int8_Mapper]),
    'RequantShift' : partial(onnxLayers.RequantShiftLayer, maps=[RequantShift_int32_Mapper, RequantShift_int16_Mapper, RequantShift_int8_Mapper]),
    'Reshape': partial(onnxLayers.ReshapeLayer, maps=[ReshapeMapper]),
    'iLayerNorm': partial(onnxLayers.iLayerNormLayer, maps=[iLayerNorm_int8_Mapper]),
    'MultiHeadSelfAttention': partial(onnxLayers.MHSALayer, maps=[MHSA_int8_Mapper]),
    'Add': partial(onnxLayers.AddLayer, maps=[Add_int8_Mapper, Add_int16_Mapper, Add_int32_Mapper]),
    'iGELU' : partial(onnxLayers.iGELULayer, maps=[GELU_int8_Mapper]),
    'MatMul': partial(onnxLayers.GEMMLayer, maps=[MatMul_int8_Mapper]),
    'Gemm': partial(onnxLayers.GEMMLayer, maps=[GEMM_int8_Mapper]),
    'Gather': partial(onnxLayers.GatherLayer, maps=[GatherMapper]),

    #'Transpose': partial(onnxLayers.ReshapeLayer, maps=[DummyMapper]),
}
