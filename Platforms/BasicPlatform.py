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


GELUMapper = lambda : NodeMapper(GELUParser, GELUChecker, iGELUTemplate.referenceTemplate)
RequantShiftMapper = lambda : NodeMapper(RequantShiftParser, RequantShiftChecker, RequantShiftTemplate.referenceTemplate)
DummyMapper = lambda : NodeMapper(DummyParser, DummyChecker, DummyTemplate.referenceTemplate)
AddMapper = lambda : NodeMapper(AddParser, AddChecker, AddTemplate.referenceTemplate)
iLayerNormMapper = lambda : NodeMapper(iLayerNormParser, iLayerNormChecker, DummyTemplate.referenceTemplate)
MatMulMapper = lambda : NodeMapper(MatMulParser, GEMMChecker, GEMMTemplate.referenceTemplate)
GEMMMapper = lambda : NodeMapper(GEMMParser, GEMMChecker, GEMMTemplate.referenceTemplate)
ConvMapper = lambda : NodeMapper(ConvParser, ConvChecker, DummyTemplate.referenceTemplate)
MHSAMapper = lambda : NodeMapper(MHSAParser, MHSAChecker, MHSATemplate.referenceTemplate)
GatherMapper = lambda : NodeMapper(GatherParser, GatherChecker, GatherTemplate.referenceTemplate)

BasicMapping = {
    'Conv' : partial(onnxLayers.ConvLayer, maps=[ConvMapper]),
    'RequantShift' : partial(onnxLayers.RequantShiftLayer, maps=[RequantShiftMapper]),
    'Reshape': partial(onnxLayers.ReshapeLayer, maps=[DummyMapper]),
    'iLayerNorm': partial(onnxLayers.iLayerNormLayer, maps=[iLayerNormMapper]),
    'MultiHeadSelfAttention': partial(onnxLayers.MHSALayer, maps=[MHSAMapper]),
    'Add': partial(onnxLayers.AddLayer, maps=[AddMapper]),
    'iGELU' : partial(onnxLayers.iGELULayer, maps=[GELUMapper]),
    'MatMul': partial(onnxLayers.GEMMLayer, maps=[MatMulMapper]),
    'Gemm': partial(onnxLayers.GEMMLayer, maps=[GEMMMapper]),
    'Gather': partial(onnxLayers.GatherLayer, maps=[GatherMapper]),

    #'Transpose': partial(onnxLayers.ReshapeLayer, maps=[DummyMapper]),
}
