# ----------------------------------------------------------------------
#
# File: PULPTiler.py
#
# Last edited: 09.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from Deeploy.Bindings.BasicBindings import BasicAddBindings, BasicReshapeBindings, BasicTransposeBindings
from Deeploy.Bindings.PULPBindings import ClusterTransformer, PULPMatMulBinding, PULPMaxPool2DBindings, \
    PULPRQAddBindings, PULPRQSBindings, PULPRQSConv2DBindings, PULPRQSDWConv2DBindings, PULPRQSGEMMBindings, \
    PULPSoftmaxBindings, PULPTransposeBindings, SimpleTransformer
from Deeploy.CodeTransformationPasses import MemoryPassthroughGeneration
from Deeploy.DeeployTypes import CodeTransformation
from Deeploy.Tiling.BasicTileConstraintFlow import AddTileConstraintFlow, NOPTileConstraintFlow, \
    TransposeTileConstraintFlow, UntiledTileConstraintFlow
from Deeploy.Tiling.PULPTileConstraintFlow import Conv2DTileConstraintFlow, DWConv2DTileConstraintFlow, \
    GEMMTileConstraintFlow, MatMulTileConstraintFlow, MaxPoolTileConstraintFlow, RequantShiftTileConstraintFlow, \
    iSoftmaxTileConstraintFlow
from Deeploy.Tiling.TilerExtension import TilingReadyNodeBindings

PULPRQSConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSConv2DBindings,
                                                           tileConstraintFlow = Conv2DTileConstraintFlow())

PULPRQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSDWConv2DBindings,
                                                             tileConstraintFlow = DWConv2DTileConstraintFlow())

PULPRQSGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSGEMMBindings,
                                                         tileConstraintFlow = GEMMTileConstraintFlow())

PULPMatMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPMatMulBinding],
                                                        tileConstraintFlow = MatMulTileConstraintFlow())

PULPRQAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQAddBindings,
                                                       tileConstraintFlow = AddTileConstraintFlow())

_BasicFlattenBindings = copy.deepcopy(BasicReshapeBindings)
for binding in _BasicFlattenBindings:
    binding.codeTransformer = CodeTransformation([MemoryPassthroughGeneration("L.*"), MemoryPassthroughGeneration()])

PULPFlattenTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _BasicFlattenBindings,
                                                         tileConstraintFlow = NOPTileConstraintFlow())

PULPMaxPool2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMaxPool2DBindings,
                                                           tileConstraintFlow = MaxPoolTileConstraintFlow())

PULPRQSTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSBindings,
                                                     tileConstraintFlow = RequantShiftTileConstraintFlow())

# _PULPTransposeBindings = BasicTransposeBindings
# for binding in _PULPTransposeBindings:
#     binding.codeTransformer = SimpleTransformer

# PULPTransposeTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _PULPTransposeBindings,
#                                                            tileConstraintFlow = UntiledTileConstraintFlow())

PULPTransposeTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPTransposeBindings,
                                                           tileConstraintFlow = TransposeTileConstraintFlow())

_PULPAddBindings = copy.deepcopy(BasicAddBindings)
for binding in _PULPAddBindings:
    binding.codeTransformer = SimpleTransformer

PULPAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _PULPAddBindings,
                                                     tileConstraintFlow = UntiledTileConstraintFlow())

PULPiSoftmaxTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPSoftmaxBindings,
                                                          tileConstraintFlow = iSoftmaxTileConstraintFlow())
