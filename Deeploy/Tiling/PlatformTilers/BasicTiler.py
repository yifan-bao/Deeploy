# ----------------------------------------------------------------------
#
# File: BasicTiler.py
#
# Last edited: 01.06.2023
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
from Deeploy.Tiling.BasicTileConstraintFlow import AddTileConstraintFlow, NOPTileConstraintFlow, \
    TransposeTileConstraintFlow, UntiledTileConstraintFlow
from Deeploy.Tiling.TilerExtension import TilingReadyNodeBindings

BasicTransposeTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = BasicTransposeBindings,
                                                            tileConstraintFlow = TransposeTileConstraintFlow())

BasicFlattenTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = BasicReshapeBindings,
                                                          tileConstraintFlow = NOPTileConstraintFlow())

BasicAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = BasicAddBindings,
                                                      tileConstraintFlow = AddTileConstraintFlow())
