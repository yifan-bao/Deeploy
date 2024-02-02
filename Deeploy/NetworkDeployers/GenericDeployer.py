# ----------------------------------------------------------------------
#
# File: GenericDeployer.py
#
# Last edited: 04.01.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
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

from typing import Callable, Dict

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import PointerType
from Deeploy.DeeployTypes import DeploymentPlatform, NetworkDeployer, TopologyOptimizer
from Deeploy.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.BasicPasses import TransposeConstOptPass, TransposeMergePass
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.DebugPasses import DebugPrintMergePass
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import NCHWtoNHWCPass, \
    TransposeMatmulInputsPass


class GenericDeployer(SignPropDeployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, PointerType],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first = False,
                 deeployStateDir: str = "DeeployStateDir",
                 inputOffsets: Dict[str, int] = {}):

        super().__init__(graph,
                         deploymentPlatform,
                         inputTypes,
                         loweringOptimizer,
                         scheduler,
                         name,
                         default_channels_first = default_channels_first,
                         deeployStateDir = deeployStateDir)

        self.inputOffsets = inputOffsets

        self.loweringOptimizer.passes += [
            TransposeMatmulInputsPass(),
            NCHWtoNHWCPass(self.default_channels_first),
            TransposeMergePass(),
            TransposeConstOptPass(),
            DebugPrintMergePass()
        ]
