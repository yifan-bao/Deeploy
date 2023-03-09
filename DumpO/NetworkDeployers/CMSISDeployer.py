# ----------------------------------------------------------------------
#
# File: NetworkDeployer.py
#
# Last edited: 26.12.2021
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

from DumpO.DumpOTypes import *

from DumpO.Parsers.BasicParsers import *

from DumpO.Layers.BasicLayers import *
from DumpO.Layers.CMSISLayers import *

from DumpO.OptimizationPasses.BasicPasses import *
from DumpO.OptimizationPasses.CMSISPasses import *

from DumpO.OptimizationPasses.LoweringOptimizationPasses import *

class CMSISDeployer(NetworkDeployer):

    def __init__(self, graph: gs.Graph, deploymentPlatform: DeploymentPlatform, loweringOptimizer: NetworkOptimizer, scheduler: Callable = lambda x: x, name: str = 'DumpONetwork', input_n_levels : Dict[str, int] = {'input_0': 256}, input_signed : Dict[str, bool] = {'input_0':False}, channels_first=False):
        super().__init__(graph, deploymentPlatform, loweringOptimizer, scheduler, name, input_n_levels, input_signed)

    def postLoweringOptimization(self):
        # Insert appropriate transposes

        TransposeMatMulInputs(self.graph)
        NCHWtoNHWC(self.graph)
        newCtxt, ret = self.baseParse() # This sanity checks the graph and generates a base context for lowering/optimization
        postLoweringOptimizer = NetworkOptimizer([

            TransposeMergePass(),
            TransposeConstOptPass()
        ])
        postLoweringOptimizer.optimize(newCtxt, self.graph)
        # Reparse
        newCtxt, ret = self.baseParse() # This sanity checks the graph and generates a base context for lowering/optimization
        if not ret:
            raise RuntimeError("Lowering of the graph failed!")
        return newCtxt, self.graph
