# ----------------------------------------------------------------------
#
# File: BasicCheckers.py
#
# Last edited: 16.12.2021        
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

from parserTypes import *
from parserTypes import _mangleVariableName, _mangleParameterName

class DummyChecker(NodeTypeChecker):
    def __init__(self):
        super().__init__()

    # Override this. This should check that the input n_levels are appropriate for the kernel
    def typeCheckNode(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> bool:
        return True

    def typeInferOutput(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> NetworkContext:
        newCtxt = ctxt.copy()

        # Dummy value for nLevels
        nLevels = 256
        
        outputNodes = node.outputs
        outputNames = [_mangleVariableName(node.name) for node in outputNodes]
        for node, name in zip(outputNodes, outputNames):
            if not newCtxt.is_global(name):
                nb = NetworkBuffer(
                    name = name,
                    shape = node.shape,
                    nLevels = nLevels
                )
                newCtxt.add(nb, 'local')
            else:
                nb = newCtxt.lookup(name)
                assert nb.nLevels >= nLevels, f'Type mismatch at output node of {node.name}'
                
        return newCtxt
