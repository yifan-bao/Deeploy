# ----------------------------------------------------------------------
#
# File: CMSISPasses.py
#
# Last edited: 20.12.2021        
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

import copy

from DumpO.DumpOTypes import *
from DumpO.Layers.BasicLayers import *

class ConvRequantMergePass(NetworkOptimizationPass):
    def __init__(self):
        super().__init__()

    def merge(self, ctxt: NetworkContext, layerBinding : List, idx: int) -> (NetworkContext, List):
        newCtxt = ctxt.copy()
        convLayer = layerBinding[idx][1]
        RQSLayer = layerBinding[idx+1][1]
        # Binding was already done -> we know the data types work

        # Check that RQS is only user of convLayer:
        convOut = convLayer.mapper.parser.parserDict['data_out']
        convOutBuffer = newCtxt.lookup(convOut)

        #import IPython; IPython.embed()
        if len(convOutBuffer._users) == 1:
            
            # we can safely merge if parameters are constant:
            if newCtxt.is_global(RQSLayer.mapper.parser.parserDict['add']):

                convLayer.mapper.parser.parserDict['mul'] = RQSLayer.mapper.parser.parserDict['mul']
                convLayer.mapper.parser.parserDict['bias'] = RQSLayer.mapper.parser.parserDict['add']
                convLayer.mapper.parser.parserDict['log2D'] = RQSLayer.mapper.parser.parserDict['log2D']

                # now we clean up the context
                # remove RQS add 
                #del newCtxt.globalObjects[RQSLayer.mapper.parser.parserDict['add']]

                # set convLayer's output to RQSLayer's output
                old_data_out = convLayer.mapper.parser.parserDict['data_out']
                convLayer.mapper.parser.parserDict['data_out'] = RQSLayer.mapper.parser.parserDict['data_out']

                convLayer.node.outputs = RQSLayer.node.outputs
                
                # delete middle node
                del newCtxt.localObjects[old_data_out]

                # Now delete RQSLayer entirely from the layerBinding
                del layerBinding[idx+1]
                layerBinding[idx] = (layerBinding[idx][0], convLayer)
                
                return newCtxt, layerBinding
                
        return ctxt, layerBinding
    
    def run(self, ctxt: NetworkContext, layerBinding : List) -> (NetworkContext, List):
        newLayerBinding = copy.deepcopy(layerBinding)
        newCtxt = ctxt.copy()
        
        layers = [layer for name, layer in layerBinding]
        convLayers = [idx for idx, layer in enumerate(layerBinding) if isinstance(layer[1], ConvLayer)]
        requantShiftLayers = [idx for idx, layer in enumerate(layerBinding) if isinstance(layer[1], RequantShiftLayer)]
        
        for idx in reversed(convLayers):
            if (idx+1) in requantShiftLayers:
                newCtxt, newLayerBinding = self.merge(newCtxt, newLayerBinding, idx)
                
        return newCtxt, newLayerBinding
