# ----------------------------------------------------------------------
#
# File: CMSISParsers.py
#
# Last edited: 17.12.2021        
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

import numpy as np
import math

from DumpO.DumpOTypes import *
from DumpO.DumpOManglers import *
from DumpO.Parsers.BasicParsers import *

class CMSISConv2DParser(Conv2DParser):
    def __init__(self):
        super().__init__()
    
    def parseNode(self, node: gs.ir.node.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.parserDict['group'] == 1,
                self.parserDict['pads'][0] == self.parserDict['pads'][2],
                self.parserDict['pads'][1] == self.parserDict['pads'][3],
                # Don't support dilations
                all([coeff == 1 for coeff in self.parserDict['dilations']]),
            ])
            
            if ret:
                self.parserDict['dim_kernel_x'] = int(self.parserDict['kernel_shape'][0])
                self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][1])
                self.parserDict['padding_x'] = int(self.parserDict['pads'][0])
                self.parserDict['padding_y'] = int(self.parserDict['pads'][1])
                self.parserDict['stride_x'] = int(self.parserDict['strides'][0])
                self.parserDict['stride_y'] = int(self.parserDict['strides'][1])
                self.parserDict['bias_shift'] = int(0)
                self.parserDict['out_shift'] = int(0)
                
        return wellFormed
    
    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)
        
        if ret:
            data_in = newCtxt.lookup(self.parserDict['data_in'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])
            weight = newCtxt.lookup(self.parserDict['weight'])
            self.parserDict['ch_im_in'] = data_in.shape[1]
            self.parserDict['dim_im_in_x'] = data_in.shape[2]
            self.parserDict['dim_im_in_y'] = data_in.shape[3]
            self.parserDict['ch_im_out'] = data_out.shape[1]
            self.parserDict['dim_im_out_x'] = data_out.shape[2]
            self.parserDict['dim_im_out_y'] = data_out.shape[3]
            
            return newCtxt, True
        
        return ctxt, False

class CMSISLinearParser(GEMMParser):

    def __init__(self):
        super().__init__()
    
    def parseNode(self, node: gs.ir.node.Node) -> (bool):

        wellFormed = super().parseNode(node)
        ret = False
        if wellFormed:

            ret = all([
                self.parserDict['alpha'] == 1.0,
                self.parserDict['beta'] == 1.0,
                self.parserDict['transA'] == 0,
                self.parserDict['transB'] == 1,
            ])
        
        return ret
    
    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)
        
        if ret:
            
            return newCtxt, True
        return ctxt, False
    
