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
from DumpO.Parsers.BasicParsers import *

class CMSISConv2DParser(Conv2DParser):
    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)
    
    def parseNode(self, node: gs.ir.node.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                node.op == 'RequantizedConv',
                self.parserDict['group'] == 1,
                self.parserDict['pads'][0] == self.parserDict['pads'][2],
                self.parserDict['pads'][1] == self.parserDict['pads'][3],
                # Don't support dilations
                all([coeff == 1 for coeff in self.parserDict['dilations']]),
                len(node.inputs) == 4,
                'div' in node.attrs,
                'n_levels' in node.attrs,
                'signed' in node.attrs
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

                self.parserDict['n_levels'] = int(node.attrs['n_levels'].values)
                self.parserDict['signed'] = int(node.attrs['signed'].values)
                self.parserDict['log2D'] = int(math.log2(node.attrs['div'].values))
                
            return ret
    
    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)
            
        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
                
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
        return wellFormed
    
    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)
        
        if ret:
            # Try to transpose A offline if possible, else fail
            if self.parserDict['transA']:
                nameA = self.parserDict['A']
                if ctxt.is_global(nameA) and isinstance(ctxt.lookup(nameA), ConstantBuffer):
                    A = ctxt.lookup(nameA)
                    npA = np.asarray(A.values).reshape(A.shape)
                    newA = np.transpose(npA, list(range(len(A.shape)-2)) + [len(A.shape)-1, len(A.shape)-2])
                    ctxt.globalObjects[nameA].shape = newA.shape
                    ctxt.globalObjects[nameA].values = newA
                    self.parserDict['transA'] = 0
                else:
                    return ctxt, False    

            # Try to transpose B offline if possible, else fail
            if self.parserDict['transB']:
                nameB = self.parserDict['B']
                if ctxt.is_global(nameB) and isinstance(ctxt.lookup(nameB), ConstantBuffer):
                    B = ctxt.lookup(nameB)
                    npB = np.asarray(B.values).reshape(B.shape)
                    newB = np.transpose(npB, list(range(len(B.shape)-2)) + [len(B.shape)-1, len(B.shape)-2])
                    ctxt.globalObjects[nameB].values = newB
                    ctxt.globalObjects[nameB].shape = newB.shape
                    self.parserDict['transB'] = 0
                else:
                    return ctxt, False    

            # Try to scale A offline if possible, else fail
            if self.parserDict['alpha'] != 1.0:
                nameA = self.parserDict['A']
                if ctxt.is_global(nameA) and isinstance(ctxt.lookup(nameA), ConstantBuffer):
                    A = ctxt.lookup(nameA)
                    npA = np.asarray(A.values).reshape(A.shape)
                    newA = npA * alpha
                    ctxt.globalObjects[nameA].values = newA
                    self.parserDict['alpha'] = 1.0
                else:
                    return ctxt, False
                
            # Try to scale B offline if possible, else fail
            if self.parserDict['beta'] != 1.0:
                nameB = self.parserDict['B']
                if ctxt.is_global(nameB) and isinstance(ctxt.lookup(nameB), ConstantBuffer):
                    B = ctxt.lookup(nameB)
                    npB = np.asarray(B.values).reshape(B.shape)
                    newB = npB * beta
                    ctxt.globalObjects[nameB].values = newB
                    self.parserDict['beta'] = 1.0
                else:
                    return ctxt, False    
                
            return newCtxt, True
        
        return ctxt, False
    
