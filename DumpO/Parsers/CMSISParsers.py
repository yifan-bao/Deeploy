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
from DumpO.Bindings.BasicBindings import DataTypes

class CMSISMaxPool2DParser(MaxPool2DParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.ir.node.Node) -> bool:
        
        ret = super().parseNode(node)
        wellFormed = all([
            self.parserDict['pads'][0] == 0,
            self.parserDict['pads'][1] == 0,
        ])
        if wellFormed:
            self.parserDict['padding_x'] = int(self.parserDict['pads'][0])
            self.parserDict['padding_y'] = int(self.parserDict['pads'][1])
            self.parserDict['stride_x'] = int(self.parserDict['strides'][0])
            self.parserDict['stride_y'] = int(self.parserDict['strides'][1])
            self.parserDict['dim_kernel_x'] = int(self.parserDict['kernel_shape'][0])
            self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][1])
        return wellFormed

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node, channels_first: bool = True) -> (NetworkContext, bool):
        
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)
        if ret:

            data_in = newCtxt.lookup(self.parserDict['data_in'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])


            self.parserDict['batch'] = data_in.shape[0]
            if channels_first:
                self.parserDict['ch_im_in'] = data_in.shape[1]
                self.parserDict['dim_im_in_x'] = data_in.shape[2]
                self.parserDict['dim_im_in_y'] = data_in.shape[3]
                self.parserDict['ch_im_out'] = data_out.shape[1]
                self.parserDict['dim_im_out_x'] = data_out.shape[2]
                self.parserDict['dim_im_out_y'] = data_out.shape[3]
            else:
                self.parserDict['ch_im_in'] = data_in.shape[3]
                self.parserDict['dim_im_in_x'] = data_in.shape[1]
                self.parserDict['dim_im_in_y'] = data_in.shape[2]
                self.parserDict['ch_im_out'] = data_out.shape[3]
                self.parserDict['dim_im_out_x'] = data_out.shape[1]
                self.parserDict['dim_im_out_y'] = data_out.shape[2]
                
        return newCtxt, ret

class CMSISDWConv2DParser(Conv2DParser):
    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)
    
    def parseNode(self, node: gs.ir.node.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                node.op == 'RequantizedConv',
                self.parserDict['pads'][0] == self.parserDict['pads'][2],
                self.parserDict['pads'][1] == self.parserDict['pads'][3],
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                self.parserDict['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.parserDict['dilations']]),
                len(node.inputs) == 5,
                'div' in node.attrs,
                'n_levels' in node.attrs,
                'signed' in node.attrs
            ])
            
            if ret:
                self.parserDict['dim_kernel_x'] = int(self.parserDict['kernel_shape'][0])
                self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][1])
                self.parserDict['dilation_x'] = int(self.parserDict['dilations'][0])
                self.parserDict['dilation_y'] = int(self.parserDict['dilations'][1])
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
    
    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node, channels_first: bool = True) -> (NetworkContext, bool):
        
        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)
            
        if ret:

            if not self.parserDict['group'] == newCtxt.lookup(self.parserDict['weight']).shape[0]:
                return ctxt, False

            inputs = ['data_in', 'weight', 'mul', 'add', 'shift']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = newCtxt.lookup(inputNode.name).name
                
            data_in = newCtxt.lookup(self.parserDict['data_in'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])
            weight = newCtxt.lookup(self.parserDict['weight'])

            if not newCtxt.is_global(self.parserDict['weight']):
                return ctxt, False

            # SCHEREMO: Transpose weights to be num filters last
            newCtxt.globalObjects[self.parserDict['weight']].values = np.transpose(weight.values, list(range(len(weight.shape)))[1:] + [0])
            
            if channels_first:
                self.parserDict['ch_im_in'] = data_in.shape[1]
                self.parserDict['dim_im_in_x'] = data_in.shape[2]
                self.parserDict['dim_im_in_y'] = data_in.shape[3]
                self.parserDict['ch_im_out'] = data_out.shape[1]
                self.parserDict['dim_im_out_x'] = data_out.shape[2]
                self.parserDict['dim_im_out_y'] = data_out.shape[3]
            else:
                self.parserDict['ch_im_in'] = data_in.shape[3]
                self.parserDict['dim_im_in_x'] = data_in.shape[1]
                self.parserDict['dim_im_in_y'] = data_in.shape[2]
                self.parserDict['ch_im_out'] = data_out.shape[3]
                self.parserDict['dim_im_out_x'] = data_out.shape[1]
                self.parserDict['dim_im_out_y'] = data_out.shape[2]                
            
            return newCtxt, True
        
        return ctxt, False


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
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                self.parserDict['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.parserDict['dilations']]),
                len(node.inputs) == 5,
                'div' in node.attrs,
                'n_levels' in node.attrs,
                'signed' in node.attrs
            ])
            
            if ret:
                self.parserDict['dim_kernel_x'] = int(self.parserDict['kernel_shape'][0])
                self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][1])
                self.parserDict['dilation_x'] = int(self.parserDict['dilations'][0])
                self.parserDict['dilation_y'] = int(self.parserDict['dilations'][1])
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
    
    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node, channels_first: bool = True) -> (NetworkContext, bool):
        
        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)
            
        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add', 'shift']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
                
            data_in = newCtxt.lookup(self.parserDict['data_in'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])
            weight = newCtxt.lookup(self.parserDict['weight'])
            
            if channels_first:
                self.parserDict['ch_im_in'] = data_in.shape[1]
                self.parserDict['dim_im_in_x'] = data_in.shape[2]
                self.parserDict['dim_im_in_y'] = data_in.shape[3]
                self.parserDict['ch_im_out'] = data_out.shape[1]
                self.parserDict['dim_im_out_x'] = data_out.shape[2]
                self.parserDict['dim_im_out_y'] = data_out.shape[3]
            else:
                self.parserDict['ch_im_in'] = data_in.shape[3]
                self.parserDict['dim_im_in_x'] = data_in.shape[1]
                self.parserDict['dim_im_in_y'] = data_in.shape[2]
                self.parserDict['ch_im_out'] = data_out.shape[3]
                self.parserDict['dim_im_out_x'] = data_out.shape[1]
                self.parserDict['dim_im_out_y'] = data_out.shape[2]
                            
            return newCtxt, True
        
        return ctxt, False
    
class CMSISLinearParser(GEMMParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)
    
    def parseNode(self, node: gs.ir.node.Node) -> (bool):

        wellFormed = super().parseNode(node)
        return wellFormed
    
    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node, channels_first: bool = True) -> (NetworkContext, bool):
        
        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)
        if ret:
            # Try to transpose A offline if possible, else fail
            if self.parserDict['transA'] == 1:
                nameA = self.parserDict['A']
                if newCtxt.is_global(nameA) and isinstance(newCtxt.lookup(nameA), ConstantBuffer):
                    A = newCtxt.lookup(nameA)
                    npA = np.asarray(A.values).reshape(A.shape)
                    newA = np.transpose(npA, list(range(len(A.shape)-2)) + [len(A.shape)-1, len(A.shape)-2])
                    newCtxt.globalObjects[nameA].shape = newA.shape
                    newCtxt.globalObjects[nameA].values = newA
                    self.parserDict['transA'] = 0
                else:
                    return newCtxt, False    

            # Try to transpose B offline if possible, else fail
            # SCHEREMO: Magic trick - CMSIS works a bit weirdly with matmuls...
            if self.parserDict['transB'] == 0:
                nameB = self.parserDict['B']
                if newCtxt.is_global(nameB) and isinstance(newCtxt.lookup(nameB), ConstantBuffer):
                    B = newCtxt.lookup(nameB)
                    npB = np.asarray(B.values).reshape(B.shape)
                    newB = np.transpose(npB, list(range(len(B.shape)-2)) + [len(B.shape)-1, len(B.shape)-2])
                    newCtxt.globalObjects[nameB].values = newB
                    newCtxt.globalObjects[nameB].shape = newB.shape
                    self.parserDict['transB'] = 1
                else:
                    return newCtxt, False    

            # Try to scale A offline if possible, else fail
            if self.parserDict['alpha'] != 1.0:
                nameA = self.parserDict['A']
                if newCtxt.is_global(nameA) and isinstance(newCtxt.lookup(nameA), ConstantBuffer):
                    A = newCtxt.lookup(nameA)
                    npA = np.asarray(A.values).reshape(A.shape)
                    newA = npA * alpha
                    newCtxt.globalObjects[nameA].values = newA
                    self.parserDict['alpha'] = 1.0
                else:
                    return newCtxt, False
                
            # Try to scale B offline if possible, else fail
            if self.parserDict['beta'] != 1.0:
                nameB = self.parserDict['B']
                if newCtxt.is_global(nameB) and isinstance(newCtxt.lookup(nameB), ConstantBuffer):
                    B = newCtxt.lookup(nameB)
                    npB = np.asarray(B.values).reshape(B.shape)
                    newB = npB * beta
                    newCtxt.globalObjects[nameB].values = newB
                    self.parserDict['beta'] = 1.0
                else:
                    return newCtxt, False    
                
            return newCtxt, True
        
        return ctxt, False
    
class CMSISGEMMParser(CMSISLinearParser):
    def __init__(self):
        super().__init__(noBiasHoisting=True)
    
    def parseNode(self, node: gs.ir.node.Node) -> (bool):

        wellFormed = super().parseNode(node)
        
        if wellFormed:
            ret = all([
                'div' in node.attrs,
                'n_levels' in node.attrs,
                'signed' in node.attrs,
                'mul' in node.attrs,
                'shift' in node.attrs,
                len(node.inputs) == 3,
        ])

            if ret:
                self.parserDict['n_levels'] = int(node.attrs['n_levels'].values)
                self.parserDict['signed'] = int(node.attrs['signed'].values)
                self.parserDict['log2D'] = int(math.log2(node.attrs['div'].values))
                self.parserDict['mul'] = int(node.attrs['mul'].values)
                self.parserDict['shift'] = int(node.attrs['shift'].values)
                
            return ret
        
        return False
    
    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node, channels_first: bool = True) -> (NetworkContext, bool):
        
        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)
        
        if ret:
            return newCtxt, True
        
        else:
            return ctxt, False

class CMSISMHSAParser(MHSAParser):
    def __init__(self):
        super().__init__()
    
    def parseNode(self, node: gs.ir.node.Node) -> (bool):

        wellFormed = super().parseNode(node)
        self.parserDict['signed'] = 1
        return wellFormed

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.ir.node.Node, channels_first: bool = True) -> (NetworkContext, bool):
        
        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:
            return newCtxt, ret
        else:
            return ctxt, False
