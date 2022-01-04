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

def bindFCParams(ctxt, name, mul, shift, data_in, weight, parserDict,  parserDictPrefix ='', bias=True):

    newCtxt = ctxt.copy()
    
    parserDict['in_N'] = np.prod(data_in.shape[0:-1])
    parserDict['in_C'] = np.prod(data_in.shape[-1:])
    parserDict['weight_N'] = parserDict['in_C']
    parserDict['weight_C'] = np.prod(weight.shape[0:-1])
    
    ctxtDict = {
        'buf': 0, #f'{name}_ctxt_buffer',  
        'size': 0
    }

    newCtxt.hoistStruct(ctxtDict, f'{name}_ctxt', 'cmsis_nn_context')
    parserDict[f'{parserDictPrefix}ctxt'] = f'{name}_ctxt'

    # activation
    if parserDict[f'signed']:
        activationDict = {
            'min': -(parserDict[f'n_levels']//2),
            'max': (parserDict[f'n_levels']//2) - 1
        }
    else:
        activationDict = {
            'min': 0,
            'max': (parserDict[f'n_levels'])-1
        }
    newCtxt.hoistStruct(activationDict, f'{name}_activation', 'cmsis_nn_activation')

    fcParamsDict = {
        'input_offset': 0,
        'filter_offset': 0,
        'output_offset': 0,
        'activation': newCtxt._mangle(newCtxt.lookup(f'{name}_activation').name),
    }
    newCtxt.hoistStruct(fcParamsDict, f'{name}_fc_params', 'cmsis_nn_fc_params')
    parserDict[f'{parserDictPrefix}fc_params'] = newCtxt.lookup(f'{name}_fc_params').name

    gemmQuantDict = {
        'multiplier': mul,
        'shift': shift,
    }

    newCtxt.hoistStruct(gemmQuantDict, f'{name}_quant_params', 'cmsis_nn_per_tensor_quant_params')
    parserDict[f'{parserDictPrefix}quant_params'] = newCtxt.lookup(f'{name}_quant_params').name

    inputDimsDict = {
        'n': parserDict['in_N'],
        'h': 1,
        'w': 1,
        'c': parserDict['in_C'],
    }            
    newCtxt.hoistStruct(inputDimsDict, f'{name}_input_dims', 'cmsis_nn_dims')
    parserDict[f'{parserDictPrefix}input_dims'] = newCtxt.lookup(f'{name}_input_dims').name

    filterDimsDict = {
        'n': parserDict['weight_N'],
        'h': 1,
        'w': 1,
        'c': parserDict['weight_C']
    }
    newCtxt.hoistStruct(filterDimsDict, f'{name}_filter_dims', 'cmsis_nn_dims')
    parserDict[f'{parserDictPrefix}filter_dims'] = newCtxt.lookup(f'{name}_filter_dims').name

    outputDimsDict = {
        'n': parserDict['in_N'],
        'h': 1,
        'w': 1,
        'c': parserDict['weight_C']
    }
    newCtxt.hoistStruct(outputDimsDict, f'{name}_output_dims', 'cmsis_nn_dims')
    parserDict[f'{parserDictPrefix}output_dims'] = newCtxt.lookup(f'{name}_output_dims').name

    biasDimsDict = {
        'n': 1,
        'h': 1,
        'w': 1,
        'c': parserDict['weight_C'] * bias,
    }
    newCtxt.hoistStruct(biasDimsDict, f'{name}_bias_dims', 'cmsis_nn_dims')
    parserDict[f'{parserDictPrefix}bias_dims'] = newCtxt.lookup(f'{name}_bias_dims').name
    
    return newCtxt, parserDict

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
                
            ctxtDict = {
                'buf': 0, #f'{node.name}_ctxt_buffer',  
                'size': 0
            }

            newCtxt.hoistStruct(ctxtDict, f'{node.name}_ctxt', 'cmsis_nn_context')
            self.parserDict['ctxt'] = f'{node.name}_ctxt'

            strideDict = {
                'w': self.parserDict['stride_x'],
                'h': self.parserDict['stride_y']
            }
            newCtxt.hoistStruct(strideDict, f'{node.name}_stride', 'cmsis_nn_tile')
            # padding
            paddingDict = {
                'w': self.parserDict['padding_x'],
                'h': self.parserDict['padding_y']
            }
            newCtxt.hoistStruct(paddingDict, f'{node.name}_padding', 'cmsis_nn_tile')

            # SCHEREMO: Fix this at some point...
            activationDict = {
                    'min': -2**7,
                    'max': 2**7-1
            }
            newCtxt.hoistStruct(activationDict, f'{node.name}_activation', 'cmsis_nn_activation')
                
            convParamsDict = {
                'stride': ctxt._mangle(newCtxt.lookup(f'{node.name}_stride').name),
                'padding': ctxt._mangle(newCtxt.lookup(f'{node.name}_padding').name),
                'activation': ctxt._mangle(newCtxt.lookup(f'{node.name}_activation').name),
            }
            newCtxt.hoistStruct(convParamsDict, f'{node.name}_pool_params', 'cmsis_nn_pool_params')
            self.parserDict[f'pool_params'] = newCtxt.lookup(f'{node.name}_pool_params').name
            
            inputDimsDict = {
                'n': 1,
                'h': self.parserDict['dim_im_in_x'],
                'w': self.parserDict['dim_im_in_y'],
                'c': self.parserDict['ch_im_in']
            }            
            newCtxt.hoistStruct(inputDimsDict, f'{node.name}_input_dims', 'cmsis_nn_dims')
            self.parserDict['input_dims'] = newCtxt.lookup(f'{node.name}_input_dims').name

            filterDimsDict = {
                'n': 1,
                'h': self.parserDict['dim_kernel_x'],
                'w': self.parserDict['dim_kernel_y'],
                'c': 1
            }
            newCtxt.hoistStruct(filterDimsDict, f'{node.name}_filter_dims', 'cmsis_nn_dims')
            self.parserDict['filter_dims'] = newCtxt.lookup(f'{node.name}_filter_dims').name

            outputDimsDict = {
                'n': 1,
                'h': self.parserDict['dim_im_out_x'],
                'w': self.parserDict['dim_im_out_y'],
                'c': self.parserDict['ch_im_out']
            }
            newCtxt.hoistStruct(outputDimsDict, f'{node.name}_output_dims', 'cmsis_nn_dims')
            self.parserDict['output_dims'] = newCtxt.lookup(f'{node.name}_output_dims').name

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
                
            # Hoist the structs to the global ctxt

            # First the context
            # https://review.trustedfirmware.org/plugins/gitiles/mirror/ARM-software/CMSIS_5/+/refs/heads/bias_for_conv/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c
            bufferSize = 2*self.parserDict['ch_im_in']*self.parserDict['dim_kernel_x']*self.parserDict['dim_kernel_y']*2 + 4

            ctxtDict = {
                'buf': 0, #f'{node.name}_ctxt_buffer',  
                'size': bufferSize
            }

            newCtxt.hoistStruct(ctxtDict, f'{node.name}_ctxt', 'cmsis_nn_context')
            self.parserDict['ctxt'] = f'{node.name}_ctxt'

            # Next the conv params
            # stride
            strideDict = {
                'w': self.parserDict['stride_x'],
                'h': self.parserDict['stride_y']
            }
            newCtxt.hoistStruct(strideDict, f'{node.name}_stride', 'cmsis_nn_tile')
            # padding
            paddingDict = {
                'w': self.parserDict['padding_x'],
                'h': self.parserDict['padding_y']
            }
            newCtxt.hoistStruct(paddingDict, f'{node.name}_padding', 'cmsis_nn_tile')
            # dilation
            dilationDict = {
                'w': self.parserDict['dilation_x'],
                'h': self.parserDict['dilation_y']
            }
            newCtxt.hoistStruct(dilationDict, f'{node.name}_dilation', 'cmsis_nn_tile')
            # activation
            if self.parserDict['signed']:
                activationDict = {
                    'min': -(self.parserDict['n_levels']//2),
                    'max': (self.parserDict['n_levels']//2) - 1
                }
            else:
                activationDict = {
                    'min': 0,
                    'max': (self.parserDict['n_levels'])-1
                }
            newCtxt.hoistStruct(activationDict, f'{node.name}_activation', 'cmsis_nn_activation')
                
            convParamsDict = {
                'input_offset': 0,
                'output_offset': 0,
                'ch_mult': 1,
                'stride': ctxt._mangle(newCtxt.lookup(f'{node.name}_stride').name),
                'padding': ctxt._mangle(newCtxt.lookup(f'{node.name}_padding').name),
                'dilation': ctxt._mangle(newCtxt.lookup(f'{node.name}_dilation').name),
                'activation': ctxt._mangle(newCtxt.lookup(f'{node.name}_activation').name),
            }
            newCtxt.hoistStruct(convParamsDict, f'{node.name}_dw_conv_params', 'cmsis_nn_dw_conv_params')
            self.parserDict[f'dw_conv_params'] = newCtxt.lookup(f'{node.name}_dw_conv_params').name
            
            convQuantDict = {
                'multiplier': ctxt._mangle(self.parserDict['mul']),
                'shift': ctxt._mangle(self.parserDict['shift']),
            }            
            newCtxt.hoistStruct(convQuantDict, f'{node.name}_quant_params', 'cmsis_nn_per_channel_quant_params')
            self.parserDict['quant_params'] = newCtxt.lookup(f'{node.name}_quant_params').name

            inputDimsDict = {
                'n': 1,
                'h': self.parserDict['dim_im_in_x'],
                'w': self.parserDict['dim_im_in_y'],
                'c': self.parserDict['ch_im_in']
            }            
            newCtxt.hoistStruct(inputDimsDict, f'{node.name}_input_dims', 'cmsis_nn_dims')
            self.parserDict['input_dims'] = newCtxt.lookup(f'{node.name}_input_dims').name

            filterDimsDict = {
                'n': 1,
                'h': self.parserDict['dim_kernel_x'],
                'w': self.parserDict['dim_kernel_y'],
                'c': self.parserDict['ch_im_out']
            }
            newCtxt.hoistStruct(filterDimsDict, f'{node.name}_filter_dims', 'cmsis_nn_dims')
            self.parserDict['filter_dims'] = newCtxt.lookup(f'{node.name}_filter_dims').name

            outputDimsDict = {
                'n': 1,
                'h': self.parserDict['dim_im_out_x'],
                'w': self.parserDict['dim_im_out_y'],
                'c': self.parserDict['ch_im_out']
            }
            newCtxt.hoistStruct(outputDimsDict, f'{node.name}_output_dims', 'cmsis_nn_dims')
            self.parserDict['output_dims'] = newCtxt.lookup(f'{node.name}_output_dims').name

            biasDimsDict = {
                'n': 1,
                'h': 1,
                'w': 1,
                'c': self.parserDict['ch_im_out'],
            }
            newCtxt.hoistStruct(biasDimsDict, f'{node.name}_bias_dims', 'cmsis_nn_dims')
            self.parserDict['bias_dims'] = newCtxt.lookup(f'{node.name}_bias_dims').name
            
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
                
            # Hoist the structs to the global ctxt

            # First the context
            # https://review.trustedfirmware.org/plugins/gitiles/mirror/ARM-software/CMSIS_5/+/refs/heads/bias_for_conv/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c
            bufferSize = 2*self.parserDict['ch_im_in']*self.parserDict['dim_kernel_x']*self.parserDict['dim_kernel_y']*2 

            ctxtDict = {
                'buf': 0, #f'{node.name}_ctxt_buffer',  
                'size': bufferSize
            }

            newCtxt.hoistStruct(ctxtDict, f'{node.name}_ctxt', 'cmsis_nn_context')
            self.parserDict['ctxt'] = f'{node.name}_ctxt'

            # Next the conv params
            # stride
            strideDict = {
                'w': self.parserDict['stride_x'],
                'h': self.parserDict['stride_y']
            }
            newCtxt.hoistStruct(strideDict, f'{node.name}_stride', 'cmsis_nn_tile')
            # padding
            paddingDict = {
                'w': self.parserDict['padding_x'],
                'h': self.parserDict['padding_y']
            }
            newCtxt.hoistStruct(paddingDict, f'{node.name}_padding', 'cmsis_nn_tile')
            # dilation
            dilationDict = {
                'w': self.parserDict['dilation_x'],
                'h': self.parserDict['dilation_y']
            }
            newCtxt.hoistStruct(dilationDict, f'{node.name}_dilation', 'cmsis_nn_tile')
            # activation
            if self.parserDict['signed']:
                activationDict = {
                    'min': -(self.parserDict['n_levels']//2),
                    'max': (self.parserDict['n_levels']//2) - 1
                }
            else:
                activationDict = {
                    'min': 0,
                    'max': (self.parserDict['n_levels'])-1
                }
            newCtxt.hoistStruct(activationDict, f'{node.name}_activation', 'cmsis_nn_activation')
                
            convParamsDict = {
                'input_offset': 0,
                'output_offset': 0,
                'stride': ctxt._mangle(newCtxt.lookup(f'{node.name}_stride').name),
                'padding': ctxt._mangle(newCtxt.lookup(f'{node.name}_padding').name),
                'dilation': ctxt._mangle(newCtxt.lookup(f'{node.name}_dilation').name),
                'activation': ctxt._mangle(newCtxt.lookup(f'{node.name}_activation').name),
            }
            newCtxt.hoistStruct(convParamsDict, f'{node.name}_conv_params', 'cmsis_nn_conv_params')
            self.parserDict[f'conv_params'] = newCtxt.lookup(f'{node.name}_conv_params').name
            
            convQuantDict = {
                'multiplier': ctxt._mangle(self.parserDict['mul']),
                'shift': ctxt._mangle(self.parserDict['shift']),
            }            
            newCtxt.hoistStruct(convQuantDict, f'{node.name}_quant_params', 'cmsis_nn_per_channel_quant_params')
            self.parserDict['quant_params'] = newCtxt.lookup(f'{node.name}_quant_params').name

            inputDimsDict = {
                'n': 1,
                'h': self.parserDict['dim_im_in_x'],
                'w': self.parserDict['dim_im_in_y'],
                'c': self.parserDict['ch_im_in']
            }            
            newCtxt.hoistStruct(inputDimsDict, f'{node.name}_input_dims', 'cmsis_nn_dims')
            self.parserDict['input_dims'] = newCtxt.lookup(f'{node.name}_input_dims').name

            filterDimsDict = {
                'n': self.parserDict['ch_im_out'],
                'h': self.parserDict['dim_kernel_x'],
                'w': self.parserDict['dim_kernel_y'],
                'c': self.parserDict['ch_im_in']
            }
            newCtxt.hoistStruct(filterDimsDict, f'{node.name}_filter_dims', 'cmsis_nn_dims')
            self.parserDict['filter_dims'] = newCtxt.lookup(f'{node.name}_filter_dims').name

            outputDimsDict = {
                'n': 1,
                'h': self.parserDict['dim_im_out_x'],
                'w': self.parserDict['dim_im_out_y'],
                'c': self.parserDict['ch_im_out']
            }
            newCtxt.hoistStruct(outputDimsDict, f'{node.name}_output_dims', 'cmsis_nn_dims')
            self.parserDict['output_dims'] = newCtxt.lookup(f'{node.name}_output_dims').name

            biasDimsDict = {
                'n': 1,
                'h': 1,
                'w': 1,
                'c': self.parserDict['ch_im_out'],
            }
            newCtxt.hoistStruct(biasDimsDict, f'{node.name}_bias_dims', 'cmsis_nn_dims')
            self.parserDict['bias_dims'] = newCtxt.lookup(f'{node.name}_bias_dims').name
            
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

            inputs = ['A', 'B', 'add']
                            
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = newCtxt.lookup(inputNode.name).name
                
            # Hoist the structs to the global ctxt
            data_in = newCtxt.lookup(self.parserDict['A'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])
            weight = newCtxt.lookup(self.parserDict['B'])
            
            newCtxt, self.parserDict = bindFCParams(newCtxt, node.name, self.parserDict['mul'], self.parserDict['shift'], data_in, weight, self.parserDict);
            
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
            inputs = ['q', 'k', 'v', 'wq_weight', 'wq_bias','wk_weight', 'wk_bias', 'wv_weight', 'wv_bias', 'wo_weight', 'wo_bias']                    
            s = newCtxt.lookup(self.parserDict['q']).shape[1]

            data_in = newCtxt.lookup(self.parserDict['q'])
            bias = newCtxt.lookup(self.parserDict['wq_bias'])
            weight = newCtxt.lookup(self.parserDict['wq_weight'])

            # Q FC layer:
            newCtxt, self.parserDict = bindFCParams(newCtxt, node.name+"_wq", self.parserDict['wq_requant_mul'], self.parserDict['wq_requant_shift'], data_in, weight, self.parserDict, "wq_", bias = (np.prod(bias.shape) > 1))

            data_in = newCtxt.lookup(self.parserDict['k'])
            bias = newCtxt.lookup(self.parserDict['wk_bias'])
            weight = newCtxt.lookup(self.parserDict['wk_weight'])

            newCtxt, self.parserDict = bindFCParams(newCtxt, node.name+"_wk", self.parserDict['wk_requant_mul'], self.parserDict['wk_requant_shift'], data_in, weight, self.parserDict, "wk_", bias = (np.prod(bias.shape) > 1))

            data_in = newCtxt.lookup(self.parserDict['v'])
            bias = newCtxt.lookup(self.parserDict['wv_bias'])
            weight = newCtxt.lookup(self.parserDict['wv_weight'])

            newCtxt, self.parserDict = bindFCParams(newCtxt, node.name+"_wv", self.parserDict['wv_requant_mul'], self.parserDict['wv_requant_shift'], data_in, weight, self.parserDict, "wv_", bias = (np.prod(bias.shape) > 1))

            data_in= np.ones((1, data_in.shape[1], self.parserDict['heads']*self.parserDict['dim_head']))
            bias = newCtxt.lookup(self.parserDict['wo_bias'])
            weight = newCtxt.lookup(self.parserDict['wo_weight'])

            newCtxt, self.parserDict = bindFCParams(newCtxt, node.name+"_wo", self.parserDict['wo_requant_mul'], self.parserDict['wo_requant_shift'], data_in, weight, self.parserDict, "wo_", bias = (np.prod(bias.shape) > 1))

            data_in = np.ones((s,self.parserDict['dim_head']))
            # K
            weight = np.ones((s,self.parserDict['dim_head']))
            
            newCtxt, self.parserDict = bindFCParams(newCtxt, node.name+"_preattn", self.parserDict['preattn_requant_mul'], self.parserDict['preattn_requant_shift'], data_in, weight, self.parserDict, "preattn_", bias=False)

            data_in = np.ones((s,s))
            # K
            weight = np.ones((self.parserDict['dim_head'],s))
            
            newCtxt, self.parserDict = bindFCParams(newCtxt, node.name+"_postattn", self.parserDict['postattn_requant_mul'], self.parserDict['postattn_requant_shift'], data_in, weight, self.parserDict, "postattn_", bias=False)

        return newCtxt, ret
