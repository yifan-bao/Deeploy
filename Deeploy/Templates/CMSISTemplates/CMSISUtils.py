# ----------------------------------------------------------------------
#
# File: CMSISUtils.py
#
# Last edited: 10.01.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
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

from Deeploy.DeeployTypes import NetworkContext
import numpy as np


def bindConvParams(ctxt, name, repName, batch, nodeRep):
    ctxt = ctxt.copy()

    # Hoist the structs to the global ctxt

    # First the context
    # https://review.trustedfirmware.org/plugins/gitiles/mirror/ARM-software/CMSIS_5/+/refs/heads/bias_for_conv/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c
    bufferSize = 2 * nodeRep['ch_im_in'] * nodeRep['dim_kernel_x'] * nodeRep['dim_kernel_y'] * 2

    ctxtDict = {
        'buf': 0,  #f'{name}_ctxt_buffer',
        'size': bufferSize
    }

    ctxt.hoistStruct(ctxtDict, f'{name}_ctxt', 'cmsis_nn_context')
    nodeRep[f'{repName}_ctxt'] = f'{name}_ctxt'

    # Next the conv params
    # stride
    strideDict = {
        'h': nodeRep['stride_x'],
        'w': nodeRep['stride_y'],
    }
    ctxt.hoistStruct(strideDict, f'{name}_stride', 'cmsis_nn_tile')
    # padding
    paddingDict = {'h': nodeRep['padding_x'], 'w': nodeRep['padding_y']}
    ctxt.hoistStruct(paddingDict, f'{name}_padding', 'cmsis_nn_tile')
    # dilation
    dilationDict = {'h': nodeRep['dilation_x'], 'w': nodeRep['dilation_y']}
    ctxt.hoistStruct(dilationDict, f'{name}_dilation', 'cmsis_nn_tile')
    # activation
    # if nodeRep['signed']:
    #     activationDict = {
    #         'min': -(nodeRep['n_levels']//2),
    #         'max': (nodeRep['n_levels']//2) - 1
    #     }
    # else:
    #     activationDict = {
    #         'min': 0,
    #         'max': (nodeRep['n_levels'])-1
    #     }
    activationDict = {'min': -(nodeRep['n_levels'] // 2), 'max': (nodeRep['n_levels'] // 2) - 1}

    ctxt.hoistStruct(activationDict, f'{name}_activation', 'cmsis_nn_activation')

    if 'data_in' in nodeRep:
        data_in = ctxt.lookup(nodeRep['data_in'])
        data_in_signed = data_in._signed
    else:
        data_in_signed = nodeRep['data_in_signed']

    if 'data_out' in nodeRep:
        data_out = ctxt.lookup(nodeRep['data_out'])
        data_out_signed = data_out._signed
    else:
        data_out_signed = nodeRep['data_out_signed']

    assert data_in_signed is not None
    assert data_out_signed is not None

    convParamsDict = {
        'input_offset': (data_in_signed == 0) * nodeRep['n_levels'] // 2,
        'output_offset': -(data_out_signed == 0) * nodeRep['n_levels'] // 2,
        'stride': ctxt._mangle(ctxt.lookup(f'{name}_stride').name),
        'padding': ctxt._mangle(ctxt.lookup(f'{name}_padding').name),
        'dilation': ctxt._mangle(ctxt.lookup(f'{name}_dilation').name),
        'activation': ctxt._mangle(ctxt.lookup(f'{name}_activation').name),
    }
    ctxt.hoistStruct(convParamsDict, f'{name}_conv_params', 'cmsis_nn_conv_params')
    nodeRep[f'{repName}_conv_params'] = ctxt.lookup(f'{name}_conv_params').name

    convQuantDict = {
        'multiplier': ctxt._mangle(nodeRep['mul']),
        'shift': ctxt._mangle(nodeRep['shift']),
    }
    ctxt.hoistStruct(convQuantDict, f'{name}_quant_params', 'cmsis_nn_per_channel_quant_params')
    nodeRep[f'{repName}_quant_params'] = ctxt.lookup(f'{name}_quant_params').name

    inputDimsDict = {'n': batch, 'h': nodeRep['dim_im_in_x'], 'w': nodeRep['dim_im_in_y'], 'c': nodeRep['ch_im_in']}
    ctxt.hoistStruct(inputDimsDict, f'{name}_input_dims', 'cmsis_nn_dims')
    nodeRep[f'{repName}_input_dims'] = ctxt.lookup(f'{name}_input_dims').name

    filterDimsDict = {
        'n': nodeRep['ch_im_out'],
        'h': nodeRep['dim_kernel_x'],
        'w': nodeRep['dim_kernel_y'],
        'c': nodeRep['ch_im_in']
    }
    ctxt.hoistStruct(filterDimsDict, f'{name}_filter_dims', 'cmsis_nn_dims')
    nodeRep[f'{repName}_filter_dims'] = ctxt.lookup(f'{name}_filter_dims').name

    outputDimsDict = {'n': batch, 'h': nodeRep['dim_im_out_x'], 'w': nodeRep['dim_im_out_y'], 'c': nodeRep['ch_im_out']}
    ctxt.hoistStruct(outputDimsDict, f'{name}_output_dims', 'cmsis_nn_dims')
    nodeRep[f'{repName}_output_dims'] = ctxt.lookup(f'{name}_output_dims').name

    biasDimsDict = {
        'n': 1,
        'h': 1,
        'w': 1,
        'c': nodeRep['ch_im_out'],
    }
    ctxt.hoistStruct(biasDimsDict, f'{name}_bias_dims', 'cmsis_nn_dims')
    nodeRep[f'{repName}_bias_dims'] = ctxt.lookup(f'{name}_bias_dims').name

    return ctxt, nodeRep


def bindFCParams(ctxt, name, mul, shift, data_in, weight, nodeRep, nodeRepPrefix = '', bias = True):

    ctxt = ctxt.copy()

    nodeRep['in_N'] = nodeRep['M']  #np.prod(data_in.shape[0:-1])
    nodeRep['in_C'] = nodeRep['N']  #np.prod(data_in.shape[-1:])
    nodeRep['weight_N'] = nodeRep['N']  #nodeRep['in_C']
    nodeRep['weight_C'] = nodeRep['O']  #np.prod(weight.shape[0:-1])

    #     nodeRep['in_N'] = np.prod(data_in.shape[0:-1])
    #     nodeRep['in_C'] = np.prod(data_in.shape[-1:])
    #     nodeRep['weight_N'] = nodeRep['in_C']
    #     nodeRep['weight_C'] = np.prod(weight.shape[0:-1])

    ctxtDict = {'buf': 0, 'size': 0}

    ctxt.hoistStruct(ctxtDict, f'{name}_ctxt', 'cmsis_nn_context')
    nodeRep[f'{nodeRepPrefix}ctxt'] = f'{name}_ctxt'

    # activation
    activationDict = {'min': -(nodeRep[f'n_levels'] // 2), 'max': (nodeRep[f'n_levels'] // 2) - 1}

    # if nodeRep[f'signed']:
    #     activationDict = {
    #         'min': -(nodeRep[f'n_levels']//2),
    #         'max': (nodeRep[f'n_levels']//2) - 1
    #     }
    # else:
    #     activationDict = {
    #         'min': 0,
    #         'max': (nodeRep[f'n_levels'])-1
    #     }

    ctxt.hoistStruct(activationDict, f'{name}_activation', 'cmsis_nn_activation')

    data_out = ctxt.lookup(nodeRep['data_out'])

    # SCHEREMO: Workaround for MHSA:
    if not hasattr(data_in, '_signed') or not hasattr(data_out, '_signed'):

        fcParamsDict = {
            'input_offset': 0,
            'output_offset': 0,
            'filter_offset': 0,
            'activation': ctxt._mangle(ctxt.lookup(f'{name}_activation').name),
        }

    else:

        fcParamsDict = {
            'input_offset': (data_in._signed == 0) * nodeRep['n_levels'] // 2,
            'output_offset': -(data_out._signed == 0) * nodeRep['n_levels'] // 2,
            'filter_offset': 0,
            'activation': ctxt._mangle(ctxt.lookup(f'{name}_activation').name),
        }

    ctxt.hoistStruct(fcParamsDict, f'{name}_fc_params', 'cmsis_nn_fc_params')
    nodeRep[f'{nodeRepPrefix}fc_params'] = ctxt.lookup(f'{name}_fc_params').name

    gemmQuantDict = {
        'multiplier': mul if isinstance(mul, int) else "*" + ctxt._mangle(mul),
        'shift': shift if isinstance(shift, int) else "*" + ctxt._mangle(shift),
    }

    ctxt.hoistStruct(gemmQuantDict, f'{name}_quant_params', 'cmsis_nn_per_tensor_quant_params')
    nodeRep[f'{nodeRepPrefix}quant_params'] = ctxt.lookup(f'{name}_quant_params').name

    inputDimsDict = {
        'n': nodeRep['in_N'],
        'h': 1,
        'w': 1,
        'c': nodeRep['in_C'],
    }
    ctxt.hoistStruct(inputDimsDict, f'{name}_input_dims', 'cmsis_nn_dims')
    nodeRep[f'{nodeRepPrefix}input_dims'] = ctxt.lookup(f'{name}_input_dims').name

    filterDimsDict = {'n': nodeRep['weight_N'], 'h': 1, 'w': 1, 'c': nodeRep['weight_C']}
    ctxt.hoistStruct(filterDimsDict, f'{name}_filter_dims', 'cmsis_nn_dims')
    nodeRep[f'{nodeRepPrefix}filter_dims'] = ctxt.lookup(f'{name}_filter_dims').name

    outputDimsDict = {'n': nodeRep['in_N'], 'h': 1, 'w': 1, 'c': nodeRep['weight_C']}
    ctxt.hoistStruct(outputDimsDict, f'{name}_output_dims', 'cmsis_nn_dims')
    nodeRep[f'{nodeRepPrefix}output_dims'] = ctxt.lookup(f'{name}_output_dims').name

    biasDimsDict = {
        'n': 1,
        'h': 1,
        'w': 1,
        'c': nodeRep['weight_C'] * bias,
    }
    ctxt.hoistStruct(biasDimsDict, f'{name}_bias_dims', 'cmsis_nn_dims')
    nodeRep[f'{nodeRepPrefix}bias_dims'] = ctxt.lookup(f'{name}_bias_dims').name

    return ctxt, nodeRep
