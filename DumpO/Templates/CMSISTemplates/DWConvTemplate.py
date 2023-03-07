# ----------------------------------------------------------------------
#
# File: DWConvTemplate.py
#
# Last edited: 04.01.2022
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

from typing import Dict, Tuple
from mako.template import Template

from DumpO.DumpOTypes import NodeTemplate, NetworkContext
from .CMSISUtils import bindFCParams

class _Conv2DDW_8_Template(NodeTemplate):
    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()

        # Hoist the structs to the global ctxt

        # First the context
        # https://review.trustedfirmware.org/plugins/gitiles/mirror/ARM-software/CMSIS_5/+/refs/heads/bias_for_conv/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c
        bufferSize = 2*nodeRep['ch_im_in']*nodeRep['dim_kernel_x']*nodeRep['dim_kernel_y']*2 + 4

        data_out_name = nodeRep['data_out']

        ctxtDict = {
            'buf': 0, #f'{node.name}_ctxt_buffer',
            'size': bufferSize
        }

        ctxt.hoistStruct(ctxtDict, f'{data_out_name}_ctxt', 'cmsis_nn_context')
        nodeRep['ctxt'] = f'{data_out_name}_ctxt'

        # Next the conv params
        # stride
        strideDict = {
            'h': nodeRep['stride_x'],
            'w': nodeRep['stride_y']
        }
        ctxt.hoistStruct(strideDict, f'{data_out_name}_stride', 'cmsis_nn_tile')
        # padding
        paddingDict = {
            'h': nodeRep['padding_x'],
            'w': nodeRep['padding_y']
        }
        ctxt.hoistStruct(paddingDict, f'{data_out_name}_padding', 'cmsis_nn_tile')
        # dilation
        dilationDict = {
            'h': nodeRep['dilation_x'],
            'w': nodeRep['dilation_y']
        }
        ctxt.hoistStruct(dilationDict, f'{data_out_name}_dilation', 'cmsis_nn_tile')
        # activation
        activationDict = {
            'min': -(nodeRep['n_levels']//2),
            'max': (nodeRep['n_levels']//2) - 1
        }

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

        ctxt.hoistStruct(activationDict, f'{data_out_name}_activation', 'cmsis_nn_activation')

        data_in = ctxt.lookup(nodeRep['data_in'])
        data_out = ctxt.lookup(nodeRep['data_out'])

        convParamsDict = {
            'input_offset': (data_in._signed == 0) * nodeRep['n_levels']//2,
            'output_offset': -(data_out._signed == 0) * nodeRep['n_levels']//2,
            'ch_mult': 1,
            'stride': ctxt._mangle(ctxt.lookup(f'{data_out_name}_stride').name),
            'padding': ctxt._mangle(ctxt.lookup(f'{data_out_name}_padding').name),
            'dilation': ctxt._mangle(ctxt.lookup(f'{data_out_name}_dilation').name),
            'activation': ctxt._mangle(ctxt.lookup(f'{data_out_name}_activation').name),
        }
        ctxt.hoistStruct(convParamsDict, f'{data_out_name}_dw_conv_params', 'cmsis_nn_dw_conv_params')
        nodeRep[f'dw_conv_params'] = ctxt.lookup(f'{data_out_name}_dw_conv_params').name

        convQuantDict = {
            'multiplier': ctxt._mangle(nodeRep['mul']),
            'shift': ctxt._mangle(nodeRep['shift']),
        }
        ctxt.hoistStruct(convQuantDict, f'{data_out_name}_quant_params', 'cmsis_nn_per_channel_quant_params')
        nodeRep['quant_params'] = ctxt.lookup(f'{data_out_name}_quant_params').name

        inputDimsDict = {
            'n': 1,
            'h': nodeRep['dim_im_in_x'],
            'w': nodeRep['dim_im_in_y'],
            'c': nodeRep['ch_im_in']
        }
        ctxt.hoistStruct(inputDimsDict, f'{data_out_name}_input_dims', 'cmsis_nn_dims')
        nodeRep['input_dims'] = ctxt.lookup(f'{data_out_name}_input_dims').name

        filterDimsDict = {
            'n': 1,
            'h': nodeRep['dim_kernel_x'],
            'w': nodeRep['dim_kernel_y'],
            'c': nodeRep['ch_im_out']
        }
        ctxt.hoistStruct(filterDimsDict, f'{data_out_name}_filter_dims', 'cmsis_nn_dims')
        nodeRep['filter_dims'] = ctxt.lookup(f'{data_out_name}_filter_dims').name

        outputDimsDict = {
            'n': 1,
            'h': nodeRep['dim_im_out_x'],
            'w': nodeRep['dim_im_out_y'],
            'c': nodeRep['ch_im_out']
        }
        ctxt.hoistStruct(outputDimsDict, f'{data_out_name}_output_dims', 'cmsis_nn_dims')
        nodeRep['output_dims'] = ctxt.lookup(f'{data_out_name}_output_dims').name

        biasDimsDict = {
            'n': 1,
            'h': 1,
            'w': 1,
            'c': nodeRep['ch_im_out'],
        }
        ctxt.hoistStruct(biasDimsDict, f'{data_out_name}_bias_dims', 'cmsis_nn_dims')
        nodeRep['bias_dims'] = ctxt.lookup(f'{data_out_name}_bias_dims').name

        return ctxt, nodeRep


conv2D_8_Template = _Conv2DDW_8_Template("""
<%
batchSizeIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchSizeOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>
void* _DumpO__ctxtBuffer_${ctxt} = dumpo_malloc(sizeof(int8_t)*${ctxt}.size);
${ctxt}.buf = _DumpO__ctxtBuffer_${ctxt};
for(int b=0; b<${batch}; b++){
arm_depthwise_conv_wrapper_s8(&${ctxt}, &${dw_conv_params}, &${quant_params}, &${input_dims}, (${data_in} + b*${batchSizeIn}), &${filter_dims}, ${weight}, &${bias_dims}, ${add}, &${output_dims}, (${data_out} + b*${batchSizeOut}));
}
free(_DumpO__ctxtBuffer_${ctxt});
""")
# int8_t* bias = int8_t* dumpo_malloc(sizeof(int8_t) * ${ch_im_in}); \n\
#                free(bias); \


class _Conv1DDW_16_Template(NodeTemplate):
    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()
        # Hoist the structs to the global ctxt

        # First the context
        # https://review.trustedfirmware.org/plugins/gitiles/mirror/ARM-software/CMSIS_5/+/refs/heads/bias_for_conv/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c
        bufferSize = 2*nodeRep['ch_im_in']*nodeRep['dim_kernel_y']*2 + 4

        data_out_name = nodeRep['data_out']

        ctxtDict = {
            'buf': 0, #f'{node.name}_ctxt_buffer',
            'size': bufferSize
        }

        ctxt.hoistStruct(ctxtDict, f'{data_out_name}_ctxt', 'cmsis_nn_context')
        nodeRep['ctxt'] = f'{data_out_name}_ctxt'

        # Next the conv params
        # stride
        strideDict = {
            'h': 1,
            'w': nodeRep['stride_y']
        }
        ctxt.hoistStruct(strideDict, f'{data_out_name}_stride', 'cmsis_nn_tile')
        # padding
        paddingDict = {
            'h': 0,
            'w': nodeRep['padding_y']
        }
        ctxt.hoistStruct(paddingDict, f'{data_out_name}_padding', 'cmsis_nn_tile')
        # dilation
        dilationDict = {
            'h': 1,
            'w': nodeRep['dilation_y']
        }
        ctxt.hoistStruct(dilationDict, f'{data_out_name}_dilation', 'cmsis_nn_tile')
        # activation
        activationDict = {
            'min': -(nodeRep['n_levels']//2),
            'max': (nodeRep['n_levels']//2) - 1
        }

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

        ctxt.hoistStruct(activationDict, f'{data_out_name}_activation', 'cmsis_nn_activation')

        data_in = ctxt.lookup(nodeRep['data_in'])
        data_out = ctxt.lookup(nodeRep['data_out'])

        convParamsDict = {
            'input_offset': (data_in._signed == 0) * nodeRep['n_levels']//2,
            'output_offset': -(data_out._signed == 0) * nodeRep['n_levels']//2,
            'ch_mult': 1,
            'stride': ctxt._mangle(ctxt.lookup(f'{data_out_name}_stride').name),
            'padding': ctxt._mangle(ctxt.lookup(f'{data_out_name}_padding').name),
            'dilation': ctxt._mangle(ctxt.lookup(f'{data_out_name}_dilation').name),
            'activation': ctxt._mangle(ctxt.lookup(f'{data_out_name}_activation').name),
        }
        ctxt.hoistStruct(convParamsDict, f'{data_out_name}_dw_conv_params', 'cmsis_nn_dw_conv_params')
        nodeRep[f'dw_conv_params'] = ctxt.lookup(f'{data_out_name}_dw_conv_params').name

        convQuantDict = {
            'multiplier': ctxt._mangle(nodeRep['mul']),
            'shift': ctxt._mangle(nodeRep['shift']),
        }
        ctxt.hoistStruct(convQuantDict, f'{data_out_name}_quant_params', 'cmsis_nn_per_channel_quant_params')
        nodeRep['quant_params'] = ctxt.lookup(f'{data_out_name}_quant_params').name

        inputDimsDict = {
            'n': data_in.shape[0],
            'h': 1,
            'w': nodeRep['dim_im_in_y'],
            'c': nodeRep['ch_im_in']
        }
        ctxt.hoistStruct(inputDimsDict, f'{data_out_name}_input_dims', 'cmsis_nn_dims')
        nodeRep['input_dims'] = ctxt.lookup(f'{data_out_name}_input_dims').name

        filterDimsDict = {
            'n': 1,
            'h': 1,
            'w': nodeRep['dim_kernel_y'],
            'c': nodeRep['ch_im_out']
        }
        ctxt.hoistStruct(filterDimsDict, f'{data_out_name}_filter_dims', 'cmsis_nn_dims')
        nodeRep['filter_dims'] = ctxt.lookup(f'{data_out_name}_filter_dims').name

        outputDimsDict = {
            'n': data_in.shape[0],
            'h': 1,
            'w': nodeRep['dim_im_out_y'],
            'c': nodeRep['ch_im_out']
        }
        ctxt.hoistStruct(outputDimsDict, f'{data_out_name}_output_dims', 'cmsis_nn_dims')
        nodeRep['output_dims'] = ctxt.lookup(f'{data_out_name}_output_dims').name

        biasDimsDict = {
            'n': 1,
            'h': 1,
            'w': 1,
            'c': nodeRep['ch_im_out'],
        }
        ctxt.hoistStruct(biasDimsDict, f'{data_out_name}_bias_dims', 'cmsis_nn_dims')
        nodeRep['bias_dims'] = ctxt.lookup(f'{data_out_name}_bias_dims').name

        return ctxt, nodeRep


conv1D_16_Template = _Conv1DDW_16_Template("""
<%
batchSizeIn = ch_im_in * dim_im_in_y
batchSizeOut = ch_im_out * dim_im_out_y
%>
void* _DumpO__ctxtBuffer_${ctxt} = dumpo_malloc(sizeof(int16_t)*${ctxt}.size);
${ctxt}.buf = _DumpO__ctxtBuffer_${ctxt};
for(int b=0; b<${batch}; b++){
arm_depthwise_conv_s16(&${ctxt}, &${dw_conv_params}, &${quant_params}, &${input_dims}, (${data_in} + b*${batchSizeIn}), &${filter_dims}, ${weight}, &${bias_dims}, ${add}, &${output_dims}, (${data_out} + b*${batchSizeOut}));
}
free(_DumpO__ctxtBuffer_${ctxt});
""")

conv1D_8_Template = _Conv1DDW_16_Template("""
<%
batchSizeIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchSizeOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>
void* _DumpO__ctxtBuffer_${ctxt} = dumpo_malloc(sizeof(int8_t)*${ctxt}.size);
${ctxt}.buf = _DumpO__ctxtBuffer_${ctxt};
for(int b=0; b<${batch}; b++){
arm_depthwise_conv_wrapper_s8(&${ctxt}, &${dw_conv_params}, &${quant_params}, &${input_dims}, (${data_in} + b*${batchSizeIn}), &${filter_dims}, ${weight}, &${bias_dims}, ${add}, &${output_dims}, (${data_out} + b*${batchSizeOut}));
}
free(_DumpO__ctxtBuffer_${ctxt});
""")
# int8_t* bias = int8_t* dumpo_malloc(sizeof(int8_t) * ${ch_im_in}); \n\
#                free(bias); \
