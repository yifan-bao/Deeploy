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

from typing import Dict, List, Tuple

from Deeploy.DataTypes.CMSISDataTypes import CMSISDataTypes
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate


class _Conv2DDW_8_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt = ctxt.copy()
        nodeList = []

        # Hoist the structs to the global ctxt

        data_out_name = nodeRep['data_out']

        ctxtDict = {'buf': nodeRep['ctxtBuffer'], 'size': nodeRep['ctxtBufferSize']}

        nodeList += [ctxt.hoistStruct(ctxtDict, f'{data_out_name}_ctxt', CMSISDataTypes.cmsis_nn_context)]
        nodeRep['ctxt'] = f'{data_out_name}_ctxt'

        # Next the conv params
        # stride
        strideDict = {'h': nodeRep['stride_x'], 'w': nodeRep['stride_y']}
        nodeList += [ctxt.hoistStruct(strideDict, f'{data_out_name}_stride', CMSISDataTypes.cmsis_nn_tile)]
        # padding
        paddingDict = {'h': nodeRep['padding_x'], 'w': nodeRep['padding_y']}
        nodeList += [ctxt.hoistStruct(paddingDict, f'{data_out_name}_padding', CMSISDataTypes.cmsis_nn_tile)]
        # dilation
        dilationDict = {'h': nodeRep['dilation_x'], 'w': nodeRep['dilation_y']}
        nodeList += [ctxt.hoistStruct(dilationDict, f'{data_out_name}_dilation', CMSISDataTypes.cmsis_nn_tile)]
        # activation
        activationDict = {'min': -(nodeRep['n_levels'] // 2), 'max': (nodeRep['n_levels'] // 2) - 1}

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

        nodeList += [
            ctxt.hoistStruct(activationDict, f'{data_out_name}_activation', CMSISDataTypes.cmsis_nn_activation)
        ]

        data_in = ctxt.lookup(nodeRep['data_in'])
        data_out = ctxt.lookup(nodeRep['data_out'])

        convParamsDict = {
            'input_offset': (data_in._signed == 0) * nodeRep['n_levels'] // 2,
            'output_offset': -(data_out._signed == 0) * nodeRep['n_levels'] // 2,
            'ch_mult': 1,
            'stride': strideDict,
            'padding': paddingDict,
            'dilation': dilationDict,
            'activation': activationDict,
        }
        nodeList += [
            ctxt.hoistStruct(convParamsDict, f'{data_out_name}_dw_conv_params', CMSISDataTypes.cmsis_nn_dw_conv_params)
        ]
        nodeRep['dw_conv_params'] = ctxt.lookup(f'{data_out_name}_dw_conv_params').name

        convQuantDict = {
            'multiplier': nodeRep['mul'],
            'shift': nodeRep['shift'],
        }
        nodeList += [
            ctxt.hoistStruct(convQuantDict, f'{data_out_name}_quant_params',
                             CMSISDataTypes.cmsis_nn_per_channel_quant_params)
        ]
        nodeRep['quant_params'] = ctxt.lookup(f'{data_out_name}_quant_params').name

        inputDimsDict = {'n': 1, 'h': nodeRep['dim_im_in_x'], 'w': nodeRep['dim_im_in_y'], 'c': nodeRep['ch_im_in']}
        nodeList += [ctxt.hoistStruct(inputDimsDict, f'{data_out_name}_input_dims', CMSISDataTypes.cmsis_nn_dims)]
        nodeRep['input_dims'] = ctxt.lookup(f'{data_out_name}_input_dims').name

        filterDimsDict = {'n': 1, 'h': nodeRep['dim_kernel_x'], 'w': nodeRep['dim_kernel_y'], 'c': nodeRep['ch_im_out']}
        nodeList += [ctxt.hoistStruct(filterDimsDict, f'{data_out_name}_filter_dims', CMSISDataTypes.cmsis_nn_dims)]
        nodeRep['filter_dims'] = ctxt.lookup(f'{data_out_name}_filter_dims').name

        outputDimsDict = {'n': 1, 'h': nodeRep['dim_im_out_x'], 'w': nodeRep['dim_im_out_y'], 'c': nodeRep['ch_im_out']}
        nodeList += [ctxt.hoistStruct(outputDimsDict, f'{data_out_name}_output_dims', CMSISDataTypes.cmsis_nn_dims)]
        nodeRep['output_dims'] = ctxt.lookup(f'{data_out_name}_output_dims').name

        biasDimsDict = {
            'n': 1,
            'h': 1,
            'w': 1,
            'c': nodeRep['ch_im_out'],
        }
        nodeList += [ctxt.hoistStruct(biasDimsDict, f'{data_out_name}_bias_dims', CMSISDataTypes.cmsis_nn_dims)]
        nodeRep['bias_dims'] = ctxt.lookup(f'{data_out_name}_bias_dims').name

        return ctxt, nodeRep, nodeList

    def hoistTransientBuffers(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        # SCHEREMO: Hoist transient buffer
        # https://review.trustedfirmware.org/plugins/gitiles/mirror/ARM-software/CMSIS_5/+/refs/heads/bias_for_conv/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c

        bufferSize = 2 * nodeRep['ch_im_in'] * nodeRep['dim_kernel_x'] * nodeRep['dim_kernel_y'] * 2 + 4

        name = nodeRep['nodeName'] + "_buffer"
        ctxt.hoistTransientBuffer(name, bufferSize)
        nodeRep['ctxtBuffer'] = name
        nodeRep['ctxtBufferSize'] = bufferSize
        return ctxt, nodeRep, [name]


conv2D_8_Template = _Conv2DDW_8_Template("""
<%
batchSizeIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchSizeOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>
for(int b=0; b<${batch}; b++){
arm_depthwise_conv_wrapper_s8(&${ctxt}, &${dw_conv_params}, &${quant_params}, &${input_dims}, (${data_in} + b*${batchSizeIn}), &${filter_dims}, ${weight}, &${bias_dims}, ${add}, &${output_dims}, (${data_out} + b*${batchSizeOut}));
}
""")
# int8_t* bias = int8_t* deeploy_malloc(sizeof(int8_t) * ${ch_im_in}); \n\
#                free(bias); \


class _Conv1DDW_16_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt = ctxt.copy()
        nameList = []
        # Hoist the structs to the global ctxt

        # First the context
        # https://review.trustedfirmware.org/plugins/gitiles/mirror/ARM-software/CMSIS_5/+/refs/heads/bias_for_conv/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c

        data_out_name = nodeRep['data_out']

        ctxtDict = {
            'buf': nodeRep['ctxtBuffer'],  #f'{node.name}_ctxt_buffer',
            'size': nodeRep['ctxtBufferSize']
        }

        nameList += [ctxt.hoistStruct(ctxtDict, f'{data_out_name}_ctxt', CMSISDataTypes.cmsis_nn_context)]
        nodeRep['ctxt'] = f'{data_out_name}_ctxt'

        # Next the conv params
        # stride
        strideDict = {'h': 1, 'w': nodeRep['stride_y']}
        # padding
        paddingDict = {'h': 0, 'w': nodeRep['padding_y']}
        # dilation
        dilationDict = {'h': 1, 'w': nodeRep['dilation_y']}
        # activation
        activationDict = {'min': -(nodeRep['n_levels'] // 2), 'max': (nodeRep['n_levels'] // 2) - 1}

        data_in = ctxt.lookup(nodeRep['data_in'])
        data_out = ctxt.lookup(nodeRep['data_out'])

        convParamsDict = {
            'input_offset': (data_in._signed == 0) * nodeRep['n_levels'] // 2,
            'output_offset': -(data_out._signed == 0) * nodeRep['n_levels'] // 2,
            'ch_mult': 1,
            'stride': strideDict,
            'padding': paddingDict,
            'dilation': dilationDict,
            'activation': activationDict,
        }
        nameList += [
            ctxt.hoistStruct(convParamsDict, f'{data_out_name}_dw_conv_params', CMSISDataTypes.cmsis_nn_dw_conv_params)
        ]
        nodeRep['dw_conv_params'] = ctxt.lookup(f'{data_out_name}_dw_conv_params').name

        convQuantDict = {
            'multiplier': nodeRep['mul'],
            'shift': nodeRep['shift'],
        }
        nameList += [
            ctxt.hoistStruct(convQuantDict, f'{data_out_name}_quant_params',
                             CMSISDataTypes.cmsis_nn_per_channel_quant_params)
        ]
        nodeRep['quant_params'] = ctxt.lookup(f'{data_out_name}_quant_params').name

        inputDimsDict = {'n': data_in.shape[0], 'h': 1, 'w': nodeRep['dim_im_in_y'], 'c': nodeRep['ch_im_in']}
        nameList += [ctxt.hoistStruct(inputDimsDict, f'{data_out_name}_input_dims', CMSISDataTypes.cmsis_nn_dims)]
        nodeRep['input_dims'] = ctxt.lookup(f'{data_out_name}_input_dims').name

        filterDimsDict = {'n': 1, 'h': 1, 'w': nodeRep['dim_kernel_y'], 'c': nodeRep['ch_im_out']}
        nameList += [ctxt.hoistStruct(filterDimsDict, f'{data_out_name}_filter_dims', CMSISDataTypes.cmsis_nn_dims)]
        nodeRep['filter_dims'] = ctxt.lookup(f'{data_out_name}_filter_dims').name

        outputDimsDict = {'n': data_in.shape[0], 'h': 1, 'w': nodeRep['dim_im_out_y'], 'c': nodeRep['ch_im_out']}
        nameList += [ctxt.hoistStruct(outputDimsDict, f'{data_out_name}_output_dims', CMSISDataTypes.cmsis_nn_dims)]
        nodeRep['output_dims'] = ctxt.lookup(f'{data_out_name}_output_dims').name

        biasDimsDict = {
            'n': 1,
            'h': 1,
            'w': 1,
            'c': nodeRep['ch_im_out'],
        }
        nameList += [ctxt.hoistStruct(biasDimsDict, f'{data_out_name}_bias_dims', CMSISDataTypes.cmsis_nn_dims)]
        nodeRep['bias_dims'] = ctxt.lookup(f'{data_out_name}_bias_dims').name

        return ctxt, nodeRep, nameList

    def hoistTransientBuffers(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        # SCHEREMO: Hoist transient buffer
        # https://review.trustedfirmware.org/plugins/gitiles/mirror/ARM-software/CMSIS_5/+/refs/heads/bias_for_conv/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c

        size = 2 * nodeRep['ch_im_in'] * nodeRep['dim_kernel_y'] * 2
        name = nodeRep['nodeName'] + "_buffer"
        ctxt.hoistTransientBuffer(name, size)
        nodeRep['ctxtBuffer'] = name
        nodeRep['ctxtBufferSize'] = size
        return ctxt, nodeRep, [name]


conv1D_16_Template = _Conv1DDW_16_Template("""
<%
batchSizeIn = ch_im_in * dim_im_in_y
batchSizeOut = ch_im_out * dim_im_out_y
%>
for(int b=0; b<${batch}; b++){
arm_depthwise_conv_s16(&${ctxt}, &${dw_conv_params}, &${quant_params}, &${input_dims}, (${data_in} + b*${batchSizeIn}), &${filter_dims}, ${weight}, &${bias_dims}, ${add}, &${output_dims}, (${data_out} + b*${batchSizeOut}));
}
""")

conv1D_8_Template = _Conv1DDW_16_Template("""
<%
batchSizeIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchSizeOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>
for(int b=0; b<${batch}; b++){
arm_depthwise_conv_wrapper_s8(&${ctxt}, &${dw_conv_params}, &${quant_params}, &${input_dims}, (${data_in} + b*${batchSizeIn}), &${filter_dims}, ${weight}, &${bias_dims}, ${add}, &${output_dims}, (${data_out} + b*${batchSizeOut}));
}
""")
