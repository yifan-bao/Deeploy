# ----------------------------------------------------------------------
#
# File: MaxPool2DTemplate.py
#
# Last edited: 27.12.2021        
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

from typing import Dict
from mako.template import Template

from DumpO.DumpOTypes import NodeTemplate, NetworkContext
from .CMSISUtils import bindFCParams

class _MaxPool2DTemplate(NodeTemplate):
    def __init__(self, templateStr):
        self.template = Template(templateStr)

    def hoistStatic(self, ctxt: NetworkContext, nodeRep: Dict) -> (NetworkContext, Dict):
        ctxt = ctxt.copy()

        data_out_name = nodeRep['data_out']

        ctxtDict = {
            'buf': 0, #f'{node.name}_ctxt_buffer',  
            'size': 0
        }

        ctxt.hoistStruct(ctxtDict, f'{data_out_name}_ctxt', 'cmsis_nn_context')
        nodeRep['ctxt'] = f'{data_out_name}_ctxt'

        strideDict = {
            'w': nodeRep['stride_x'],
            'h': nodeRep['stride_y']
        }
        ctxt.hoistStruct(strideDict, f'{data_out_name}_stride', 'cmsis_nn_tile')
        # padding
        paddingDict = {
            'w': nodeRep['padding_x'],
            'h': nodeRep['padding_y']
        }
        ctxt.hoistStruct(paddingDict, f'{data_out_name}_padding', 'cmsis_nn_tile')

        # SCHEREMO: Fix this at some point...
        activationDict = {
                'min': -2**7,
                'max': 2**7-1
        }
        ctxt.hoistStruct(activationDict, f'{data_out_name}_activation', 'cmsis_nn_activation')

        convParamsDict = {
            'stride': ctxt._mangle(ctxt.lookup(f'{data_out_name}_stride').name),
            'padding': ctxt._mangle(ctxt.lookup(f'{data_out_name}_padding').name),
            'activation': ctxt._mangle(ctxt.lookup(f'{data_out_name}_activation').name),
        }
        ctxt.hoistStruct(convParamsDict, f'{data_out_name}_pool_params', 'cmsis_nn_pool_params')
        nodeRep[f'pool_params'] = ctxt.lookup(f'{data_out_name}_pool_params').name

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
            'c': 1
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

        return ctxt, nodeRep

cmsisTemplate = _MaxPool2DTemplate("""
arm_max_pool_s8(&${ctxt}, &${pool_params}, &${input_dims}, ${data_in}, &${filter_dims}, &${output_dims}, ${data_out});
""")
