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

from DumpO.DumpOTypes import NetworkContext
import numpy as np

def bindFCParams(ctxt, name, mul, shift, data_in, weight, nodeRep,  nodeRepPrefix ='', bias=True):

    ctxt = ctxt.copy()
    
    nodeRep['in_N'] = np.prod(data_in.shape[0:-1])
    nodeRep['in_C'] = np.prod(data_in.shape[-1:])
    nodeRep['weight_N'] = nodeRep['in_C']
    nodeRep['weight_C'] = np.prod(weight.shape[0:-1])
    
    ctxtDict = {
        'buf': 0,
        'size': 0
    }

    ctxt.hoistStruct(ctxtDict, f'{name}_ctxt', 'cmsis_nn_context')
    nodeRep[f'{nodeRepPrefix}ctxt'] = f'{name}_ctxt'

    # activation
    if nodeRep[f'signed']:
        activationDict = {
            'min': -(nodeRep[f'n_levels']//2),
            'max': (nodeRep[f'n_levels']//2) - 1
        }
    else:
        activationDict = {
            'min': 0,
            'max': (nodeRep[f'n_levels'])-1
        }
    ctxt.hoistStruct(activationDict, f'{name}_activation', 'cmsis_nn_activation')

    fcParamsDict = {
        'input_offset': 0,
        'filter_offset': 0,
        'output_offset': 0,
        'activation': ctxt._mangle(ctxt.lookup(f'{name}_activation').name),
    }
    ctxt.hoistStruct(fcParamsDict, f'{name}_fc_params', 'cmsis_nn_fc_params')
    nodeRep[f'{nodeRepPrefix}fc_params'] = ctxt.lookup(f'{name}_fc_params').name

    gemmQuantDict = {
        'multiplier': mul,
        'shift': shift,
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

    filterDimsDict = {
        'n': nodeRep['weight_N'],
        'h': 1,
        'w': 1,
        'c': nodeRep['weight_C']
    }
    ctxt.hoistStruct(filterDimsDict, f'{name}_filter_dims', 'cmsis_nn_dims')
    nodeRep[f'{nodeRepPrefix}filter_dims'] = ctxt.lookup(f'{name}_filter_dims').name

    outputDimsDict = {
        'n': nodeRep['in_N'],
        'h': 1,
        'w': 1,
        'c': nodeRep['weight_C']
    }
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
