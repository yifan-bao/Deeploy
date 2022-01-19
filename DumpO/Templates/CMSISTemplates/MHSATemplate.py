# ----------------------------------------------------------------------
#
# File: MHSATemplate.py
#
# Last edited: 01.01.2022        
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

import mako
from typing import Dict
from mako.template import Template
import numpy as np

from DumpO.DumpOTypes import NodeTemplate, NetworkContext
from .CMSISUtils import bindFCParams

class _MHSATemplate(NodeTemplate):
    def __init__(self, templateStr):
        self.template = Template(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> (NetworkContext, Dict):
        inputs = ['q', 'k', 'v', 'wq_weight', 'wq_bias','wk_weight', 'wk_bias', 'wv_weight', 'wv_bias', 'wo_weight', 'wo_bias']
        
        s = ctxt.lookup(nodeRep['q']).shape[1]

        data_in = ctxt.lookup(nodeRep['q'])
        bias = ctxt.lookup(nodeRep['wq_bias'])
        weight = ctxt.lookup(nodeRep['wq_weight'])

        # Q FC layer:
        ctxt, nodeRep = bindFCParams(ctxt, nodeRep['data_out']+"_wq", nodeRep['wq_requant_mul'], nodeRep['wq_requant_shift'], data_in, weight, nodeRep, "wq_", bias = (np.prod(bias.shape) > 1))

        data_in = ctxt.lookup(nodeRep['k'])
        bias = ctxt.lookup(nodeRep['wk_bias'])
        weight = ctxt.lookup(nodeRep['wk_weight'])

        ctxt, nodeRep = bindFCParams(ctxt, nodeRep['data_out']+"_wk", nodeRep['wk_requant_mul'], nodeRep['wk_requant_shift'], data_in, weight, nodeRep, "wk_", bias = (np.prod(bias.shape) > 1))

        data_in = ctxt.lookup(nodeRep['v'])
        bias = ctxt.lookup(nodeRep['wv_bias'])
        weight = ctxt.lookup(nodeRep['wv_weight'])

        ctxt, nodeRep = bindFCParams(ctxt, nodeRep['data_out']+"_wv", nodeRep['wv_requant_mul'], nodeRep['wv_requant_shift'], data_in, weight, nodeRep, "wv_", bias = (np.prod(bias.shape) > 1))


        new_shape = (1, data_in.shape[1], nodeRep['heads']*nodeRep['dim_head'])
        data_in = ctxt.VariableBuffer(name = 'data_in', shape = new_shape, nLevels=nodeRep['n_levels'])
        data_in._signed = True
        bias = ctxt.lookup(nodeRep['wo_bias'])
        weight = ctxt.lookup(nodeRep['wo_weight'])

        ctxt, nodeRep = bindFCParams(ctxt, nodeRep['data_out']+"_wo", nodeRep['wo_requant_mul'], nodeRep['wo_requant_shift'], data_in, weight, nodeRep, "wo_", bias = (np.prod(bias.shape) > 1))
        #*nodeRep['heads']
        new_shape = (s,nodeRep['dim_head'])
        data_in = ctxt.VariableBuffer(name = 'data_in', shape = new_shape, nLevels=nodeRep['n_levels'])
        data_in._signed = True
        # K
        weight = np.ones((s,nodeRep['dim_head']))
        
        ctxt, nodeRep = bindFCParams(ctxt, nodeRep['data_out']+"_preattn", nodeRep['preattn_requant_mul'], nodeRep['preattn_requant_shift'], data_in, weight, nodeRep, "preattn_", bias=False)
        #*nodeRep['heads']
        new_shape = (s,s)
        data_in = ctxt.VariableBuffer(name = 'data_in', shape = new_shape, nLevels=nodeRep['n_levels'])
        data_in._signed = False
        # K
        weight = np.ones((nodeRep['dim_head'],s))
        
        ctxt, nodeRep = bindFCParams(ctxt, nodeRep['data_out']+"_postattn", nodeRep['postattn_requant_mul'], nodeRep['postattn_requant_shift'], data_in, weight, nodeRep, "postattn_", bias=False)

        return ctxt, nodeRep
        

referenceTemplate = _MHSATemplate("""
do{
<%
    sequenceLength = q_shape[1]
%>
// W_Q * q -> Q
int32_t* _null = malloc(${dim}*${heads});
memset(_null, 0, ${dim}*${heads});

int8_t* wq_buffer = malloc(${wq_output_dims}.n * ${wq_output_dims}.c);
arm_fully_connected_s8(&${wq_ctxt}, &${wq_fc_params}, &${wq_quant_params}, &${wq_input_dims}, ${q}, &${wq_filter_dims}, ${wq_weight}, &${wq_bias_dims}, ${wq_bias}, &${wq_output_dims}, wq_buffer);
<%
dim1 = sequenceLength
dim2 = heads
dim3 = dim_head
%>
// Q: NSHD-> NHSD
int8_t* wq_buffer_transposed = malloc(${wq_output_dims}.n * ${wq_output_dims}.c);
for(int k=0;k<${dim2};k++){
for (int i=0;i<${dim1};i++){
for(int j=0;j<${dim3};j++){
wq_buffer_transposed[k*${dim3}*${dim1} + i*${dim3} + j] = wq_buffer[i*${dim2}*${dim3} + k*${dim3} + j];
}
}
}
free(wq_buffer);

// W_K * k -> K
int8_t* wk_buffer = malloc(${wk_output_dims}.n * ${wk_output_dims}.c);
arm_fully_connected_s8(&${wk_ctxt}, &${wk_fc_params}, &${wk_quant_params}, &${wk_input_dims}, ${k}, &${wk_filter_dims}, ${wk_weight}
, &${wk_bias_dims}, ${wk_bias}, &${wk_output_dims}, wk_buffer);


// K: NSHD-> NHSD
int8_t* wk_buffer_transposed = malloc(${wk_output_dims}.n * ${wk_output_dims}.c);
for(int k=0;k<${dim2};k++){
for (int i=0;i<${dim1};i++){
for(int j=0;j<${dim3};j++){
wk_buffer_transposed[k*${dim3}*${dim1} + i*${dim3} + j] = wk_buffer[i*${dim2}*${dim3} + k*${dim3} + j];
}
}
}
free(wk_buffer);


// ATTN Matrix -> Q*KT

// QKT -> NHSS
int8_t* preattn_buffer = malloc(${heads} * ${sequenceLength} * ${sequenceLength});
for(int i=0; i<${heads}; i++){
arm_fully_connected_s8(&${preattn_ctxt}, &${preattn_fc_params}, &${preattn_quant_params}, &${preattn_input_dims}, &wq_buffer_transposed[i * ${preattn_input_dims}.n * ${preattn_input_dims}.c], &${preattn_filter_dims}, &wk_buffer_transposed[i * ${preattn_filter_dims}.n * ${preattn_filter_dims}.c], &${preattn_bias_dims}, _null, &${preattn_output_dims}, &preattn_buffer[i*${preattn_output_dims}.c*${preattn_output_dims}.n]);
}
free(wq_buffer_transposed);
free(wk_buffer_transposed);
int8_t* postattn_buffer = malloc(${heads} * ${sequenceLength} * ${sequenceLength});
SoftmaxKernel_s8(preattn_buffer, postattn_buffer, ${heads} * ${sequenceLength} * ${sequenceLength}, ${sequenceLength}, ${isoftmaxA}, ${isoftmaxB}, ${isoftmaxC}, ${isoftmaxlog2}, ${n_levels});
free(preattn_buffer);

int8_t* wv_buffer = malloc(${wv_output_dims}.n * ${wv_output_dims}.c);
arm_fully_connected_s8(&${wv_ctxt}, &${wv_fc_params}, &${wv_quant_params}, &${wv_input_dims}, ${v}, &${wv_filter_dims}, ${wv_weight}, &${wv_bias_dims}, ${wv_bias}, &${wv_output_dims}, wv_buffer);

<%
dim1 = sequenceLength
dim2 = heads
dim3 = dim_head
%>
// NSHD-> NHDS
// 
int8_t* wv_buffer_transposed = malloc(${wv_output_dims}.n * ${wv_output_dims}.c);
for(int k=0;k<${dim2};k++){
for(int j=0;j<${dim3};j++){
for (int i=0;i<${dim1};i++){
wv_buffer_transposed[k*${dim3}*${dim1} + j*${dim1} + i] = wv_buffer[i*${dim2}*${dim3} + k*${dim3} + j];
}
}
}
free(wv_buffer);

int8_t* out_buffer = malloc(${heads} * ${sequenceLength} * ${dim_head});

for(int i=0; i<${heads}; i++){
arm_fully_connected_s8(&${postattn_ctxt}, &${postattn_fc_params}, &${postattn_quant_params}, 
&${postattn_input_dims}, &postattn_buffer[i*${postattn_input_dims}.n*${postattn_input_dims}.c], 
&${postattn_filter_dims}, &wv_buffer_transposed[i*${postattn_filter_dims}.n * ${postattn_filter_dims}.c], 
&${postattn_bias_dims}, _null, 
&${postattn_output_dims}, &out_buffer[i*${postattn_output_dims}.n*${postattn_output_dims}.c]);  
}
free(postattn_buffer);
free(wv_buffer_transposed);
<%
dim1 = heads
dim2 = sequenceLength
dim3 = dim_head
%>

// NHSD-> NSHD
int8_t* out_buffer_transposed = malloc(${heads} * ${sequenceLength} * ${dim_head});
for(int k=0;k<${dim2};k++){
for (int i=0;i<${dim1};i++){
for(int j=0;j<${dim3};j++){
out_buffer_transposed[k*${dim3}*${dim1} + i*${dim3} + j] = out_buffer[i*${dim2}*${dim3} + k*${dim3} + j];
}
}
}
free(out_buffer);
free(_null);

arm_fully_connected_s8(&${wo_ctxt}, &${wo_fc_params}, &${wo_quant_params}, &${wo_input_dims}, out_buffer_transposed, &${wo_filter_dims}, ${wo_weight}, &${wo_bias_dims}, ${wo_bias}, &${wo_output_dims}, ${data_out});

free(out_buffer_transposed);

}while(0);
""")




