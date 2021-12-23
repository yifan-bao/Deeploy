# ----------------------------------------------------------------------
#
# File: ConvTemplate.py
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

from DumpO.DumpOTypes import NodeTemplate

conv2DTemplate = NodeTemplate("\
void* _DumpO__ctxtBuffer_${ctxt} = malloc(sizeof(int8_t)*${ctxt}->size);\n\
${ctxt}_UL.buf = _DumpO__ctxtBuffer_${ctxt};\n\
arm_convolve_s8(${ctxt}, ${conv_params}, ${quant_params}, ${input_dims}, ${data_in}, ${filter_dims}, ${weight}, ${bias_dims}, ${add}, ${output_dims}, ${data_out}); \n\
free(_DumpO__ctxtBuffer_${ctxt});\
")
# int8_t* bias = int8_t* malloc(sizeof(int8_t) * ${ch_im_in}); \n\
#                free(bias); \
