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

conv2DBasicTemplate = NodeTemplate("\
static int8_t* bias = int8_t* malloc(sizeof(int8_t) * ${ch_im_in}); \n\
static int8_t bufferB[1]; \n\
static int16_t bufferA[2*${ch_im_in}*${dim_kernel_x}*${dim_kernel_y}]; \n\
arm_convolve_HWC_q7_basic_nonsquare(${data_in}, ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in}, ${weight}, ${ch_im_out}, ${dim_kernel_x}, ${dim_kernel_y}, ${padding_x}, ${padding_y}, ${stride_x}, ${stride_y}, bias, ${bias_shift}, ${out_shift}, ${data_out}, ${dim_im_out_x}, ${dim_im_out_y}, bufferA, bufferB); \n\
free(bufferA); \n\
free(bufferB); \n\
free(bias); \
")
