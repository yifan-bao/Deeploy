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

from DumpO.DumpOTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
MHSAKernel_s8( ${q},${k},${v},${wq_weight},${wq_bias},${wk_weight},${wk_bias},${wv_weight},${wv_bias},${wo_weight},${wo_bias}, ${wq_requant_mul}, ${wq_requant_div},${wk_requant_mul}, ${wk_requant_div},${wv_requant_mul}, ${wv_requant_div},${wo_requant_mul}, ${wo_requant_div}, ${attn_requant_div}, ${attn_requant_mul}, ${dim}, ${dim_head}, ${heads}, ${isoftmaxA}, ${isoftmaxB}, ${isoftmaxC}, ${isoftmaxlog2}, ${n_levels});
""")
