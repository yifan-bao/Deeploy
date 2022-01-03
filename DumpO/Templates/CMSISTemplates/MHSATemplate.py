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
// W_Q * Q
// arm_fully_connected_s8(&${wq_ctxt}, &${wq_fc_params}, &${wq_quant_params}, &${wq_input_dims}, ${q}, &${wq_filter_dims}, ${wq_weight}, &${wq_bias_dims}, ${wq_bias}, &${wq_output_dims}, ${data_out});
""")
