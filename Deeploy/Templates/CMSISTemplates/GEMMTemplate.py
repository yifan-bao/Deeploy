# ----------------------------------------------------------------------
#
# File: GEMMTemplate.py
#
# Last edited: 20.12.2021
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

from typing import Dict, Tuple
from mako.template import Template

from Deeploy.DeeployTypes import NodeTemplate, NetworkContext
from .CMSISUtils import bindFCParams


class _GEMM_8_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        inputs = ['A', 'B', 'add']

        # Hoist the structs to the global ctxt
        data_in = ctxt.lookup(nodeRep['A'])
        data_out = ctxt.lookup(nodeRep['data_out'])
        weight = ctxt.lookup(nodeRep['B'])

        ctxt, nodeRep = bindFCParams(ctxt, nodeRep['data_out'], nodeRep['mul'], nodeRep['shift'], data_in, weight,
                                     nodeRep)

        return ctxt, nodeRep


Linear_8_Template = _GEMM_8_Template("""
// GEMM
int8_t* ref_${data_out}_${A} = ${A};
int8_t* ref_${data_out}_${B} = ${B};
int8_t* ref_${data_out}_${data_out} = ${data_out};
for(int i=0;i<${batch};i++){
    arm_fully_connected_s8(&${ctxt}, &${fc_params}, &${quant_params}, &${input_dims}, ref_${data_out}_${A}, &${filter_dims}, ref_${data_out}_${B}, &${bias_dims}, ${C}, &${output_dims}, ref_${data_out}_${data_out});
    ref_${data_out}_${A} += ${M} * ${N};
    ref_${data_out}_${B} += ${N} * ${O};
    ref_${data_out}_${data_out} += ${M} * ${O};
}
""")


class _GEMM_16_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        inputs = ['A', 'B', 'add']

        # Hoist the structs to the global ctxt
        data_in = ctxt.lookup(nodeRep['A'])
        data_out = ctxt.lookup(nodeRep['data_out'])
        weight = ctxt.lookup(nodeRep['B'])

        ctxt, nodeRep = bindFCParams(ctxt, nodeRep['data_out'], nodeRep['mul'], nodeRep['shift'], data_in, weight,
                                     nodeRep)

        return ctxt, nodeRep


Linear_16_Template = _GEMM_16_Template("""
// FC
arm_fully_connected_s16(&${ctxt}, &${fc_params}, &${quant_params}, &${input_dims}, ${A}, &${filter_dims}, ${B}, &${bias_dims}, ${C}, &${output_dims}, ${data_out});
""")
