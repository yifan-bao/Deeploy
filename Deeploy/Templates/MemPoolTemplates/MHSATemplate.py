# ----------------------------------------------------------------------
#
# File: MHSATemplate.py
#
# Last edited: 08.02.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
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

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes as DataTypes
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTemplate

# ITA Configuration
_SPLIT = 4
_ITA_S = 64
_ITA_E = 64
_ITA_P = 64


class _MHSATemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        nameList = []

        nodeName = nodeRep['nodeName']

        data_out = ctxt.lookup(nodeRep['data_out'])

        wq_bias = ctxt.lookup(nodeRep['wq_bias'])
        wk_bias = ctxt.lookup(nodeRep['wk_bias'])
        wv_bias = ctxt.lookup(nodeRep['wv_bias'])
        wo_bias = ctxt.lookup(nodeRep['wo_bias'])
        wq_weight = ctxt.lookup(nodeRep['wq_weight'])
        wk_weight = ctxt.lookup(nodeRep['wk_weight'])
        wv_weight = ctxt.lookup(nodeRep['wv_weight'])
        wo_weight = ctxt.lookup(nodeRep['wo_weight'])
        q = ctxt.lookup(nodeRep['q'])
        k = ctxt.lookup(nodeRep['k'])
        # v = ctxt.lookup(nodeRep['v'])

        # Disable buffers
        wq_bias._deploy = False
        wk_bias._deploy = False
        wv_bias._deploy = False
        wo_bias._deploy = False
        wq_weight._deploy = False
        wk_weight._deploy = False
        wv_weight._deploy = False
        wo_weight._deploy = False

        nodeRep['S'] = nodeRep['dim']
        nodeRep['P'] = nodeRep['dim_head']

        N = nodeRep['heads']
        S = nodeRep['S']
        E = nodeRep['E']
        P = nodeRep['P']

        PAD_S = _ITA_S - S
        PAD_E = _ITA_E - E
        PAD_P = _ITA_P - P

        # Extract values and transform them to layout required by ITA
        wq_bias_ita = wq_bias.values.reshape(N, S, P)
        wq_bias_ita = np.pad(wq_bias_ita, ((0, 0), (0, PAD_S), (0, PAD_P)))
        wq_bias_ita = np.reshape(np.split(wq_bias_ita, _SPLIT, axis = 2), (N, _ITA_S, _ITA_P))

        wk_bias_ita = wk_bias.values.reshape(N, S, P)
        wk_bias_ita = np.pad(wk_bias_ita, ((0, 0), (0, PAD_S), (0, PAD_P)))
        wk_bias_ita = np.reshape(np.split(wk_bias_ita, _SPLIT, axis = 2), (N, _ITA_S, _ITA_P))

        wv_bias_ita = wv_bias.values.reshape(N, S, P)
        wv_bias_ita = np.pad(wv_bias_ita, ((0, 0), (0, PAD_S), (0, PAD_P)))
        wv_bias_ita = np.reshape(np.split(np.reshape(np.transpose(wv_bias_ita), (N, _ITA_P, _ITA_S)), _SPLIT, axis = 2),
                                 (N, _ITA_P, _ITA_S))

        wo_bias_ita = wo_bias.values.reshape(N, S, E)
        wo_bias_ita = np.pad(wo_bias_ita, ((0, 0), (0, PAD_S), (0, PAD_E)))
        wo_bias_ita = np.reshape(np.split(wo_bias_ita, _SPLIT, axis = 2), (N, _ITA_S, _ITA_E))

        wq_weight_ita = wq_weight.values.reshape(N, E, P)
        wq_weight_ita = np.pad(wq_weight_ita, ((0, 0), (0, PAD_E), (0, PAD_P)))
        wq_weight_ita = np.concatenate(
            np.split(np.concatenate([np.transpose(wq_weight_ita[i]) for i in range(N)]), _SPLIT, axis = 1))
        wq_weight_ita = np.reshape(wq_weight_ita, (N, _ITA_P, _ITA_E))

        wk_weight_ita = wk_weight.values.reshape(N, E, P)
        wk_weight_ita = np.pad(wk_weight_ita, ((0, 0), (0, PAD_E), (0, PAD_P)))
        wk_weight_ita = np.concatenate([np.transpose(wk_weight_ita[i]) for i in range(N)])
        wk_weight_ita = np.reshape(wk_weight_ita, (N, _ITA_P, _ITA_E))

        wv_weight_ita = wv_weight.values.reshape(N, E, P)
        wv_weight_ita = np.pad(wv_weight_ita, ((0, 0), (0, PAD_E), (0, PAD_P)))
        wv_weight_ita = np.concatenate([np.transpose(wv_weight_ita[i]) for i in range(N)])
        wv_weight_ita = np.reshape(wv_weight_ita, (N, _ITA_P, _ITA_E))

        wo_weight_ita = wo_weight.values.reshape(N, P, E)
        wo_weight_ita = np.pad(wo_weight_ita, ((0, 0), (0, PAD_P), (0, PAD_E)))
        wo_weight_ita = np.concatenate([np.transpose(wo_weight_ita[i]) for i in range(N)])
        wo_weight_ita = np.reshape(wo_weight_ita, (N, _ITA_E, _ITA_P))

        # Create dummy array for key and values
        q_ita = np.zeros((1, _ITA_S, _ITA_E))
        k_ita = np.zeros((1, _ITA_S, _ITA_E))

        # Fuse all inputs together and store in L2
        data = np.stack([
            wo_weight_ita,
            wv_weight_ita,
            wk_weight_ita,
            q_ita,
            k_ita,
            wq_weight_ita,
            wo_bias_ita,
            wv_bias_ita,
            wk_bias_ita,
            wq_bias_ita,
        ])

        data_in = ctxt.ConstantBuffer(name = f'{nodeName}_input', shape = data.shape, values = data)
        ctxt.add(data_in, 'global')
        data_in._type = Pointer(DataTypes.int8_t)
        nodeRep['data_in'] = data_in.name
        nameList += [data_in.name]

        def unsignedToSigned(n, byte_count):
            return int.from_bytes(n.to_bytes(byte_count, 'little', signed = False), 'little', signed = True)

        requant_mult_data = np.array([
            unsignedToSigned(nodeRep['wq_requant_mul'], 1),
            unsignedToSigned(nodeRep['wk_requant_mul'], 1),
            unsignedToSigned(nodeRep['preattn_requant_mul'], 1),
            unsignedToSigned(nodeRep['wv_requant_mul'], 1),
            unsignedToSigned(nodeRep['postattn_requant_mul'], 1),
            unsignedToSigned(nodeRep['wo_requant_mul'], 1),
            0,
            0,
        ])
        requant_mult = ctxt.ConstantBuffer(name = f'{nodeName}_requant_mult',
                                           shape = requant_mult_data.shape,
                                           values = requant_mult_data)
        ctxt.add(requant_mult, 'global')
        requant_mult._type = Pointer(DataTypes.int8_t)
        nodeRep['requant_mult'] = requant_mult.name
        nameList += [requant_mult.name]

        requant_shift_data = np.array([
            unsignedToSigned(int(np.log2(nodeRep['wq_requant_div'])), 1),
            unsignedToSigned(int(np.log2(nodeRep['wk_requant_div'])), 1),
            unsignedToSigned(int(np.log2(nodeRep['preattn_requant_div'])), 1),
            unsignedToSigned(int(np.log2(nodeRep['wv_requant_div'])), 1),
            unsignedToSigned(int(np.log2(nodeRep['postattn_requant_div'])), 1),
            unsignedToSigned(int(np.log2(nodeRep['wo_requant_div'])), 1),
            0,
            0,
        ])
        requant_shift = ctxt.ConstantBuffer(name = f'{nodeName}_requant_shift',
                                            shape = requant_shift_data.shape,
                                            values = requant_shift_data)
        ctxt.add(requant_shift, 'global')
        requant_shift._type = Pointer(DataTypes.int8_t)
        nodeRep['requant_shift'] = requant_shift.name
        nameList += [requant_shift.name]

        requant_add_data = np.array([
            nodeRep['wq_requant_add'],
            nodeRep['wk_requant_add'],
            nodeRep['preattn_requant_add'],
            nodeRep['wv_requant_add'],
            nodeRep['postattn_requant_add'],
            nodeRep['wo_requant_add'],
            0,
            0,
        ])
        requant_add = ctxt.ConstantBuffer(name = f'{nodeName}_requant_add',
                                          shape = requant_add_data.shape,
                                          values = requant_add_data)
        ctxt.add(requant_add, 'global')
        requant_add._type = Pointer(DataTypes.int8_t)
        nodeRep['requant_add'] = requant_add.name
        nameList += [requant_add.name]

        nodeRep['q_offset'] = (q._signed == 0) * int(q.nLevels // 2)
        nodeRep['k_offset'] = (k._signed == 0) * int(k.nLevels // 2)
        nodeRep['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            nodeRep['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        # import IPython; IPython.embed()
        return ctxt, nodeRep, nameList


MemPoolParallelTemplate = _MHSATemplate("""


// ITA MHSA (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);
MHSA_s8_ITA(
    ${q}, ${k}, ${data_in},
    ${S}, ${E},
    ${requant_mult},
    ${requant_shift},
    ${requant_add},
    ${data_out},
    ${q_offset}, ${k_offset}, ${output_offset},
    core_id,
    numThreads
);
mempool_barrier(numThreads);
""")
