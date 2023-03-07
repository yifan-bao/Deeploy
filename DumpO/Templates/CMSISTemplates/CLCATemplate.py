# ----------------------------------------------------------------------
#
# File: CLCATemplate.py
#
# Last edited: 26.08.2022
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
from typing import Dict, Tuple
from mako.template import Template
import numpy as np
import copy

from DumpO.DumpOTypes import NodeTemplate, NetworkContext, VariableBuffer
from DumpO.DataTypes.BasicDataTypes import DataTypes
from .CMSISUtils import bindFCParams, bindConvParams

from DumpO.Templates.BasicTemplates import ReduceMeanTemplate, RequantShiftTemplate, TransposeTemplate
from . import ConvTemplate, GEMMTemplate

class _CLCATemplate(NodeTemplate):
    def __init__(self, templateStr):
        super().__init__(templateStr)

        reduceMeanTemplate = ReduceMeanTemplate.referenceTemplate
        convTemplate = ConvTemplate.cmsis2D_8_Template
        RQSMMTemplate = GEMMTemplate.Linear_8_Template
        rqsTemplate = RequantShiftTemplate.referenceTemplate
        transposeTemplate = TransposeTemplate.referenceTemplate

        self.subTemplates["reduceMean"] = (reduceMeanTemplate, self.reduceMeanGenerator)

        self.subTemplates["convQ"] = (convTemplate, self.convQGenerator)
        self.subTemplates["convV"] = (convTemplate, self.convVGenerator)
        self.subTemplates["convO"] = (convTemplate, self.convOGenerator)

        self.subTemplates["RQK"] = (rqsTemplate, self.rqsKGenerator)
        self.subTemplates["RQDelta"] = (rqsTemplate, self.rqsDeltaGenerator)

        self.subTemplates["PreTransposeV"] = (transposeTemplate, self.transQGenerator)
        self.subTemplates["PostTransposeV"] = (transposeTemplate, self.transQGenerator)

        self.subTemplates["TransposeQ"] = (transposeTemplate, self.transQGenerator)
        self.subTemplates["TransposeO"] = (transposeTemplate, self.transOGenerator)

        self.subTemplates["MMA"] = (RQSMMTemplate, self.MMAGenerator)

    def MMAGenerator(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:

        ctxt, nodeRep = copy.deepcopy(ctxt), copy.deepcopy(nodeRep)

        K = ctxt.lookup(f"K", _id=nodeRep['id'])
        V = ctxt.lookup(f"V", _id=nodeRep['id'])
        A = ctxt.lookup(f"A", _id=nodeRep['id'])

        nodeRep['A'] = K.name
        nodeRep['B'] = V.name
        nodeRep['data_out'] = A.name
        nodeRep['C'] = nodeRep['preattn_requant_add']

        nodeRep['size'] = np.prod(K.shape) // nodeRep['heads']
        nodeRep['alpha'] = 1.0
        nodeRep['beta'] = 1.0
        nodeRep['transA'] = 0
        nodeRep['transB'] = 1

        nodeRep['mul'] = nodeRep['preattn_requant_mul']
        nodeRep['shift'] = nodeRep['preattn_requant_div']
        nodeRep['channels'] = 1

        return ctxt, nodeRep


    def transQGenerator(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        Q = ctxt.lookup(f"Q", _id=nodeRep['id'])
        QT = ctxt.lookup(f"QT", _id=nodeRep['id'])

        nodeRep['data_in'] = Q.name
        nodeRep['data_in_type'] = Q._type
        nodeRep['data_in_shape'] = [1, nodeRep['heads'], nodeRep['dim_head'], nodeRep['q_shape'][-1]]
        nodeRep['data_out'] = QT.name
        nodeRep['data_out_type'] = QT._type
        nodeRep['data_out_shape'] = [1, nodeRep['heads'], nodeRep['q_shape'][-1], nodeRep['dim_head']]
        nodeRep['perm'] = [0,1,3,2]

        return ctxt, nodeRep

    def transOGenerator(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        O = ctxt.lookup(f"O", _id=nodeRep['id'])
        OT = ctxt.lookup(f"OT", _id=nodeRep['id'])

        nodeRep['data_in'] = O.name
        nodeRep['data_in_type'] = O._type
        nodeRep['data_in_shape'] = [1, nodeRep['heads'], nodeRep['q_shape'][-1], nodeRep['dim_head']]
        nodeRep['data_out'] = OT.name
        nodeRep['data_out_type'] = OT._type
        nodeRep['data_out_shape'] = [1, nodeRep['heads'], nodeRep['dim_head'], nodeRep['q_shape'][-1]]
        nodeRep['perm'] = [0,1,3,2]

        return ctxt, nodeRep

    def rqsDeltaGenerator(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:

        K = ctxt.lookup(f"K", _id=nodeRep['id'])
        RK = ctxt.lookup(f"RK", _id=nodeRep['id'])

        nodeRep['data_in'] = K.name
        nodeRep['data_in_type'] = K._type
        nodeRep['data_out'] = RK.name
        nodeRep['size'] = nodeRep['input_size_KV']
        nodeRep['mul'] = nodeRep['kdiv_requant_mul']
        nodeRep['add'] = nodeRep['kdiv_requant_add']
        nodeRep['log2D'] = nodeRep['kdiv_requant_div']
        nodeRep['channels'] = 1

        return ctxt, nodeRep


    def rqsKGenerator(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:

        V = ctxt.lookup(f"V", _id=nodeRep['id'])
        K = ctxt.lookup(f"K", _id=nodeRep['id'])

        nodeRep['data_in'] = V.name
        nodeRep['data_in_type'] = V._type
        nodeRep['data_out'] = K.name
        nodeRep['size'] = nodeRep['input_size_KV']
        nodeRep['mul'] = nodeRep['wk_requant_mul']
        nodeRep['add'] = nodeRep['wk_requant_add']
        nodeRep['log2D'] = nodeRep['wk_requant_div']
        nodeRep['channels'] = 1

        return ctxt, nodeRep

    def convOGenerator(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        convCtxt, convRep = self.convGenerator(ctxt, nodeRep)

        OT = ctxt.lookup(f"OT", _id=nodeRep['id'])

        convRep['data_in'] = OT.name
        convRep['weight'] = convRep['wo_weight']
        convRep['add'] = convRep['wo_bias']
        convRep['dim_im_in_x'] = convRep['q_shape'][2]
        convRep['dim_im_out_x'] = convRep['q_shape'][2]
        convRep['mul'] = convRep['wo_requant_mul']
        convRep['shift'] = convRep['wo_requant_div']
        convRep['ch_im_in'] = convRep['dim_head'] * convRep['heads']
        convRep['ch_im_out'] = convRep['out_dim']

        return convCtxt, convRep

    def convVGenerator(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        convCtxt, convRep = self.convGenerator(ctxt, nodeRep)

        V = ctxt.lookup(f"V", _id=nodeRep['id'])

        convRep['data_in'] = convRep['k']
        convRep['weight'] = convRep['wk_weight']
        convRep['add'] = convRep['wk_bias']
        convRep['data_out'] = V.name
        convRep['dim_im_in_x'] = convRep['kv_shape'][2]
        convRep['dim_im_out_x'] = convRep['kv_shape'][2]
        convRep['mul'] = convRep['wv_requant_mul']
        convRep['shift'] = convRep['wv_requant_div']
        convRep['ch_im_in'] = convRep['kv_shape'][1]
        convRep['ch_im_out'] = convRep['dim_head'] * convRep['heads']

        return convCtxt, convRep

    def convQGenerator(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        convCtxt, convRep = self.convGenerator(ctxt, nodeRep)

        Q = ctxt.lookup(f"Q", _id=nodeRep['id'])

        convRep['data_in'] = convRep['q']
        convRep['weight'] = convRep['wq_weight']
        convRep['add'] = convRep['wq_bias']
        convRep['data_out'] = Q.name
        convRep['dim_im_in_x'] = convRep['q_shape'][2]
        convRep['dim_im_out_x'] = convRep['q_shape'][2]
        convRep['mul'] = convRep['wq_requant_mul']
        convRep['shift'] = convRep['wq_requant_div']
        convRep['ch_im_in'] = convRep['q_shape'][1]
        convRep['ch_im_out'] = convRep['dim_head'] * convRep['heads']

        return convCtxt, convRep

    def convGenerator(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:

        convRep = copy.deepcopy(nodeRep)
        convCtxt = copy.deepcopy(ctxt)

        # Same for all convs
        convRep['dilation_x'] = 1
        convRep['dilation_y'] = 1
        convRep['padding_x'] = 0
        convRep['padding_y'] = 0
        convRep['stride_x'] = 1
        convRep['stride_y'] = 1
        convRep['dim_kernel_x'] = 1
        convRep['dim_kernel_y'] = 1
        convRep['dim_im_in_y'] = 1
        convRep['dim_im_out_y'] = 1

        return convCtxt, convRep

    def reduceMeanGenerator(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        K = ctxt.lookup(f"K", _id=nodeRep['id'])
        E = ctxt.lookup(f"E", _id=nodeRep['id'])

        nodeRep['data_in'] = K.name
        nodeRep['data_in_type'] = K._type
        nodeRep['data_out'] = E.name
        nodeRep['data_out_type'] = E._type

        nodeRep['data_in_shape'] = [1, nodeRep['heads'], nodeRep['dim_head'], nodeRep['kv_shape'][-1]]
        nodeRep['data_out_shape'] = [1, nodeRep['heads'], nodeRep['dim_head']]
        nodeRep['size'] = nodeRep['heads']*nodeRep['dim_head']*nodeRep['kv_shape'][2]
        nodeRep['axisLength'] = nodeRep['kv_shape'][-1]
        nodeRep['axes'] = [3]

        return ctxt, nodeRep

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:

        Q = VariableBuffer(f"Q", [nodeRep['heads'], nodeRep['dim_head'], nodeRep['q_shape'][-1]], 256)
        Q._type = DataTypes.int8_t
        Q._signed = False
        Q._deploy = False

        QT = VariableBuffer(f"QT", [nodeRep['heads'], nodeRep['q_shape'][-1], nodeRep['dim_head']], 256)
        QT._type = DataTypes.int8_t
        QT._signed = False
        QT._deploy = False

        K = VariableBuffer(f"K", [nodeRep['heads'], nodeRep['dim_head'], nodeRep['kv_shape'][-1]], 256)
        K._type = DataTypes.int8_t
        K._signed = False
        K._deploy = False

        RK = VariableBuffer(f"RK", [nodeRep['heads'], nodeRep['dim_head'], nodeRep['kv_shape'][-1]], 256)
        RK._type = DataTypes.int8_t
        RK._signed = True
        RK._deploy = False

        V = VariableBuffer(f"V", [nodeRep['heads'], nodeRep['dim_head'], nodeRep['kv_shape'][-1]], 256)
        V._type = DataTypes.int8_t
        V._signed = True
        V._deploy = False

        VT = VariableBuffer(f"VT", [nodeRep['heads'], nodeRep['kv_shape'][-1], nodeRep['dim_head']], 256)
        VT._type = DataTypes.int8_t
        VT._signed = True
        VT._deploy = False

        E = VariableBuffer(f"E", [nodeRep['heads'],nodeRep['dim_head'],1], 256)
        E._type = DataTypes.int8_t
        E._signed = False
        E._deploy = False

        A = VariableBuffer(f"A", [nodeRep['heads'],nodeRep['dim_head'],nodeRep['dim_head']], 256)
        A._type = DataTypes.int8_t
        A._signed = False
        A._deploy = False

        AA = VariableBuffer(f"AA", [nodeRep['heads'],nodeRep['dim_head'],nodeRep['q_shape'][-1]], 2**32)
        AA._type = DataTypes.int32_t
        AA._signed = True
        AA._deploy = False

        B = VariableBuffer(f"B", [nodeRep['heads'], nodeRep['q_shape'][-1],1], 2**32)
        B._type = DataTypes.int32_t
        B._signed = True
        B._deploy = False

        O = VariableBuffer(f"O", [1, nodeRep['heads'], nodeRep['q_shape'][-1], nodeRep['dim_head']], 256)
        O._type = DataTypes.int8_t
        O._signed = True
        O._deploy = False

        OT = VariableBuffer(f"OT", [1, nodeRep['heads'] * nodeRep['dim_head'], nodeRep['q_shape'][-1],], 256)
        OT._type = DataTypes.int8_t
        OT._signed = True
        OT._deploy = False

        nodeRep['id'] = nodeRep['data_out']

        ctxt.add(Q, 'internal', _id=nodeRep['id'])
        nodeRep['Q'] = Q.name
        ctxt.add(QT, 'internal', _id=nodeRep['id'])
        nodeRep['QT'] = QT.name
        ctxt.add(K, 'internal', _id=nodeRep['id'])
        nodeRep['K'] = K.name
        ctxt.add(RK, 'internal', _id=nodeRep['id'])
        nodeRep['RK'] = RK.name
        ctxt.add(V, 'internal', _id=nodeRep['id'])
        nodeRep['V'] = V.name
        ctxt.add(VT, 'internal', _id=nodeRep['id'])
        nodeRep['VT'] = VT.name
        ctxt.add(E, 'internal', _id=nodeRep['id'])
        nodeRep['E'] = E.name
        ctxt.add(A, 'internal', _id=nodeRep['id'])
        nodeRep['A'] = A.name
        ctxt.add(AA, 'internal', _id=nodeRep['id'])
        nodeRep['AA'] = AA.name
        ctxt.add(B, 'internal', _id=nodeRep['id'])
        nodeRep['B'] = B.name
        ctxt.add(O, 'internal', _id=nodeRep['id'])
        nodeRep['O'] = O.name
        ctxt.add(OT, 'internal', _id=nodeRep['id'])
        nodeRep['OT'] = OT.name

        return ctxt, nodeRep


referenceTemplate = _CLCATemplate("""
<%
sizeV = heads*dim_head*kv_shape[2]
%>
// alloc V
int8_t* ${V} = dumpo_malloc(${sizeV});
am_util_stdio_printf("Alloc V at 0x\%x\\n", ${V});
// V <- k * WKV
${RENDER_convV}
//am_util_stdio_printf("Comp V\\n");
// alloc K
<%
sizeK = heads*dim_head*kv_shape[2]
%>
int8_t* ${K} = dumpo_malloc(${sizeK});
//am_util_stdio_printf("Alloc K at 0x\%x\\n", ${K});
// K <- RQ(V)
${RENDER_RQK}
// alloc A
<%
sizeA = heads*dim_head*dim_head
%>
// RK <- RQ(K)
int8_t* ${RK} = dumpo_malloc(${sizeK});
//am_util_stdio_printf("Alloc RK at 0x\%x\\n", ${RK});
${RENDER_RQDelta}
int8_t* ${A} = (int8_t*)dumpo_malloc(sizeof(int8_t) * ${sizeA});
//am_util_stdio_printf("Alloc A at 0x\%x\\n", ${A});
// A <- RQS(KT x V)
// Headwise MMA
int8_t* OG_${A} = ${A};
int8_t* OG_${RK} = ${RK};
int8_t* OG_${V} = ${V};
for (int head=0; head<${heads}; head++){
${RENDER_MMA}
${A} += ${dim_head}*${dim_head};
${RK} += ${kv_shape[-1]}*${dim_head};
${V} += ${kv_shape[-1]}*${dim_head};
}
${A} = OG_${A};
${RK} = OG_${RK};
${V} = OG_${V};
//am_util_stdio_printf("Comp A\\n");
free(${RK});
//am_util_stdio_printf("Free RK at 0x\%x\\n", ${RK});
// alloc E
<%
sizeE = heads*dim_head
%>
int8_t* ${E} = dumpo_malloc(${sizeE});
//am_util_stdio_printf("Alloc E at 0x\%x\\n", ${E});
// E <- mean(K)
${RENDER_reduceMean}
//am_util_stdio_printf("Comp E\\n");
// free K
free(${K});
//am_util_stdio_printf("Free K at 0x\%x\\n", ${K});
// free V
free(${V});
//am_util_stdio_printf("Free V at 0x\%x\\n", ${V});
// alloc Q
<%
sizeQ = heads*dim_head*q_shape[2]
%>
int8_t* ${Q} = dumpo_malloc(${sizeQ});
//am_util_stdio_printf("Alloc Q at 0x\%x\\n", ${Q});
// Q <- q * WQ
${RENDER_convQ}
// alloc QT
int8_t* ${QT} = dumpo_malloc(${sizeQ});
//am_util_stdio_printf("Alloc QT at 0x\%x\\n", ${QT});
// transpose Q -> QT
${RENDER_TransposeQ}
// free Q
free(${Q});
//am_util_stdio_printf("Free Q at 0x\%x\\n", ${Q});
// alloc AA
<%
sizeAA = heads*dim_head*dim_head
%>
int32_t* ${AA} = (int32_t*)dumpo_malloc((sizeof(int32_t)) * ${sizeAA});
//am_util_stdio_printf("Alloc AA at 0x\%x\\n", ${AA});
// AA <- Q x A
int32_t* OG_${AA} = ${AA};
int8_t* OG_${QT} = ${QT};
for (int head=0; head<${heads}; head++){
MatMul_s8_s8_s32(${QT}, ${A}, ${AA}, ${q_shape[-1]}, ${dim_head}, ${dim_head});
${QT} += ${q_shape[-1]} * ${dim_head};
${A} += ${dim_head} * ${dim_head};
${AA} += ${q_shape[-1]} * ${dim_head};
}
${AA} = OG_${AA};
${A} = OG_${A};
${QT} = OG_${QT};
//am_util_stdio_printf("Comp AA\\n");
// free A
free(${A});
//am_util_stdio_printf("Free A at 0x\%x\\n", ${A});
  //am_util_delay_ms(5);
// alloc B
<%
sizeB = heads*dim_head
%>
int32_t* ${B} = (int32_t*)dumpo_malloc((sizeof(int32_t)) * ${sizeB});
//am_util_stdio_printf("Alloc B at 0x\%x\\n", ${B});
  //am_util_delay_ms(5);
// B <- Q x E
int8_t* OG_${E} = ${E};
int32_t* OG_${B} = ${B};
for (int head=0; head<${heads}; head++){
MatMul_s8_s8_s32(${QT}, ${E}, ${B}, ${q_shape[-1]}, ${dim_head}, 1);
${QT} += ${q_shape[-1]} * ${dim_head};
${E} += ${dim_head};
${B} += ${q_shape[-1]};
}
${E} = OG_${E};
${B} = OG_${B};
${QT} = OG_${QT};
//am_util_stdio_printf("QT: 0x%x \\n", ${QT});
//am_util_stdio_printf("Comp B \\n");
  //am_util_delay_ms(5);
// free o
free(${QT});
//am_util_stdio_printf("Free QT at 0x\%x\\n", ${QT});
  //am_util_delay_ms(5);
// free E
free(${E});
//am_util_stdio_printf("Free E at 0x\%x\\n", ${E});
  //am_util_delay_ms(5);
// alloc _o
<%
sizeO = sizeQ
%>
int8_t* ${O} = dumpo_malloc(${sizeO});
//am_util_stdio_printf("Alloc O at 0x\%x\\n", ${O});
RQDivKernel_s32_s8(${AA}, ${B}, ${sizeAA}, ${sizeB}, ${O}, ${Delta}, ${eps}, ${eta}, *${postattn_requant_mul}, *${postattn_requant_add}, *${postattn_requant_div});
//am_util_stdio_printf("Comp O\\n");
  //am_util_delay_ms(5);
// free AA
free(${AA});
//am_util_stdio_printf("Free AA at 0x\%x\\n", ${AA});
  //am_util_delay_ms(5);
// free B
free(${B});
//am_util_stdio_printf("Free B at 0x\%x\\n", ${B});
  //am_util_delay_ms(5);

// alloc OT
int8_t* ${OT} = dumpo_malloc(${sizeQ});
//am_util_stdio_printf("Alloc OT at 0x\%x\\n", ${OT});
  //am_util_delay_ms(5);
// transpose O -> OT
${RENDER_TransposeO}
// free O
free(${O});
//am_util_stdio_printf("Free O at 0x\%x\\n", ${O});
  //am_util_delay_ms(5);
// data_out <- o * WO
${RENDER_convO}
//am_util_stdio_printf("Comp Output \\n");
  //am_util_delay_ms(5);
// free o
free(${OT});
am_util_stdio_printf("Free OT at 0x\%x\\n", ${OT});
am_util_delay_ms(15);

""")
