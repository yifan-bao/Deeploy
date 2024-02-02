# ----------------------------------------------------------------------
#
# File: ConvTemplate.py
#
# Last edited: 10.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate


class PULP2DConvTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(nodeRep['weight'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(nodeRep['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(nodeRep['data_out'])._type.referencedType.typeMin < 0
        nodeRep['weight_signed'] = signedW
        nodeRep['input_signed'] = signedI
        nodeRep['output_signed'] = signedO

        return ctxt, nodeRep, []

    @staticmethod
    def computeTransientBuffersSize(ctxt: NetworkContext, nodeRep: Dict) -> List[Tuple[str, Union[int, IntVar]]]:
        im2col_dim = 2 * 8 * (nodeRep['ch_im_in'] * nodeRep['dim_kernel_x'] * nodeRep['dim_kernel_y'])
        im2col_name = nodeRep['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP2DConvTemplate.computeTransientBuffersSize(ctxt, nodeRep)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        nodeRep['ctxtBuffer'] = im2col_name
        nodeRep['ctxtBufferSize'] = im2col_dim
        return ctxt, nodeRep, [im2col_name]


class PULP2DDWConvTemplate(PULP2DConvTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(nodeRep['weight'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(nodeRep['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(nodeRep['data_out'])._type.referencedType.typeMin < 0
        nodeRep['weight_signed'] = signedW
        nodeRep['input_signed'] = signedI
        nodeRep['output_signed'] = signedO

        return ctxt, nodeRep, []


class PULP1DConvTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(nodeRep['weight'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(nodeRep['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(nodeRep['data_out'])._type.referencedType.typeMin < 0
        nodeRep['weight_signed'] = signedW
        nodeRep['input_signed'] = signedI
        nodeRep['output_signed'] = signedO

        nodeRep['pad_x_left'] = nodeRep['pads'][0]
        nodeRep['pad_x_right'] = nodeRep['pads'][1]
        nodeRep['stride_x'] = nodeRep['strides'][0]

        return ctxt, nodeRep, []

    @staticmethod
    def computeTransientBuffersSize(ctxt: NetworkContext, nodeRep: Dict) -> List[Tuple[str, Union[int, IntVar]]]:
        im2col_dim = 8 * (1 * (1 + nodeRep['pads'][0]) + nodeRep['dim_kernel_y'])
        im2col_name = nodeRep['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP1DConvTemplate.computeTransientBuffersSize(ctxt, nodeRep)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)
        nodeRep['ctxtBuffer'] = im2col_name
        nodeRep['ctxtBufferSize'] = im2col_dim
        return ctxt, nodeRep, [im2col_name]


class PULP1DDWConvTemplate(PULP1DConvTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


PULPConv2D_8_Template = PULP2DConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>

<%
operatorString = ''
if dim_kernel_x == 1 and dim_kernel_y == 1:
    operatorString = 'pointwise'
else:
    operatorString = 'conv'
operatorString = 'conv'
%>

pulp_nn_${operatorString}${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, ${mul}, ${add}, 1, ${log2D}, ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in}, ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out}, ${dim_kernel_x}, ${dim_kernel_y}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}, ${stride_x}, ${stride_y}, 1, 1);
""")

PULPDWConv2D_8_Template = PULP2DDWConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
pulp_nn_depthwise${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, NULL, ${mul}, ${add}, 1, ${log2D}, ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in}, ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out}, ${dim_kernel_x}, ${dim_kernel_y}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}, ${stride_x}, ${stride_y}, 1, 1);
""")

PULPConv1D_8_Template = PULP1DConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>

<%
operatorString = ''
if dim_kernel_x == 1 and dim_kernel_y == 1:
    operatorString = 'pointwise'
else:
    operatorString = 'conv'
operatorString = 'conv'
%>

pulp_nn_${operatorString}${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, ${mul}, ${add}, 1, ${log2D}, 1, ${dim_im_in_y}, ${ch_im_in}, 1, ${dim_im_out_y}, ${ch_im_out}, 1, ${dim_kernel_y}, ${padding_y_top}, ${padding_y_bottom}, 0, 0, 1, ${stride_y}, 1, 1);
""")

PULPDWConv1D_8_Template = PULP1DDWConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
pulp_nn_depthwise${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, NULL, ${mul}, ${add}, 1, ${log2D}, 1, ${dim_im_in_y}, ${ch_im_in}, 1, ${dim_im_out_y}, ${ch_im_out}, 1, ${dim_kernel_y}, ${padding_y_top}, ${padding_y_bottom}, 0, 0, 1, ${stride_y}, 1, 1);
""")
