# ----------------------------------------------------------------------
#
# File: MaxPool2DTemplate.py
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

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate


class PULPMaxPoolTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        signedI = ctxt.lookup(nodeRep['data_in'])._type.referencedType.typeMin < 0
        nodeRep['input_signed'] = signedI
        return ctxt, nodeRep, []


PULPMaxPool2D_8_Template = PULPMaxPoolTemplate("""
// PULP NN MaxPool 2D
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
pulp_nn_maxpool${signatureString}(${data_in}, ${data_out}, ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in}, ${dim_im_out_x}, ${dim_im_out_y}, ${dim_kernel_x}, ${dim_kernel_y}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}, ${stride_x}, ${stride_y});
""")
