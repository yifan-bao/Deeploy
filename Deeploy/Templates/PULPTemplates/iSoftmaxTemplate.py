# ----------------------------------------------------------------------
#
# File: iSoftmaxTemplate.py
#
# Last edited: 13.11.2023
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


class PULPiSoftmaxTemplate(NodeTemplate):

    @staticmethod
    def computeTransientBuffersSize(ctxt: NetworkContext, nodeRep: Dict) -> List[Tuple[str, Union[int, IntVar]]]:

        lastDimBuffer_dim = 8 * 4 * nodeRep['lastDimLength']
        lastDimBuffer_name = nodeRep['nodeName'] + "_lastDimBuffer"
        return [(lastDimBuffer_name, lastDimBuffer_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        lastDimBuffer_name, lastDimBuffer_dim = PULPiSoftmaxTemplate.computeTransientBuffersSize(ctxt, nodeRep)[0]
        ctxt.hoistTransientBuffer(lastDimBuffer_name, lastDimBuffer_dim)

        nodeRep['lastDimBuffer'] = lastDimBuffer_name
        nodeRep['lastDimBufferSize'] = lastDimBuffer_dim
        return ctxt, nodeRep, [lastDimBuffer_name]

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt = ctxt.copy()

        signedI = ctxt.lookup(nodeRep['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(nodeRep['data_out'])._type.referencedType.typeMin < 0

        nodeRep['input_signed'] = signedI
        nodeRep['output_signed'] = signedO

        return ctxt, nodeRep, []


referenceTemplate = PULPiSoftmaxTemplate("""
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
%>
PULPSoftmax${signatureString}(${data_in}, ${data_out}, ${lastDimBuffer}, ${size}, ${lastDimLength}, ${coeffB}, ${coeffC}, ${log2});
""")
