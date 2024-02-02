# ----------------------------------------------------------------------
#
# File: MemPoolParsers.py
#
# Last edited: 13.11.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
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

from typing import Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Parsers.BasicParsers import MHSAParser


class MemPoolMHSAParser(MHSAParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)
        if wellFormed:
            wellFormed = all([
                self.parserDict['dim_head'] <= 64,  # Projection Size
                self.parserDict['dim'] <= 64,  # Sequence Length
                self.parserDict['n_levels'] == 256,
                'preattn_requant_add' in node.attrs,
                'postattn_requant_add' in node.attrs,
                'wo_requant_add' in node.attrs,
                'wq_requant_add' in node.attrs,
                'wk_requant_add' in node.attrs,
                'wv_requant_add' in node.attrs,
            ])

        if wellFormed:
            self.parserDict['preattn_requant_add'] = int(node.attrs['preattn_requant_add'])
            self.parserDict['postattn_requant_add'] = int(node.attrs['postattn_requant_add'])
            self.parserDict['wo_requant_add'] = int(node.attrs['wo_requant_add'])
            self.parserDict['wq_requant_add'] = int(node.attrs['wq_requant_add'])
            self.parserDict['wk_requant_add'] = int(node.attrs['wk_requant_add'])
            self.parserDict['wv_requant_add'] = int(node.attrs['wv_requant_add'])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        K = ctxt.lookup(self.parserDict['k'])
        V = ctxt.lookup(self.parserDict['v'])

        self.parserDict['E'] = int(K.shape[-1])  # Embedding size

        wellFormed = all([
            self.parserDict['E'] <= 64,
            K.name == V.name  # K and V has to be the same for ITA
        ])

        return newCtxt, wellFormed


# Test parser for currently limited implementation of ITA in MemPool
class DebugMemPoolMHSAParser(MemPoolMHSAParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)
        if wellFormed:
            wellFormed = all([
                self.parserDict['heads'] == 1,
            ])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        return newCtxt, ret
