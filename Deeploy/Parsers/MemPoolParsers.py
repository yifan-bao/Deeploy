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

import numpy as np
import math
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import *
from Deeploy.Parsers.BasicParsers import *
from Deeploy.Bindings.BasicBindings import DataTypes


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
            ])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

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
                self.parserDict['wq_requant_div'] == 2**14,
                self.parserDict['wk_requant_div'] == 2**14,
                self.parserDict['wv_requant_div'] == 2**14,
                self.parserDict['preattn_requant_div'] == 2**14,
                self.parserDict['postattn_requant_div'] == 2**14,
                self.parserDict['wo_requant_div'] == 2**14,
                self.parserDict['wq_requant_mul'] == 52,
                self.parserDict['wk_requant_mul'] == 66,
                self.parserDict['wv_requant_mul'] == 54,
                self.parserDict['preattn_requant_mul'] == 19,
                self.parserDict['postattn_requant_mul'] == 76,
                self.parserDict['wo_requant_mul'] == 6,
            ])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        return newCtxt, ret