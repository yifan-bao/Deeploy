# ----------------------------------------------------------------------
#
# File: GenericParsers.py
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

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext
from Deeploy.Parsers.BasicParsers import Conv1DParser, Conv2DParser, GEMMParser, MaxPool2DParser


class GenericMaxPool2DParser(MaxPool2DParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([all([pad == 0 for pad in self.parserDict['pads']]), self.parserDict['ceil_mode'] == 0],)

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        return newCtxt, ret


class GenericConv1DParser(Conv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)

        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.parserDict['group'] == 1,
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                self.parserDict['pads'][0] == 0,
                all([coeff == 1 for coeff in self.parserDict['dilations']]),
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class GenericDWConv1DParser(Conv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)

        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                self.parserDict['pads'][0] == 0,
                all([coeff == 1 for coeff in self.parserDict['dilations']]),
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name

            if self.parserDict['group'] == self.parserDict['ch_im_in']:
                return newCtxt, True

        return ctxt, False


class GenericConv2DParser(Conv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)

        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.parserDict['group'] == 1,
                self.parserDict['pads'][0] == self.parserDict['pads'][2],
                self.parserDict['pads'][1] == self.parserDict['pads'][3],
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                self.parserDict['pads'][0] == 0,
                all([coeff == 1 for coeff in self.parserDict['dilations']]),
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
            return newCtxt, True

        return ctxt, False


class GenericDWConv2DParser(Conv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)

        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.parserDict['pads'][0] == self.parserDict['pads'][2],
                self.parserDict['pads'][1] == self.parserDict['pads'][3],
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                self.parserDict['pads'][0] == 0,
                all([coeff == 1 for coeff in self.parserDict['dilations']]),
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
            if self.parserDict['group'] == self.parserDict['ch_im_in']:
                return newCtxt, True

        return ctxt, False


class GenericGEMMParser(GEMMParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if ret:
            # Try to scale A offline if possible, else fail
            if not self.parserDict['alpha'].is_integer():
                nameA = self.parserDict['A']
                if newCtxt.is_global(nameA) and isinstance(newCtxt.lookup(nameA), ConstantBuffer):
                    A = newCtxt.lookup(nameA)
                    npA = np.asarray(A.values).reshape(A.shape)
                    newA = npA * self.parserDict['beta']
                    newCtxt.globalObjects[nameA].values = newA
                    self.parserDict['alpha'] = 1.0
                else:
                    return newCtxt, False
            # Try to scale B offline if possible, else fail
            if not self.parserDict['beta'].is_integer():
                nameB = self.parserDict['B']
                if newCtxt.is_global(nameB) and isinstance(newCtxt.lookup(nameB), ConstantBuffer):
                    B = newCtxt.lookup(nameB)
                    npB = np.asarray(B.values).reshape(B.shape)
                    newB = npB * self.parserDict['beta']
                    newCtxt.globalObjects[nameB].values = newB
                    self.parserDict['beta'] = 1.0
                else:
                    return newCtxt, False

            self.parserDict['alpha'] = int(self.parserDict['alpha'])
            self.parserDict['beta'] = int(self.parserDict['beta'])
            return newCtxt, True

        return ctxt, False
