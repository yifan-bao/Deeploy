# ----------------------------------------------------------------------
#
# File: CMSISParsers.py
#
# Last edited: 17.12.2021
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

import math
from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext
from Deeploy.Parsers.BasicParsers import CLCAParser, GEMMParser, LinearAttentionParser, MaxPool2DParser, MHSAParser, \
    RQSConv1DParser, RQSConv2DParser, RQSParserInterface


class CMSISMaxPool2DParser(MaxPool2DParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                self.parserDict['pads'][0] == 0,
                self.parserDict['pads'][1] == 0,
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        return super().parseNodeCtxt(ctxt, node, channels_first)


class CMSISDWConv2DParser(RQSConv2DParser):

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
                # Don't support dilations
                #all([coeff == 1 for coeff in self.parserDict['dilations']]),
                len(node.inputs) == 5,
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:

            if not self.parserDict['group'] == newCtxt.lookup(self.parserDict['weight']).shape[0]:
                return ctxt, False

            inputs = ['data_in', 'weight', 'mul', 'add', 'shift']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            data_in = newCtxt.lookup(self.parserDict['data_in'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])
            weight = newCtxt.lookup(self.parserDict['weight'])

            if not newCtxt.is_global(self.parserDict['weight']):
                return ctxt, False

            # SCHEREMO: Transpose weights to be num filters last
            newCtxt.globalObjects[self.parserDict['weight']].values = np.transpose(
                weight.values,
                list(range(len(weight.shape)))[1:] + [0])

            return newCtxt, True

        return ctxt, False


class CMSISConv2DParser(RQSConv2DParser):

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
                #self.parserDict['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.parserDict['dilations']]),
                len(node.inputs) == 5,
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add', 'shift']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class CMSISDWConv1DParser(RQSConv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                #self.parserDict['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.parserDict['dilations']]),
                len(node.inputs) == 5,
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:

            inputs = ['data_in', 'weight', 'mul', 'add', 'shift']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            if not self.parserDict['group'] == newCtxt.lookup(self.parserDict['weight']).shape[-1]:
                return ctxt, False

            return newCtxt, True

        return ctxt, False


class CMSISConv1DParser(RQSConv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                self.parserDict['group'] == 1,
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                #self.parserDict['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.parserDict['dilations']]),
                len(node.inputs) == 5,
            ])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add', 'shift']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class CMSISLinearParser(GEMMParser):

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
            # Try to transpose A offline if possible, else fail
            if self.parserDict['transA'] == 1:
                nameA = self.parserDict['A']
                if newCtxt.is_global(nameA) and isinstance(newCtxt.lookup(nameA), ConstantBuffer):
                    A = newCtxt.lookup(nameA)
                    npA = np.asarray(A.values).reshape(A.shape)
                    newA = np.transpose(npA, list(range(len(A.shape) - 2)) + [len(A.shape) - 1, len(A.shape) - 2])
                    newCtxt.globalObjects[nameA].shape = newA.shape
                    newCtxt.globalObjects[nameA].values = newA
                    self.parserDict['transA'] = 0
                else:
                    return newCtxt, False

            # Try to transpose B offline if possible, else fail
            # SCHEREMO: Magic trick - CMSIS works a bit weirdly with matmuls...
            if self.parserDict['transB'] == 0:
                nameB = self.parserDict['B']
                if newCtxt.is_global(nameB) and isinstance(newCtxt.lookup(nameB), ConstantBuffer):
                    B = newCtxt.lookup(nameB)
                    npB = np.asarray(B.values).reshape(B.shape)
                    newB = np.transpose(npB, list(range(len(B.shape) - 2)) + [len(B.shape) - 1, len(B.shape) - 2])
                    newCtxt.globalObjects[nameB].values = newB
                    newCtxt.globalObjects[nameB].shape = newB.shape
                    self.parserDict['transB'] = 1
                else:
                    return newCtxt, False

            # Try to scale A offline if possible, else fail
            if self.parserDict['alpha'] != 1.0:
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
            if self.parserDict['beta'] != 1.0:
                nameB = self.parserDict['B']
                if newCtxt.is_global(nameB) and isinstance(newCtxt.lookup(nameB), ConstantBuffer):
                    B = newCtxt.lookup(nameB)
                    npB = np.asarray(B.values).reshape(B.shape)
                    newB = npB * self.parserDict['beta']
                    newCtxt.globalObjects[nameB].values = newB
                    self.parserDict['beta'] = 1.0
                else:
                    return newCtxt, False

            return newCtxt, True

        return ctxt, False


class CMSISGEMMParser(CMSISLinearParser, RQSParserInterface):

    def __init__(self):
        super().__init__(noBiasHoisting = True)

    def parseNode(self, node: gs.Node) -> (bool):

        ret_linear = CMSISLinearParser.parseNode(self, node)
        ret_rqs = RQSParserInterface.parseNode(self, node)

        ret = all([
            ret_linear == True,
            ret_rqs == True,
            'shift' in node.attrs,
            len(node.inputs) == 4,
        ])

        if ret:
            self.parserDict['shift'] = int(node.attrs['shift'].values)

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['A', 'B', 'C', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            return newCtxt, True

        else:
            return ctxt, False


class CMSISMHSAParser(MHSAParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)

        if wellFormed:
            ret = all([
                'isoftmaxA' in node.attrs,
                'isoftmaxB' in node.attrs,
                'isoftmaxC' in node.attrs,
                'isoftmaxlog2' in node.attrs,
            ])

            if ret:
                self.parserDict['signed'] = 1
                self.parserDict['preattn_requant_shift'] = int(node.attrs['preattn_requant_shift'].values)
                self.parserDict['preattn_requant_div'] = int(math.log2(int(node.attrs['preattn_requant_div'].values)))
                self.parserDict['postattn_requant_shift'] = int(node.attrs['postattn_requant_shift'].values)
                self.parserDict['postattn_requant_div'] = int(math.log2(int(node.attrs['postattn_requant_div'].values)))
                self.parserDict['wo_requant_shift'] = int(node.attrs['wo_requant_shift'].values)
                self.parserDict['wo_requant_div'] = int(math.log2(int(node.attrs['wo_requant_div'].values)))
                self.parserDict['wq_requant_shift'] = int(node.attrs['wq_requant_shift'].values)
                self.parserDict['wq_requant_div'] = int(math.log2(int(node.attrs['wq_requant_div'].values)))
                self.parserDict['wk_requant_shift'] = int(node.attrs['wk_requant_shift'].values)
                self.parserDict['wk_requant_div'] = int(math.log2(int(node.attrs['wk_requant_div'].values)))
                self.parserDict['wv_requant_shift'] = int(node.attrs['wv_requant_shift'].values)
                self.parserDict['wv_requant_div'] = int(math.log2(int(node.attrs['wv_requant_div'].values)))
                self.parserDict['isoftmaxA'] = int(node.attrs['isoftmaxA'].values)
                self.parserDict['isoftmaxB'] = int(node.attrs['isoftmaxB'].values)
                self.parserDict['isoftmaxC'] = int(node.attrs['isoftmaxC'].values)
                self.parserDict['isoftmaxlog2'] = int(node.attrs['isoftmaxlog2'].values)

            return ret

        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            return newCtxt, ret
        else:
            return ctxt, False


class CMSISLinearAttentionParser(LinearAttentionParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        self.parserDict['signed'] = 1
        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            return newCtxt, ret
        else:
            return ctxt, False


class CMSISCLCAParser(CLCAParser):

    def __init__(self):
        super().__init__()

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
            # Div to shift:
            newCtxt.globalObjects[self.parserDict['wq_requant_div']].values = np.log2(
                newCtxt.globalObjects[self.parserDict['wq_requant_div']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['wk_requant_div']].values = np.log2(
                newCtxt.globalObjects[self.parserDict['wk_requant_div']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['wv_requant_div']].values = np.log2(
                newCtxt.globalObjects[self.parserDict['wv_requant_div']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['wo_requant_div']].values = np.log2(
                newCtxt.globalObjects[self.parserDict['wo_requant_div']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['kdiv_requant_div']].values = np.log2(
                newCtxt.globalObjects[self.parserDict['kdiv_requant_div']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['preattn_requant_div']].values = np.log2(
                newCtxt.globalObjects[self.parserDict['preattn_requant_div']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['postattn_requant_div']].values = np.log2(
                newCtxt.globalObjects[self.parserDict['postattn_requant_div']].values).astype('int')

            # Fold additions:
            newCtxt.globalObjects[
                self.parserDict['wo_bias']].values = newCtxt.globalObjects[self.parserDict['wo_bias']].values + (
                    newCtxt.globalObjects[self.parserDict['wo_requant_add']].values /
                    newCtxt.globalObjects[self.parserDict['wo_requant_mul']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['wo_requant_add']]._deploy = False
            newCtxt.globalObjects[
                self.parserDict['wq_bias']].values = newCtxt.globalObjects[self.parserDict['wq_bias']].values + (
                    newCtxt.globalObjects[self.parserDict['wq_requant_add']].values /
                    newCtxt.globalObjects[self.parserDict['wq_requant_mul']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['wq_requant_add']]._deploy = False
            newCtxt.globalObjects[
                self.parserDict['wk_bias']].values = newCtxt.globalObjects[self.parserDict['wk_bias']].values + (
                    newCtxt.globalObjects[self.parserDict['wv_requant_add']].values /
                    newCtxt.globalObjects[self.parserDict['wv_requant_mul']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['wv_requant_add']]._deploy = False

            # Rescale requant adds:
            newCtxt.globalObjects[self.parserDict['postattn_requant_add']].values = (
                newCtxt.globalObjects[self.parserDict['postattn_requant_add']].values /
                newCtxt.globalObjects[self.parserDict['postattn_requant_mul']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['preattn_requant_add']].values = (
                newCtxt.globalObjects[self.parserDict['preattn_requant_add']].values /
                newCtxt.globalObjects[self.parserDict['preattn_requant_mul']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['kdiv_requant_add']].values = (
                newCtxt.globalObjects[self.parserDict['kdiv_requant_add']].values /
                newCtxt.globalObjects[self.parserDict['kdiv_requant_mul']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['wk_requant_add']].values = (
                newCtxt.globalObjects[self.parserDict['wk_requant_add']].values /
                newCtxt.globalObjects[self.parserDict['wk_requant_mul']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['wo_requant_add']].values = (
                newCtxt.globalObjects[self.parserDict['wo_requant_add']].values /
                newCtxt.globalObjects[self.parserDict['wo_requant_mul']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['wq_requant_add']].values = (
                newCtxt.globalObjects[self.parserDict['wq_requant_add']].values /
                newCtxt.globalObjects[self.parserDict['wq_requant_mul']].values).astype('int')
            newCtxt.globalObjects[self.parserDict['wv_requant_add']].values = (
                newCtxt.globalObjects[self.parserDict['wv_requant_add']].values /
                newCtxt.globalObjects[self.parserDict['wv_requant_mul']].values).astype('int')

            # Delta into mul
            newCtxt.globalObjects[self.parserDict['kdiv_requant_mul']].values = newCtxt.globalObjects[
                self.parserDict['kdiv_requant_mul']].values * self.parserDict['Delta']

            return newCtxt, ret
        else:
            return ctxt, False
