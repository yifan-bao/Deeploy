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

import numpy as np
import math
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import *
from Deeploy.Parsers.BasicParsers import *
from Deeploy.Bindings.BasicBindings import DataTypes


class GenericMaxPool2DParser(MaxPool2DParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:

        wellFormed = super().parseNode(node)
        if wellFormed:
            wellFormed = all([all([pad == 0 for pad in self.parserDict['pads']]), self.parserDict['ceil_mode'] == 0])
            if wellFormed:
                self.parserDict['padding_x'] = int(self.parserDict['pads'][0])
                self.parserDict['padding_y'] = int(self.parserDict['pads'][1])
                self.parserDict['stride_x'] = int(self.parserDict['strides'][0])
                self.parserDict['stride_y'] = int(self.parserDict['strides'][1])
                self.parserDict['dim_kernel_x'] = int(self.parserDict['kernel_shape'][0])
                self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][1])

        return wellFormed

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:

            data_in = newCtxt.lookup(self.parserDict['data_in'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])

            self.parserDict['batch'] = data_in.shape[0]
            if channels_first:
                self.parserDict['ch_im_in'] = data_in.shape[1]
                self.parserDict['dim_im_in_x'] = data_in.shape[2]
                self.parserDict['dim_im_in_y'] = data_in.shape[3]
                self.parserDict['ch_im_out'] = data_out.shape[1]
                self.parserDict['dim_im_out_x'] = data_out.shape[2]
                self.parserDict['dim_im_out_y'] = data_out.shape[3]
            else:
                self.parserDict['ch_im_in'] = data_in.shape[3]
                self.parserDict['dim_im_in_x'] = data_in.shape[1]
                self.parserDict['dim_im_in_y'] = data_in.shape[2]
                self.parserDict['ch_im_out'] = data_out.shape[3]
                self.parserDict['dim_im_out_x'] = data_out.shape[1]
                self.parserDict['dim_im_out_y'] = data_out.shape[2]

        return newCtxt, ret


class GenericConv1DParser(Conv1DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):
        wellFormed = super().parseNode(node)

        if wellFormed:
            ret = all([
                # Make sure padding is square
                node.op == 'Conv',
                self.parserDict['group'] == 1,
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                self.parserDict['pads'][0] == 0,
                all([coeff == 1 for coeff in self.parserDict['dilations']]),
            ])

            if ret:
                self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][0])
                self.parserDict['dilation_y'] = int(self.parserDict['dilations'][0])
                self.parserDict['padding_y'] = int(self.parserDict['pads'][0])
                self.parserDict['stride_y'] = int(self.parserDict['strides'][0])
            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:
            inputs = ['data_in', 'weight']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name

            data_in = newCtxt.lookup(self.parserDict['data_in'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])

            self.parserDict['batch'] = data_in.shape[0]
            if channels_first:
                self.parserDict['ch_im_in'] = data_in.shape[1]
                self.parserDict['dim_im_in_y'] = data_in.shape[2]
                self.parserDict['ch_im_out'] = data_out.shape[1]
                self.parserDict['dim_im_out_y'] = data_out.shape[2]
            else:
                self.parserDict['ch_im_in'] = data_in.shape[2]
                self.parserDict['dim_im_in_y'] = data_in.shape[1]
                self.parserDict['ch_im_out'] = data_out.shape[2]
                self.parserDict['dim_im_out_y'] = data_out.shape[1]
            # import IPython;IPython.embed()

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
                node.op == 'Conv',
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                self.parserDict['pads'][0] == 0,
                all([coeff == 1 for coeff in self.parserDict['dilations']]),
            ])

            if ret:
                self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][0])
                self.parserDict['dilation_y'] = int(self.parserDict['dilations'][0])
                self.parserDict['padding_y'] = int(self.parserDict['pads'][0])
                self.parserDict['stride_y'] = int(self.parserDict['strides'][0])
            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:
            inputs = ['data_in', 'weight']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name

            data_in = newCtxt.lookup(self.parserDict['data_in'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])

            self.parserDict['batch'] = data_in.shape[0]
            if channels_first:
                self.parserDict['ch_im_in'] = data_in.shape[1]
                self.parserDict['dim_im_in_y'] = data_in.shape[2]
                self.parserDict['ch_im_out'] = data_out.shape[1]
                self.parserDict['dim_im_out_y'] = data_out.shape[2]
            else:
                self.parserDict['ch_im_in'] = data_in.shape[2]
                self.parserDict['dim_im_in_y'] = data_in.shape[1]
                self.parserDict['ch_im_out'] = data_out.shape[2]
                self.parserDict['dim_im_out_y'] = data_out.shape[1]

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
                node.op == 'Conv',
                self.parserDict['group'] == 1,
                self.parserDict['pads'][0] == self.parserDict['pads'][2],
                self.parserDict['pads'][1] == self.parserDict['pads'][3],
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                self.parserDict['pads'][0] == 0,
                all([coeff == 1 for coeff in self.parserDict['dilations']]),
            ])

            if ret:
                self.parserDict['dim_kernel_x'] = int(self.parserDict['kernel_shape'][0])
                self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][1])
                self.parserDict['dilation_x'] = int(self.parserDict['dilations'][0])
                self.parserDict['dilation_y'] = int(self.parserDict['dilations'][1])
                self.parserDict['padding_x'] = int(self.parserDict['pads'][0])
                self.parserDict['padding_y'] = int(self.parserDict['pads'][1])
                self.parserDict['stride_x'] = int(self.parserDict['strides'][0])
                self.parserDict['stride_y'] = int(self.parserDict['strides'][1])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:
            inputs = ['data_in', 'weight']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name

            data_in = newCtxt.lookup(self.parserDict['data_in'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])

            self.parserDict['batch'] = data_in.shape[0]
            if channels_first:
                self.parserDict['ch_im_in'] = data_in.shape[1]
                self.parserDict['dim_im_in_x'] = data_in.shape[2]
                self.parserDict['dim_im_in_y'] = data_in.shape[3]
                self.parserDict['ch_im_out'] = data_out.shape[1]
                self.parserDict['dim_im_out_x'] = data_out.shape[2]
                self.parserDict['dim_im_out_y'] = data_out.shape[3]
            else:
                self.parserDict['ch_im_in'] = data_in.shape[3]
                self.parserDict['dim_im_in_x'] = data_in.shape[1]
                self.parserDict['dim_im_in_y'] = data_in.shape[2]
                self.parserDict['ch_im_out'] = data_out.shape[3]
                self.parserDict['dim_im_out_x'] = data_out.shape[1]
                self.parserDict['dim_im_out_y'] = data_out.shape[2]
            # import IPython;IPython.embed()

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
                node.op == 'Conv',
                self.parserDict['pads'][0] == self.parserDict['pads'][2],
                self.parserDict['pads'][1] == self.parserDict['pads'][3],
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                self.parserDict['pads'][0] == 0,
                all([coeff == 1 for coeff in self.parserDict['dilations']]),
            ])

            if ret:
                self.parserDict['dim_kernel_x'] = int(self.parserDict['kernel_shape'][0])
                self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][1])
                self.parserDict['dilation_x'] = int(self.parserDict['dilations'][0])
                self.parserDict['dilation_y'] = int(self.parserDict['dilations'][1])
                self.parserDict['padding_x'] = int(self.parserDict['pads'][0])
                self.parserDict['padding_y'] = int(self.parserDict['pads'][1])
                self.parserDict['stride_x'] = int(self.parserDict['strides'][0])
                self.parserDict['stride_y'] = int(self.parserDict['strides'][1])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:
            inputs = ['data_in', 'weight']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name

            data_in = newCtxt.lookup(self.parserDict['data_in'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])

            self.parserDict['batch'] = data_in.shape[0]
            if channels_first:
                self.parserDict['ch_im_in'] = data_in.shape[1]
                self.parserDict['dim_im_in_x'] = data_in.shape[2]
                self.parserDict['dim_im_in_y'] = data_in.shape[3]
                self.parserDict['ch_im_out'] = data_out.shape[1]
                self.parserDict['dim_im_out_x'] = data_out.shape[2]
                self.parserDict['dim_im_out_y'] = data_out.shape[3]
            else:
                self.parserDict['ch_im_in'] = data_in.shape[3]
                self.parserDict['dim_im_in_x'] = data_in.shape[1]
                self.parserDict['dim_im_in_y'] = data_in.shape[2]
                self.parserDict['ch_im_out'] = data_out.shape[3]
                self.parserDict['dim_im_out_x'] = data_out.shape[1]
                self.parserDict['dim_im_out_y'] = data_out.shape[2]

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
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)
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
