# ----------------------------------------------------------------------
#
# File: PULPParsers.py
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

import math
from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeParser
from Deeploy.Parsers.BasicParsers import AddParser, RQSConv1DParser, RQSConv2DParser


class PULPRQAddParser(AddParser):

    def parseNode(self, node: gs.Node) -> bool:

        if not super().parseNode(node):
            return False

        ret = all([
            'rqs1_mul' in node.attrs,
            'rqs1_add' in node.attrs,
            'rqs1_div' in node.attrs,
            'rqs1_signed' in node.attrs,
            any(['rqs1_n_levels' in node.attrs, 'rqs1_n_levels_out' in node.attrs]),
            'rqs2_mul' in node.attrs,
            'rqs2_add' in node.attrs,
            'rqs2_div' in node.attrs,
            'rqs2_signed' in node.attrs,
            any(['rqs2_n_levels' in node.attrs, 'rqs2_n_levels_out' in node.attrs]),
            'rqsOut_mul' in node.attrs,
            'rqsOut_add' in node.attrs,
            'rqsOut_div' in node.attrs,
            'rqsOut_signed' in node.attrs,
            any(['rqsOut_n_levels' in node.attrs, 'rqsOut_n_levels_out' in node.attrs]),
        ])

        if ret:
            if 'rqs1_n_levels' in node.attrs:
                self.parserDict['rqs1_n_levels'] = int(node.attrs['rqs1_n_levels'].values)
            else:
                self.parserDict['rqs1_n_levels'] = int(node.attrs['rqs1_n_levels_out'].values)
            self.parserDict['rqs1_mul'] = int(node.attrs['rqs1_mul'])
            self.parserDict['rqs1_add'] = int(node.attrs['rqs1_add'])
            self.parserDict['rqs1_signed'] = int(node.attrs['rqs1_signed'].values)
            self.parserDict['rqs1_log2D'] = int(math.log2(node.attrs['rqs1_div'].values))

            if 'rqs2_n_levels' in node.attrs:
                self.parserDict['rqs2_n_levels'] = int(node.attrs['rqs2_n_levels'].values)
            else:
                self.parserDict['rqs2_n_levels'] = int(node.attrs['rqs2_n_levels_out'].values)
            self.parserDict['rqs2_mul'] = int(node.attrs['rqs2_mul'])
            self.parserDict['rqs2_add'] = int(node.attrs['rqs2_add'])
            self.parserDict['rqs2_signed'] = int(node.attrs['rqs2_signed'].values)
            self.parserDict['rqs2_log2D'] = int(math.log2(node.attrs['rqs2_div'].values))

            if 'rqsOut_n_levels' in node.attrs:
                self.parserDict['rqsOut_n_levels'] = int(node.attrs['rqsOut_n_levels'].values)
            else:
                self.parserDict['rqsOut_n_levels'] = int(node.attrs['rqsOut_n_levels_out'].values)
            self.parserDict['rqsOut_mul'] = int(node.attrs['rqsOut_mul'])
            self.parserDict['rqsOut_add'] = int(node.attrs['rqsOut_add'])
            self.parserDict['rqsOut_signed'] = int(node.attrs['rqsOut_signed'].values)
            self.parserDict['rqsOut_log2D'] = int(math.log2(node.attrs['rqsOut_div'].values))

        return ret


class PULPConv2DParser(RQSConv2DParser):

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
                len(node.inputs) == 4,
                'shift' in node.attrs,
            ])

            self.parserDict['dim_kernel_x'] = int(self.parserDict['kernel_shape'][0])
            self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][1])
            self.parserDict['dilation_x'] = int(self.parserDict['dilations'][0])
            self.parserDict['dilation_y'] = int(self.parserDict['dilations'][1])
            self.parserDict['padding_y_top'] = int(self.parserDict['pads'][0])
            self.parserDict['padding_x_left'] = int(self.parserDict['pads'][1])
            self.parserDict['padding_y_bottom'] = int(self.parserDict['pads'][2])
            self.parserDict['padding_x_right'] = int(self.parserDict['pads'][3])
            self.parserDict['stride_x'] = int(self.parserDict['strides'][0])
            self.parserDict['stride_y'] = int(self.parserDict['strides'][1])

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False


class PULPDWConv1DParser(RQSConv1DParser):

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
                len(node.inputs) == 4,
            ])

            if ret:

                self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][0])
                self.parserDict['dilation_y'] = int(self.parserDict['dilations'][0])
                self.parserDict['padding_y_top'] = int(self.parserDict['pads'][0])
                self.parserDict['padding_y_bottom'] = int(self.parserDict['pads'][1])
                self.parserDict['stride_y'] = int(self.parserDict['strides'][0])

                if 'n_levels' in node.attrs:
                    self.parserDict['n_levels'] = int(node.attrs['n_levels'].values)
                else:
                    self.parserDict['n_levels'] = int(node.attrs['n_levels_out'].values)

                self.parserDict['signed'] = int(node.attrs['signed'].values)
                self.parserDict['log2D'] = int(math.log2(node.attrs['div'].values))
            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:

            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            if not self.parserDict['group'] == newCtxt.lookup(self.parserDict['weight']).shape[0]:
                return ctxt, False

            # if not newCtxt.is_global(self.parserDict['weight']):
            #     return ctxt, False

            # SCHEREMO: Transpose weights to be num filters last
            # newCtxt.globalObjects[self.parserDict['weight']].values = np.transpose(weight.values, list(range(len(weight.shape)))[1:] + [0])

            return newCtxt, True

        return ctxt, False


class PULPDWConv2DParser(RQSConv2DParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        if wellFormed:
            ret = all([
                # Make sure padding is square
                node.op == 'RequantizedConv',
                self.parserDict['pads'][0] == self.parserDict['pads'][2],
                self.parserDict['pads'][1] == self.parserDict['pads'][3],
                self.parserDict['pads'][0] == self.parserDict['pads'][1],
                #self.parserDict['pads'][0] == 0,
                # Don't support dilations
                #all([coeff == 1 for coeff in self.parserDict['dilations']]),
                len(node.inputs) == 4,
                'shift' in node.attrs,
                any(['n_levels' in node.attrs, 'n_levels_out' in node.attrs]),
                'signed' in node.attrs
            ])

            if ret:
                self.parserDict['dim_kernel_x'] = int(self.parserDict['kernel_shape'][0])
                self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][1])
                self.parserDict['dilation_x'] = int(self.parserDict['dilations'][0])
                self.parserDict['dilation_y'] = int(self.parserDict['dilations'][1])
                self.parserDict['padding_y_top'] = int(self.parserDict['pads'][0])
                self.parserDict['padding_x_left'] = int(self.parserDict['pads'][1])
                self.parserDict['padding_y_bottom'] = int(self.parserDict['pads'][2])
                self.parserDict['padding_x_right'] = int(self.parserDict['pads'][3])
                self.parserDict['stride_x'] = int(self.parserDict['strides'][0])
                self.parserDict['stride_y'] = int(self.parserDict['strides'][1])

                if 'n_levels' in node.attrs:
                    self.parserDict['n_levels'] = int(node.attrs['n_levels'].values)
                else:
                    self.parserDict['n_levels'] = int(node.attrs['n_levels_out'].values)
                self.parserDict['signed'] = int(node.attrs['signed'].values)
                self.parserDict['log2D'] = int(math.log2(node.attrs['div'].values))

            return ret
        return False

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:

            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = newCtxt.lookup(inputNode.name).name

            if not self.parserDict['group'] == newCtxt.lookup(self.parserDict['weight']).shape[0]:
                return ctxt, False

            data_in = newCtxt.lookup(self.parserDict['data_in'])
            data_out = newCtxt.lookup(self.parserDict['data_out'])
            _ = newCtxt.lookup(self.parserDict['weight'])

            # if not newCtxt.is_global(self.parserDict['weight']):
            #     return ctxt, False

            # SCHEREMO: Transpose weights to be num filters last
            # newCtxt.globalObjects[self.parserDict['weight']].values = np.transpose(weight.values, list(range(len(weight.shape)))[1:] + [0])

            if channels_first:
                self.parserDict['ch_im_in'] = data_in.shape[1]
                self.parserDict['dim_im_in_x'] = data_in.shape[2]
                self.parserDict['dim_im_in_y'] = data_in.shape[3]
                self.parserDict['ch_im_out'] = data_out.shape[1]
                self.parserDict['dim_im_out_x'] = data_out.shape[2]
                self.parserDict['dim_im_out_y'] = data_out.shape[3]
            else:
                self.parserDict['ch_im_in'] = data_in.shape[1]
                self.parserDict['dim_im_in_x'] = data_in.shape[2]
                self.parserDict['dim_im_in_y'] = data_in.shape[3]
                self.parserDict['ch_im_out'] = data_out.shape[3]
                self.parserDict['dim_im_out_x'] = data_out.shape[1]
                self.parserDict['dim_im_out_y'] = data_out.shape[2]

            return newCtxt, True

        return ctxt, False


class PULPConv1DParser(RQSConv1DParser):

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
                len(node.inputs) == 4,
            ])

            self.parserDict['dim_kernel_y'] = int(self.parserDict['kernel_shape'][0])
            self.parserDict['dilation_y'] = int(self.parserDict['dilations'][0])
            self.parserDict['padding_y_top'] = int(self.parserDict['pads'][0])
            self.parserDict['padding_y_bottom'] = int(self.parserDict['pads'][1])
            self.parserDict['stride_y'] = int(self.parserDict['strides'][0])

            return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['data_in', 'weight', 'mul', 'add']
            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name

            return newCtxt, True

        return ctxt, False
