# ----------------------------------------------------------------------
#
# File: BasicParsers.py
#
# Last edited: 15.12.2021
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

import numpy as np
import math
import onnx_graphsurgeon as gs

from DumpO.DumpOTypes import *

class TransposeParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        ret = all([
            'perm' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1
        ])

        if ret:
            self.parserDict['perm'] = node.attrs['perm']
        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.parserDict['data_in_shape'] = data_in.shape
        self.parserDict['data_out_shape'] = data_out.shape
        self.parserDict['data_in'] = data_in.name
        self.parserDict['data_out'] = data_out.name
        self.parserDict['data_in_size'] = np.prod(data_in.shape)
        self.parserDict['data_out_size'] = np.prod(data_out.shape)

        return ctxt, True

class MaxPoolParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        ret = all([
            'ceil_mode' in node.attrs,
            'kernel_shape' in node.attrs,
            'pads' in node.attrs,
            'strides' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) >= 1
        ])

        if ret:
            self.parserDict['ceil_mode'] = node.attrs['ceil_mode']
            self.parserDict['pads'] = node.attrs['pads']
            self.parserDict['kernel_shape'] = node.attrs['kernel_shape']
            self.parserDict['strides'] = node.attrs['strides']

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.parserDict['data_in'] = data_in.name
        self.parserDict['data_out'] = data_out.name
        self.parserDict['data_in_size'] = np.prod(data_in.shape)
        self.parserDict['data_out_size'] = np.prod(data_out.shape)

        return ctxt, True

class MaxPool2DParser(MaxPoolParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        ret = super().parseNode(node)
        wellFormed = False
        if ret:
            pads = self.parserDict['pads']
            kernel_shape = self.parserDict['kernel_shape']
            strides = self.parserDict['strides']
            if len(pads) == 4 and len(kernel_shape) == 2 and len(strides) == 2:
                wellFormed = True
        return wellFormed

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt, ret = super().parseNodeCtxt(ctxt, node)
        wellFormed = False
        if ret:
            data_in = ctxt.lookup(node.inputs[0].name)
            data_out = ctxt.lookup(node.outputs[0].name)
            if len(data_in.shape) == 4 and len(data_out.shape) == 4:
                wellFormed = True

        return ctxt, wellFormed


class PadParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        ret = all([
            'mode' in node.attrs,
            'pads' in node.attrs,
            'value' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1
        ])

        if ret:
            self.parserDict['mode'] = node.attrs['mode']
            self.parserDict['pads'] = node.attrs['pads']
            self.parserDict['value'] = node.attrs['value']

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        ctxt = ctxt.copy()

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.parserDict['data_in'] = data_in.name
        self.parserDict['data_out'] = data_out.name
        self.parserDict['data_in_size'] = np.prod(data_in.shape)
        self.parserDict['data_out_size'] = np.prod(data_out.shape)

        return ctxt, True

class Pad2DParser(PadParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        ret = super().parseNode(node)
        wellFormed = False
        if ret:
            pads = self.parserDict['pads']
            if len(pads) == 8 and pads[0] == 0 and pads[4] == 0 \
            and pads[1] == 0 and pads[5] == 0:
                wellFormed = True
                self.parserDict['pad_x'] = int(pads[3])
                self.parserDict['pad_y'] = int(pads[2])

        return wellFormed

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt, ret = super().parseNodeCtxt(ctxt, node)
        wellFormed = False
        if ret:
            data_in = ctxt.lookup(node.inputs[0].name)
            data_out = ctxt.lookup(node.outputs[0].name)
            if len(data_in.shape) == 4:
                wellFormed = True
                self.parserDict['batch'] = data_in.shape[0]
                if channels_first:
                    self.parserDict['dim_im_in_x'] = data_in.shape[2]
                    self.parserDict['dim_im_in_y'] = data_in.shape[3]
                    self.parserDict['dim_im_in_ch'] = data_in.shape[1]
                    self.parserDict['dim_im_out_x'] = data_out.shape[2]
                    self.parserDict['dim_im_out_y'] = data_out.shape[3]
                    self.parserDict['dim_im_out_ch'] = data_out.shape[1]
                else:
                    self.parserDict['dim_im_in_x'] = data_in.shape[1]
                    self.parserDict['dim_im_in_y'] = data_in.shape[2]
                    self.parserDict['dim_im_in_ch'] = data_in.shape[3]
                    self.parserDict['dim_im_out_x'] = data_out.shape[1]
                    self.parserDict['dim_im_out_y'] = data_out.shape[2]
                    self.parserDict['dim_im_out_ch'] = data_out.shape[3]
        return ctxt, wellFormed

class Pad1DParser(PadParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        ret = super().parseNode(node)
        wellFormed = False
        if ret:
            pads = self.parserDict['pads']
            if len(pads) == 6 and pads[0] == 0 and pads[3] == 0 \
            and pads[1] == 0 and pads[4] == 0:
                wellFormed = True
                self.parserDict['pad_y'] = pads[2]
                self.parserDict['pad_x'] = 0

        return wellFormed

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt, ret = super().parseNodeCtxt(ctxt, node)
        wellFormed = False
        if ret:
            data_in = ctxt.lookup(node.inputs[0].name)
            data_out = ctxt.lookup(node.outputs[0].name)
            if len(data_in.shape) == 3:
                wellFormed = True
                self.parserDict['batch'] = data_in.shape[0]
                self.parserDict['dim_im_in_x'] = 1
                self.parserDict['dim_im_out_x'] = 1
                if channels_first:
                    self.parserDict['dim_im_in_y'] = data_in.shape[2]
                    self.parserDict['dim_im_in_ch'] = data_in.shape[1]
                    self.parserDict['dim_im_out_y'] = data_out.shape[2]
                    self.parserDict['dim_im_out_ch'] = data_out.shape[1]
                else:
                    self.parserDict['dim_im_in_y'] = data_in.shape[1]
                    self.parserDict['dim_im_in_ch'] = data_in.shape[2]
                    self.parserDict['dim_im_out_y'] = data_out.shape[1]
                    self.parserDict['dim_im_out_ch'] = data_out.shape[2]
        return ctxt, wellFormed


class AddParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        ret = all([
            len(node.inputs) == 2,
            len(node.outputs) == 1
        ])

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        data_in_1 = ctxt.lookup(node.inputs[0].name)
        data_in_2 = ctxt.lookup(node.inputs[1].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.parserDict['data_in_1'] = data_in_1.name
        self.parserDict['data_in_2'] = data_in_2.name
        self.parserDict['data_out'] = data_out.name
        self.parserDict['size'] = np.prod(data_in_1.shape)

        return ctxt, True

class ReduceMeanParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        ret = all([
            'axes' in node.attrs,
            'keepdims' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1
        ])

        if ret:
            if isinstance(node.attrs['axes'], int):
                self.parserDict['axes'] = [node.attrs['axes']]
            else:
                self.parserDict['axes'] = node.attrs['axes']
            self.parserDict['keepdims'] = int(node.attrs['keepdims'])

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.parserDict['data_in'] = data_in.name
        self.parserDict['data_out'] = data_out.name
        self.parserDict['data_in_shape'] = data_in.shape
        self.parserDict['data_out_shape'] = data_out.shape
        self.parserDict['size'] = np.prod(data_in.shape)
        self.parserDict['axisLength'] = data_in.shape[self.parserDict['axes'][0]]

        return ctxt, True

class iSoftmaxParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        ret = all([
            'coeffA' in node.attrs,
            'coeffB' in node.attrs,
            'coeffC' in node.attrs,
            'log2' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1
        ])

        if ret:
            self.parserDict['coeffA'] = int(node.attrs['coeffA'].values)
            self.parserDict['coeffB'] = int(math.log2(node.attrs['coeffB'].values))
            self.parserDict['coeffC'] = int(node.attrs['coeffC'].values)
            self.parserDict['log2'] = int(node.attrs['log2'].values)

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.parserDict['data_in'] = data_in.name
        self.parserDict['data_out'] = data_out.name
        self.parserDict['size'] = np.prod(data_in.shape)
        self.parserDict['lastDimLength'] = data_in.shape[-1]

        return ctxt, True

class iGELUParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        ret = all([
            'b' in node.attrs,
            'one' in node.attrs,
            len(node.inputs) >= 1,
            len(node.outputs) == 1
        ])

        if ret:
            self.parserDict['b'] = node.attrs['b']
            self.parserDict['one'] = node.attrs['one']

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.parserDict['data_in'] = data_in.name
        self.parserDict['data_out'] = data_out.name
        self.parserDict['size'] = np.prod(data_in.shape)

        return ctxt, True

class RQSiGELUParser(iGELUParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        wellFormed = all([
            len(node.inputs) == 4,
        ])
        ret = super().parseNode(node)

        return (ret and wellFormed)

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:

            inputs = ['data_in', 'mul', 'add', 'shift']
            outputs = ['data_out']

            for idx, inputNode in enumerate(node.inputs):
                self.parserDict[inputs[idx]] = newCtxt.lookup(inputNode.name).name
            for idx, outputNode in enumerate(node.outputs):
                self.parserDict[outputs[idx]] = newCtxt.lookup(outputNode.name).name

            return newCtxt, True
        return ctxt, False


class GatherParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> (bool):

        ret = all([
            'axis' in node.attrs,
            len(node.inputs) == 2,
            len(node.outputs) == 1
        ])


        if ret:
            self.parserDict['axis'] = node.attrs['axis']

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        inputs = ['data_in', 'indices']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name


        axis = self.parserDict['axis']
        self.parserDict['numIndices'] = np.prod(ctxt.lookup(self.parserDict['indices']).values.shape)
        self.parserDict['offset'] = np.prod(ctxt.lookup(node.inputs[0].name).shape[axis+1:])
        self.parserDict['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)

        return ctxt, True

class FlattenParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> (bool):

        ret = all([
            'axis' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1
        ])

        if ret:
            self.parserDict['axis'] = node.attrs['axis']

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        inputs = ['data_in']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name


        return ctxt, True

class UnsqueezeParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> (bool):

        ret = all([
            'axes' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1
        ])

        if ret:
            self.parserDict['axes'] = node.attrs['axes']

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        inputs = ['data_in']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name


        return ctxt, True


class ReshapeParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> (bool):

        ret = all([
            len(node.inputs) == 2,
            len(node.outputs) == 1
        ])

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        inputs = ['data_in', 'indices']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name


        self.parserDict['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)

        return ctxt, True

class RequantShiftParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> (bool):

        ret = all([
            'div' in node.attrs,
            any(['n_levels' in node.attrs,
                'n_levels_out' in node.attrs
                 ]),
            'signed' in node.attrs,
            len(node.inputs) == 3,
            len(node.outputs) == 1
        ])

        if ret:
            if 'n_levels' in node.attrs:
                self.parserDict['n_levels'] = int(node.attrs['n_levels'].values)
            else:
                self.parserDict['n_levels'] = int(node.attrs['n_levels_out'].values)
            self.parserDict['signed'] = int(node.attrs['signed'].values)
            self.parserDict['log2D'] = int(math.log2(node.attrs['div'].values))

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        inputs = ['data_in', 'mul', 'add']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.parserDict['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.parserDict['channels'] = ctxt.lookup(node.inputs[0].name).shape[1]

        return ctxt, True

class MulParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> (bool):

        wellFormed = all([
            len(node.inputs) == 2,
            len(node.outputs) == 1
        ])

        return wellFormed

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        inputs = ['A', 'B']
        outputs = ['C']

        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.parserDict['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.parserDict['sizeB'] = np.prod(ctxt.lookup(node.inputs[1].name).shape)

        return ctxt, True

class ConvParser(NodeParser):
    def __init__(self, noBiasHoisting):
        super().__init__()
        self.noBiasHoisting = noBiasHoisting

    def parseNode(self, node:  gs.Node) -> (bool):

        wellFormed = all([
            'dilations' in node.attrs,
            'group' in node.attrs,
            'kernel_shape' in node.attrs,
            'pads' in node.attrs,
            'strides' in node.attrs,
            len(node.outputs) == 1
        ])
        if self.noBiasHoisting:
            wellFormed = wellFormed

        if wellFormed:
            self.parserDict['group'] = node.attrs['group']
            self.parserDict['kernel_shape'] = node.attrs['kernel_shape']
            self.parserDict['pads'] = node.attrs['pads']
            self.parserDict['strides'] = node.attrs['strides']
            self.parserDict['dilations'] = node.attrs['dilations']

        return wellFormed

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        inputs = ['data_in', 'weight']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            if idx < len(inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name

        if len(node.inputs) == 3:
            self.parserDict['bias'] = ctxt.lookup(node.inputs[2].name).name
        else:
            if not self.noBiasHoisting:
                values = np.zeros((1))
                zeroTensor = gs.Constant(f'{node.name}_Bias_Tensor', values=values)
                ctxt.hoistConstant(zeroTensor)
                node.inputs.append(zeroTensor)
                self.parserDict['bias'] = f'{node.name}_Bias_Tensor'

        self.parserDict['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)

        return ctxt, True

class Conv2DParser(ConvParser):
    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node:  gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        ret = False

        if wellFormed:
            ret = all([
                # Make sure kernel is 2D
                len(node.attrs['kernel_shape']) == 2,
                # Make sure strides are 2D
                len(node.attrs['strides']) == 2,
                len(node.attrs['pads']) == 4,
                len(node.attrs['dilations']) == 2,
            ])

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:
            data_in = newCtxt.lookup(self.parserDict['data_in'])
            weight = newCtxt.lookup(self.parserDict['weight'])
            if len(data_in.shape) == 4 and len(weight.shape) == 4:
                return newCtxt, True

        return ctxt, False

class Conv1DParser(ConvParser):
    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node:  gs.Node) -> (bool):

        wellFormed = super().parseNode(node)
        ret = False

        if wellFormed:
            ret = all([
                # Make sure kernel is 2D
                len(node.attrs['kernel_shape']) == 1,
                # Make sure strides are 2D
                len(node.attrs['strides']) == 1,
                len(node.attrs['pads']) == 2,
                len(node.attrs['dilations']) == 1,
            ])

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        newCtxt, ret = super().parseNodeCtxt(ctxt, node)

        if ret:
            data_in = newCtxt.lookup(self.parserDict['data_in'])
            weight = newCtxt.lookup(self.parserDict['weight'])
            if len(data_in.shape) == 3 and len(weight.shape) == 3:
                return newCtxt, True

        return ctxt, False


class MHSAParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> (bool):

        ret = all([
            'preattn_requant_mul' in node.attrs,
            'preattn_requant_div' in node.attrs,
            'postattn_requant_mul' in node.attrs,
            'postattn_requant_div' in node.attrs,
            'wo_requant_mul' in node.attrs,
            'wo_requant_div' in node.attrs,
            'wq_requant_mul' in node.attrs,
            'wq_requant_div' in node.attrs,
            'wk_requant_mul' in node.attrs,
            'wk_requant_div' in node.attrs,
            'wv_requant_mul' in node.attrs,
            'wv_requant_div' in node.attrs,
            'isoftmaxA' in node.attrs,
            'isoftmaxB' in node.attrs,
            'isoftmaxC' in node.attrs,
            'isoftmaxlog2' in node.attrs,
            'n_levels' in node.attrs,
            'dim' in node.attrs,
            'dim_head' in node.attrs,
            'heads' in node.attrs,
            len(node.inputs) == 11,
            len(node.outputs) == 1
        ])


        if ret:
            self.parserDict['preattn_requant_mul'] = int(node.attrs['preattn_requant_mul'].values)
            self.parserDict['preattn_requant_shift'] = int(node.attrs['preattn_requant_shift'].values)
            self.parserDict['preattn_requant_div'] = int(math.log2(int(node.attrs['preattn_requant_div'].values)))
            self.parserDict['postattn_requant_mul'] = int(node.attrs['postattn_requant_mul'].values)
            self.parserDict['postattn_requant_shift'] = int(node.attrs['postattn_requant_shift'].values)
            self.parserDict['postattn_requant_div'] = int(math.log2(int(node.attrs['postattn_requant_div'].values)))
            self.parserDict['wo_requant_mul'] = int(node.attrs['wo_requant_mul'].values)
            self.parserDict['wo_requant_shift'] = int(node.attrs['wo_requant_shift'].values)
            self.parserDict['wo_requant_div'] = int(math.log2(int(node.attrs['wo_requant_div'].values)))
            self.parserDict['wq_requant_mul'] = int(node.attrs['wq_requant_mul'].values)
            self.parserDict['wq_requant_shift'] = int(node.attrs['wq_requant_shift'].values)
            self.parserDict['wq_requant_div'] = int(math.log2(int(node.attrs['wq_requant_div'].values)))
            self.parserDict['wk_requant_mul'] = int(node.attrs['wk_requant_mul'].values)
            self.parserDict['wk_requant_shift'] = int(node.attrs['wk_requant_shift'].values)
            self.parserDict['wk_requant_div'] = int(math.log2(int(node.attrs['wk_requant_div'].values)))
            self.parserDict['wv_requant_mul'] = int(node.attrs['wv_requant_mul'].values)
            self.parserDict['wv_requant_shift'] = int(node.attrs['wv_requant_shift'].values)
            self.parserDict['wv_requant_div'] = int(math.log2(int(node.attrs['wv_requant_div'].values)))
            self.parserDict['isoftmaxA'] = int(node.attrs['isoftmaxA'].values)
            self.parserDict['isoftmaxB'] = int(node.attrs['isoftmaxB'].values)
            self.parserDict['isoftmaxC'] = int(node.attrs['isoftmaxC'].values)
            self.parserDict['isoftmaxlog2'] = int(node.attrs['isoftmaxlog2'].values)
            self.parserDict['n_levels'] = int(node.attrs['n_levels'].values)
            self.parserDict['dim'] = int(node.attrs['dim'].values)
            self.parserDict['dim_head'] = int(node.attrs['dim_head'].values)
            self.parserDict['heads'] = int(node.attrs['heads'].values)

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        inputs = ['q', 'k', 'v', 'wq_weight', 'wq_bias', 'wk_weight', 'wk_bias' , 'wv_weight', 'wv_bias', 'wo_weight', 'wo_bias']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.parserDict['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.parserDict['q_shape'] = ctxt.lookup(node.inputs[0].name).shape

        return ctxt, True

class LinearAttentionParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> (bool):

        ret = all([
            'preattn_requant_mul' in node.attrs,
            'preattn_requant_div' in node.attrs,
            'normalizer_requant_mul' in node.attrs,
            'normalizer_requant_div' in node.attrs,
            'postattn_requant_mul' in node.attrs,
            'postattn_requant_div' in node.attrs,
            'wo_requant_mul' in node.attrs,
            'wo_requant_div' in node.attrs,
            'wq_requant_mul' in node.attrs,
            'wq_requant_div' in node.attrs,
            'wk_requant_mul' in node.attrs,
            'wk_requant_div' in node.attrs,
            'wv_requant_mul' in node.attrs,
            'wv_requant_div' in node.attrs,
            'Delta' in node.attrs,
            'eps' in node.attrs,
            'act_type' in node.attrs,
            'n_levels' in node.attrs,
            'dim' in node.attrs,
            'dim_head' in node.attrs,
            'heads' in node.attrs,
            len(node.inputs) == 11,
            len(node.outputs) == 1
        ])


        if ret:
            self.parserDict['preattn_requant_mul'] = int(node.attrs['preattn_requant_mul'].values)
            self.parserDict['preattn_requant_shift'] = int(node.attrs['preattn_requant_shift'].values)
            self.parserDict['preattn_requant_div'] = int(math.log2(int(node.attrs['preattn_requant_div'].values)))
            self.parserDict['normalizer_requant_mul'] = int(node.attrs['normalizer_requant_mul'].values)
            self.parserDict['normalizer_requant_shift'] = int(node.attrs['normalizer_requant_shift'].values)
            self.parserDict['normalizer_requant_div'] = int(math.log2(int(node.attrs['normalizer_requant_div'].values)))
            self.parserDict['postattn_requant_mul'] = int(node.attrs['postattn_requant_mul'].values)
            self.parserDict['postattn_requant_shift'] = int(node.attrs['postattn_requant_shift'].values)
            self.parserDict['postattn_requant_div'] = int(math.log2(int(node.attrs['postattn_requant_div'].values)))
            self.parserDict['wo_requant_mul'] = int(node.attrs['wo_requant_mul'].values)
            self.parserDict['wo_requant_shift'] = int(node.attrs['wo_requant_shift'].values)
            self.parserDict['wo_requant_div'] = int(math.log2(int(node.attrs['wo_requant_div'].values)))
            self.parserDict['wq_requant_mul'] = int(node.attrs['wq_requant_mul'].values)
            self.parserDict['wq_requant_shift'] = int(node.attrs['wq_requant_shift'].values)
            self.parserDict['wq_requant_div'] = int(math.log2(int(node.attrs['wq_requant_div'].values)))
            self.parserDict['wk_requant_mul'] = int(node.attrs['wk_requant_mul'].values)
            self.parserDict['wk_requant_shift'] = int(node.attrs['wk_requant_shift'].values)
            self.parserDict['wk_requant_div'] = int(math.log2(int(node.attrs['wk_requant_div'].values)))
            self.parserDict['wv_requant_mul'] = int(node.attrs['wv_requant_mul'].values)
            self.parserDict['wv_requant_shift'] = int(node.attrs['wv_requant_shift'].values)
            self.parserDict['wv_requant_div'] = int(math.log2(int(node.attrs['wv_requant_div'].values)))
            self.parserDict['Delta'] = int(node.attrs['Delta'])
            self.parserDict['eps'] = int(node.attrs['eps'])
            self.parserDict['act_type'] = int(node.attrs['act_type'])
            self.parserDict['n_levels'] = int(node.attrs['n_levels'].values)
            self.parserDict['dim'] = int(node.attrs['dim'].values)
            self.parserDict['dim_head'] = int(node.attrs['dim_head'].values)
            self.parserDict['heads'] = int(node.attrs['heads'].values)

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        inputs = ['q', 'k', 'v', 'wq_weight', 'wq_bias', 'wk_weight', 'wk_bias' , 'wv_weight', 'wv_bias', 'wo_weight', 'wo_bias']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.parserDict['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.parserDict['q_shape'] = ctxt.lookup(node.inputs[0].name).shape

        return ctxt, True

class CLCAParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> (bool):

        ret = all([
            'Delta' in node.attrs,
            'eps' in node.attrs,
            'eta' in node.attrs,
            'act_type' in node.attrs,
            'n_levels' in node.attrs,
            'dim' in node.attrs,
            'dim_head' in node.attrs,
            'out_dim' in node.attrs,
            'heads' in node.attrs,
            len(node.inputs) == 29,
            len(node.outputs) == 1
        ])


        if ret:
            self.parserDict['Delta'] = int(node.attrs['Delta'])
            self.parserDict['eps'] = int(node.attrs['eps'])
            self.parserDict['eta'] = int(node.attrs['eta'])
            self.parserDict['act_type'] = int(node.attrs['act_type'])
            self.parserDict['n_levels'] = int(node.attrs['n_levels'].values)
            self.parserDict['dim'] = int(node.attrs['dim'].values)
            self.parserDict['dim_head'] = int(node.attrs['dim_head'].values)
            self.parserDict['out_dim'] = int(node.attrs['out_dim'].values)
            self.parserDict['heads'] = int(node.attrs['heads'].values)

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        inputs = ['q', 'k', 'wq_weight', 'wq_bias', 'wk_weight', 'wk_bias' , 'wo_weight', 'wo_bias',
                  'wq_requant_mul', 'wq_requant_add', 'wq_requant_div',
                  'wk_requant_mul', 'wk_requant_add', 'wk_requant_div',
                  'wv_requant_mul', 'wv_requant_add', 'wv_requant_div',
                  'kdiv_requant_mul', 'kdiv_requant_add', 'kdiv_requant_div',
                  'preattn_requant_mul', 'preattn_requant_add', 'preattn_requant_div',
                  'postattn_requant_mul', 'postattn_requant_add', 'postattn_requant_div',
                  'wo_requant_mul', 'wo_requant_add', 'wo_requant_div']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.parserDict['input_size_Q'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.parserDict['input_size_KV'] = np.prod(ctxt.lookup(node.inputs[1].name).shape)
        self.parserDict['q_shape'] = ctxt.lookup(node.inputs[0].name).shape
        self.parserDict['kv_shape'] = ctxt.lookup(node.inputs[1].name).shape

        return ctxt, True


class iLayerNormParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> (bool):

        ret = all([
            'D' in node.attrs,
            'n_levels' in node.attrs,
            len(node.inputs) == 3,
            len(node.outputs) == 1
        ])


        if ret:
            self.parserDict['n_levels'] = int(node.attrs['n_levels'].values)
            self.parserDict['log2D'] = int(math.log2(node.attrs['D'].values))

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()

        inputs = ['data_in', 'weight', 'bias']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.parserDict['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.parserDict['lastDimLength'] = ctxt.lookup(node.inputs[0].name).shape[-1]

        return ctxt, True

class MatMulParser(NodeParser):
    def __init__(self, noBiasHoisting = True):
        super().__init__()
        self.noBiasHoisting = noBiasHoisting
    def parseNode(self, node:  gs.Node) -> (bool):

        ret = all([
            len(node.inputs) >= 2,
            len(node.outputs) == 1
        ])

        # Assign GEMM-like attributes to be able to reuse same kernel binding
        if ret:
            self.parserDict['alpha'] = 1
            self.parserDict['beta'] = 1
            self.parserDict['transB'] = 0
            self.parserDict['transA'] = 0

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt = ctxt.copy()
        ret = True

        inputs = ['A', 'B']
        outputs = ['data_out']

        for idx, inputNode in enumerate(node.inputs):
            if idx < len(inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name

        # Create fake C node for GEMM-compatibility and hoist it
        if not self.noBiasHoisting:
            values = np.zeros((1))
            zeroTensor = gs.Constant(f'{node.name}_C_Tensor', values=values)
            ctxt.hoistConstant(zeroTensor)
            node.inputs.append(zeroTensor)
            self.parserDict['C'] = f'{node.name}_C_Tensor'

        self.parserDict['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)
        self.parserDict['A_shape'] = ctxt.lookup(node.inputs[0].name).shape
        self.parserDict['B_shape'] = ctxt.lookup(node.inputs[1].name).shape
        self.parserDict['M'] = ctxt.lookup(node.inputs[0].name).shape[(-2 + self.parserDict['transA'])]
        self.parserDict['N'] = ctxt.lookup(node.inputs[0].name).shape[(-1 - self.parserDict['transA'])]
        self.parserDict['O'] = ctxt.lookup(node.inputs[1].name).shape[(-1 - self.parserDict['transB'])]

        # SCHEREMO: Assert that reduction dimension is the same on both matrices
        ret = ret and (self.parserDict['N'] == ctxt.lookup(node.inputs[1].name).shape[-2 + self.parserDict['transB']])

        self.parserDict['batch'] = np.prod(ctxt.lookup(node.inputs[0].name).shape[:-2])

        # SCHEREMO: Assert that batch is the same on both matrices
        ret = ret and (self.parserDict['batch'] == np.prod(ctxt.lookup(node.inputs[1].name).shape[:-2]))

        return ctxt, ret

# This parser combines Matmul nodes and GEMM nodes to the more general GEMM nodes
class GEMMParser(MatMulParser):
    def __init__(self, noBiasHoisting = True):
        self.noBiasHoisting = noBiasHoisting
        super().__init__()

    def parseNode(self, node:  gs.Node) -> (bool):

        ret = all([
            len(node.inputs) >= 2,
            len(node.outputs) == 1,
            'alpha' in node.attrs,
            'beta' in node.attrs,
        ])

        # This is a GEMM node:
        if ret:
            self.parserDict['alpha'] = node.attrs['alpha']
            self.parserDict['beta'] = node.attrs['beta']

            if 'transA' in node.attrs:
                self.parserDict['transA'] = node.attrs['transA']
            else:
                self.parserDict['transA'] = 0

            if 'transB' in node.attrs:
                self.parserDict['transB'] = node.attrs['transB']
            else:
                self.parserDict['transB'] = 0

            return True
        # This might be a matmul node -> Cast up
        else:
            return False


    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        ctxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)

        # We are a true GEMM
        if wellFormed:
            inputs = ['A', 'B']
            outputs = ['data_out']

            for idx, inputNode in enumerate(node.inputs):
                if idx < len(inputs):
                    self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
            for idx, outputNode in enumerate(node.outputs):
                self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name

            if len(node.inputs) == 3:
                self.parserDict['C'] = ctxt.lookup(node.inputs[2].name).name
            elif not self.noBiasHoisting:
                values = np.zeros((1))
                zeroTensor = gs.Constant(f'{node.name}_C_Tensor', values=values)
                ctxt.hoistConstant(zeroTensor)
                self.parserDict['C'] = f'{node.name}_C_Tensor'

            self.parserDict['size'] = np.prod(ctxt.lookup(node.inputs[0].name).shape)

            return ctxt, True

        # We are a matmul, so behave like one
        else:
            return ctxt, False

class DummyParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        return True

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = []
        outputs = []
        for i in node.inputs:
            inputs.append(ctxt.lookup(i.name))
        for i in node.outputs:
            outputs.append(ctxt.lookup(i.name))

        self.parserDict['data_in'] = inputs[0].name
        self.parserDict['data_out'] = outputs[0].name
        self.parserDict['size'] = np.prod(inputs[0].shape)

        return ctxt, True

class IntegerDivParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:

        ret = all([
            len(node.inputs) >= 2,
            len(node.outputs) == 1,
            'Delta' in node.attrs,
            'eps' in node.attrs,
            'eta' in node.attrs,
        ])

        if ret:
            self.parserDict['Delta'] = node.attrs['Delta']
            self.parserDict['eps'] = node.attrs['eps']
            self.parserDict['eta'] = node.attrs['eta']

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        inputs = ["A", "B"]
        outputs = ["C"]
        for idx, inputNode in enumerate(node.inputs):
            if idx < len(inputs):
                self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name

        self.parserDict['sizeA'] = np.prod(ctxt.lookup(self.parserDict['A']).shape)
        self.parserDict['sizeB'] = np.prod(ctxt.lookup(self.parserDict['B']).shape)

        for idx,(a,b) in enumerate(zip(ctxt.lookup(self.parserDict['A']).shape, ctxt.lookup(self.parserDict['B']).shape)):
            if a != b:
                self.parserDict['nomStep'] = np.prod(ctxt.lookup(self.parserDict['A']).shape[idx:])
                self.parserDict['denomStep'] = np.prod(ctxt.lookup(self.parserDict['B']).shape[idx:])
                break

        return ctxt, True

class RQIntegerDivParser(IntegerDivParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node:  gs.Node) -> bool:
        ret = super().parseNode(node)

        wellFormed = all([
            len(node.inputs) == 5,
        ])

        if ret:
            return wellFormed

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node:  gs.Node, channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        ctxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        inputs = ["A", "B", "requant_mul", "requant_add", "requant_div"]
        outputs = ["C"]
        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(inputNode.name).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(outputNode.name).name

        return ctxt, ret
