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

from parserTypes import *
from parserTypes import _mangleVariableName, _mangleParameterName
from templates import *
import numpy as np
import math

class AddParser(NodeParser):
    def __init__(self):
        super().__init__()

    def nodeParse(self, node: gs.ir.node.Node) -> bool:

        ret = all([
            len(node.inputs) == 2,
            len(node.outputs) == 1
        ])

        return ret

    def nodeCtxtParse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):

        ctxt = ctxt.copy()
        
        data_in_1 = ctxt.lookup(_mangleVariableName(node.inputs[0].name))
        data_in_2 = ctxt.lookup(_mangleVariableName(node.inputs[1].name))
        data_out = ctxt.lookup(_mangleVariableName(node.outputs[0].name))
        self.parserDict['data_in_1'] = data_in_1.name
        self.parserDict['data_in_2'] = data_in_2.name
        self.parserDict['data_out'] = data_out.name
        self.parserDict['size'] = np.prod(data_in_1.shape)
        
        return ctxt, True


class GELUParser(NodeParser):
    def __init__(self):
        super().__init__()

    def nodeParse(self, node: gs.ir.node.Node) -> bool:

        ret = all([
            'b' in node.attrs,
            'D' in node.attrs,
            'n_levels' in node.attrs,
            'one' in node.attrs,
            'totScaler' in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1
        ])

        if ret:
            self.parserDict['b'] = int(node.attrs['b'].values)
            self.parserDict['log2D'] = int(math.log2(node.attrs['D'].values))
            self.parserDict['n_levels'] = int(node.attrs['n_levels'].values)
            self.parserDict['one'] = int(node.attrs['one'].values)
            self.parserDict['totScaler'] = int(node.attrs['totScaler'].values)

        return ret

    def nodeCtxtParse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):

        ctxt = ctxt.copy()
        
        data_in = ctxt.lookup(_mangleVariableName(node.inputs[0].name))
        data_out = ctxt.lookup(_mangleVariableName(node.outputs[0].name))
        self.parserDict['data_in'] = data_in.name
        self.parserDict['data_out'] = data_out.name
        self.parserDict['size'] = np.prod(data_in.shape)
        
        return ctxt, True
        
class RequantShiftParser(NodeParser):
    def __init__(self):
        super().__init__()

    def nodeParse(self, node: gs.ir.node.Node) -> (bool):

        ret = all([
            'div' in node.attrs,
            'n_levels' in node.attrs,
            'signed' in node.attrs,
            len(node.inputs) == 3,
            len(node.outputs) == 1
        ])
        

        if ret:
            self.parserDict['n_levels'] = int(node.attrs['n_levels'].values)
            self.parserDict['signed'] = int(node.attrs['signed'].values)
            self.parserDict['log2D'] = int(math.log2(node.attrs['div'].values))

        return ret
    
    def nodeCtxtParse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):

        ctxt = ctxt.copy()

        inputs = ['data_in', 'mul', 'add']
        outputs = ['data_out']
        
        for idx, inputNode in enumerate(node.inputs):
            self.parserDict[inputs[idx]] = ctxt.lookup(_mangleVariableName(inputNode.name)).name
        for idx, outputNode in enumerate(node.outputs):
            self.parserDict[outputs[idx]] = ctxt.lookup(_mangleVariableName(outputNode.name)).name
            
        self.parserDict['size'] = np.prod(ctxt.lookup(_mangleVariableName(node.inputs[0].name)).shape)

        return ctxt, True

class DummyParser(NodeParser):
    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.ir.node.Node) -> bool:
        return ret
        
    def nodeCtxtParse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):
        inputs = []
        outputs = []
        for i in node.inputs:
            inputs.append(ctxt.lookup(_mangleVariableName(i.name)))
        for i in node.outputs:
            outputs.append(ctxt.lookup(_mangleVariableName(i.name)))
            
        self.parserDict['data_in'] = inputs[0].name
        self.parserDict['data_out'] = outputs[0].name
        self.parserDict['size'] = np.prod(inputs[0].shape)
        
        return ctxt, True
