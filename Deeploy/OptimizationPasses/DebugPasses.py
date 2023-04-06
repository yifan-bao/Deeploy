# ----------------------------------------------------------------------
#
# File: DebugPasses.py
#
# Last edited: 14.12.20122
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
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

import copy
from typing import NamedTuple, Literal
import onnx_graphsurgeon as gs
from functools import partial

from Deeploy.DeeployTypes import *
from Deeploy.Layers.BasicLayers import *
from Deeploy.OptimizationPasses.PassClasses import *


def debug_fun(ctxt: NetworkContext,
              graph: gs.Graph,
              match: Match,
              name: str,
              position: Literal["before", "after"] = "before"):
    assert position in ["before", "after"], f"'{position}' is not a valid position for the debug node!"

    ctxt = ctxt.copy()

    matched_nodes = [m for k, m in match.nodes_map.items()]

    node = matched_nodes[0]

    if position == 'before' and "DEBUG" not in node.inputs[0].name:
        newNodeInput = gs.Variable(name + '_input', dtype = np.float32, shape = node.inputs[0].shape)
        newDebugNode = gs.Node(op = 'Debug',
                               name = node.name + '_input',
                               inputs = [node.inputs[0]],
                               outputs = [newNodeInput])

        node1 = ctxt.VariableBuffer().fromNode(newNodeInput, ctxt.lookup(node.inputs[0].name).nLevels)
        node1._type = ctxt.lookup(node.inputs[0].name)._type
        ctxt.add(node1, 'local')

        node.inputs[0] = newNodeInput

        graph.nodes.append(newDebugNode)
        graph.cleanup().toposort()

    if position == 'after' and "DEBUG" not in node.outputs[0].name:
        newNodeOutput = gs.Variable(name + '_output', dtype = np.float32, shape = node.outputs[0].shape)
        # newDebugNode = gs.Node(op='Debug', name=name+'_debug',inputs=[newNodeOutput], outputs= [node.outputs[0]])
        newDebugNode = gs.Node(op = 'Debug',
                               name = node.name + '_output',
                               inputs = [newNodeOutput],
                               outputs = [node.outputs[0]])

        node1 = ctxt.VariableBuffer().fromNode(newNodeOutput, ctxt.lookup(node.outputs[0].name).nLevels)
        node1._type = ctxt.lookup(node.outputs[0].name)._type
        ctxt.add(node1, 'local')

        node.outputs[0] = newNodeOutput

        graph.nodes.append(newDebugNode)
        graph.cleanup().toposort()

    return ctxt, graph


class DebugPass(ReplaceSequentialPatternPass):

    def __init__(self, op: str = "", position = 'before'):

        if op == "":
            raise ValueError(f'Operator not set!')
        if position not in ['before', 'after']:
            ValueError(f'Invalid position "{position}"!')

        graph = gs.Graph()
        _input = gs.Variable(name = 'input_0')
        output = graph.layer(inputs = [_input], outputs = ['output0'], op = op, name = op.lower() + '1')
        graph.outputs.append(output)
        graph.inputs = [_input]

        name = "_DEBUG_" + position.upper() + "_" + op.upper() + "_PASS"
        super().__init__(graph, partial(debug_fun, position = position), name)


def merge_debug_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    d1 = matched_nodes[0]
    d2 = matched_nodes[1]

    _inputs = list(d1.inputs)
    _outputs = list(d2.outputs)

    newDebugNode = gs.Node(op = 'Debug', name = name)
    graph.replaceInsertNode(_inputs, _outputs, newDebugNode)

    graph.cleanup().toposort()
    return ctxt, graph


class DebugMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['d1_out'], op = 'Debug', name = 'd1')
        output = graph.layer(inputs = output, outputs = ['d2_out'], op = 'Debug', name = 'd2')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_DEBUG_PASS"
        super().__init__(graph, merge_debug_fun, name)


class EmulateCMSISRequantPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['output0'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = f"_EMULATE_CMSIS_REQUANT_PASS"
        super().__init__(graph, convert_requant_to_cmsis_fun, name)


def convert_requant_to_cmsis_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    rqs = matched_nodes[0]

    # Make sure pass is only applied once
    if 'Emulate_CMSIS_RequantShift' in rqs.attrs:
        return ctxt, graph

    # WIESEP: Because CMSIS performs add-multiply-divide and we normally do multiply-add-divide
    #         we can emulate the same behavior by rounding the MUL value
    rqs.inputs[-1].values = np.round(rqs.inputs[-1].values / (rqs.inputs[-2].values + 1e-3)) * rqs.inputs[-2].values
    rqs.attrs['emulate_CMSIS_requantShift'] = True

    return ctxt, graph
