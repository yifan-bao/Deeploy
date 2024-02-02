# ----------------------------------------------------------------------
#
# File: BasicPasses.py
#
# Last edited: 28.04.2023
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

import copy
from functools import partial
from typing import List

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.OptimizationPasses.Matchers import BranchingMatcher, Match
from Deeploy.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


def _merge_integerdiv_rq_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    integerdiv = matched_nodes[0]
    rqs = matched_nodes[1]
    totalShift = np.round(np.log2(rqs.attrs['div'].values))

    rqs.inputs[-1].values = np.round(rqs.inputs[-1].values / rqs.inputs[-2].values + 1e-3)  # normalize add

    shiftNode = gs.Constant(f'{integerdiv.name}_shift', np.array(totalShift))
    _inputs = list(integerdiv.inputs) + list(rqs.inputs[1:]) + [shiftNode]
    _outputs = rqs.outputs

    #import IPython; IPython.embed()

    rqsIntegerDiv = gs.Node(op = 'RQIntegerDiv', name = name, attrs = {**integerdiv.attrs, **rqs.attrs})
    graph.replaceInsertNode(_inputs, _outputs, rqsIntegerDiv)

    return graph


@contextagnostic
class IntegerDivRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['integerdiv_out'], op = 'IntegerDiv', name = 'integerdiv')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_INTEGERDIV_PASS"
        super().__init__(graph, _merge_integerdiv_rq_fun, name)


def _merge_igelu_rq_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    igelu = matched_nodes[0]
    rqs = matched_nodes[1]
    totalShift = np.round(np.log2(rqs.attrs['div'].values))

    rqs.inputs[-1].values = np.round(rqs.inputs[-1].values / rqs.inputs[-2].values + 1e-3)  # normalize add

    shiftNode = gs.Constant(f'{igelu.name}_shift', np.array(totalShift))
    _inputs = list(igelu.inputs) + list(rqs.inputs[1:]) + [shiftNode]
    _outputs = rqs.outputs

    #import IPython; IPython.embed()

    rqsiGELU = gs.Node(op = 'RequantizediGELU', name = name, attrs = {**igelu.attrs, **rqs.attrs})
    graph.replaceInsertNode(_inputs, _outputs, rqsiGELU)

    return graph


@contextagnostic
class iGELURequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['igelu_out'], op = 'iGELU', name = 'igelu')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = f"_MERGE_iGELURQ_PASS"
        super().__init__(graph, _merge_igelu_rq_fun, name)


def _merge_rqs_add_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    add = matched_nodes[0]
    rqs = matched_nodes[1]

    if (isinstance(add.inputs[0], gs.Constant) or isinstance(add.inputs[1], gs.Constant)) and isinstance(
            rqs.inputs[2], gs.Constant):
        if isinstance(add.inputs[0], gs.Constant):
            idx = 1  # Non-constant idx
            constantTensor = add.inputs[0]
        else:
            idx = 0  # non-constant idx
            constantTensor = add.inputs[1]
        if constantTensor.values.shape != tuple(add.outputs[0].shape):
            rqs.inputs[2].values = (rqs.inputs[1].values * constantTensor.values) + rqs.inputs[2].values
            add.inputs[(idx + 1) % 2].values = add.inputs[(idx + 1) % 2].values * 0
            rqs.inputs[0] = add.inputs[idx]
        return graph
    else:
        return graph


@contextagnostic
class MergeConstAddAndRequantPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['add_out'], op = 'Add', name = 'add1')
        output = graph.layer(inputs = output, outputs = ['rqs_out'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs = [_input]

        name = "_MERGE_RQS_ADD_PASS"
        super().__init__(graph, _merge_rqs_add_fun, name)


def _extract_padding_fun(graph: gs.Graph, match: Match, name: str, value = 0):

    matched_nodes = [m for k, m in match.nodes_map.items()]
    conv = matched_nodes[0]
    if 'pads' in conv.attrs and np.sum(conv.attrs['pads']) > 1:
        pads = copy.deepcopy(conv.attrs['pads'])
        shape = copy.deepcopy(conv.inputs[0].shape)
        newPads = np.zeros(2 * len(shape))
        assert len(shape) - 2 == len(pads) / 2, "Conv padding dims do not match!"
        newShape = shape

        beginPads = pads[0:len(pads) // 2]
        endPads = pads[len(pads) // 2:]
        for idx, i in enumerate(beginPads):
            newShape[2 + idx] = newShape[2 + idx] + i
            newPads[2 + idx] = i

        for idx, i in enumerate(endPads):
            newShape[2 + idx] = newShape[2 + idx] + i
            newPads[len(newPads) // 2 + 2 + idx] = i

        newConvInput = gs.Variable(name + '_padded_input', dtype = np.float32, shape = newShape)
        #valConst = gs.Constant('value', np.array(0))
        conv.attrs['pads'] = [0 for pad in conv.attrs['pads']]
        newPad = gs.Node(op = 'Pad',
                         name = name + '_pad',
                         attrs = {
                             'pads': newPads,
                             'mode': 'constant',
                             'value': value
                         },
                         inputs = [conv.inputs[0]],
                         outputs = [newConvInput])

        conv.inputs[0] = newConvInput
        graph.nodes.append(newPad)
        graph.cleanup().toposort()
        #import IPython; IPython.embed()

    return graph


@contextagnostic
class ExtractPaddingFromPoolPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['pool_out'], op = 'MaxPool', name = 'maxpool1')
        graph.outputs.append(output)
        graph.inputs = [_input]

        name = "_EXTRACT_POOL_PASS"
        # SCHEREMO: This is a workaround!!!
        super().__init__(graph, partial(_extract_padding_fun, value = -128), name)


@contextagnostic
class ExtractPaddingFromConvPass(ReplaceSequentialPatternPass):

    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['conv_out'], op = 'Conv', name = 'conv1')
        graph.outputs.append(output)
        graph.inputs = [_input]

        name = "_EXTRACT_CONV_PASS"
        super().__init__(graph, _extract_padding_fun, name)


def _merge_matmul_add_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    gemm = matched_nodes[0]
    add = matched_nodes[1]
    _bias = add.inputs[0] if isinstance(add.inputs[0], gs.Constant) else add.inputs[1]
    _inputs = gemm.inputs + [_bias]
    _outputs = add.outputs

    rqsGemm = gs.Node(op = 'Gemm', name = name, attrs = {'alpha': 1.0, 'beta': 1.0})
    graph.replaceInsertNode(_inputs, _outputs, rqsGemm)

    return graph


@contextagnostic
class MatMulAddMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['gemm_out'], op = 'MatMul', name = 'gemm')
        output = graph.layer(inputs = output, outputs = ['add_out'], op = 'Add', name = 'add')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_MATMUL_ADD_PASS"
        super().__init__(graph, _merge_matmul_add_fun, name)


def _propagate_requant_fun(graph: gs.Graph, match: Match, name: str):

    matched_nodes = [m for k, m in match.nodes_map.items()]
    add = matched_nodes[0]
    rqs = matched_nodes[1]

    inputNode1 = add.inputs[0]
    inputNode2 = add.inputs[1]

    newAdd1 = gs.Constant(name = name + '_rqs1_add', values = rqs.inputs[2].values)
    newAdd2 = gs.Constant(name = name + '_rqs2_add', values = rqs.inputs[2].values)
    newMul1 = gs.Constant(name = name + '_rqs1_mul', values = rqs.inputs[1].values)
    newMul2 = gs.Constant(name = name + '_rqs2_mul', values = rqs.inputs[1].values)

    newAddInput1 = gs.Variable(name + '_add_in_1', dtype = np.float32, shape = inputNode1.shape)
    newAddInput2 = gs.Variable(name + '_add_in_2', dtype = np.float32, shape = inputNode2.shape)

    newRQS1 = gs.Node(op = 'RequantShift',
                      name = name + '_rqs1',
                      attrs = rqs.attrs,
                      inputs = [inputNode1, newMul1, newAdd1],
                      outputs = [newAddInput1])
    newRQS2 = gs.Node(op = 'RequantShift',
                      name = name + '_rqs2',
                      attrs = rqs.attrs,
                      inputs = [inputNode2, newMul2, newAdd2],
                      outputs = [newAddInput2])

    graph.nodes.append(newRQS1)
    graph.nodes.append(newRQS2)

    add.inputs = [newAddInput1, newAddInput2]
    graph.deleteNode(rqs)

    return graph


@contextagnostic
class PropagateRequantThroughAddPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        _input2 = gs.Variable(name = 'input_2')
        output = graph.layer(inputs = [_input, _input2], outputs = ['add_out'], op = 'Add', name = 'add1')
        output = graph.layer(inputs = output, outputs = ['r1_out'], op = 'RequantShift', name = 'r1')
        graph.outputs.append(output)
        graph.inputs = [_input, _input2]

        name = "_OPT_ADD_RQS_PASS"
        super().__init__(graph, _propagate_requant_fun, name)


def _merge_requant_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    attrs = {}
    rqs1 = matched_nodes[0]
    rqs2 = matched_nodes[1]

    div1 = rqs1.attrs['div'].values
    div2 = rqs2.attrs['div'].values
    newDiv = max(div1, div2)
    minDiv = min(div1, div2)
    nLevels = max(rqs1.attrs['n_levels_out'].values, rqs2.attrs['n_levels_out'].values)
    signed = max(rqs1.attrs['signed'].values, rqs2.attrs['signed'].values)

    attrs['div'] = gs.Constant(name = 'div', values = newDiv)
    attrs['n_levels'] = gs.Constant(name = 'n_levels', values = nLevels)
    attrs['signed'] = gs.Constant(name = 'signed', values = signed)

    if isinstance(rqs1.inputs[1], gs.Constant) and isinstance(rqs1.inputs[2], gs.Constant) and \
       isinstance(rqs2.inputs[1], gs.Constant) and isinstance(rqs2.inputs[2], gs.Constant):
        mul1 = rqs1.inputs[1].values
        mul2 = rqs2.inputs[1].values
        add1 = rqs1.inputs[2].values
        add2 = rqs2.inputs[2].values

        newMul = (mul1 * mul2)
        newAdd = (add1 * mul2) + (div1 * add2)

        newMul = gs.Constant(name = rqs1.name + name + '_mul', values = np.array(np.round(newMul / minDiv)))
        newAdd = gs.Constant(name = rqs1.name + name + '_add', values = np.array(np.round(newAdd / minDiv)))

        _inputs = [rqs1.inputs[0], newMul, newAdd]
        _outputs = rqs2.outputs
        newTrans = gs.Node(op = 'RequantShift', name = name, attrs = attrs)
        graph.replaceInsertNode(_inputs, _outputs, newTrans)
        return graph
    else:
        return graph


@contextagnostic
class MergeRequantPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['r1_out'], op = 'RequantShift', name = 'r1')
        output = graph.layer(inputs = output, outputs = ['r2_out'], op = 'RequantShift', name = 'r2')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_OPT_RQS_PASS"
        super().__init__(graph, _merge_requant_fun, name)


def _merge_transposes_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    t1 = matched_nodes[0]
    t2 = matched_nodes[1]

    #Transpose forth and back - delete both nodes

    if (t1.inputs[0].shape == t2.outputs[0].shape):
        # Find Nodes-to-be-replaced
        graph.deleteNode(t2)
        graph.deleteNode(t1)
        graph.cleanup().toposort()
        return graph
    # Net the transpose
    else:
        p1 = t1.attrs['perm']
        p2 = t2.attrs['perm']
        newPerm = [p1[idx] for idx in p2]

    _inputs = list(t1.inputs)
    _outputs = list(t2.outputs)

    newTrans = gs.Node(op = 'Transpose', name = name, attrs = {"perm": newPerm})
    graph.replaceInsertNode(_inputs, _outputs, newTrans)
    graph.cleanup().toposort()
    return graph


@contextagnostic
class TransposeMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['t1_out'], op = 'Transpose', name = 't1')
        output = graph.layer(inputs = output, outputs = ['t2_out'], op = 'Transpose', name = 't2')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_TRANSPOSES_PASS"
        super().__init__(graph, _merge_transposes_fun, name)


def _split_transposes_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    t1 = matched_nodes[0]

    if len(t1.outputs[0].outputs) <= 1:
        return graph

    perm = t1.attrs['perm']
    inputVar = t1.inputs[0]
    inputNode = t1.inputs[0].inputs[0]

    originalNode = t1.outputs[0]

    postSplitOutput = gs.Variable(name = f"{t1.outputs[0].name}_split", dtype = np.float32, shape = t1.inputs[0].shape)
    inputNode.outputs = [postSplitOutput]

    for node in originalNode.outputs.copy():
        nodeName = node.name + f"_transpose_in"
        varName = node.name + f"_transpose_in_var"
        newOutput = gs.Variable(name = varName, dtype = np.float32, shape = t1.outputs[0].shape)

        transposeNode = gs.Node(name = nodeName,
                                op = "Transpose",
                                inputs = [postSplitOutput],
                                outputs = [newOutput],
                                attrs = {'perm': perm})

        graph.nodes.append(transposeNode)

        newNodeInputs = []
        for _input in node.inputs:
            if _input != originalNode:
                newNodeInputs.append(_input)
            else:
                newNodeInputs.append(newOutput)

        node.inputs = newNodeInputs

    t1.outputs = []
    t1.inputs = []

    graph.cleanup().toposort()
    return graph


@contextagnostic
class TransposeSplitPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['t1_out'], op = 'Transpose', name = 't1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_SPLIT_TRANSPOSES_PASS"
        super().__init__(graph, _split_transposes_fun, name)


def _const_opt_transposes_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    t1 = matched_nodes[0]

    if isinstance(t1.inputs[0], gs.Constant):
        t1.inputs[0].values = np.transpose(t1.inputs[0].values, t1.attrs['perm'])
        graph.deleteNode(t1)

    return graph


@contextagnostic
class TransposeConstOptPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['t1_out'], op = 'Transpose', name = 't1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_CONST_OPT_TRANSPOSES_PASS"
        super().__init__(graph, _const_opt_transposes_fun, name)
