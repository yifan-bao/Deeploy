# ----------------------------------------------------------------------
#
# File: MemPoolPasses.py
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

import copy
import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import *
from Deeploy.Layers.BasicLayers import *
from Deeploy.OptimizationPasses.PassClasses import *


def fuse_mhsa_fun(ctxt: NetworkContext, graph: gs.Graph, match: Match, name: str):
    # matched_nodes = [m for k, m in match.nodes_map.items()]

    def get_named_node(nodes_map: Dict, name: str):
        if name in nodes_map:
            return nodes_map[name]
        return None

    Projection_q = get_named_node(match.nodes_map, 'Projection_q')
    Bias_Pq = get_named_node(match.nodes_map, 'Bias_Pq')
    RequantShift_Pq = get_named_node(match.nodes_map, 'RequantShift_Pq')
    Reshape_Pq = get_named_node(match.nodes_map, 'Reshape_Pq')
    Transpose_Pq = get_named_node(match.nodes_map, 'Transpose_Pq')
    Projection_k = get_named_node(match.nodes_map, 'Projection_k')
    Bias_Pk = get_named_node(match.nodes_map, 'Bias_Pk')
    RequantShift_Pk = get_named_node(match.nodes_map, 'RequantShift_Pk')
    Reshape_Pk = get_named_node(match.nodes_map, 'Reshape_Pk')
    Transpose_Pk = get_named_node(match.nodes_map, 'Transpose_Pk')
    Projection_v = get_named_node(match.nodes_map, 'Projection_v')
    Bias_Pv = get_named_node(match.nodes_map, 'Bias_Pv')
    RequantShift_Pv = get_named_node(match.nodes_map, 'RequantShift_Pv')
    Reshape_Pv = get_named_node(match.nodes_map, 'Reshape_Pv')
    Transpose_Pv = get_named_node(match.nodes_map, 'Transpose_Pv')
    MatMul_a = get_named_node(match.nodes_map, 'MatMul_a')
    RequantShift_a = get_named_node(match.nodes_map, 'RequantShift_a')
    IntegerDiv_a = get_named_node(match.nodes_map, 'IntegerDiv_a')
    Softmax_a = get_named_node(match.nodes_map, 'Softmax_a')
    MatMul_o = get_named_node(match.nodes_map, 'MatMul_o')
    RequantShift_o = get_named_node(match.nodes_map, 'RequantShift_o')
    # Transpose_o = get_named_node(match.nodes_map,'Transpose_o')
    # Reshape_Po = get_named_node(match.nodes_map,'Reshape_Po')
    Projection_Po = get_named_node(match.nodes_map, 'Projection_Po')
    Bias_Po = get_named_node(match.nodes_map, 'Bias_Po')
    RequantShift_Po = get_named_node(match.nodes_map, 'RequantShift_Po')
    ReduceSum_Po = get_named_node(match.nodes_map, 'ReduceSum_Po')

    # Check if we accidentally swapped Q and K
    if Transpose_Pq.attrs['perm'] != Transpose_Pv.attrs['perm']:
        Projection_q = get_named_node(match.nodes_map, 'Projection_k')
        Bias_Pq = get_named_node(match.nodes_map, 'Bias_Pk')
        RequantShift_Pq = get_named_node(match.nodes_map, 'RequantShift_Pk')
        Reshape_Pq = get_named_node(match.nodes_map, 'Reshape_Pk')
        Transpose_Pq = get_named_node(match.nodes_map, 'Transpose_Pk')
        Projection_k = get_named_node(match.nodes_map, 'Projection_q')
        Bias_Pk = get_named_node(match.nodes_map, 'Bias_Pq')
        RequantShift_Pk = get_named_node(match.nodes_map, 'RequantShift_Pq')
        Reshape_Pk = get_named_node(match.nodes_map, 'Reshape_Pq')
        Transpose_Pk = get_named_node(match.nodes_map, 'Transpose_Pq')

        assert Transpose_Pq.attrs['perm'] == Transpose_Pv.attrs[
            'perm'], "[FuseMHSAPass] MHSA key and value permutation is not the same!"

    attrs = {}
    attrs['n_levels'] = RequantShift_Po.attrs['n_levels_out'].values.reshape(1)
    attrs['signed'] = RequantShift_Po.attrs['signed'].values.reshape(1)
    attrs['heads'] = Reshape_Pq.inputs[1].values[2]
    attrs['dim_head'] = Reshape_Pq.inputs[1].values[-1]  # Projection Size
    attrs['dim'] = Bias_Pq.inputs[1].shape[-2]  # Sequence Length

    # attrs['S'] = Bias_Pq.inputs[1].shape[-2] # Sequence Length
    # attrs['E'] = Projection_q.inputs[1].shape[0] # Embedding Size
    # attrs['P'] = Reshape_Pq.inputs[1].values[-1] # Projection Size

    attrs['wq_requant_mul'] = RequantShift_Pq.inputs[2].values.reshape(1)
    attrs['wk_requant_mul'] = RequantShift_Pk.inputs[2].values.reshape(1)
    attrs['wv_requant_mul'] = RequantShift_Pv.inputs[2].values.reshape(1)
    attrs['wo_requant_mul'] = RequantShift_Po.inputs[2].values.reshape(1)

    # WIESEP: We also have to handle the integer div node!
    if IntegerDiv_a is not None:
        attrs['preattn_requant_mul'] = np.round(RequantShift_a.inputs[2].values.reshape(1) /
                                                IntegerDiv_a.inputs[1].values.reshape(1))
    else:
        attrs['preattn_requant_mul'] = RequantShift_a.inputs[2].values.reshape(1)

    attrs['postattn_requant_mul'] = RequantShift_o.inputs[2].values.reshape(1)

    attrs['wq_requant_div'] = RequantShift_Pq.attrs['div'].values.reshape(1)
    attrs['wk_requant_div'] = RequantShift_Pk.attrs['div'].values.reshape(1)
    attrs['wv_requant_div'] = RequantShift_Pv.attrs['div'].values.reshape(1)
    attrs['wo_requant_div'] = RequantShift_Po.attrs['div'].values.reshape(1)
    attrs['preattn_requant_div'] = RequantShift_a.attrs['div'].values.reshape(1)
    attrs['postattn_requant_div'] = RequantShift_o.attrs['div'].values.reshape(1)

    _inputs = []
    _inputs.append(Projection_q.inputs[0])
    _inputs.append(Projection_k.inputs[0])
    _inputs.append(Projection_v.inputs[0])

    def get_constant_input(n: gs.Node):
        for input in n.inputs:
            if isinstance(input, gs.Constant):
                return input.values
        assert False, f"Did not find constant input for {n} node"

    # WIESEP: Should we also handle these requant shift add parameters or assume them to be zero?
    # WIEESP: Not yet handled by ITA
    attrs['wq_requant_add'] = RequantShift_Pq.inputs[1].values.reshape(1)
    attrs['wk_requant_add'] = RequantShift_Pk.inputs[1].values.reshape(1)
    attrs['wv_requant_add'] = RequantShift_Pv.inputs[1].values.reshape(1)
    attrs['wo_requant_add'] = RequantShift_Po.inputs[1].values.reshape(1)
    attrs['preattn_requant_add'] = RequantShift_a.inputs[1].values.reshape(1)
    attrs['postattn_requant_add'] = RequantShift_o.inputs[1].values.reshape(1)

    if not np.all([
            attrs['wq_requant_add'] == 0,
            attrs['wk_requant_add'] == 0,
            attrs['wv_requant_add'] == 0,
            attrs['wo_requant_add'] == 0,
            attrs['preattn_requant_add'] == 0,
            attrs['postattn_requant_add'] == 0,
    ]):
        print(
            "[FuseMHSAPass] Non-zero bias for pre and post requantization is not supported! Inference result may differ from the expected value."
        )

    _inputs += [gs.Constant(name = name + '_wq_weight', values = get_constant_input(Projection_q))]
    _inputs += [gs.Constant(name = name + '_wq_bias', values = get_constant_input(Bias_Pq))]

    _inputs += [gs.Constant(name = name + '_wk_weight', values = get_constant_input(Projection_k))]
    _inputs += [gs.Constant(name = name + '_wk_bias', values = get_constant_input(Bias_Pk))]

    _inputs += [gs.Constant(name = name + '_wv_weight', values = get_constant_input(Projection_v))]
    _inputs += [gs.Constant(name = name + '_wv_bias', values = get_constant_input(Bias_Pv))]

    _inputs += [gs.Constant(name = name + '_wo_weight', values = get_constant_input(Projection_Po))]
    _inputs += [gs.Constant(name = name + '_wo_bias', values = get_constant_input(Bias_Po))]

    _outputs = ReduceSum_Po.outputs

    mhsa = gs.Node(op = 'MHSA', name = name, attrs = attrs)
    graph.replaceInsertNode(_inputs, _outputs, mhsa)

    return ctxt, graph


class FuseMHSAPass(ReplaceSequentialPatternPass):

    def __init__(self, IntegerDiv = False, preSoftMaxRQ = True):
        graph = gs.Graph()
        _input_q = gs.Variable(name = 'input_q')
        _input_k = gs.Variable(name = 'input_k')
        _input_v = gs.Variable(name = 'input_v')

        # Query Projection
        output_q = graph.layer(inputs = [_input_q], outputs = ['pQ'], op = 'MatMul', name = 'Projection_q')
        output_q = graph.layer(inputs = output_q, outputs = ['pQ_b'], op = 'Add', name = 'Bias_Pq')
        output_q = graph.layer(inputs = output_q, outputs = ['pQ_rq'], op = 'RequantShift', name = 'RequantShift_Pq')
        output_q = graph.layer(inputs = output_q, outputs = ['pQ_r'], op = 'Reshape', name = 'Reshape_Pq')
        output_q = graph.layer(inputs = output_q, outputs = ['pQ_t'], op = 'Transpose', name = 'Transpose_Pq')

        # Key Projection
        output_k = graph.layer(inputs = [_input_k], outputs = ['pK'], op = 'MatMul', name = 'Projection_k')
        output_k = graph.layer(inputs = output_k, outputs = ['pK_b'], op = 'Add', name = 'Bias_Pk')
        output_k = graph.layer(inputs = output_k, outputs = ['pK_rq'], op = 'RequantShift', name = 'RequantShift_Pk')
        output_k = graph.layer(inputs = output_k, outputs = ['pK_r'], op = 'Reshape', name = 'Reshape_Pk')
        output_k = graph.layer(inputs = output_k, outputs = ['pK_t'], op = 'Transpose', name = 'Transpose_Pk')

        # Value Projection
        output_v = graph.layer(inputs = [_input_v], outputs = ['pV'], op = 'MatMul', name = 'Projection_v')
        output_v = graph.layer(inputs = output_v, outputs = ['pV_b'], op = 'Add', name = 'Bias_Pv')
        output_v = graph.layer(inputs = output_v, outputs = ['pV_rq'], op = 'RequantShift', name = 'RequantShift_Pv')
        output_v = graph.layer(inputs = output_v, outputs = ['pV_r'], op = 'Reshape', name = 'Reshape_Pv')
        output_v = graph.layer(inputs = output_v, outputs = ['pV_t'], op = 'Transpose', name = 'Transpose_Pv')

        # Attention Matrix
        output_a = graph.layer(inputs = output_q + output_k, outputs = ['a'], op = 'MatMul', name = 'MatMul_a')
        if preSoftMaxRQ:
            output_a = graph.layer(inputs = output_a, outputs = ['a_rq'], op = 'RequantShift', name = 'RequantShift_a')

        if IntegerDiv:
            output_a = graph.layer(inputs = output_a, outputs = ['a_d'], op = 'IntegerDiv', name = 'IntegerDiv_a')

        output_a = graph.layer(inputs = output_a, outputs = ['a_s'], op = 'ITAPartialMax', name = 'Softmax_a')

        # Attention
        output = graph.layer(inputs = output_v + output_a, outputs = ['o'], op = 'MatMul', name = 'MatMul_o')
        output = graph.layer(inputs = output, outputs = ['o_rq'], op = 'RequantShift', name = 'RequantShift_o')
        # output = graph.layer(inputs=output, outputs=['o_t'], op='Transpose', name='Transpose_o')
        # output = graph.layer(inputs=output, outputs=['o_r'], op='Reshape', name='Reshape_Po')
        output = graph.layer(inputs = output, outputs = ['pO'], op = 'MatMul', name = 'Projection_Po')
        output = graph.layer(inputs = output, outputs = ['pO_b'], op = 'Add', name = 'Bias_Po')
        output = graph.layer(inputs = output, outputs = ['pO_rq'], op = 'RequantShift', name = 'RequantShift_Po')
        output = graph.layer(inputs = output, outputs = ['pO_rqs'], op = 'ReduceSum', name = 'ReduceSum_Po')

        graph.outputs.append(output)
        graph.inputs.append(_input_q)
        graph.inputs.append(_input_k)
        graph.inputs.append(_input_v)

        name = f"_FUSE_MHSA_PASS"
        super().__init__(graph, fuse_mhsa_fun, name, matcher = BranchingMatcher)
