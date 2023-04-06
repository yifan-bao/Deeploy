# ----------------------------------------------------------------------
#
# File: PassClasses.py
#
# Last edited: 28.12.2021
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

import copy
from typing import NamedTuple, Literal, Union
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import *
from Deeploy.Layers.BasicLayers import *


@gs.Graph.register()
def deleteNode(self, node):

    inputNode = node.inputs[0]
    outputNode = node.outputs[0]

    # SCHEREMO : Ugly workaround, assume single output
    while (len(node.outputs[0].outputs) > 0):
        newInputs = []
        outputLayer = node.outputs[0].outputs[0]
        for i in outputLayer.inputs:
            if i == outputNode:
                newInputs.append(inputNode)
            else:
                newInputs.append(i)
        outputLayer.inputs = newInputs

    node.inputs.clear()
    node.outputs.clear()


@gs.Graph.register()
def replaceInsertNode(self, inputs, outputs, node):
    # Disconnect output nodes of all input tensors
    # for inp in inputs:
    #     inp.outputs.clear()

    ret = self.layer(op = node.op, name = node.name, attrs = node.attrs, inputs = inputs, outputs = outputs)

    # Disconnect input nodes of all output tensors
    for out in outputs:
        out.inputs = out.inputs[-1:]

    return ret


class Match(NamedTuple):
    anchor: gs.Node
    nodes_map: Dict[str, gs.Node]


class GSPass(NetworkOptimizationPass):

    def __init__(self):
        self.parent = None
        self._subpasses = {}

    def __setattr__(self, attribute, value):
        if isinstance(value, GSPass) and attribute != 'parent':
            self.register_subpass(attribute, value)
        super(GSPass, self).__setattr__(attribute, value)

    def register_subpass(self, name, value):
        subpasses = self.__dict__.get('_subpasses')
        if subpasses is None:
            raise AttributeError("Cannot assign sub-pass before calling GSPass.__init__!")
        if name in self._subpasses.keys():
            del self._subpasses[name]

        value.parent = self
        self._subpasses[name] = value

    def remove_subpass(self, name):
        try:
            del self._subpasses[name]
        except KeyError:
            print(f"No subpass with name {name}, cannot remove!")
        except AttributeError:
            raise AttributeError("Cannot remove sub-pass before calling GSPass.__init__!")

    def __getattr__(self, attribute):
        subpasses = self.__dict__.get('_subpasses')
        if subpasses is not None and attribute in subpasses.keys():
            return subpasses[attribute]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attribute}")

    def named_subpasses(self):
        return self._subpasses.copy()

    # overwrite this function in custom pass subclasses!
    # run_pass should return the modified graphModule.
    def run_pass(self, ctxt: NetworkContext, graph: gs.Graph):
        raise NotImplementedError("Can't apply GSPass base class. Inherit from this class")

    # DO NOT OVERWRITE this function in custom pass subclasses unless you have
    # a very good reason!
    def apply(self, ctxt: NetworkContext, graph: gs.Graph):
        ctxt, graph = self.retarget(ctxt, graph)
        ctxt, graph = self.run_pass(ctxt, graph)
        return ctxt, graph

    def __call__(self, ctxt: NetworkContext, graph: gs.Graph):
        return self.apply(ctxt, graph)

    # overwrite this if your pass is specific to a graph instance (e.g., most
    # "dynamic" SequentialPass derivatives will be, as the list of passes to
    # execute probably depends on the graph. See e.g.
    # ReplaceSequentialPatternPass for an example)
    def retarget(self, ctxt: NetworkContext, graph: gs.Graph):
        return ctxt, graph


class SequentialPass(GSPass):

    def __init__(self, *passes, name_prefix = ''):
        super(SequentialPass, self).__init__()
        self.name_prefix = name_prefix
        self.setup_passes(passes)

    def run_pass(self, ctxt: NetworkContext, graph: gs.Graph):
        for p in self.named_subpasses().values():
            ctxt, graph = p.apply(ctxt, graph)
        return ctxt, graph

    def setup_passes(self, passes):
        for i, p in enumerate(passes):
            self.register_subpass(self.name_prefix + '_' + str(i), p)


class NonBranchingMatcher:
    # simplified matcher which matches call_module ops more reasonably
    def __init__(self, pattern: gs.Graph):
        # This checking is sufficient - iff the graph is acyclic and connected (checked by parser)
        # and every node has one output, the graph is sequential

        assert len(pattern.outputs) == 1, "Found more than one output"
        for node in pattern.nodes:
            assert len(node.outputs) == 1, "Graph needs to be purely sequential!"

        # we need to have access to both the pattern graph (using the output
        # node as an entry point) as well as the
        # enclosing GraphModule to correctly match call_module ops
        self.p = pattern
        self.pattern_anchor = next(iter(self.p.nodes))
        # this will be the graph module that we search for the pattern
        self.searched_graph: gs.Graph = None

    @property
    def searched_modules(self):
        # a dictionary of the modules contained in the searched GraphModule
        return dict(self.searched_graph.nodes)

    def matches_subgraph_from_anchor(self, ctxt: NetworkContext, anchor: gs.Node):
        # similar to the fx method, except the nodes_map is not a member of the
        # class
        #TODO: is this really a nice way to return the results? (None if no
        # match, Match object if match)
        match = self._match_nodes(ctxt, self.pattern_anchor, anchor, len(self.p.nodes))

        if match is not None:
            mm = Match(anchor = anchor, nodes_map = match)
        else:
            mm = None
        return mm

    def _match_nodes(self,
                     ctxt: NetworkContext,
                     pn: gs.Node,
                     gn: gs.Node,
                     remaining_pattern_length: int,
                     nodes_map: dict = None):
        nodes_map = {} if nodes_map is None else nodes_map
        last_active_node = remaining_pattern_length == 1

        # as we do sequential traversal, the first step (checking if nodes
        # already traversed) of the original _match_nodes function can be
        # discarded

        # the following submethod is a modified version of the one from the
        # original SubgraphMatcher
        def attributes_are_equal(pn: gs.Node, gn: gs.Node) -> bool:
            return pn.op == gn.op

        # from here on, proceed as in the original implementation.
        if not attributes_are_equal(pn, gn):
            return None

        nodes_map[pn.name] = gn

        # if we are in the "active" pattern, the graph node has to be
        # single-output and single-use
        # if (pn.op not in ("output", "placeholder") and
        # (len(gn.all_input_nodes) != 1) or (len(gn.users) > 1 and not
        # first_active_node)):
        if ((len(ctxt.lookup(gn.outputs[0].name)._users) > 1 and not last_active_node) or len(gn.outputs) > 1):
            # if the gn has >1 users, the pattern is "leaking" and we don't
            # want to match it
            return None

        if remaining_pattern_length == 1:
            return nodes_map

        # otherwise we are on a "matching track", so move one node down in
        # pattern and graph. We know that gn has only 1 input!
        if len(pn.outputs[0].outputs) < 1 or len(gn.outputs[0].outputs) < 1:
            return None

        return self._match_nodes(ctxt, pn.outputs[0].outputs[0], gn.outputs[0].outputs[0], remaining_pattern_length - 1,
                                 nodes_map)

    def match_graph(self, ctxt: NetworkContext, graph: gs.Graph):
        # this function returns a list of non-overlapping matches of self.p
        # in gm, which is first traced with self.trace. Any matches which
        # overlap previous matches are discarded.
        self.searched_graph = graph
        all_matches = []
        matched_nodes = set()

        def match_overlaps_with_previous(match):
            return any(n.name in matched_nodes for k, n in match.nodes_map.items())

        for node in self.searched_graph.nodes:
            match = self.matches_subgraph_from_anchor(ctxt, node)
            if match is not None:
                if not match_overlaps_with_previous(match):
                    all_matches.append(match)
                    for k, n in match.nodes_map.items():
                        matched_nodes.add(n.name)
        return all_matches


# WIESEP: TODO Rename to BranchingMather
class BranchingMatcher:
    # simplified matcher which matches call_module ops more reasonably
    def __init__(self, pattern: gs.Graph):
        assert len(pattern.outputs) == 1, "Found more than one output"

        # we need to have access to both the pattern graph (using the output
        # node as an entry point) as well as the
        # enclosing GraphModule to correctly match call_module ops
        self.p = pattern
        self.pattern_anchor = next(iter(self.p.nodes))
        # this will be the graph module that we search for the pattern
        self.searched_graph: gs.Graph = None

    @property
    def searched_modules(self):
        # a dictionary of the modules contained in the searched GraphModule
        return dict(self.searched_graph.nodes)

    def matches_subgraph_from_anchor(self, ctxt: NetworkContext, anchor: gs.Node):
        # similar to the fx method, except the nodes_map is not a member of the
        # class
        #TODO: is this really a nice way to return the results? (None if no
        # match, Match object if match)
        match = self._match_nodes(ctxt, self.pattern_anchor, anchor)

        if match is not None:
            if len(match.keys()) != len(self.p.nodes):
                mm = None
            else:
                mm = Match(anchor = anchor, nodes_map = match)
        else:
            mm = None
        return mm

    def _match_nodes(self,
                     ctxt: NetworkContext,
                     pn: gs.Node,
                     gn: gs.Node,
                     nodes_map: dict = None,
                     direction: Literal["Forward", "Reverse"] = 'Forward'):
        assert direction in ["Forward", "Reverse"], f"'{direction}' is not a valid matching direction!"

        nodes_map = {} if nodes_map is None else nodes_map

        # Check if nodes are identical
        def attributes_are_equal(pn: gs.Node, gn: gs.Node) -> bool:
            ret = True
            pn_inputs = [pn_input for pn_input in pn.inputs if len(pn_input.inputs) > 0]
            gn_inputs = [gn_input for gn_input in gn.inputs if len(gn_input.inputs) > 0]
            if len(pn.inputs) != 1 or len(pn.inputs[0].inputs) != 0:
                ret &= (len(pn_inputs) == len(gn_inputs))

            pn_outputs = [pn_output for pn_output in pn.outputs if len(pn_output.outputs) > 0]
            gn_outputs = [gn_output for gn_output in gn.outputs if len(gn_output.outputs) > 0]
            if len(pn.outputs) != 1 or len(pn.outputs[0].outputs) != 0:
                ret &= (len(pn_outputs) == len(gn_outputs))

            ret &= (pn.op == gn.op)
            return ret, (pn_inputs, gn_inputs, pn_outputs, gn_outputs)

        ret, (pn_inputs, gn_inputs, pn_outputs, gn_outputs) = attributes_are_equal(pn, gn)
        if not ret:
            return None

        # Add candidate match to list
        nodes_map[pn.name] = gn

        # Check if we are done
        if direction == 'Reverse':
            # If our pattern is fully matched until here
            if len(pn_inputs) == 0:
                return nodes_map
            # If our pattern is not fully matched completely but we reached leaf node
            if len(pn_inputs) > 0 and len(gn_inputs) == 0:
                return None

        if direction == 'Forward':
            # If our pattern is fully matched until here
            if len(pn_outputs) == 0:
                return nodes_map
            # If our pattern is not fully matched completely but we reached leaf node
            if len(pn_outputs) > 0 and len(gn_outputs) == 0:
                return None

        # If pn and gn have multiple parent nodes, we have want to traverse upwards
        if len(pn.inputs) > 0 and len(gn.inputs) > 0:
            for pn_input in pn.inputs:
                # Check if parent node of pn is constant or input node (in this case it has no additional inputs)
                # and if node was already matched
                if len(pn_input.inputs) > 0 and pn_input.inputs[0].name not in nodes_map.keys():
                    tmp = None
                    for gn_input in gn.inputs:
                        # Check if parent node of gn is constant or input node (in this case it has no additional inputs)
                        # and if node was already matched
                        if len(gn_input.inputs) > 0 and gn_input.inputs[0] not in nodes_map.values():
                            # Search for valid subgraphs
                            tmp = self._match_nodes(ctxt,
                                                    pn_input.inputs[0],
                                                    gn_input.inputs[0],
                                                    nodes_map,
                                                    direction = 'Reverse')
                            if tmp is not None:
                                nodes_map = tmp

                    # If it was not possible to map parent node of pn to a parent node of gn
                    if tmp == None:
                        return None

            # If it was possible to map all parent nodes of pn to a parent node of gn
            if direction == 'Reverse':
                return nodes_map

        # If pn and gn have multiple child nodes, we have want to traverse downwards
        if len(pn.outputs) > 0 and len(gn.outputs) > 0:
            for pn_input in pn.outputs:
                # Check if parent node of pn is is output node (in this case it has no additional outputs)
                # and if node was already matched
                if len(pn_input.outputs) > 0 and pn_input.outputs[0].name not in nodes_map.keys():
                    tmp = None
                    for gn_input in gn.outputs:
                        # Check if parent node of gn is is output node (in this case it has no additional outputs)
                        # and if node was already matched
                        if len(gn_input.outputs) > 0 and gn_input.outputs[0] not in nodes_map.values():
                            # Search for valid subgraphs
                            tmp = self._match_nodes(ctxt,
                                                    pn_input.outputs[0],
                                                    gn_input.outputs[0],
                                                    nodes_map,
                                                    direction = 'Forward')
                            if tmp is not None:
                                nodes_map = tmp

                    # If it was not possible to map parent node of pn to a parent node of gn
                    if tmp == None:
                        return None

            # If it was possible to map all child nodea of pn to a child node of gn
            if direction == 'Forward':
                return nodes_map

        assert False, "This statement should never be reached!"

    def match_graph(self, ctxt: NetworkContext, graph: gs.Graph):
        # this function returns a list of non-overlapping matches of self.p
        # in gm, which is first traced with self.trace. Any matches which
        # overlap previous matches are discarded.
        self.searched_graph = graph
        all_matches = []
        matched_nodes = set()

        def match_overlaps_with_previous(match):
            return any(n.name in matched_nodes for k, n in match.nodes_map.items())

        for node in self.searched_graph.nodes:
            match = self.matches_subgraph_from_anchor(ctxt, node)
            if match is not None:
                if not match_overlaps_with_previous(match):
                    all_matches.append(match)
                    for k, n in match.nodes_map.items():
                        matched_nodes.add(n.name)
        return all_matches


class ReplaceMatchWithModulePass(GSPass):
    #Matches are specific to graph instances, so don't use this type of pass on its
    #own if you want to reuse it!
    def __init__(self, match: Match, module: gs.Node):
        # this class needs a name field because the inserted submodules will be named
        super(ReplaceMatchWithModulePass, self).__init__()
        self.match = match
        self.replacementNode = module

    def run_pass(self, ctxt: NetworkContext, graph: gs.Graph):
        matched_nodes = list(self.match.nodes_map.values())
        if self.replacementNode is not None:
            graph.replaceInsertNode(self.replacementNode)
        return ctxt, graph


class ReplaceSequentialPatternPass(SequentialPass):
    # finds all instances of pattern in the graph, calls the replacement_fn on
    # the matches and replaces the matched nodes with the module returned by
    # replacement_fn.
    def __init__(self,
                 pattern: callable,
                 replacement_fn: callable,
                 name: str,
                 matcher: Union[NonBranchingMatcher, BranchingMatcher] = NonBranchingMatcher,
                 **kwargs):
        super().__init__(name_prefix = name)
        self.matcher = matcher(pattern)
        self.replacement_fn = replacement_fn
        self.name = name
        self.kwargs = kwargs

    def retarget(self, ctxt: NetworkContext, graph: gs.Graph):
        # to retarget to a new graph, clear all registered subpasses.
        for k in self.named_subpasses().keys():
            self.remove_subpass(k)
        self.matches = self.matcher.match_graph(ctxt, graph)
        passes = []
        for i, m in enumerate(self.matches):
            ctxt, graph = self.replacement_fn(ctxt, graph, m, f"{self.name}_{i}", **self.kwargs)
        graph.cleanup().toposort()
        return ctxt, graph
