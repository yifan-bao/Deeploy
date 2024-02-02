# ----------------------------------------------------------------------
#
# File: ITAMaxTemplate.py
#
# Last edited: 27.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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

from typing import Dict, List, Tuple

from mako.template import Template

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate


class _ITAMaxTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        ctxt = ctxt.copy()
        return ctxt, nodeRep, []

    def hoistTransientBuffers(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:
        # WIESEP: Hack: Allocate a buffer for each core
        size = nodeRep['lastDimLength'] * 192
        name = nodeRep['nodeName'] + f"_buffer"
        ctxt.hoistTransientBuffer(name, size)
        nodeRep['ctxtBuffer'] = ctxt._mangle(name)
        nodeRep['ctxtBufferSize'] = size

        return ctxt, nodeRep, [name]


MemPoolParallelTemplate = _ITAMaxTemplate("""
// ITAMax Parallel (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);

ITAMax_parallel_s${data_in_type.referencedType.typeWidth}(
    ${data_in},
    ${data_out},
    ${ctxtBuffer},
    ${size},
    ${lastDimLength},
    ${n_levels},
    core_id,
    numThreads
);
mempool_barrier(numThreads);
""")
