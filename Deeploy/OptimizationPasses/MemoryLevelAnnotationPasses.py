# ----------------------------------------------------------------------
#
# File: MemoryLevelAnnotationPasses.py
#
# Last edited: 10.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Victor Jung, ETH Zurich
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

from typing import List, Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevels import MemoryHierarchy
from Deeploy.OptimizationPasses.PassClasses import SequentialPass


class AnnotateDefaultMemoryLevel(SequentialPass):

    def __init__(self, memoryHierarchy: MemoryHierarchy):
        self.memoryHierarchy = memoryHierarchy
        super().__init__()

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:

        newCtxt = ctxt.copy()

        defaultMemoryLevel = self.memoryHierarchy.getDefaultMemoryLevel()

        for tensor_name in {**newCtxt.localObjects, **newCtxt.globalObjects}.keys():
            bufferObject = newCtxt.lookup(tensor_name)
            if isinstance(bufferObject, VariableBuffer) and not (hasattr(bufferObject, "_memoryLevel")):
                bufferObject._memoryLevel = defaultMemoryLevel.name

        return newCtxt, graph


class AnnotateIOMemoryLevel(SequentialPass):

    def __init__(self, IOLevel: str):
        self.ioLevel = IOLevel
        super().__init__()

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:

        def getIOTensors(ctxt: NetworkContext, graph: gs.Graph) -> List[str]:
            graphInputs = [tensor.name for tensor in graph.inputs]
            graphOutputs = [tensor.name for tensor in graph.outputs]

            inputs = []
            outputs = []

            for key, value in ctxt.globalObjects.items():
                if not isinstance(value, ctxt.VariableBuffer) or value._users == []:
                    continue
                if key not in graphInputs:
                    continue

                inputs += [key]

            for key, value in ctxt.globalObjects.items():

                if not isinstance(value, ctxt.VariableBuffer) or value._users != []:
                    continue
                if key not in graphOutputs:
                    continue

                outputs += [key]

            return inputs + outputs

        newCtxt = ctxt.copy()

        ioTensors = getIOTensors(ctxt, graph)
        for tensor in ioTensors:
            ctxt.lookup(tensor)._memoryLevel = self.ioLevel

        return ctxt, graph
