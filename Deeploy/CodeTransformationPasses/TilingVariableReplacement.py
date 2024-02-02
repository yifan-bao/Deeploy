# ----------------------------------------------------------------------
#
# File: TilingVariableReplacement.py
#
# Last edited: 28.09.2023
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
from typing import Dict, List, Tuple, Type

import numpy as np
from mako.parsetree import Expression, Node, TemplateNode, Text

from Deeploy.AbstractDataTypes import Pointer, PointerClass
from Deeploy.CodeTransformationPasses.Closure import ClosureExecutionBlock
from Deeploy.CodeTransformationPasses.IntrospectiveCodeTransformation import IntrospectiveCodeTransformationMixIn
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes
from Deeploy.DataTypes.PULPDataTypes import PULPStructDataTypes
from Deeploy.DeeployTypes import CodeTransformationPass, ExecutionBlock, NetworkContext, NodeTemplate, \
    TransientBuffer, VariableBuffer, _TemplateNode
from Deeploy.Tiling.MemoryConstraints import NodeMemoryConstraint
from Deeploy.Tiling.TilingCodegen import TilingSchedule, VariableReplacementScheme


class TilingVariableReplacement(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

    _prefix = "TILING_REPLACED_"

    def __init__(self, targetMemLevel: str):
        self.targetMemLevel = targetMemLevel
        self._name: str

    @property
    def prefix(self):
        return self._prefix + f"{self._name}_" + self.targetMemLevel + "_"

    def _dereferencePointer(self, nodes: List[Node], name: str) -> List[Node]:
        instanceIdxs = [idx for idx, node in enumerate(nodes) if isinstance(node, Expression) and node.text == name]

        for offset, idx in enumerate(instanceIdxs):
            text = Text("*", source = "*", lineno = 0, pos = 0, filename = None)
            nodes.insert(offset + idx, text)

        return nodes

    def _replaceImmediate(self, ctxt: NetworkContext, nodeRep: Dict, variableReplacement: Tuple[str, List],
                          dataType: Type[PointerClass]) -> Tuple[NetworkContext, Dict]:

        varName = variableReplacement[0]
        varVal = variableReplacement[1]

        newConstName = self.prefix + varName
        newRefName = self.prefix + "ref_" + varName

        cb = ctxt.ConstantBuffer(newConstName, shape = (len(varVal),), values = varVal)
        ctxt.add(cb, "global")

        cb._type = dataType
        cb._instance = dataType(newConstName, ctxt)
        cb._memoryLevel = self.targetMemLevel

        reference = ctxt.hoistReference(newConstName, newRefName)
        ctxt.lookup(reference)._memoryLevel = self.targetMemLevel

        nodeRep[varName] = reference

        return ctxt, nodeRep

    def _hoistTileReference(self, ctxt: NetworkContext, reference: str, name: str, offset: int) -> NetworkContext:

        refName = ctxt.hoistReference(reference, name)
        refBuf = ctxt.lookup(refName)

        staticBuf = ctxt.lookup(f"MEMORYARENA_{self.targetMemLevel}")

        refBuf.allocTemplate = NodeTemplate(" \
        ${type.typeName} ${name} = (${type.typeName}) " + f"((char*){str(staticBuf._instance)} + {offset});")
        refBuf._memoryLevel = self.targetMemLevel

        return ctxt

    def _replaceReferences(self, ctxt: NetworkContext, nodeRep: Dict, tilingSchedule: TilingSchedule,
                           name: str) -> Tuple[NetworkContext, Dict]:

        def unravelOldRef(refName):
            oldBuf = ctxt.lookup(refName)
            if hasattr(oldBuf, "_referenceName"):
                return unravelOldRef(oldBuf._referenceName)
            return oldBuf.name

        newRefName = self.prefix + "ref_" + name
        oldRefName = nodeRep[name]

        if name in tilingSchedule.inputBaseOffsets:
            offset = tilingSchedule.inputBaseOffsets[name]
        elif name in tilingSchedule.outputBaseOffsets:
            offset = tilingSchedule.outputBaseOffsets[name]
        else:
            raise RuntimeError(f"Name {name} not found in TilingSchedule {tilingSchedule}")

        unravelRef = unravelOldRef(oldRefName)

        ctxt = self._hoistTileReference(ctxt, unravelRef, newRefName, offset[0])
        nodeRep[name] = newRefName

        return ctxt, nodeRep

    def _replaceTransients(self, ctxt: NetworkContext, nodeRep: Dict, nodeMemoryConstraint: NodeMemoryConstraint,
                           name: str) -> Tuple[NetworkContext, Dict]:

        memoryConstraints = nodeMemoryConstraint.tensorMemoryConstraints[nodeRep[name]].memoryConstraints
        assert len(memoryConstraints) == 1, f"Tiled transient buffer {nodeRep[name]} has more than one memory level!"
        key = list(memoryConstraints.keys())[0]
        constraint = memoryConstraints[key]
        assert constraint.addrSpace is not None, f"Address space of {constraint} cannot be None!"
        offset = constraint.addrSpace[0]

        refBuf = ctxt.lookup(nodeRep[name])

        if refBuf._memoryLevel != self.targetMemLevel:
            return ctxt, nodeRep

        staticBuf = ctxt.lookup(f"MEMORYARENA_{self.targetMemLevel}")

        refBuf.allocTemplate = NodeTemplate(" \
        ${type.typeName} ${name} = (${type.typeName}) " + f"((char*){str(staticBuf._instance)} + {offset});")
        refBuf.deallocTemplate = NodeTemplate("")
        refBuf._memoryLevel = self.targetMemLevel

        return ctxt, nodeRep

    def _replaceTiledExpressions(self, ctxt: NetworkContext, templateNode: _TemplateNode,
                                 variableReplacement: VariableReplacementScheme, tilingSchedule: TilingSchedule,
                                 nodeMemoryConstraint: NodeMemoryConstraint) -> NetworkContext:

        nodeRep = templateNode.nodeRep
        template = templateNode.template

        immediateList = [
            (key, value) for key, value in variableReplacement.perTileReplacements.items() if type(nodeRep[key]) != str
        ]

        inoutSchedule = {**tilingSchedule.inputBaseOffsets, **tilingSchedule.outputBaseOffsets}
        variableList = [key for key, value in inoutSchedule.items() if type(nodeRep[key]) == str]

        transientBufferList = []
        for key, value in nodeRep.items():
            if not isinstance(value, str):
                continue
            if (ctxt.is_local(value) and isinstance(ctxt.lookup(value), TransientBuffer)):
                transientBufferList.append(key)

        parseTree = IntrospectiveCodeTransformationMixIn._generateParseTree(template)
        newParseTree = copy.copy(parseTree)
        nodes = parseTree.nodes

        newNodes = copy.copy(nodes)

        for rep in immediateList:
            ctxt, nodeRep = self._replaceImmediate(ctxt, nodeRep, rep, variableReplacement.replacementTypes[rep[0]])
            newNodes = self._dereferencePointer(newNodes, rep[0])

        for rep in variableList:
            ctxt, nodeRep = self._replaceReferences(ctxt, nodeRep, tilingSchedule, rep)

        for rep in transientBufferList:
            ctxt, nodeRep = self._replaceTransients(ctxt, nodeRep, nodeMemoryConstraint, rep)

        newParseTree.nodes = newNodes
        IntrospectiveCodeTransformationMixIn._reconstructCode(template, newParseTree)

        return ctxt

    def apply(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
              name: str) -> Tuple[NetworkContext, ExecutionBlock]:

        def unravelReference(ctxt: NetworkContext, name: str) -> str:

            if name not in ctxt.localObjects.keys() and name not in ctxt.globalObjects.keys():
                return name

            refBuffer = ctxt.lookup(name)
            if not hasattr(refBuffer, "_referenceName"):
                return name

            return unravelReference(ctxt, refBuffer._referenceName)

        self._name = name

        if isinstance(executionBlock, ClosureExecutionBlock):
            baseExecutionBlock = executionBlock.baseBlock
        else:
            baseExecutionBlock = executionBlock

        patternMemoryConstraint = baseExecutionBlock.patternMemoryConstraint

        if patternMemoryConstraint is None:
            return ctxt, executionBlock

        assert len(patternMemoryConstraint.nodeConstraints) == 1, "Only layerwise supported for now!"
        #assert len(executionBlock.nodeTemplates) == 1, "Only layerwise supported for now!"

        nodeMemoryConstraint = patternMemoryConstraint.nodeConstraints[0]
        templateNode = baseExecutionBlock.nodeTemplates[0]
        nodeRep = templateNode.nodeRep

        unravelRep = nodeRep.copy()
        for key in unravelRep.keys():

            val = unravelRep[key]
            if not isinstance(val, str):
                continue

            unravelRep[key] = unravelReference(ctxt, val)

        template = templateNode.template

        variableReplacement, tilingSchedules = template.tileConstraintFlow.wrapTilingSolution(
            nodeMemoryConstraint, self.targetMemLevel, ctxt, unravelRep)

        flatTilingSchedule = copy.copy(tilingSchedules[0])
        for tilingSchedule in tilingSchedules[1:]:
            flatTilingSchedule += tilingSchedule

        ctxt = self._replaceTiledExpressions(ctxt, templateNode, variableReplacement, flatTilingSchedule,
                                             nodeMemoryConstraint)

        for template, nRep in executionBlock.nodeTemplates:
            if not "closureStructArgs" in nRep:
                continue

            keyList = {}

            for key in list(flatTilingSchedule.inputBaseOffsets.keys()) + list(
                    flatTilingSchedule.outputBaseOffsets.keys()):
                keyList[unravelRep[key]] = nodeRep[key]

            for key in copy.copy(nRep['closureStructArgs'].value).keys():
                if nRep['closureStructArgs'].value[key].referenceName in keyList.keys():
                    nRep['closureStructArgs'].value[key] = type(nRep['closureStructArgs'].value[key])(
                        keyList[nRep['closureStructArgs'].value[key].referenceName], ctxt)

        return ctxt, executionBlock
