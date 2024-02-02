# ----------------------------------------------------------------------
#
# File: TilingCodeGeneration.py
#
# Last edited: 24.10.2023
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
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, Literal, Tuple, Type

from mako.parsetree import Expression, Node, TemplateNode, Text

from Deeploy.AbstractDataTypes import Pointer, PointerClass
from Deeploy.CodeTransformationPasses.Closure import ClosureExecutionBlock
from Deeploy.CodeTransformationPasses.IntrospectiveCodeTransformation import IntrospectiveCodeTransformationMixIn
from Deeploy.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes
from Deeploy.DataTypes.PULPDataTypes import PULPStructDataTypes
from Deeploy.DeeployTypes import CodeTransformationPass, ExecutionBlock, NetworkContext, NodeTemplate, StructBuffer, \
    VariableBuffer, _TemplateNode
from Deeploy.Tiling.MemoryConstraints import NodeMemoryConstraint
from Deeploy.Tiling.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme, \
    calculateRectangleOffset, minimizeRectangleDims


class TilingCodeGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

    def __init__(self, targetMemLevel: str):
        self.targetMemLevel = targetMemLevel
        self.argStructGeneration = ArgumentStructGeneration()

    @abstractmethod
    def generateTilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                           nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                           variableReplacement: VariableReplacementScheme,
                           nodeRep: Dict) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        return ctxt, executionBlock, False

    # SCHEREMO: internalPtr refers to the HIGHER memory level of a transfer,
    # e.g. in both an L2 -> L1 and L1 -> L2 transfer, the internalPtr is in L1.
    @staticmethod
    def isFinalMemoryLevel(nodeMemoryConstraint: NodeMemoryConstraint, internalPtr: VariableBuffer) -> bool:
        externalName = internalPtr._referenceName
        tensorMemoryConstraint = nodeMemoryConstraint.tensorMemoryConstraints[externalName]
        if len(tensorMemoryConstraint.memoryConstraints.keys()) <= 2:
            return True

        finalMemoryLevels = list(tensorMemoryConstraint.memoryConstraints.keys())[:2]
        memoryLevel = internalPtr._memoryLevel

        return memoryLevel in finalMemoryLevels

    def _hoistTileIdxPtr(self, ctxt: NetworkContext, nodeRep: Dict, sourceMemoryLevel: str = "L2") -> str:

        newPtrName = self.prefix + nodeRep['nodeName'] + "_tileIdxPtr"

        tilePtrBuffer = ctxt.VariableBuffer(newPtrName, shape = [1])
        ctxt.add(tilePtrBuffer, "local")

        tilePtrBuffer._type = Pointer(IntegerDataTypes.uint32_t)
        tilePtrBuffer._instance = tilePtrBuffer._type(newPtrName, ctxt)
        tilePtrBuffer._memoryLevel = sourceMemoryLevel

        tilePtrBuffer.allocTemplate = NodeTemplate("")
        tilePtrBuffer.deallocTemplate = NodeTemplate("")
        tilePtrBuffer.initTemplate = NodeTemplate("""
        ${type.referencedType.typeName} bu_${name} = 0;
        ${type.referencedType.typeName}* ${name} = &bu_${name};""")

        return newPtrName

    def _hoistNumTiles(self,
                       ctxt: NetworkContext,
                       nodeName: str,
                       tilingSchedules: List[TilingSchedule],
                       sourceMemoryLevel: str = "L2") -> str:

        newPtrName = self.prefix + nodeName + "_numTiles"

        numTiles = [len(tilingSchedule.outputLoadSchedule) for tilingSchedule in tilingSchedules]
        cumNumTiles = [0]
        for idx in list(range(len(numTiles))):
            cumNumTiles.append(cumNumTiles[-1] + numTiles[idx])

        cb = ctxt.ConstantBuffer(newPtrName, [len(cumNumTiles)], values = cumNumTiles)
        ctxt.add(cb, "global")

        cb._type = Pointer(IntegerDataTypes.uint32_t)
        cb._instance = cb._type(newPtrName, ctxt)
        cb._memoryLevel = sourceMemoryLevel

        return newPtrName

    def apply(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
              name: str) -> Tuple[NetworkContext, ExecutionBlock]:

        def unravelReference(ctxt: NetworkContext, name: str) -> str:

            if name not in ctxt.localObjects.keys() and name not in ctxt.globalObjects.keys():
                return name

            refBuffer = ctxt.lookup(name)
            if not hasattr(refBuffer, "_referenceName"):
                return name

            return unravelReference(ctxt, refBuffer._referenceName)

        if isinstance(executionBlock, ClosureExecutionBlock):
            baseExecutionBlock = executionBlock.baseBlock
        else:
            baseExecutionBlock = executionBlock

        patternMemoryConstraint = baseExecutionBlock.patternMemoryConstraint

        if patternMemoryConstraint is None:
            return ctxt, executionBlock

        assert len(patternMemoryConstraint.nodeConstraints) == 1, "Only layerwise supported for now!"
        #assert len(baseExecutionBlock.nodeTemplates) == 1, "Only layerwise supported for now!"

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

        ctxt, executionBlock, applicable = self.generateTilingLoop(ctxt, executionBlock, nodeMemoryConstraint,
                                                                   tilingSchedules, variableReplacement, nodeRep)
        if applicable:
            ctxt, executionBlock = self.argStructGeneration.apply(ctxt, executionBlock, name)

        return ctxt, executionBlock
