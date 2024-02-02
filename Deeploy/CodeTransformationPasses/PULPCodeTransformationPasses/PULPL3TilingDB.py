# ----------------------------------------------------------------------
#
# File: PULPClusterTiling.py
#
# Last edited: 17.10.2023
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
from collections import namedtuple
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import numpy as np
from mako.parsetree import Expression, Node, TemplateNode, Text

from Deeploy.AbstractDataTypes import ImmediateClass, Pointer, PointerClass
from Deeploy.CodeTransformationPasses.PULPCodeTransformationPasses import AutoTransposeUtils, PULPL3TilingGeneration
from Deeploy.CodeTransformationPasses.PULPCodeTransformationPasses.PULPL3Tiling import _DMAUpdate
from Deeploy.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes
from Deeploy.DataTypes.PULPDataTypes import PULPStructDataTypes
from Deeploy.DeeployTypes import CodeTransformationPass, ConstantBuffer, ExecutionBlock, NetworkContext, NodeTemplate, \
    _TemplateNode
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import _invertPermutation, \
    _permuteList
from Deeploy.Tiling.MemoryConstraints import NodeMemoryConstraint
from Deeploy.Tiling.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme, \
    calculateRectangleOffset, minimizeRectangleDims

_openTileLoopTemplate = NodeTemplate("""
// I AM A DOUBLEBUFFER!
// TILING LOOP
for (int TILING_I=1 + ${numTiles}[*${tileIdxPtr}]; TILING_I<1 + ${numTiles}[(*${tileIdxPtr})+1]; TILING_I++){
""")

_moveTileInTemplate = NodeTemplate("""

// IMPORT TILE ${innerTilePtr} from ${outerTilePtr}
if (${tileNum} < ${numTiles}[*${tileIdxPtr}+1]){
pi_cl_ram_copy_2d(get_ram_ptr(), ${stateReference}.pi_ram_addr, ${stateReference}.addr, ${stateReference}.size, ${stateReference}.stride, ${stateReference}.length, ${stateReference}.ext2loc, &${stateReference});
}

""")

_moveTileOutTemplate = NodeTemplate("""

// EXPORT TILE ${innerTilePtr} to ${outerTilePtr}
if((${tileNum}) % 2 == 0){
pi_cl_ram_copy_2d(get_ram_ptr(), ${stateReference}.pi_ram_addr, ${stateReference}.addr, ${stateReference}.size, ${stateReference}.stride, ${stateReference}.length, ${stateReference}.ext2loc, &${stateReference});
} else {
pi_cl_ram_copy_2d(get_ram_ptr(), ${_stateReference}.pi_ram_addr, ${_stateReference}.addr, ${_stateReference}.size, ${_stateReference}.stride, ${_stateReference}.length, ${_stateReference}.ext2loc, &${_stateReference});
}

""")

_blockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
if((${tileNum}) > 2){
if((${tileNum}) % 2 == 1){
pi_cl_ram_copy_wait(&${stateReference});
} else {
pi_cl_ram_copy_wait(&${_stateReference});
}
}

""")

_finalBlockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
% if numTiles < 3:
pi_cl_ram_copy_wait(&${stateReference});
pi_cl_ram_copy_wait(&${_stateReference});
% else:
if(${tileNum} > 2){
if(${tileNum} % 2 == 1){
pi_cl_ram_copy_wait(&${stateReference});
} else {
pi_cl_ram_copy_wait(&${_stateReference});
}
}
% endif
""")

_updateDMATransferStructTemplate = NodeTemplate("""

// UPDATE DMA STRUCT ${stateReference}
${stateReference}.pi_ram_addr = ((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}];
${stateReference}.size = ${length1dPtr}[${tileNum}];
${stateReference}.length = ${number1dPtr}[${tileNum}];
${stateReference}.addr = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
${locPtr} = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}-1]);

""")

_outUpdateDMATransferStructTemplate = NodeTemplate("""

if ((${tileNum}) % 2 == 0){
// UPDATE DMA STRUCT ${stateReference}
${stateReference}.pi_ram_addr = ((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}];
${stateReference}.size = ${length1dPtr}[${tileNum}];
${stateReference}.length = ${number1dPtr}[${tileNum}];
${stateReference}.addr = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
} else {
${_stateReference}.pi_ram_addr = ((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}];
${_stateReference}.size = ${length1dPtr}[${tileNum}];
${_stateReference}.length = ${number1dPtr}[${tileNum}];
${_stateReference}.addr = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
}
${locPtr} = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);

""")


class PULPL3TilingGenerationDB(PULPL3TilingGeneration):

    _prefix = "TILING_REPLACED_"

    _openTileLoopTemplate = _openTileLoopTemplate

    _blockTileOutTemplate = _blockTileOutTemplate

    _updateDMATransferStructTemplate = _updateDMATransferStructTemplate

    _moveTileOutTemplate = _moveTileOutTemplate
    _moveTileInTemplate = _moveTileInTemplate

    def _hoistDMAUpdates(self, ctxt: NetworkContext, tensorName: str, updateList: List[_DMAUpdate],
                         nodeRep: Dict) -> Tuple[NetworkContext, Dict]:

        nodeName = nodeRep['nodeName']

        nodeRep = nodeRep.copy()

        dmaName = self._DMAStructName(tensorName, nodeName)
        # nodeRep['stateReference'] = dmaName
        # nodeRep['tileNum'] = "TILING_I"
        nodeRep['locPtr'] = ctxt.lookup(nodeRep[tensorName]).name
        nodeRep['baseLocPtr'] = ctxt.hoistReference(nodeRep['locPtr'], nodeRep['locPtr'] + "_ref")
        nodeRep['_stateReference'] = self._DMAStructName(tensorName, nodeName) + "_1"
        ctxt.lookup(nodeRep['baseLocPtr'])._memoryLevel = self.targetMemLevel

        namePrefix = self.prefix + f"{nodeName}_{tensorName}"

        ctxt, nodeRep = super()._hoistDMAUpdates(ctxt, tensorName, updateList, nodeRep)

        locOffsetList = []
        locBaseOffset = updateList[0].locOffset
        for update in updateList:
            locOffsetList.append(int(update.locOffset) - locBaseOffset)

        name = namePrefix + "_locOffset"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], locOffsetList)
        ctxt, nodeRep = self._hoistConstantAndReference(ctxt, cb, nodeRep, nodeName, 'locOffsetPtr')

        return ctxt, nodeRep

    def _generateEgressPointerUpdates(self, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
                                      nodeRep: Dict) -> Tuple[NetworkContext, List[_TemplateNode]]:

        updates = []
        newCtxt = ctxt.copy()

        updateDict = self._generatePointerUpdates(ctxt, nodeRep, tilingSchedule.outputLoadSchedule, tilingSchedule)

        for key, updateList in updateDict.items():

            newCtxt, newNodeRep = self._hoistDMAUpdates(newCtxt, key, updateList, nodeRep)
            updates.append(_TemplateNode(_outUpdateDMATransferStructTemplate, newNodeRep))

        return newCtxt, updates

    def _generateEgressDMACode(self, tilingSchedule: TilingSchedule, nodeMemoryConstraint: NodeMemoryConstraint,
                               ctxt: NetworkContext, nodeRep: Dict) -> Tuple[List[_TemplateNode], List[_TemplateNode]]:

        egressDMATransferCalls = []
        egressDMAWaitStatements = []
        exportLoadStep = tilingSchedule.outputLoadSchedule[0]

        for key, rectangle in exportLoadStep.items():
            externalPtr = ctxt.lookup(ctxt.lookup(nodeRep[key])._referenceName)
            internalPtr = ctxt.lookup(nodeRep[key])

            tensorName = key
            nodeName = nodeRep['nodeName']
            dmaName = self._DMAStructName(tensorName, nodeName)

            struct = self._rectToDMAStruct(ctxt, rectangle, "FromL2", internalPtr.name, externalPtr.name)
            _ = ctxt.hoistStruct(struct, dmaName, PULPStructDataTypes.pi_cl_ram_req_t)
            ctxt.lookup(dmaName)._users += [nodeRep['nodeName']]

            tensorName = key + "_1"
            nodeName = nodeRep['nodeName']
            _dmaName = self._DMAStructName(tensorName, nodeName)

            struct = self._rectToDMAStruct(ctxt, rectangle, "FromL2", internalPtr.name, externalPtr.name)
            _ = ctxt.hoistStruct(struct, _dmaName, PULPStructDataTypes.pi_cl_ram_req_t)
            ctxt.lookup(_dmaName)._users += [nodeRep['nodeName']]

            egressDMATransferCalls.append(
                _TemplateNode(
                    self._moveTileOutTemplate, {
                        'innerTilePtr': str(internalPtr._instance),
                        "outerTilePtr": str(externalPtr._instance),
                        "stateReference": dmaName,
                        "_stateReference": _dmaName
                    }))

            egressDMAWaitStatements.append(
                _TemplateNode(
                    self._blockTileOutTemplate, {
                        'innerTilePtr': str(internalPtr._instance),
                        "outerTilePtr": str(externalPtr._instance),
                        "stateReference": dmaName,
                        "_stateReference": _dmaName
                    }))

        return egressDMATransferCalls, egressDMAWaitStatements

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    nodeRep: Dict) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        tileIdxPtr = self._hoistTileIdxPtr(ctxt, nodeRep)

        ingressDMATransferCalls, ingressDMAWaitStatements = self._generateIngressDMACode(tilingSchedule, ctxt, nodeRep)

        egressDMATransferCalls, egressDMAWaitStatements = self._generateEgressDMACode(
            tilingSchedule, nodeMemoryConstraint, ctxt, nodeRep)

        ctxt, ingressDMAUpdates = self._generateIngressPointerUpdates(tilingSchedule, ctxt, nodeRep)
        ctxt, egressDMAUpdates = self._generateEgressPointerUpdates(tilingSchedule, ctxt, nodeRep)

        variableUpdates = self._generateVariableUpdates(tilingSchedule, variableReplacement, ctxt, nodeRep)

        for transaction in egressDMAUpdates:
            _nodeRep = transaction.nodeRep
            _nodeRep["tileNum"] = f"TILING_I - 1"
            executionBlock.addLeft(transaction.template, _nodeRep)

        # SCHEREMO: use for debug -> should be left w double buffering
        for transaction in egressDMAWaitStatements:
            _nodeRep = transaction.nodeRep
            _nodeRep["tileNum"] = f"TILING_I"
            executionBlock.addLeft(transaction.template, _nodeRep)

        for transaction in ingressDMATransferCalls:
            _nodeRep = transaction.nodeRep.copy()
            _nodeRep["tileNum"] = f"TILING_I"
            _nodeRep["numTiles"] = nodeRep['numTiles']
            _nodeRep["tileIdxPtr"] = tileIdxPtr
            executionBlock.addLeft(transaction.template, _nodeRep)

        for transaction in ingressDMAUpdates:
            executionBlock.addLeft(transaction.template, transaction.nodeRep)

        for transaction in ingressDMAWaitStatements:
            _nodeRep = transaction.nodeRep
            _nodeRep["tileNum"] = f"TILING_I"
            executionBlock.addLeft(transaction.template, _nodeRep)

        # SCHEREMO: add export loads RIGHT
        # SCHEREMO: add export load blocks RIGHT
        # SCHEREMO: Advance tile pointer RIGHT

        # SCHEREMO: Kernel is here

        # SCHEREMO: Updates are delayed by one

        for transaction in egressDMATransferCalls:
            _nodeRep = transaction.nodeRep
            _nodeRep["tileNum"] = f"TILING_I - 1"
            _nodeRep["numTiles"] = nodeRep['numTiles']
            _nodeRep["tileIdxPtr"] = tileIdxPtr
            executionBlock.addRight(transaction.template, _nodeRep)

        # SCHEREMO: Wrap it all in a for loop

        executionBlock.addLeft(self._openTileLoopTemplate, {"numTiles": nodeRep["numTiles"], "tileIdxPtr": tileIdxPtr})
        executionBlock.addRight(self._closeTileLoopTemplate, {
            "numTiles": nodeRep["numTiles"],
            "tileIdxPtr": tileIdxPtr
        })

        for transaction in ingressDMATransferCalls:
            _nodeRep = transaction.nodeRep.copy()
            _nodeRep["tileNum"] = 0
            _nodeRep["numTiles"] = nodeRep['numTiles']
            _nodeRep["tileIdxPtr"] = tileIdxPtr
            executionBlock.addLeft(transaction.template, _nodeRep)
        # for transaction in ingressDMAUpdates:
        #     executionBlock.addLeft(transaction.template, transaction.nodeRep)
        # executionBlock.addLeft(self._initTilingTemplate, {})

        for transaction in egressDMAWaitStatements:
            _nodeRep = transaction.nodeRep.copy()
            _nodeRep['tileNum'] = ctxt.lookup(nodeRep["numTiles"]).values[-1]
            executionBlock.addRight(transaction.template, _nodeRep)

        return ctxt, executionBlock, True

    def generateTilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                           nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedules: List[TilingSchedule],
                           variableReplacement: VariableReplacementScheme,
                           nodeRep: Dict) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        flatTilingSchedule = copy.copy(tilingSchedules[0])
        for tilingSchedule in tilingSchedules[1:]:
            flatTilingSchedule += tilingSchedule

        offsetLists = list({**flatTilingSchedule.inputBaseOffsets, **flatTilingSchedule.outputBaseOffsets}.values())

        if len(offsetLists) == 0:
            return ctxt, executionBlock, False

        for offsetList in offsetLists:
            if not len(offsetList) == 2:
                return ctxt, executionBlock, False

        allNumTiles = [len(schedule.outputLoadSchedule) for schedule in tilingSchedules]
        nodeRep["numTiles"] = self._hoistNumTiles(ctxt, nodeRep['nodeName'], tilingSchedules)

        if any([numTiles < 2 for numTiles in allNumTiles]):
            generator = PULPL3TilingGeneration(self.targetMemLevel)
            return generator._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule,
                                         variableReplacement, nodeRep)

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                nodeRep)
