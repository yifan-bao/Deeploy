# ----------------------------------------------------------------------
#
# File: PULPClusterTilingDB.py
#
# Last edited: 25.10.2023
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
from typing import Dict, List, Tuple

from Deeploy.CodeTransformationPasses.PULPCodeTransformationPasses import PULPClusterTilingGeneration
from Deeploy.CodeTransformationPasses.PULPCodeTransformationPasses.PULPClusterTiling import _DMAUpdate
from Deeploy.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.DataTypes.PULPDataTypes import PULPStructDataTypes
from Deeploy.DeeployTypes import ExecutionBlock, NetworkContext, NodeTemplate, _TemplateNode
from Deeploy.Tiling.MemoryConstraints import NodeMemoryConstraint
from Deeploy.Tiling.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme, \
    calculateRectangleOffset, minimizeRectangleDims

_openTileLoopTemplate = NodeTemplate("""

// TILING LOOP
for (int TILING_I=1 + ${numTiles}[*${tileIdxPtr}]; TILING_I<1 + ${numTiles}[(*${tileIdxPtr})+1]; TILING_I++){
""")

_moveTileInTemplate = NodeTemplate("""

// IMPORT TILE ${innerTilePtr} from ${outerTilePtr}
if (${tileNum} < ${numTiles}[*${tileIdxPtr}+1]){
dory_dma_memcpy_async(&${stateReference});
}

""")

_iteratedMoveTileInTemplate = NodeTemplate("""

// IMPORT TILE ${innerTilePtr} from ${outerTilePtr}
// ITERATED

<%
_extStrides = [stride * stateStruct.value['length_1d_copy'].value for stride in remainderStrides]
_locStride = f"{stateReference}.length_1d_copy  * {stateReference}.number_of_1d_copies  *  {stateReference}.number_of_2d_copies"

stateStruct.value['ext'] = str(stateReference) + ".ext"
stateStruct.value['loc'] = str(stateReference) + ".loc"
stateStruct.value['tid'] = str(stateReference) + ".tid"
stateStruct.value['stride_2d'] = str(stateReference) + ".stride_2d"
stateStruct.value['stride_1d'] = str(stateReference) + ".stride_1d"
stateStruct.value['number_of_2d_copies'] = str(stateReference) + ".number_of_2d_copies"
stateStruct.value['number_of_1d_copies'] = str(stateReference) + ".number_of_1d_copies"
stateStruct.value['length_1d_copy'] = str(stateReference) + ".length_1d_copy"
%>

int8_t * bu_${stateReference}_loc = ${stateReference}.loc;
int8_t * bu_${stateReference}_ext = ${stateReference}.ext;

% for idx, dimLen in enumerate(dimLens):
uint16_t ${nodeName}_${tensorName}_dimLen_${idx} = ${dimLen}[${tileNum}];
for(int i_${idx} = 0; i_${idx} < ${nodeName}_${tensorName}_dimLen_${idx}; i_${idx}++){
%endfor
${stateStruct.typeName} trans_${stateReference} = (${stateStruct.typeName}) ${str(stateStruct)};
dory_dma_memcpy_async(&trans_${stateReference});
${stateStruct.value['loc']} = (((int8_t*) ${stateStruct.value['loc']}) + ${_locStride});
% for idx, _ in enumerate(dimLens):
${stateStruct.value['ext']} = (((int8_t*) ${stateStruct.value['ext']}) + (${_extStrides[idx]}));
}
${stateStruct.value['ext']} = (((int8_t*) ${stateStruct.value['ext']}) - ${nodeName}_${tensorName}_dimLen_${len(dimLens) -1 - idx} * ${_extStrides[idx]});
%endfor

${stateStruct.value['loc']} = bu_${stateReference}_loc;
${stateStruct.value['ext']} = bu_${stateReference}_ext;

""")

_moveTileOutTemplate = NodeTemplate("""

// EXPORT TILE ${innerTilePtr} to ${outerTilePtr}
if((${tileNum}) % 2 == 0){
dory_dma_memcpy_async(&${stateReference});
} else {
dory_dma_memcpy_async(&${_stateReference});
}
""")

_blockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
if((${tileNum}) > 2){
if((${tileNum}) % 2 == 1){
dory_dma_barrier(&${stateReference});
} else {
dory_dma_barrier(&${_stateReference});
}
}
""")

_finalBlockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
% if numTiles < 3:
dory_dma_barrier(&${stateReference});
dory_dma_barrier(&${_stateReference});
% else:
if(${tileNum} > 2){
if(${tileNum} % 2 == 1){
dory_dma_barrier(&${stateReference});
} else {
dory_dma_barrier(&${_stateReference});
}
}
% endif
""")

_blockTileInTemplate = NodeTemplate("""

// BLOCKING IMPORT TILE ${innerTilePtr}
dory_dma_barrier(&${stateReference});
""")

_updateDMATransferStructTemplate = NodeTemplate("""

// UPDATE DMA STRUCT ${stateReference}, ${_stateReference}
${stateReference}.ext = (((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}]);
${stateReference}.length_1d_copy = ${length1dPtr}[${tileNum}];
${stateReference}.number_of_1d_copies = ${number1dPtr}[${tileNum}];
${stateReference}.number_of_2d_copies = ${number2dPtr}[${tileNum}];
${stateReference}.loc = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
${locPtr} = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}-1]);
""")

_outUpdateDMATransferStructTemplate = NodeTemplate("""

if ((${tileNum}) % 2 == 0){
// UPDATE DMA STRUCT ${stateReference}
${stateReference}.ext = ((char*)${extPtr} + ${extOffsetPtr}[${tileNum}]);
${stateReference}.length_1d_copy = ${length1dPtr}[${tileNum}];
${stateReference}.number_of_1d_copies = ${number1dPtr}[${tileNum}];
${stateReference}.number_of_2d_copies = ${number2dPtr}[${tileNum}];
${stateReference}.loc = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
} else {
${_stateReference}.ext = ((char*)${extPtr} + ${extOffsetPtr}[${tileNum}]);
${_stateReference}.length_1d_copy = ${length1dPtr}[${tileNum}];
${_stateReference}.number_of_1d_copies = ${number1dPtr}[${tileNum}];
${_stateReference}.number_of_2d_copies = ${number2dPtr}[${tileNum}];
${_stateReference}.loc = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
}
${locPtr} = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);

""")


class PULPClusterTilingGenerationDB(PULPClusterTilingGeneration):

    _openTileLoopTemplate = _openTileLoopTemplate
    #_initTilingTemplate = _initTilingTemplate
    _blockTileOutTemplate = _blockTileOutTemplate
    _blockTileInTemplate = _blockTileInTemplate
    _updateDMATransferStructTemplate = _updateDMATransferStructTemplate
    _moveTileOutTemplate = _moveTileOutTemplate
    _moveTileInTemplate = _moveTileInTemplate

    _iteratedMoveTileInTemplate = _iteratedMoveTileInTemplate

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

    def _generateEgressPointerUpdates(self, nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                                      ctxt: NetworkContext,
                                      nodeRep: Dict) -> Tuple[NetworkContext, List[_TemplateNode]]:

        updates = []
        newCtxt = ctxt.copy()

        updateDict = PULPClusterTilingGeneration._generatePointerUpdates(ctxt, nodeRep,
                                                                         tilingSchedule.outputLoadSchedule,
                                                                         nodeMemoryConstraint, tilingSchedule)

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

            finalMemoryLevel = TilingCodeGeneration.isFinalMemoryLevel(nodeMemoryConstraint, internalPtr)
            struct = self._rectToDMAStruct(ctxt, rectangle, "FromL1", internalPtr.name, externalPtr.name,
                                           finalMemoryLevel)
            _ = ctxt.hoistStruct(struct, dmaName, PULPStructDataTypes.DMA_copy)
            ctxt.lookup(dmaName)._users += [nodeRep['nodeName']]

            tensorName = key + "_1"
            nodeName = nodeRep['nodeName']
            _dmaName = self._DMAStructName(tensorName, nodeName)

            struct = self._rectToDMAStruct(ctxt, rectangle, "FromL1", internalPtr.name, externalPtr.name,
                                           finalMemoryLevel)
            _ = ctxt.hoistStruct(struct, _dmaName, PULPStructDataTypes.DMA_copy)
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

        ingressDMATransferCalls, ingressDMAWaitStatements = self._generateIngressDMACode(
            tilingSchedule, nodeMemoryConstraint, ctxt, nodeRep)

        egressDMATransferCalls, egressDMAWaitStatements = self._generateEgressDMACode(
            tilingSchedule, nodeMemoryConstraint, ctxt, nodeRep)

        ctxt, ingressDMAUpdates = self._generateIngressPointerUpdates(nodeMemoryConstraint, tilingSchedule, ctxt,
                                                                      nodeRep)
        ctxt, egressDMAUpdates = self._generateEgressPointerUpdates(nodeMemoryConstraint, tilingSchedule, ctxt, nodeRep)

        variableUpdates = self._generateVariableUpdates(tilingSchedule, variableReplacement, ctxt, nodeRep)

        # SCHEREMO: Advance tile pointer LEFT
        # SCHEREMO: add import load blocks LEFT
        # SCHEREMO: add import loads LEFT

        # for transaction in ingressDMATransferCalls:
        #     executionBlock.addLeft(transaction.template, transaction.nodeRep)
        # for transaction in ingressDMAUpdates:
        #     executionBlock.addLeft(transaction.template, transaction.nodeRep)

        # SCHEREMO: use for debug -> should be left w double buffering
        for transaction in egressDMAWaitStatements:
            _nodeRep = transaction.nodeRep
            _nodeRep["tileNum"] = f"TILING_I"
            executionBlock.addLeft(transaction.template, _nodeRep)

        for transaction in egressDMAUpdates:
            _nodeRep = transaction.nodeRep
            _nodeRep["tileNum"] = f"TILING_I - 1"
            executionBlock.addLeft(transaction.template, _nodeRep)

        for transaction in ingressDMATransferCalls:
            _nodeRep = transaction.nodeRep.copy()
            _nodeRep["tileNum"] = f"TILING_I"
            _nodeRep["numTiles"] = nodeRep['numTiles']
            _nodeRep["tileIdxPtr"] = tileIdxPtr
            executionBlock.addLeft(transaction.template, _nodeRep)
        for transaction in ingressDMAWaitStatements:
            _nodeRep = transaction.nodeRep
            _nodeRep["tileNum"] = f"TILING_I"
            executionBlock.addLeft(transaction.template, _nodeRep)

        for transaction in ingressDMAUpdates:
            executionBlock.addLeft(transaction.template, transaction.nodeRep)

        # SCHEREMO: Updates are delayed by one

        for transaction in variableUpdates:
            _nodeRep = transaction.nodeRep
            _nodeRep["tileNum"] = f"TILING_I - 1"
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

        for transaction in ingressDMAUpdates:
            _nodeRep = transaction.nodeRep.copy()
            _nodeRep["channelName"] = "dma_channel_in"
            executionBlock.addLeft(self._setDMAChannelTemplate, _nodeRep)
        for transaction in egressDMAUpdates:
            _nodeRep = transaction.nodeRep.copy()
            _nodeRep["channelName"] = "dma_channel_out"
            executionBlock.addLeft(self._setDMAChannelTemplate, _nodeRep.copy())
            _nodeRep["channelName"] = "dma_channel_out_2"
            _nodeRep["stateReference"] = _nodeRep["_stateReference"]
            executionBlock.addLeft(self._setDMAChannelTemplate, _nodeRep.copy())
        executionBlock.addLeft(self._initDMATemplate, {"channelName": "dma_channel_in"})
        executionBlock.addLeft(self._initDMATemplate, {"channelName": "dma_channel_out"})
        executionBlock.addLeft(self._initDMATemplate, {"channelName": "dma_channel_out_2"})
        executionBlock.addRight(self._releaseDMATemplate,
                                {"stateReference": ingressDMAUpdates[0].nodeRep["stateReference"]})
        executionBlock.addRight(self._releaseDMATemplate,
                                {"stateReference": egressDMAUpdates[0].nodeRep["stateReference"]})
        executionBlock.addRight(self._releaseDMATemplate,
                                {"stateReference": egressDMAUpdates[0].nodeRep["_stateReference"]})

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
            generator = PULPClusterTilingGeneration(self.targetMemLevel)
            return generator._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule,
                                         variableReplacement, nodeRep)

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                nodeRep)
