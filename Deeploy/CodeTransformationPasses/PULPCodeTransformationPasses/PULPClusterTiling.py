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
from typing import Any, Dict, List, Literal, Tuple, Type

import numpy as np
from mako.parsetree import Expression, Node, TemplateNode, Text

from Deeploy.AbstractDataTypes import ImmediateClass, Pointer, PointerClass
from Deeploy.CodeTransformationPasses.PULPCodeTransformationPasses import AutoTransposeUtils
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

// TILING LOOP
for (int TILING_I=${numTiles}[*${tileIdxPtr}]; TILING_I<${numTiles}[(*${tileIdxPtr})+1]; TILING_I++){
""")

_closeTileLoopTemplate = NodeTemplate("""

// CLOSE TILING LOOP
}
*${tileIdxPtr} += 1;

""")

_moveTileInTemplate = NodeTemplate("""

// IMPORT TILE ${innerTilePtr} from ${outerTilePtr}
dory_dma_memcpy_async(&${stateReference});

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

_blockTileInTemplate = NodeTemplate("""

// BLOCKING IMPORT TILE ${innerTilePtr}
dory_dma_barrier(&${stateReference});

""")

_moveTileOutTemplate = NodeTemplate("""

// EXPORT TILE ${innerTilePtr} to ${outerTilePtr}
dory_dma_memcpy_async(&${stateReference});

""")

_blockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
dory_dma_barrier(&${stateReference});

""")

_updateDMATransferStructTemplate = NodeTemplate("""

// UPDATE DMA STRUCT ${stateReference}
${stateReference}.ext = ((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}];
${stateReference}.length_1d_copy = ${length1dPtr}[${tileNum}];
${stateReference}.number_of_1d_copies = ${number1dPtr}[${tileNum}];
${stateReference}.number_of_2d_copies = ${number2dPtr}[${tileNum}];

${stateReference}.stride_1d = ${stride1dPtr}[${tileNum}];
${stateReference}.stride_2d = ${stride2dPtr}[${tileNum}];

""")

_updateReferenceTemplate = NodeTemplate("""

// UPDATE VARIABLE ${reference}
*${reference} = ${baseReference}[${tileNum}];
""")

_initDMATemplate = NodeTemplate("""
int32_t ${channelName} = dory_dma_allocate();
""")

_setDMAChannelTemplate = NodeTemplate("""
${stateReference}.tid = ${channelName};
""")

_releaseDMATemplate = NodeTemplate("""
dory_dma_free(&${stateReference});
""")

# ADD NUM TRANSFERS VARIABLE

_DMAUpdate = namedtuple(
    "_DMAUpdate", "extOffset locOffset length_1d_copy number_of_1d_copies number_of_2d_copies stride_1d stride_2d")


class PULPClusterTilingGeneration(TilingCodeGeneration):

    _prefix = "TILING_REPLACED_"

    _openTileLoopTemplate = _openTileLoopTemplate
    _closeTileLoopTemplate = _closeTileLoopTemplate

    _moveTileInTemplate = _moveTileInTemplate
    _iteratedMoveTileInTemplate = _iteratedMoveTileInTemplate
    _blockTileInTemplate = _blockTileInTemplate

    _moveTileOutTemplate = _moveTileOutTemplate
    _blockTileOutTemplate = _blockTileOutTemplate

    _updateDMATransferStructTemplate = _updateDMATransferStructTemplate
    _updateReferenceTemplate = _updateReferenceTemplate

    _initDMATemplate = _initDMATemplate
    _setDMAChannelTemplate = _setDMAChannelTemplate
    _releaseDMATemplate = _releaseDMATemplate

    @property
    def prefix(self):
        return self._prefix + self.targetMemLevel + "_"

    def _DMAStructName(self, tensorName: str, nodeName: str) -> str:
        return f"{self.prefix}_DMA_{nodeName}_{tensorName}"

    @staticmethod
    def _generatePointerUpdates(ctxt: NetworkContext, nodeRep: Dict[str, Any], loadSchedule: List[Dict[str,
                                                                                                       HyperRectangle]],
                                nodeMemoryConstraint: NodeMemoryConstraint,
                                tilingSchedule: TilingSchedule) -> Dict[str, _DMAUpdate]:
        updateDict = {}
        deltaOffsets = {}

        for idx, loadStep in enumerate(loadSchedule):
            for stepIdx, (key, rect) in enumerate(loadStep.items()):

                if key in tilingSchedule.outputBaseOffsets.keys():
                    baseOffsets = tilingSchedule.outputBaseOffsets[key]
                    direction = "FromL1"
                else:
                    baseOffsets = tilingSchedule.inputBaseOffsets[key]
                    direction = "ToL1"

                if key not in updateDict.keys():
                    updateDict[key] = []
                if key not in deltaOffsets.keys():
                    deltaOffsets[key] = 0

                referenceBuffer = ctxt.lookup(ctxt.lookup(nodeRep[key])._referenceName)
                l1Buffer = ctxt.lookup(nodeRep[key])

                finalMemoryLevel = TilingCodeGeneration.isFinalMemoryLevel(nodeMemoryConstraint, l1Buffer)

                if (f"in{stepIdx}_perm" in nodeRep
                        and key in tilingSchedule.inputBaseOffsets.keys()) and (finalMemoryLevel == False):
                    perm = nodeRep[f"in{stepIdx}_perm"]
                    struct, _, _ = AutoTransposeUtils.generateTransposedDMAStruct(ctxt, rect, direction, perm,
                                                                                  l1Buffer.name,
                                                                                  l1Buffer._referenceName)

                    _invPerm = _invertPermutation(perm)
                    _rect = copy.copy(rect)
                    _referenceBuffer = copy.copy(referenceBuffer)
                    _rect.offset = _permuteList(rect.offset, _invPerm)
                    _rect.dims = _permuteList(rect.dims, _invPerm)
                    _referenceBuffer.shape = _permuteList(referenceBuffer.shape, _invPerm)

                    accOffset = calculateRectangleOffset(_rect, _referenceBuffer)

                else:
                    struct = PULPClusterTilingGeneration._rectToDMAStruct(ctxt, rect, direction, l1Buffer.name,
                                                                          l1Buffer._referenceName, finalMemoryLevel)
                    accOffset = calculateRectangleOffset(rect, referenceBuffer)

                length_1d_copy = struct.value['length_1d_copy'].value
                number_of_1d_copies = struct.value['number_of_1d_copies'].value
                number_of_2d_copies = struct.value['number_of_2d_copies'].value
                stride_1d = struct.value['stride_1d'].value
                stride_2d = struct.value['stride_2d'].value

                lIdx = idx % len(baseOffsets)

                sol = _DMAUpdate(accOffset, baseOffsets[lIdx], length_1d_copy, number_of_1d_copies, number_of_2d_copies,
                                 stride_1d, stride_2d)

                deltaOffsets[key] = accOffset
                updateDict[key].append(sol)

        return updateDict

    @staticmethod
    def _rectToDMAStruct(ctxt: NetworkContext, rectangle: HyperRectangle, direction: Literal["ToL1", "FromL1"],
                         L1Name: str, L2Name: str, finalMemoryLevel: bool) -> PULPStructDataTypes.DMA_copy:

        referenceBuffer = ctxt.lookup(L2Name)

        rect, referenceRect = minimizeRectangleDims(rectangle, referenceBuffer)
        assert len(rect.dims) <= 3, "PULP: Only 2D transfers are supported!"

        if direction == "ToL1":
            _dir = 1
        else:
            _dir = 0

        length_1d_copy = rect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)

        number_of_1d_copies = 1
        stride_1d = 0

        if len(rect.dims) > 1:
            number_of_1d_copies = rect.dims[-2]
            stride_1d = referenceRect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)

            if not finalMemoryLevel:
                stride_1d = length_1d_copy

        number_of_2d_copies = 1
        stride_2d = 0

        if len(rect.dims) > 2:
            number_of_2d_copies = rect.dims[-3]
            stride_2d = referenceRect.dims[-2] * stride_1d

        struct = PULPStructDataTypes.DMA_copy(
            {
                "ext": referenceBuffer.name,
                "loc": L1Name,
                "hwc_to_chw": 0,
                "stride_2d": stride_2d,
                "number_of_2d_copies": number_of_2d_copies,
                "stride_1d": stride_1d,
                "number_of_1d_copies": number_of_1d_copies,
                "length_1d_copy": length_1d_copy,
                "dir": _dir,
                "tid": 0
            }, ctxt)

        return struct

    def _hoistConstantAndReference(self,
                                   ctxt: NetworkContext,
                                   constBuf: ConstantBuffer,
                                   nodeRep: Dict,
                                   nodeName: str,
                                   nodeRepName: str,
                                   immediateType: ImmediateClass = None) -> Tuple[NetworkContext, Dict]:

        if immediateType is None:
            _type = Pointer(IntegerDataTypes.int32_t)
        else:
            _type = Pointer(immediateType)

        name = constBuf.name

        ctxt.add(constBuf, "global")
        constBuf._type = _type
        constBuf._instance = constBuf._type(name, ctxt)
        constBuf._users = [nodeName]
        constBuf._memoryLevel = self.targetMemLevel

        refName = name + "_ref"
        reference = ctxt.hoistReference(name, refName)
        ctxt.lookup(reference)._memoryLevel = self.targetMemLevel

        nodeRep[nodeRepName] = refName

        return ctxt, nodeRep

    def _hoistDMAUpdates(self, ctxt: NetworkContext, tensorName: str, updateList: List[_DMAUpdate],
                         nodeRep: Dict) -> Tuple[NetworkContext, Dict]:

        nodeRep = nodeRep.copy()

        nodeName = nodeRep['nodeName']

        offsetList = []
        len1dList = []
        num1dList = []
        num2dList = []
        stride1dList = []
        stride2dList = []
        for update in updateList:
            offsetList.append(int(update.extOffset))
            len1dList.append(int(update.length_1d_copy))
            num1dList.append(int(update.number_of_1d_copies))
            num2dList.append(int(update.number_of_2d_copies))
            stride1dList.append(int(update.stride_1d))
            stride2dList.append(int(update.stride_2d))

        dmaName = self._DMAStructName(tensorName, nodeName)
        nodeRep['stateReference'] = dmaName
        nodeRep['tileNum'] = "TILING_I"
        nodeRep['extPtr'] = ctxt.lookup(nodeRep[tensorName])._referenceName

        namePrefix = self.prefix + f"{nodeName}_{tensorName}"

        name = namePrefix + "_offset"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], offsetList)
        ctxt, nodeRep = self._hoistConstantAndReference(ctxt, cb, nodeRep, nodeName, 'extOffsetPtr')

        name = namePrefix + "_length_1d_copy"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], len1dList)
        ctxt, nodeRep = self._hoistConstantAndReference(ctxt, cb, nodeRep, nodeName, 'length1dPtr',
                                                        PULPStructDataTypes.DMA_copy.structTypeDict['length_1d_copy'])

        name = namePrefix + "_number_of_1d_copies"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], num1dList)
        ctxt, nodeRep = self._hoistConstantAndReference(
            ctxt, cb, nodeRep, nodeName, 'number1dPtr',
            PULPStructDataTypes.DMA_copy.structTypeDict['number_of_1d_copies'])

        name = namePrefix + "_number_of_2d_copies"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], num2dList)
        ctxt, nodeRep = self._hoistConstantAndReference(
            ctxt, cb, nodeRep, nodeName, 'number2dPtr',
            PULPStructDataTypes.DMA_copy.structTypeDict['number_of_2d_copies'])

        name = namePrefix + "_stride_1d"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], stride1dList)
        ctxt, nodeRep = self._hoistConstantAndReference(ctxt, cb, nodeRep, nodeName, 'stride1dPtr',
                                                        PULPStructDataTypes.DMA_copy.structTypeDict['stride_1d'])

        name = namePrefix + "_stride_2d"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], stride2dList)
        ctxt, nodeRep = self._hoistConstantAndReference(ctxt, cb, nodeRep, nodeName, 'stride2dPtr',
                                                        PULPStructDataTypes.DMA_copy.structTypeDict['stride_2d'])

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
            updates.append(_TemplateNode(self._updateDMATransferStructTemplate, newNodeRep))

        return newCtxt, updates

    def _generateIngressPointerUpdates(self, nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                                       ctxt: NetworkContext,
                                       nodeRep: Dict) -> Tuple[NetworkContext, List[_TemplateNode]]:

        updates = []
        newCtxt = ctxt.copy()

        updateDict = PULPClusterTilingGeneration._generatePointerUpdates(ctxt, nodeRep,
                                                                         tilingSchedule.inputLoadSchedule,
                                                                         nodeMemoryConstraint, tilingSchedule)

        for key, updateList in updateDict.items():

            newCtxt, newNodeRep = self._hoistDMAUpdates(newCtxt, key, updateList, nodeRep)
            updates.append(_TemplateNode(self._updateDMATransferStructTemplate, newNodeRep))

        return newCtxt, updates

    def _generateVariableUpdates(self, tilingSchedule: TilingSchedule, variableReplacement: VariableReplacementScheme,
                                 ctxt: NetworkContext, nodeRep: Dict) -> List[_TemplateNode]:

        updates = []

        for key in variableReplacement.perTileReplacements.keys():

            buf = ctxt.lookup(nodeRep[key])
            reference = str(buf._instance)

            updates.append(
                _TemplateNode(self._updateReferenceTemplate, {
                    "reference": reference,
                    "tileNum": "TILING_I",
                    "baseReference": buf._referenceName
                }))

        return updates

    def _generateDMACode(self, nodeMemoryConstraint: NodeMemoryConstraint, ctxt: NetworkContext,
                         nodeRep: Dict[str, Any], loadSchedule: List[Dict[str, HyperRectangle]],
                         direction: Literal["ToL1", "FromL1"]) -> Tuple[List[_TemplateNode], List[_TemplateNode]]:

        DMATransferCalls = []
        DMAWaitStatements = []

        allNumTransfers = AutoTransposeUtils.allNumTransfers(ctxt, nodeRep, loadSchedule, direction)

        transferNodeRep = {}

        if allNumTransfers != []:

            dimLens = []

            for dim in range(len(allNumTransfers[0])):
                dimVec = [transfer[dim] for transfer in allNumTransfers]
                namePrefix = nodeRep["nodeName"] + "_"
                vecName = f"dimLen_{dim}"

                cb = ctxt.ConstantBuffer(namePrefix + vecName, [len(dimVec)], dimVec)
                ctxt, transferNodeRep = self._hoistConstantAndReference(ctxt, cb, transferNodeRep, nodeRep['nodeName'],
                                                                        vecName)

                dimLens.append(str(cb._instance))

            transferNodeRep['nodeName'] = nodeRep['nodeName']
            transferNodeRep['dimLens'] = dimLens
            transferNodeRep['tileNum'] = "TILING_I"

        loadStep = loadSchedule[0]

        for idx, (key, rectangle) in enumerate(loadStep.items()):

            permName = f"in{idx}_perm"

            externalPtr = ctxt.lookup(ctxt.lookup(nodeRep[key])._referenceName)
            internalPtr = ctxt.lookup(nodeRep[key])

            tensorName = key
            nodeName = nodeRep['nodeName']
            dmaName = self._DMAStructName(tensorName, nodeName)

            transferNodeRep = {
                **transferNodeRep,
                **{
                    'innerTilePtr': str(internalPtr._instance),
                    "outerTilePtr": str(externalPtr._instance),
                    "stateReference": dmaName
                }
            }

            if permName in nodeRep and direction == "ToL1":
                perm = nodeRep[permName]
                struct, remainderStrides, numTransfers = AutoTransposeUtils.generateTransposedDMAStruct(
                    ctxt, rectangle, direction, perm, internalPtr.name, externalPtr.name)
                locStride = np.prod(
                    rectangle.dims) // np.prod(numTransfers) * (externalPtr._type.referencedType.typeWidth // 8)

                transferNodeRep['tensorName'] = nodeRep[key]

                transferNodeRep = {**transferNodeRep, **{"remainderStrides": remainderStrides, "locStride": locStride}}

            else:
                finalMemoryLevel = TilingCodeGeneration.isFinalMemoryLevel(nodeMemoryConstraint, internalPtr)

                struct = self._rectToDMAStruct(ctxt, rectangle, direction, internalPtr.name, externalPtr.name,
                                               finalMemoryLevel)

            transferNodeRep["stateStruct"] = struct
            _ = ctxt.hoistStruct(struct, dmaName, PULPStructDataTypes.DMA_copy)
            ctxt.lookup(dmaName)._users += [nodeRep['nodeName']]

            if permName in nodeRep and direction == "ToL1":

                DMATransferCalls.append(_TemplateNode(self._iteratedMoveTileInTemplate, transferNodeRep))
            else:
                DMATransferCalls.append(_TemplateNode(self._moveTileInTemplate, transferNodeRep))

            DMAWaitStatements.append(_TemplateNode(self._blockTileInTemplate, transferNodeRep))

        return DMATransferCalls, DMAWaitStatements

    def _generateIngressDMACode(self, tilingSchedule: TilingSchedule, nodeMemoryConstraint: NodeMemoryConstraint,
                                ctxt: NetworkContext, nodeRep: Dict) -> Tuple[List[_TemplateNode], List[_TemplateNode]]:

        importLoadStep = tilingSchedule.inputLoadSchedule
        ingressDMATransferCalls, ingressDMAWaitStatements = self._generateDMACode(nodeMemoryConstraint, ctxt, nodeRep,
                                                                                  importLoadStep, "ToL1")
        return ingressDMATransferCalls, ingressDMAWaitStatements

    def _generateEgressDMACode(self, tilingSchedule: TilingSchedule, nodeMemoryConstraint: NodeMemoryConstraint,
                               ctxt: NetworkContext, nodeRep: Dict) -> Tuple[List[_TemplateNode], List[_TemplateNode]]:

        exportLoadStep = tilingSchedule.outputLoadSchedule
        egressDMATransferCalls, egressDMAWaitStatements = self._generateDMACode(nodeMemoryConstraint, ctxt, nodeRep,
                                                                                exportLoadStep, "FromL1")

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

        for transaction in ingressDMAWaitStatements:
            executionBlock.addLeft(transaction.template, transaction.nodeRep)
        for transaction in variableUpdates:
            executionBlock.addLeft(transaction.template, transaction.nodeRep)
        for transaction in ingressDMATransferCalls:
            executionBlock.addLeft(transaction.template, transaction.nodeRep)
        for transaction in ingressDMAUpdates:
            executionBlock.addLeft(transaction.template, transaction.nodeRep)

        # SCHEREMO: add export loads RIGHT
        # SCHEREMO: add export load blocks RIGHT
        # SCHEREMO: Advance tile pointer RIGHT

        for transaction in egressDMAUpdates:
            executionBlock.addRight(transaction.template, transaction.nodeRep)
        for transaction in egressDMATransferCalls:
            executionBlock.addRight(transaction.template, transaction.nodeRep)
        for transaction in egressDMAWaitStatements:
            executionBlock.addRight(transaction.template, transaction.nodeRep)

        # SCHEREMO: Wrap it all in a for loop

        executionBlock.addLeft(self._openTileLoopTemplate, {"numTiles": nodeRep["numTiles"], "tileIdxPtr": tileIdxPtr})
        executionBlock.addRight(self._closeTileLoopTemplate, {
            "numTiles": nodeRep["numTiles"],
            "tileIdxPtr": tileIdxPtr
        })

        # SCHEREMO: And set the DMA init

        for transaction in ingressDMAUpdates:
            nodeRep = transaction.nodeRep.copy()
            nodeRep["channelName"] = "dma_channel_in"
            executionBlock.addLeft(self._setDMAChannelTemplate, nodeRep)
        for transaction in egressDMAUpdates:
            nodeRep = transaction.nodeRep.copy()
            nodeRep["channelName"] = "dma_channel_out"
            executionBlock.addLeft(self._setDMAChannelTemplate, nodeRep)
        executionBlock.addLeft(self._initDMATemplate, {"channelName": "dma_channel_in"})
        executionBlock.addLeft(self._initDMATemplate, {"channelName": "dma_channel_out"})
        executionBlock.addRight(self._releaseDMATemplate,
                                {"stateReference": ingressDMAUpdates[0].nodeRep["stateReference"]})
        executionBlock.addRight(self._releaseDMATemplate,
                                {"stateReference": egressDMAUpdates[0].nodeRep["stateReference"]})

        return ctxt, executionBlock, True

    def generateTilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                           nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedules: List[TilingSchedule],
                           variableReplacement: VariableReplacementScheme,
                           nodeRep: Dict) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        flatTilingSchedule = copy.copy(tilingSchedules[0])
        for tilingSchedule in tilingSchedules[1:]:
            flatTilingSchedule += tilingSchedule

        # SCHEREMO: hoist numTiles

        offsetLists = list({**flatTilingSchedule.inputBaseOffsets, **flatTilingSchedule.outputBaseOffsets}.values())

        if len(offsetLists) == 0:
            return ctxt, executionBlock, False

        for offsetList in offsetLists:
            if not len(offsetList) == 1:
                return ctxt, executionBlock, False

        nodeRep["numTiles"] = self._hoistNumTiles(ctxt, nodeRep['nodeName'], tilingSchedules)

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                nodeRep)
