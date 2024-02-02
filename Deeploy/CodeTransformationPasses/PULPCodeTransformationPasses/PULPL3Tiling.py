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
// for (int TILING_I=0; TILING_I<${numTiles}; TILING_I++){
for (int TILING_I=${numTiles}[*${tileIdxPtr}]; TILING_I<${numTiles}[(*${tileIdxPtr})+1]; TILING_I++){
""")

_closeTileLoopTemplate = NodeTemplate("""

// CLOSE TILING LOOP
}
*${tileIdxPtr} += 1;

""")

_moveTileInTemplate = NodeTemplate("""

// IMPORT TILE ${innerTilePtr} from ${outerTilePtr}
pi_cl_ram_copy_2d(get_ram_ptr(), ${stateReference}.pi_ram_addr, ${stateReference}.addr, ${stateReference}.size, ${stateReference}.stride, ${stateReference}.length, ${stateReference}.ext2loc, &${stateReference});

""")

_blockTileInTemplate = NodeTemplate("""

// BLOCKING IMPORT TILE ${innerTilePtr}
pi_cl_ram_copy_wait(&${stateReference});
""")

_moveTileOutTemplate = NodeTemplate("""

// EXPORT TILE ${innerTilePtr} to ${outerTilePtr}
pi_cl_ram_copy_2d(get_ram_ptr(), ${stateReference}.pi_ram_addr, ${stateReference}.addr, ${stateReference}.size, ${stateReference}.stride, ${stateReference}.length, ${stateReference}.ext2loc, &${stateReference});

""")

_blockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
pi_cl_ram_copy_wait(&${stateReference});

""")

_updateDMATransferStructTemplate = NodeTemplate("""

// UPDATE DMA STRUCT ${stateReference}
${stateReference}.pi_ram_addr = ((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}];
${stateReference}.size = ${length1dPtr}[${tileNum}];
${stateReference}.length = ${number1dPtr}[${tileNum}];

""")

# ${stateReference}.number_of_2d_copies = ${number2dPtr}[${tileNum}];

_updateReferenceTemplate = NodeTemplate("""

// UPDATE VARIABLE ${reference}
*${reference} = ${baseReference}[${tileNum}];
""")

# ADD NUM TRANSFERS VARIABLE

_DMAUpdate = namedtuple("_DMAUpdate", "extOffset locOffset length_1d_copy number_of_1d_copies number_of_2d_copies")


class PULPL3TilingGeneration(TilingCodeGeneration):

    _prefix = "TILING_REPLACED_"

    _openTileLoopTemplate = _openTileLoopTemplate
    _closeTileLoopTemplate = _closeTileLoopTemplate

    _moveTileInTemplate = _moveTileInTemplate
    _blockTileInTemplate = _blockTileInTemplate

    _moveTileOutTemplate = _moveTileOutTemplate
    _blockTileOutTemplate = _blockTileOutTemplate

    _updateDMATransferStructTemplate = _updateDMATransferStructTemplate
    _updateReferenceTemplate = _updateReferenceTemplate

    @property
    def prefix(self):
        return self._prefix + self.targetMemLevel + "_"

    def _DMAStructName(self, tensorName: str, nodeName: str) -> str:
        return f"{self.prefix}_DMA_{nodeName}_{tensorName}"

    @staticmethod
    def _generatePointerUpdates(ctxt: NetworkContext, nodeRep: Dict[str, Any], loadSchedule: List[Dict[str,
                                                                                                       HyperRectangle]],
                                tilingSchedule: TilingSchedule) -> Dict[str, _DMAUpdate]:
        updateDict = {}
        deltaOffsets = {}

        for idx, loadStep in enumerate(loadSchedule):
            for stepIdx, (key, rect) in enumerate(loadStep.items()):

                if key in tilingSchedule.outputBaseOffsets.keys():
                    baseOffsets = tilingSchedule.outputBaseOffsets[key]
                    direction = "FromL2"
                else:
                    baseOffsets = tilingSchedule.inputBaseOffsets[key]
                    direction = "ToL2"

                if key not in updateDict.keys():
                    updateDict[key] = []
                if key not in deltaOffsets.keys():
                    deltaOffsets[key] = 0

                referenceBuffer = ctxt.lookup(ctxt.lookup(nodeRep[key])._referenceName)
                l1Buffer = ctxt.lookup(nodeRep[key])

                struct = PULPL3TilingGeneration._rectToDMAStruct(ctxt, rect, direction, l1Buffer.name,
                                                                 l1Buffer._referenceName)
                accOffset = calculateRectangleOffset(rect, referenceBuffer)

                length_1d_copy = struct.value['size'].value
                number_of_1d_copies = struct.value['length'].value

                lIdx = idx % len(baseOffsets)

                sol = _DMAUpdate(accOffset, baseOffsets[lIdx], length_1d_copy, number_of_1d_copies, 0)

                deltaOffsets[key] = accOffset
                updateDict[key].append(sol)

        return updateDict

    @staticmethod
    def _rectToDMAStruct(ctxt: NetworkContext, rectangle: HyperRectangle, direction: Literal["ToL2", "FromL2"],
                         L1Name: str, L2Name: str) -> PULPStructDataTypes.pi_cl_ram_req_t:

        referenceBuffer = ctxt.lookup(L2Name)

        rect, referenceRect = minimizeRectangleDims(rectangle, referenceBuffer)
        assert len(rect.dims) <= 2, "PULP: Only 2D transfers are supported!"

        if direction == "ToL2":
            _dir = 1
        else:
            _dir = 0

        length_1d_copy = rect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)

        if len(rect.dims) > 1:
            number_of_1d_copies = rect.dims[-2]
            stride_1d = referenceRect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)
        else:
            number_of_1d_copies = 1
            stride_1d = 0

        struct = PULPStructDataTypes.pi_cl_ram_req_t(
            {
                "pi_ram_addr": referenceBuffer.name,
                "addr": L1Name,
                "stride": stride_1d,
                "length": length_1d_copy,
                "size": number_of_1d_copies * length_1d_copy,
                "ext2loc": _dir,
                "is_2d": 1
            }, ctxt)

        return struct

    def _hoistConstantAndReference(self,
                                   ctxt: NetworkContext,
                                   constBuf: ConstantBuffer,
                                   nodeRep: Dict,
                                   nodeName: str,
                                   nodeRepName: str,
                                   immediateType: Optional[type[ImmediateClass]] = None) -> Tuple[NetworkContext, Dict]:
        if immediateType is None:
            _type = Pointer(IntegerDataTypes.int32_t)
        else:
            _type = Pointer(immediateType)

        constBuf._users = [nodeName]
        constBuf._memoryLevel = self.targetMemLevel

        refName = ctxt.hoistConstantAndReference(constBuf, _type)

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
        for update in updateList:
            offsetList.append(int(update.extOffset))
            len1dList.append(int(update.length_1d_copy))
            num1dList.append(int(update.number_of_1d_copies))
            num2dList.append(int(update.number_of_2d_copies))

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
                                                        PULPStructDataTypes.pi_cl_ram_req_t.structTypeDict['size'])

        name = namePrefix + "_number_of_1d_copies"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], num1dList)
        ctxt, nodeRep = self._hoistConstantAndReference(ctxt, cb, nodeRep, nodeName, 'number1dPtr',
                                                        PULPStructDataTypes.pi_cl_ram_req_t.structTypeDict['length'])

        return ctxt, nodeRep

    def _generateEgressPointerUpdates(self, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
                                      nodeRep: Dict) -> Tuple[NetworkContext, List[_TemplateNode]]:

        updates = []
        newCtxt = ctxt.copy()

        updateDict = PULPL3TilingGeneration._generatePointerUpdates(ctxt, nodeRep, tilingSchedule.outputLoadSchedule,
                                                                    tilingSchedule)

        for key, updateList in updateDict.items():

            newCtxt, newNodeRep = self._hoistDMAUpdates(newCtxt, key, updateList, nodeRep)
            updates.append(_TemplateNode(self._updateDMATransferStructTemplate, newNodeRep))

        return newCtxt, updates

    def _generateIngressPointerUpdates(self, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
                                       nodeRep: Dict) -> Tuple[NetworkContext, List[_TemplateNode]]:

        updates = []
        newCtxt = ctxt.copy()

        updateDict = PULPL3TilingGeneration._generatePointerUpdates(ctxt, nodeRep, tilingSchedule.inputLoadSchedule,
                                                                    tilingSchedule)

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

    def _generateDMACode(self, ctxt: NetworkContext, nodeRep: Dict[str, Any], loadSchedule: List[Dict[str,
                                                                                                      HyperRectangle]],
                         direction: Literal["ToL2", "FromL2"]) -> Tuple[List[_TemplateNode], List[_TemplateNode]]:

        DMATransferCalls = []
        DMAWaitStatements = []

        allNumTransfers = AutoTransposeUtils.allNumTransfers(ctxt, nodeRep, loadSchedule, direction)

        transferNodeRep = {}

        loadStep = loadSchedule[0]

        for idx, (key, rectangle) in enumerate(loadStep.items()):

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

            struct = PULPL3TilingGeneration._rectToDMAStruct(ctxt, rectangle, direction, internalPtr.name,
                                                             externalPtr.name)
            transferNodeRep["stateStruct"] = struct
            _ = ctxt.hoistStruct(struct, dmaName, PULPStructDataTypes.pi_cl_ram_req_t)
            ctxt.lookup(dmaName)._users += [nodeRep['nodeName']]

            DMATransferCalls.append(_TemplateNode(self._moveTileInTemplate, transferNodeRep))

            DMAWaitStatements.append(_TemplateNode(self._blockTileInTemplate, transferNodeRep))

        return DMATransferCalls, DMAWaitStatements

    def _generateIngressDMACode(self, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
                                nodeRep: Dict) -> Tuple[List[_TemplateNode], List[_TemplateNode]]:

        importLoadStep = tilingSchedule.inputLoadSchedule
        ingressDMATransferCalls, ingressDMAWaitStatements = self._generateDMACode(ctxt, nodeRep, importLoadStep, "ToL2")
        return ingressDMATransferCalls, ingressDMAWaitStatements

    def _generateEgressDMACode(self, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
                               nodeRep: Dict) -> Tuple[List[_TemplateNode], List[_TemplateNode]]:

        exportLoadStep = tilingSchedule.outputLoadSchedule
        egressDMATransferCalls, egressDMAWaitStatements = self._generateDMACode(ctxt, nodeRep, exportLoadStep, "FromL2")

        return egressDMATransferCalls, egressDMAWaitStatements

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    nodeRep: Dict) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        tileIdxPtr = self._hoistTileIdxPtr(ctxt, nodeRep)

        ingressDMATransferCalls, ingressDMAWaitStatements = self._generateIngressDMACode(tilingSchedule, ctxt, nodeRep)

        egressDMATransferCalls, egressDMAWaitStatements = self._generateEgressDMACode(tilingSchedule, ctxt, nodeRep)

        ctxt, ingressDMAUpdates = self._generateIngressPointerUpdates(tilingSchedule, ctxt, nodeRep)
        ctxt, egressDMAUpdates = self._generateEgressPointerUpdates(tilingSchedule, ctxt, nodeRep)

        variableUpdates = self._generateVariableUpdates(tilingSchedule, variableReplacement, ctxt, nodeRep)

        # SCHEREMO: Advance tile pointer LEFT
        # SCHEREMO: add import load blocks LEFT
        # SCHEREMO: add import loads LEFT

        for transaction in ingressDMAWaitStatements:
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
            if not len(offsetList) == 1:
                return ctxt, executionBlock, False

        nodeRep["numTiles"] = self._hoistNumTiles(ctxt, nodeRep['nodeName'], tilingSchedules)

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                nodeRep)
