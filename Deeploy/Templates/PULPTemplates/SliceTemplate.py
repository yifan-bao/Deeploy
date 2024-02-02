# ----------------------------------------------------------------------
#
# File: SliceTemplate.py
#
# Last edited: 01.06.2023
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

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DataTypes.PULPDataTypes import PULPStructDataTypes
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate


class _SliceTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict, List[str]]:

        assert ctxt.lookup(nodeRep['data_in'])._memoryLevel in ["L2", "L1"], "input data needs to be on-chip!"
        assert ctxt.lookup(nodeRep['data_out'])._memoryLevel in ["L2", "L1"], "output data needs to be on-chip!"
        assert ctxt.lookup(nodeRep['data_in'])._memoryLevel != ctxt.lookup(
            nodeRep['data_out'])._memoryLevel, "Can't move on same memory level with Cluster DMA!"

        bufferList = []

        def _downSample(starts, ends, axes, steps, data_in_shape, idx) -> bool:
            return steps[idx] != 1 or starts[idx] > 0 or ends[idx] < data_in_shape[axes[idx]]

        # Immediate-ify start
        startsBuffer = ctxt.lookup(nodeRep['starts'])
        axesBuffer = ctxt.lookup(nodeRep['axes'])
        endsBuffer = ctxt.lookup(nodeRep['ends'])
        stepsBuffer = ctxt.lookup(nodeRep['steps'])

        startsBuffer._deploy = False
        axesBuffer._deploy = False
        endsBuffer._deploy = False
        stepsBuffer._deploy = False

        nodeRep['starts'] = startsBuffer.values
        nodeRep['ends'] = endsBuffer.values
        nodeRep['axes'] = axesBuffer.values
        nodeRep['steps'] = stepsBuffer.values

        nodeRep['data_in_size'] = np.prod(nodeRep['data_in_shape'])

        data_in_shape = nodeRep['data_in_shape']
        data_in_size = nodeRep['data_in_size']
        axes = nodeRep['axes']
        starts = nodeRep['starts']
        ends = nodeRep['ends']
        steps = nodeRep['steps']

        dimSteps = []
        dimSteps.append(data_in_size // data_in_shape[0])
        for dim in data_in_shape[1:]:
            dimSteps.append(dimSteps[-1] // dim)

        number_of_1d_copies = 1
        number_of_2d_copies = 1
        stride_1d = 0
        stride_2d = 0

        numCopies = []
        strides = []
        downSample = []

        switchIdx = 0

        for i in range(len(axes)):
            numCopies.append(ends[i] - starts[i])
            strides.append(dimSteps[axes[i]])
            downSample.append(_downSample(starts, ends, axes, steps, data_in_shape, i))

        for idx, switch in enumerate(downSample):
            if switch == True:
                switchIdx = idx
                break
            switchIdx = axes[idx] + 1

        nodeRep["offset"] = starts[switchIdx] * dimSteps[axes[switchIdx]]

        nodeRep['numberIterations'] = np.prod(data_in_shape[:axes[switchIdx]])

        inputOffset = dimSteps[axes[switchIdx]] * data_in_shape[axes[switchIdx]]
        outputOffset = int(inputOffset * ((ends[switchIdx] - starts[switchIdx]) / data_in_shape[axes[switchIdx]]))
        consecutiveCopies = outputOffset
        transferSize1D = consecutiveCopies * nodeRep['data_in_type'].referencedType.typeWidth // 8

        if ctxt.lookup(nodeRep['data_in'])._memoryLevel == "L2":
            # Target address:
            ext = nodeRep['data_in']
            # Source address:
            loc = nodeRep['data_out']
            _dir = 1
            nodeRep["extOffset"] = inputOffset
            nodeRep["locOffset"] = outputOffset
        else:
            # Target address:
            ext = nodeRep['data_out']
            # Source address:
            loc = nodeRep['data_in']
            _dir = 0
            nodeRep["locOffset"] = inputOffset
            nodeRep["extOffset"] = outputOffset

        nodeRep["dir"] = _dir

        bufferList += [
            ctxt.hoistStruct(
                {
                    "ext": ext,
                    "loc": loc,
                    "hwc_to_chw": 0,
                    "stride_2d": stride_2d,
                    "number_of_2d_copies": number_of_2d_copies,
                    "stride_1d": stride_1d,
                    "number_of_1d_copies": number_of_1d_copies,
                    "length_1d_copy": transferSize1D,
                    "dir": _dir,
                    "tid": 0
                }, nodeRep['nodeName'] + "_stateReference", PULPStructDataTypes.DMA_copy)
        ]

        nodeRep['stateReference'] = nodeRep['nodeName'] + "_stateReference"

        return ctxt, nodeRep, bufferList


referenceTemplate = _SliceTemplate("""
// Slice (Name: ${nodeName}, Op: ${nodeOp})
// data_in : ${data_in_shape}
// data_out : ${data_out_shape}
% if dir == 1:
${stateReference}.ext += ${offset};
% else:
${stateReference}.loc += ${offset};
% endif
for(int j=0;j<${numberIterations};j++){
dory_dma_memcpy_async(&${stateReference});
${stateReference}.ext += ${extOffset};
${stateReference}.loc += ${locOffset};
}
""")
