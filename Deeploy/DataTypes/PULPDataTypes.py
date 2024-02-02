# ----------------------------------------------------------------------
#
# File: PULPDataTypes.py
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

from functools import partial

from Deeploy.AbstractDataTypes import DataTypeCollection, HelperTypes, Pointer
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes
from Deeploy.DataTypes.Future import Future
from Deeploy.DeeployTypes import NodeTemplate

_DMAResolveTemplate = NodeTemplate("""
// PULP CLUSTER DMA Resolve
dory_dma_barrier(&${stateReference});
""")

_DMADispatchTemplate = NodeTemplate("""
// PULP CLUSTER DMA Dispatch
// No dispatch necessary
""")


class PULPStructDataTypes(DataTypeCollection):
    DMA_copy = {
        "ext": Pointer(HelperTypes.void),
        "loc": Pointer(HelperTypes.void),
        "hwc_to_chw": IntegerDataTypes.uint16_t,
        "stride_2d": IntegerDataTypes.uint16_t,
        "number_of_2d_copies": IntegerDataTypes.uint16_t,
        "stride_1d": IntegerDataTypes.uint16_t,
        "number_of_1d_copies": IntegerDataTypes.uint16_t,
        "length_1d_copy": IntegerDataTypes.uint16_t,
        "dir": IntegerDataTypes.int32_t,
        "tid": IntegerDataTypes.int32_t
    }

    pi_cl_ram_req_t = {
        "addr": Pointer(HelperTypes.void),
        "pi_ram_addr": Pointer(HelperTypes.void),
        "size": IntegerDataTypes.uint32_t,
        "stride": IntegerDataTypes.uint32_t,
        "length": IntegerDataTypes.uint32_t,
        "is_2d": IntegerDataTypes.uint8_t,
        "ext2loc": IntegerDataTypes.uint8_t,
    }


PULPDMAFuture = partial(Future,
                        stateReferenceType = PULPStructDataTypes.DMA_copy,
                        resolveCheckTemplate = _DMAResolveTemplate,
                        dispatchCheckTemplate = _DMADispatchTemplate)
