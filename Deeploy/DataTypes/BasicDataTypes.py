# ----------------------------------------------------------------------
#
# File: BasicDataTypes.py
#
# Last edited: 31.08.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
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

from Deeploy.AbstractDataTypes import DataTypeCollection, IntegerType


class SignedIntegerDataTypes(DataTypeCollection):
    int8_t = IntegerType(width = 8, signed = True)
    int16_t = IntegerType(width = 16, signed = True)
    int32_t = IntegerType(width = 32, signed = True)
    int64_t = IntegerType(width = 64, signed = True)


class UnsignedIntegerDataTypes(DataTypeCollection):
    uint8_t = IntegerType(width = 8, signed = False)
    uint16_t = IntegerType(width = 16, signed = False)
    uint32_t = IntegerType(width = 32, signed = False)
    uint64_t = IntegerType(width = 64, signed = False)


IntegerDataTypes = SignedIntegerDataTypes + UnsignedIntegerDataTypes
