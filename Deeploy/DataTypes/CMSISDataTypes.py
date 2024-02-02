# ----------------------------------------------------------------------
#
# File: CMSISDataTypes.py
#
# Last edited: 01.05.2023
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

from Deeploy.AbstractDataTypes import DataTypeCollection, HelperTypes, Pointer
from Deeploy.DataTypes.BasicDataTypes import IntegerDataTypes


class CMSISBaseStructDataTypes(DataTypeCollection):
    cmsis_nn_context = {"buf": Pointer(HelperTypes.void), "size": IntegerDataTypes.int32_t}
    cmsis_nn_tile = {"w": IntegerDataTypes.int32_t, "h": IntegerDataTypes.int32_t}
    cmsis_nn_activation = {"min": IntegerDataTypes.int32_t, "max": IntegerDataTypes.int32_t}
    cmsis_nn_dims = {
        "n": IntegerDataTypes.int32_t,
        "h": IntegerDataTypes.int32_t,
        "w": IntegerDataTypes.int32_t,
        "c": IntegerDataTypes.int32_t
    }
    cmsis_nn_per_channel_quant_params = {
        "multiplier": Pointer(IntegerDataTypes.int32_t),
        "shift": Pointer(IntegerDataTypes.int32_t)
    }
    cmsis_nn_per_tensor_quant_params = {"multiplier": IntegerDataTypes.int32_t, "shift": IntegerDataTypes.int32_t}


class CMSISCompositeStructDataTypes(DataTypeCollection):
    cmsis_nn_conv_params = {
        "input_offset": IntegerDataTypes.int32_t,
        "output_offset": IntegerDataTypes.int32_t,
        "stride": CMSISBaseStructDataTypes.cmsis_nn_tile,
        "padding": CMSISBaseStructDataTypes.cmsis_nn_tile,
        "dilation": CMSISBaseStructDataTypes.cmsis_nn_tile,
        "activation": CMSISBaseStructDataTypes.cmsis_nn_activation
    }
    cmsis_nn_fc_params = {
        "input_offset": IntegerDataTypes.int32_t,
        "filter_offset": IntegerDataTypes.int32_t,
        "output_offset": IntegerDataTypes.int32_t,
        "activation": CMSISBaseStructDataTypes.cmsis_nn_activation
    }
    cmsis_nn_pool_params = {
        "stride": CMSISBaseStructDataTypes.cmsis_nn_tile,
        "padding": CMSISBaseStructDataTypes.cmsis_nn_tile,
        "activation": CMSISBaseStructDataTypes.cmsis_nn_activation
    }
    cmsis_nn_dw_conv_params = {
        "input_offset": IntegerDataTypes.int32_t,
        "output_offset": IntegerDataTypes.int32_t,
        "ch_mult": IntegerDataTypes.int32_t,
        "stride": CMSISBaseStructDataTypes.cmsis_nn_tile,
        "padding": CMSISBaseStructDataTypes.cmsis_nn_tile,
        "dilation": CMSISBaseStructDataTypes.cmsis_nn_tile,
        "activation": CMSISBaseStructDataTypes.cmsis_nn_activation
    }


CMSISDataTypes = IntegerDataTypes + CMSISBaseStructDataTypes + CMSISCompositeStructDataTypes
