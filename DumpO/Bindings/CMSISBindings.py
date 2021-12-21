# ----------------------------------------------------------------------
#
# File: CMSISBindings.py
#
# Last edited: 21.12.2021        
# 
# Copyright (C) 2021, ETH Zurich and University of Bologna.
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

from DumpO.DumpOTypes import *
from DumpO.TypeCheckers.BasicCheckers import *
from DumpO.Templates.BasicTemplates import *

from DumpO.TypeCheckers.CMSISCheckers import *
from DumpO.Templates.CMSISTemplates import *

class CMSISDataTypes(Enum):
    int8_t = 8
    int16_t = 16
    int32_t = 32

CMSISConv2DBinding = NodeBinding(ConvChecker([CMSISDataTypes.int8_t,CMSISDataTypes.int8_t], [CMSISDataTypes.int32_t]), ConvTemplate.conv2DTemplate)
CMSISSaturatingAddBindings = [NodeBinding(CMSISSaturatingAddChecker([CMSISDataTypes.int8_t],[CMSISDataTypes.int8_t]), AddTemplate.AddInt8Template),
                              NodeBinding(CMSISSaturatingAddChecker([CMSISDataTypes.int16_t],[CMSISDataTypes.int16_t]), AddTemplate.AddInt16Template),
                              NodeBinding(CMSISSaturatingAddChecker([CMSISDataTypes.int32_t],[CMSISDataTypes.int32_t]), AddTemplate.AddInt32Template)] 
