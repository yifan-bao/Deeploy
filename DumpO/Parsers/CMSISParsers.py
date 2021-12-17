# ----------------------------------------------------------------------
#
# File: CMSISParsers.py
#
# Last edited: 17.12.2021        
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
from DumpO.DumpOManglers import *
import numpy as np
import math
from DumpO.Parsers.BasicParsers import *

class CMSISConv2DParser(Conv2DParser):
    def __init__(self):
        super().__init__()
    
    def nodeParse(self, node: gs.ir.node.Node) -> (bool):

        wellFormed = super().nodeParse(node)

        return wellFormed
    
    def nodeCtxtParse(self, ctxt: NetworkContext, node: gs.ir.node.Node) -> (NetworkContext, bool):

        ctxt = ctxt.copy()

        newCtxt, ret = super().nodeCtxtParse(ctxt, node)
        
        if ret:
            return newCtxt, True
            
        return ctxt, False

