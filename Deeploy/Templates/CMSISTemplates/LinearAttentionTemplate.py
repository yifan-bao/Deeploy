# ----------------------------------------------------------------------
#
# File: LinearAttentionTemplate.py
#
# Last edited: 05.06.2022
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

import mako
from typing import Dict, Tuple
from mako.template import Template
import numpy as np

from Deeploy.DeeployTypes import NodeTemplate, NetworkContext
from .CMSISUtils import bindFCParams


class _LinearAttentionTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> Tuple[NetworkContext, Dict]:
        return ctxt, nodeRep


referenceTemplate = _LinearAttentionTemplate("""
// PLACEHOLDER LINEAR ATTENTION
""")
