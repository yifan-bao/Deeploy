# ----------------------------------------------------------------------
#
# File: ReshapeTemplate.py
#
# Last edited: 16.12.2021
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

from typing import Dict
from mako.template import Template

from DumpO.DumpOTypes import NodeTemplate, NetworkContext

class _ReshapeTemplate(NodeTemplate):
    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, nodeRep: Dict) -> (NetworkContext, Dict):
        ctxt = ctxt.copy()

        # SCHEREMO: Selectively mark 'indices' dead, since we don't need them
        if 'indices' in nodeRep.keys():
            ctxt.globalObjects[nodeRep['indices']]._deploy = False
            ctxt.globalObjects[nodeRep['indices']]._live = False

        return ctxt, nodeRep

referenceTemplate = _ReshapeTemplate("${data_out} = ${data_in};")
