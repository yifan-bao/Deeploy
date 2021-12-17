# ----------------------------------------------------------------------
#
# File: onnxParse.py
#
# Last edited: 13.12.2021        
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

import onnx
import onnx_graphsurgeon as gs

from parserTypes import *
from Platforms.BasicPlatform import *

model = onnx.load_model('./quantlib_trial.onnx')
graph = gs.import_onnx(model)

def scheduler(graph: gs.Graph):
    return graph.nodes

model = NetworkContainer(graph, BasicPlatform, scheduler)
model.parse()
inferenceCode = model.generateInferenceCode()
initializationCode = model.generateBufferInitializationCode()
deAllocationCode = model.generateBufferDeAllocationCode()
import IPython; IPython.embed()

