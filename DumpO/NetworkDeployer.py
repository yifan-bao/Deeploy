# ----------------------------------------------------------------------
#
# File: NetworkDeployer.py
#
# Last edited: 26.12.2021        
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
from DumpO.Parsers.BasicParsers import *
from DumpO.Layers.BasicLayers import *
from DumpO.OptimizationPasses.BasicPasses import *

class NetworkDeployer(NetworkContainer):
    def __init__(self, graph: gs.Graph, deploymentPlatform: DeploymentPlatform,loweringOptimizer: NetworkOptimizer, scheduler: Callable = lambda x: x,name: str = 'DumpONetwork',input_n_levels : Dict[str, int] = {'input_0': 256}, input_signed : Dict[str, bool] = {'input_0':False}):
        super().__init__(graph, deploymentPlatform, scheduler, name)
        self.name = name
        self.prepared = False
        self.baseParser = NodeParser()
        self.optimizer = loweringOptimizer
        self.input_n_levels = input_n_levels
        self.input_signed = input_signed
        
    # Don't override this
    def lower(self, ctxt: NetworkContext, graph: gs.Graph) -> (NetworkContext, gs.Graph):
        return self.optimizer.optimize(ctxt, graph)

    # Don't override this
    def baseParse(self) -> (NetworkContext, bool):
        newCtxt = NetworkContext(VariableBuffer, ConstantBuffer, StructBuffer, {}, {})
        newCtxt = self._createIOBindings(newCtxt, self.graph)

        for node in self.graph.nodes:
            newCtxt, ret = self.baseParser.parse(newCtxt, node)

        return newCtxt, ret

    def postLoweringOptimization(self):
        pass
    
    # Don't Override this
    def middleWare(self):
        
        # Rename graph inputs and outputs:
        for idx, inputNode in enumerate(self.graph.inputs):
            inputNode.name = "input_"+str(idx)
        for idx, outputNode in enumerate(self.graph.outputs):
            outputNode.name = "output_"+str(idx)
            
        # sanity check the graph and generate a base context for lowering/optimization
        self.ctxt, ret = self.baseParse() 
        if not ret:
            raise RuntimeError("The given graph was not valid - check that it is acyclic!")
        
        self.ctxt, self.graph = self.lower(self.ctxt, self.graph) # This lowers the graph to a deployable format
        onnx.save_model(gs.export_onnx(self.graph), "test_preturn.onnx")

        self.postLoweringOptimization()
    
    # Don't override this
    def exportGraph(self, f):
        model = gs.export_onnx(self.graph)
        convert_model_to_external_data(model, location="model.data")
        onnx.save(model, f)

    # Don't override this unless you know what you are doin
    def backEnd(self, channels_first=True):
        self.parse(channels_first) # This reparses the lowered graph
        self.broadcast(channels_first) # This broadcasts all tensors offline
        self.bind() # This binds the graph to the node templates
        
    # Don't override this
    def prepare(self):
        # MIDDLE END
        self.middleWare()
        # BACK END - Inherited from NetworkContainer
        self.backEnd(channels_first=False)
        # FINAL TRANSFORMS 
        self.prepared = True

    # Don't override this
    def generateFunction(self) -> str:
        if not self.prepared:
            self.prepare()

        return self.generateInferenceCode()
