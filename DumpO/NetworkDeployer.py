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
        
class NetworkDeployer(NetworkContainer):
    def __init__(self, graph: gs.Graph, deploymentPlatform: DeploymentPlatform, loweringOptimizer: NetworkOptimizer, scheduler: Callable = lambda x: x, name: str = "DumpONetwork"):
        super().__init__(graph, deploymentPlatform, scheduler, name)
        self.name = name
        self.prepared = False
        self.baseParser = NodeParser()
        self.optimizer = loweringOptimizer
        
    # Don't override this
    def lower(self, ctxt: NetworkContext, graph: gs.Graph) -> (gs.Graph, bool):
        return (self.optimizer.optimize(ctxt, graph), True)

    # Don't override this
    def baseParse(self) -> (NetworkContext, bool):
        newCtxt = NetworkContext(VariableBuffer, ConstantBuffer, StructBuffer, {}, {})
        newCtxt = self._createIOBindings(newCtxt, self.graph)

        for node in self.graph.nodes:
            newCtxt, ret = self.baseParser.parse(newCtxt, node)

        return newCtxt, ret

    def middleEnd(self):
        baseCtxt, ret = self.baseParse() # This sanity checks the graph and generates a base context for lowering/optimization
        if not ret:
            raise RuntimeError("The given graph was not valid - check that it is acyclic!")
        self.graph, ret = self.lower(baseCtxt, self.graph) # This lowers the graph to a deployable format
        if not ret:
            raise RuntimeError("Lowering of the graph failed!")

        
    def exportGraph(self, f):
        model = gs.export_onnx(self.graph)
        convert_model_to_external_data(model, location="model.data")
        onnx.save(model, f)

    # Works purely on the context
    # Currently flips all convolutions and the FIRST linear layer
    # Assumes all convolution weights are constants!!!
    # SCHEREMO: This can be done smarter and more robust, but for now it's okayish
    def realignDims(self):
        
        for idx, nameLayer in enumerate(self.layerBinding.items()):
            name = nameLayer[0]
            layer = nameLayer[1]
            # Check if is conv
            if isinstance(layer.mapper.parser, Conv2DParser):
                # Check if conv2D
                if len(self.ctxt.lookup(layer.mapper.parser.parserDict['data_in']).shape) == 4:
                    print("SWITCHED ONE CONV")
                    weightName = layer.mapper.parser.parserDict['weight']
                    weightTensor = self.ctxt.lookup(weightName)
                    # Flip weight buffer
                    assert self.ctxt.is_global(weightName), "Weight tensor was not constant! Cannot flip the weight tensor's dimensions!"
                    self.ctxt.globalObjects[weightName].values = np.transpose(weightTensor.values, (0,2,3,1)) # Channels last!
                    self.ctxt.globalObjects[weightName].shape = self.ctxt.globalObjects[weightName].values.shape
                    print(self.ctxt.globalObjects[weightName].shape)
            # Check if Linear layer
            elif isinstance(layer.mapper.parser, GEMMParser):
                print("SWITCHED ONE GEMM")
                weightName = layer.mapper.parser.parserDict['B']
                # THIS ONE IS NEEDED!
                self.ctxt.globalObjects[weightName].values = np.transpose(self.ctxt.globalObjects[weightName].values, (1,0))
                self.ctxt.globalObjects[weightName].shape = self.ctxt.globalObjects[weightName].values.shape
                # Assume you follow CONV
                # SCHEREMO: Very fragile, look out in the future to fix this better!
                weightTensor = self.ctxt.lookup(weightName)
                assert self.ctxt.is_global(weightName), "Weight tensor was not constant! Cannot flip the weight tensor's dimensions!"

                #SCHEREMO: THIS WILL BREAK VERY SOON
                flattener = self.ctxt.lookup(layer.node.inputs[0].inputs[0].inputs[0].name)
                
                newDim2 = flattener.shape[1:]
                intermediateValue = np.reshape(weightTensor.values, list(weightTensor.shape[:-1]) + newDim2)
                numAxes = len(intermediateValue.shape)
                axes = list(range(numAxes))
                intermediateValue = np.transpose(intermediateValue, axes[:-3] + [-2,-1,-3]) # Channels last!
                intermediateValue = np.reshape(intermediateValue, weightTensor.shape) # Channels last!
                self.ctxt.globalObjects[weightName].values = intermediateValue
                self.ctxt.globalObjects[weightName].shape = self.ctxt.globalObjects[weightName].values.shape
                return
        
    def backEnd(self):
        self.parse() # This reparses the lowered graph
        self.broadcast() # This broadcasts all tensors offline
        self.bind() # This binds the graph to the node templates
        self.realignDims() # Performs the NCHW -> NHWC conversion
        self.prepared = True
        
    # Don't override this
    def prepare(self):
        # MIDDLE END
        self.middleEnd()
        # BACK END - Inherited from NetworkContainer
        self.backEnd()
        
    def generateFunction(self) -> str:
        if not self.prepared:
            self.prepare()

        return self.generateInferenceCode()
