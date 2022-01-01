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
    def __init__(self, graph: gs.Graph, deploymentPlatform: DeploymentPlatform, loweringOptimizer: NetworkOptimizer, scheduler: Callable = lambda x: x, name: str = "DumpONetwork"):
        super().__init__(graph, deploymentPlatform, scheduler, name)
        self.name = name
        self.prepared = False
        self.baseParser = NodeParser()
        self.optimizer = loweringOptimizer
        
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
        # Insert appropriate transposes
        self.NCHWtoNHWC()
        # Remove duplicate transposes
        self.ctxt, ret = self.baseParse() # This sanity checks the graph and generates a base context for lowering/optimization
        mergeOptimizer = TransposeMergePass()
        constOptimizer = TransposeConstOptPass()
        _, self.graph = mergeOptimizer.apply(self.ctxt, self.graph)
        _, self.graph = constOptimizer.apply(self.ctxt, self.graph)
        self.graph.cleanup().toposort()
        onnx.save_model(gs.export_onnx(self.graph), "test.onnx")
        if not ret:
            raise RuntimeError("Lowering of the graph failed!")
        
    def exportGraph(self, f):
        model = gs.export_onnx(self.graph)
        convert_model_to_external_data(model, location="model.data")
        onnx.save(model, f)


    def NCHWtoNHWC(self):

        def newShape(node, shape):
            newShape = []
            for i in shape:
                newShape.append(node.shape[i])

            return newShape
        
        newlayerBindings = []
        transposeIdx = 0
        self._bindLayers()
        # Insert Transpose nodes for NCHW to NHWC conversion
        for idx, layerName in enumerate(self.layerBinding):
            
            layer = self.layerBinding[layerName]
            
            if isinstance(layer, (ConvLayer, MaxPoolLayer, PadLayer)):
                
                inputNode = layer.node.inputs[0]
                outputNode = layer.node.outputs[0]
                shape = list(range(len(inputNode.shape)))
                inPermute = shape[0:1] + shape[2:] + shape[1:2]
                outPermute = inPermute[0:1] + inPermute[2:] + inPermute[1:2]
                # Transpose conv input
                inputTransposeOutput = gs.Variable("TransposeIn"+str(transposeIdx), dtype=np.float32, shape=newShape(inputNode, np.array(inPermute)))
                outputTransposeInput = gs.Variable("TransposeOut"+str(transposeIdx+1), dtype=np.float32, shape=newShape(outputNode, np.array(inPermute)))

                inputTransposeNode = gs.Node(name='Transpose'+str(transposeIdx),op="Transpose", inputs=[inputNode], outputs=[inputTransposeOutput], attrs={'perm': inPermute})
                outputTransposeNode = gs.Node(name='Transpose'+str(transposeIdx+1),op="Transpose", inputs=[outputTransposeInput], outputs=[outputNode], attrs={'perm': outPermute})

                layer.node.inputs[0] = inputTransposeOutput
                layer.node.outputs[0] = outputTransposeInput

                self.graph.nodes.append(inputTransposeNode)
                self.graph.nodes.append(outputTransposeNode)

                if isinstance(layer, ConvLayer):
                    weightNode = layer.node.inputs[1]
                    weightTransposeOutput = gs.Variable("TransposeWeight"+str(transposeIdx+2), dtype=np.float32, shape=newShape(weightNode, np.array(inPermute)))
                    weightTransposeNode = gs.Node(name='Transpose'+str(transposeIdx+2),op="Transpose", inputs=[weightNode], outputs=[weightTransposeOutput], attrs={'perm': inPermute})
                    layer.node.inputs[1] = weightTransposeOutput
                    self.graph.nodes.append(weightTransposeNode)

                    transposeIdx += 1
                    
                transposeIdx += 2
                
        self.graph.cleanup().toposort()
                
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
        
    def generateFunction(self) -> str:
        if not self.prepared:
            self.prepare()

        return self.generateInferenceCode()
