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
from DumpO.Layers.CMSISLayers import *
from DumpO.OptimizationPasses.BasicPasses import *

class CMSISDeployer(NetworkDeployer):

    def __init__(self, graph: gs.Graph, deploymentPlatform: DeploymentPlatform, loweringOptimizer: NetworkOptimizer, scheduler: Callable = lambda x: x, name: str = 'DumpONetwork', input_n_levels : Dict[str, int] = {'input_0': 256}, input_signed : Dict[str, bool] = {'input_0':False}):
        super().__init__(graph, deploymentPlatform, loweringOptimizer, scheduler, name, input_n_levels, input_signed)

    def postLoweringOptimization(self):
        # Insert appropriate transposes
        self.TransposeMatMulInputs()

        # Reparse
        self.ctxt, ret = self.baseParse() # This sanity checks the graph and generates a base context for lowering/optimization
        self.ctxt, self.graph = self.lower(self.ctxt, self.graph) # This lowers the graph to a deployable format
        # NCHW -> NHWC Transposes
        self.NCHWtoNHWC()
        # Remove duplicate transposes
        self.ctxt, ret = self.baseParse() # This sanity checks the graph and generates a base context for lowering/optimization

        mergeOptimizer = TransposeMergePass()
        constOptimizer = TransposeConstOptPass()
        _, self.graph = mergeOptimizer.apply(self.ctxt, self.graph)
        _, self.graph = constOptimizer.apply(self.ctxt, self.graph)
        self.graph.cleanup().toposort()

        if not ret:
            raise RuntimeError("Lowering of the graph failed!")

    def TransposeMatMulInputs(self):
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

            if isinstance(layer, (RQSGEMMLayer)):
                # SCHEREMO: Enforce transA = 0, transB = 1
                inputA = layer.node.inputs[0]
                inputB = layer.node.inputs[1]

                if 'transA' not in layer.node.attrs:
                    layer.node.attrs['transA'] = 0
                if 'transB' not in layer.node.attrs:
                    layer.node.attrs['transB'] = 0
                if 'alpha' not in layer.node.attrs:
                    layer.node.attrs['alpha'] = 1.0
                if 'beta' not in layer.node.attrs:
                    layer.node.attrs['beta'] = 1.0

                # Prepend transpose on A if it's a variable and transposed
                if layer.node.attrs['transA'] != 0 and isinstance(inputA, gs.ir.tensor.Variable):
                    inShapeA = inputA.shape
                    permute = list(range(len(inShapeA)))
                    permute = permute[0:-2] + [permute[-1]] + [permute[-2]]
                    outShapeA = inShapeA[0:-2]+ [inShapeA[-1]] + [inShapeA[-2]]
                    inputATransposeOutput = gs.Variable("MatMulTransposeIn"+str(transposeIdx), dtype=np.float32, shape=outShapeA)
                    inputATransposeNode = gs.Node(name='MatMulTranspose'+str(transposeIdx),op="Transpose", inputs=[inputA], outputs=[inputATransposeOutput], attrs={'perm': permute})
                    layer.node.inputs[0] = inputATransposeOutput
                    layer.node.attrs['transA'] = 0
                    self.graph.nodes.append(inputATransposeNode)
                    transposeIdx += 1

                # Prepend transpose on B if it's a variable and not transposed
                if layer.node.attrs['transB'] != 1 and isinstance(inputB, gs.ir.tensor.Variable):
                    inShapeB = inputB.shape
                    permute = list(range(len(inShapeB)))
                    permute = permute[0:-2] + [permute[-1]] + [permute[-2]]
                    outShapeB = inShapeB[0:-2]+ [inShapeB[-1]] + [inShapeB[-2]]
                    inputBTransposeOutput = gs.Variable("MatMulTransposeIn"+str(transposeIdx), dtype=np.float32, shape=outShapeB)
                    inputBTransposeNode = gs.Node(name='MatMulTranspose'+str(transposeIdx),op="Transpose", inputs=[inputB], outputs=[inputBTransposeOutput], attrs={'perm': permute})
                    layer.node.inputs[1] = inputBTransposeOutput
                    layer.node.attrs['transB'] = 1
                    self.graph.nodes.append(inputBTransposeNode)
                    transposeIdx += 1

        self.graph.cleanup().toposort()

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
                inPermute = [shape[0]] + shape[2:] + [shape[1]]
                if len(inputNode.shape) > 3:
                    outPermute = [inPermute[0]] + inPermute[2:] + [inPermute[1]]
                else:
                    outPermute = inPermute
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

                    #import IPython; IPython.embed()
                    # Non DW-Type:
                    if layer.node.attrs['group'] == 1:
                        weightNode = layer.node.inputs[1]
                        weightTransposeOutput = gs.Variable("TransposeWeight"+str(transposeIdx+2), dtype=np.float32, shape=newShape(weightNode, np.array(inPermute)))
                        weightTransposeNode = gs.Node(name='Transpose'+str(transposeIdx+2),op="Transpose", inputs=[weightNode], outputs=[weightTransposeOutput], attrs={'perm': inPermute})
                        layer.node.inputs[1] = weightTransposeOutput
                        self.graph.nodes.append(weightTransposeNode)

                        transposeIdx += 1

                    else:

                        #import IPython; IPython.embed()
                        DWPermute = [inPermute[-1]] + inPermute[1:-1] + [inPermute[0]]

                        weightNode = layer.node.inputs[1]
                        weightTransposeOutput = gs.Variable("TransposeWeight"+str(transposeIdx+2), dtype=np.float32, shape=newShape(weightNode, np.array(DWPermute)))
                        weightTransposeNode = gs.Node(name='Transpose'+str(transposeIdx+2),op="Transpose", inputs=[weightNode], outputs=[weightTransposeOutput], attrs={'perm': DWPermute})
                        layer.node.inputs[1] = weightTransposeOutput
                        self.graph.nodes.append(weightTransposeNode)

                        transposeIdx += 1


                transposeIdx += 2

        self.graph.cleanup().toposort()
