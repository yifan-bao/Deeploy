# ----------------------------------------------------------------------
#
# File: LoweringOptimizationPasses.py
#
# Last edited: 07.03.2023
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

from DumpO.DumpOTypes import *

def TransposeMatMulInputs(graph: gs.Graph):
    def newShape(node, shape):
        newShape = []
        for i in shape:
            newShape.append(node.shape[i])
        return newShape

    newlayerBindings = []
    transposeIdx = 0
    # Insert Transpose nodes for NCHW to NHWC conversion
    for idx, node in enumerate(graph.nodes):

        #layer = self.layerBinding[layerName]

        if node.op == "RequantizedGemm":
            # SCHEREMO: Enforce transA = 0, transB = 1
            inputA = node.inputs[0]
            inputB = node.inputs[1]

            if 'transA' not in node.attrs:
                node.attrs['transA'] = 0
            if 'transB' not in node.attrs:
                node.attrs['transB'] = 0
            if 'alpha' not in node.attrs:
                node.attrs['alpha'] = 1.0
            if 'beta' not in node.attrs:
                node.attrs['beta'] = 1.0

            # Prepend transpose on A if it's a variable and transposed
            if node.attrs['transA'] != 0 and isinstance(inputA, gs.ir.tensor.Variable):
                inShapeA = inputA.shape
                permute = list(range(len(inShapeA)))
                permute = permute[0:-2] + [permute[-1]] + [permute[-2]]
                outShapeA = inShapeA[0:-2]+ [inShapeA[-1]] + [inShapeA[-2]]
                inputATransposeOutput = gs.Variable("MatMulTransposeIn"+str(transposeIdx), dtype=np.float32, shape=outShapeA)
                inputATransposeNode = gs.Node(name='MatMulTranspose'+str(transposeIdx),op="Transpose", inputs=[inputA], outputs=[inputATransposeOutput], attrs={'perm': permute})
                node.inputs[0] = inputATransposeOutput
                node.attrs['transA'] = 0
                graph.nodes.append(inputATransposeNode)
                transposeIdx += 1

            # Prepend transpose on B if it's a variable and not transposed
            if node.attrs['transB'] != 1 and isinstance(inputB, gs.ir.tensor.Variable):
                inShapeB = inputB.shape
                permute = list(range(len(inShapeB)))
                permute = permute[0:-2] + [permute[-1]] + [permute[-2]]
                outShapeB = inShapeB[0:-2]+ [inShapeB[-1]] + [inShapeB[-2]]
                inputBTransposeOutput = gs.Variable("MatMulTransposeIn"+str(transposeIdx), dtype=np.float32, shape=outShapeB)
                inputBTransposeNode = gs.Node(name='MatMulTranspose'+str(transposeIdx),op="Transpose", inputs=[inputB], outputs=[inputBTransposeOutput], attrs={'perm': permute})
                node.inputs[1] = inputBTransposeOutput
                node.attrs['transB'] = 1
                graph.nodes.append(inputBTransposeNode)
                transposeIdx += 1

    graph.cleanup().toposort()


def NCHWtoNHWC(graph: gs.Graph):

    def newShape(node, shape):
        newShape = []
        for i in shape:
            newShape.append(node.shape[i])
        return newShape

    newlayerBindings = []
    transposeIdx = 0
    # Insert Transpose nodes for NCHW to NHWC conversion
    for idx, node in enumerate(graph.nodes):

        if (node.op == "RequantizedConv" or node.op == "MaxPool" or node.op == "Pad") and (("channels_first" in node.attrs and node.attrs["channels_first"] == 1) or ("channels_first" not in node.attrs)):

            inputNode = node.inputs[0]
            outputNode = node.outputs[0]
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

            node.inputs[0] = inputTransposeOutput
            node.outputs[0] = outputTransposeInput

            graph.nodes.append(inputTransposeNode)
            graph.nodes.append(outputTransposeNode)

            if node.op == "RequantizedConv":

                #import IPython; IPython.embed()
                # Non DW-Type:
                if node.attrs['group'] == 1:
                    weightNode = node.inputs[1]
                    weightTransposeOutput = gs.Variable("TransposeWeight"+str(transposeIdx+2), dtype=np.float32, shape=newShape(weightNode, np.array(inPermute)))
                    weightTransposeNode = gs.Node(name='Transpose'+str(transposeIdx+2),op="Transpose", inputs=[weightNode], outputs=[weightTransposeOutput], attrs={'perm': inPermute})
                    node.inputs[1] = weightTransposeOutput
                    graph.nodes.append(weightTransposeNode)

                    transposeIdx += 1

                else:

                    #import IPython; IPython.embed()
                    DWPermute = [inPermute[-1]] + inPermute[1:-1] + [inPermute[0]]

                    weightNode = node.inputs[1]
                    weightTransposeOutput = gs.Variable("TransposeWeight"+str(transposeIdx+2), dtype=np.float32, shape=newShape(weightNode, np.array(DWPermute)))
                    weightTransposeNode = gs.Node(name='Transpose'+str(transposeIdx+2),op="Transpose", inputs=[weightNode], outputs=[weightTransposeOutput], attrs={'perm': DWPermute})
                    node.inputs[1] = weightTransposeOutput
                    graph.nodes.append(weightTransposeNode)

                    transposeIdx += 1


            transposeIdx += 2
            node.attrs["channels_first"] = 0

    graph.cleanup().toposort()
