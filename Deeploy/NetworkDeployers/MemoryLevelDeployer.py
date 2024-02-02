# ----------------------------------------------------------------------
#
# File: MemoryLevelAnnotation.py
#
# Last edited: 04.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# Moritz Scherer, ETH Zurich
# Victor Jung, ETH Zurich
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

from typing import Callable, Dict, Optional

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import PointerType
from Deeploy.DeeployTypes import DeploymentPlatform, NetworkContext, NetworkDeployer, NetworkOptimizer, \
    TopologyOptimizer
from Deeploy.MemoryLevels import MemoryHierarchy
from Deeploy.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel


class MemoryLevelAwareDeployer(NetworkDeployer):

    def __init__(self,
                 memoryHierarchy: MemoryHierarchy,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, PointerType],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState",
                 annotationOptimizer: Optional[NetworkOptimizer] = None):
        super().__init__(graph, deploymentPlatform, inputTypes, loweringOptimizer, scheduler, name,
                         default_channels_first, deeployStateDir)

        self.memoryHierarchy = memoryHierarchy
        if annotationOptimizer is None:
            self.memoryLevelAnnotationOptimizer = NetworkOptimizer([AnnotateDefaultMemoryLevel(self.memoryHierarchy)])
        else:
            self.memoryLevelAnnotationOptimizer = annotationOptimizer

    def bind(self):

        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        ret = super().bind()
        if not ret:
            return False

        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        return ret

    def codeTransform(self):
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        super().codeTransform()


class MemoryLevelAwareSignPropDeployer(SignPropDeployer):

    def __init__(self,
                 memoryHierarchy: MemoryHierarchy,
                 graph: gs.Graph,
                 inputTypes: Dict[str, PointerType],
                 deploymentPlatform: DeploymentPlatform,
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState",
                 inputOffsets: Dict[str, int] = {},
                 annotationOptimizer: Optional[NetworkOptimizer] = None):
        super().__init__(graph, deploymentPlatform, inputTypes, loweringOptimizer, scheduler, name,
                         default_channels_first, deeployStateDir, inputOffsets)

        self.memoryHierarchy = memoryHierarchy
        if annotationOptimizer is None:
            self.memoryLevelAnnotationOptimizer = NetworkOptimizer([AnnotateDefaultMemoryLevel(self.memoryHierarchy)])
        else:
            self.memoryLevelAnnotationOptimizer = annotationOptimizer

    def bind(self):

        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        ret = super().bind()
        if not ret:
            return False

        # SCHEREMO: There might be hoisting; reassign memoryLevel preferences
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        return ret

    def codeTransform(self):
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        super().codeTransform()


def memoryLevelAnnotation(cls: NetworkDeployer,
                          memoryHierachy: MemoryHierarchy,
                          annotationOptimizer: Optional[NetworkOptimizer] = None) -> DeploymentPlatform:

    class MemoryLevelAnnotationWrapper(MemoryLevelAwareDeployer):

        def __init__(self, deployer: NetworkDeployer, memoryHierarchy: MemoryHierarchy):

            if not isinstance(deployer, NetworkDeployer):
                raise RuntimeError("Must pass cls argument of type NetworkDeployer!")

            self._innerObject = deployer
            self.memoryHierarchy = memoryHierarchy
            if annotationOptimizer is None:
                self.memoryLevelAnnotationOptimizer = NetworkOptimizer(
                    [AnnotateDefaultMemoryLevel(self.memoryHierarchy)])
            else:
                self.memoryLevelAnnotationOptimizer = annotationOptimizer

        def bind(self):
            self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
            ret = self._innerObject.bind()

            return ret

        def codeTransform(self):
            self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
            self._innerObject.codeTransform()

        def generateBufferAllocationCode(self) -> str:
            return self._innerObject.generateBufferAllocationCode()

        def registerInnerAttribute(attribute):

            def fget(self):
                return getattr(self._innerObject, attribute)

            def fset(self, value):
                # Make sure the user is not creating a new attribute with this setter
                assert hasattr(self._innerObject,
                               attribute), f"Attribute {attribute} does not exist in the inner deployer!"

                setattr(self._innerObject, attribute, value)

            return property(fget, fset)

        def _createIOBindings(self, ctxt: NetworkContext, graph: gs.Graph):
            return self._innerObject._createIOBindings(ctxt, graph)

        # Define the interface as properties
        ctxt = registerInnerAttribute("ctxt")
        graph = registerInnerAttribute("graph")
        Platform = registerInnerAttribute("Platform")
        scheduler = registerInnerAttribute("scheduler")
        prepared = registerInnerAttribute("prepared")
        baseParser = registerInnerAttribute("baseParser")
        loweringOptimizer = registerInnerAttribute("loweringOptimizer")
        inputTypes = registerInnerAttribute("inputTypes")
        default_channels_first = registerInnerAttribute("default_channels_first")
        parsed = registerInnerAttribute("parsed")
        bound = registerInnerAttribute("bound")
        worstCaseBufferSize = registerInnerAttribute("worstCaseBufferSize")
        deeployStateDir = registerInnerAttribute("deeployStateDir")
        inputOffsets = registerInnerAttribute("inputOffsets")
        layerBinding = registerInnerAttribute("layerBinding")

    return MemoryLevelAnnotationWrapper(cls, memoryHierachy)
