# ----------------------------------------------------------------------
#
# File: BasicPlatform.py
#
# Last edited: 17.12.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author:
# - Moritz Scherer, ETH Zurich
# - Philip Wiese, ETH Zurich
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

import onnx_graphsurgeon as gs

from DumpO.Parsers.BasicParsers import *

from DumpO.Layers.BasicLayers import *

from DumpO.Bindings.BasicBindings import *

from DumpO.OptimizationPasses.BasicPasses import *

AddMapper = NodeMapper(AddParser(), BasicAddBindings)
FlattenMapper = NodeMapper(FlattenParser(), BasicReshapeBindings)
GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
MatMulMapper = NodeMapper(MatMulParser(), [BasicMatMulBinding])
RequantShiftMapper = NodeMapper(RequantShiftParser(), BasicRQSBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

BasicMapping = {
    'Add': AddLayer([AddMapper]),
    'Flatten': ReshapeLayer([FlattenMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'MatMul': GEMMLayer([MatMulMapper]),
    'RequantShift': RequantShiftLayer([RequantShiftMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),

    # # For example, you can use the DummpyMapper, in case you want to test
    # # deployment or optimizations with GlobalAveragePool nodes but did not yet
    # # implement the corresponding kernel
    # 'GlobalAveragePool': ConvLayer([DummyMapper]),
}


def BasicTypeInfer(node:  gs.Node):
    if hasattr(node, 'values'):
        assert len(node.outputs) <= 1, "Expected node for type inference to only have ONE output!"
        outNode = node
    else:
        import IPython
        IPython.embed()
        raise ValueError("TypeInfer was given a wrong type of node!")

    if hasattr(outNode, 'signed') and outNode.attrs['signed']:
        signed = True
    else:
        signed = False

    for _type in DataTypes:
        if signed and outNode.values.max() < 2**(_type._value_ - 1) and outNode.values.min() >= -2**(_type._value_ - 1):
            return _type
        # For nor we only have signed kernels
        elif not signed and outNode.values.max() < 2**(_type._value_ - 1):
            return _type

    raise TypeError(f'Could not infer type of node {node.name}')


class SimpleNetworkBuffer(VariableBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        return AllocateTemplate.referenceInitTemplate.generate(type = self._type._name_, name = self.name)

    def alloc(self):
        return AllocateTemplate.referenceAllocateTemplate.generate(type = self._type._name_,
                                                                   name = self.name,
                                                                   size = np.prod(self.shape))

    def dealloc(self):
        return FreeTemplate.referenceLocalTemplate.generate(name = self.name)


class SimpleGlobalBuffer(ConstantBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        values = list(self.values.reshape(-1))
        strValues = [str(value) for value in values]
        valueString = ', '.join(strValues)
        return AllocateTemplate.referenceGlobalInitTemplate.generate(type = self._type._name_,
                                                                     name = self.name,
                                                                     size = int(np.prod(self.shape)),
                                                                     values = valueString)

    def alloc(self):
        values = list(self.values.reshape(-1))
        strValues = [str(value) for value in values]
        valueString = ', '.join(strValues)
        return AllocateTemplate.referenceGlobalAllocateTemplate.generate(type = self._type._name_,
                                                                         name = self.name,
                                                                         size = int(np.prod(self.shape)),
                                                                         values = valueString)

    def dealloc(self):
        return FreeTemplate.referenceGlobalTemplate.generate(name = self.name)


class SimpleStructBuffer(StructBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        return AllocateTemplate.referenceStructInitTemplate.generate(type = self._type,
                                                                     name = self.name,
                                                                     structDict = self.structDict)

    def alloc(self) -> str:
        return AllocateTemplate.referenceStructAllocateTemplate.generate(type = self._type,
                                                                         name = self.name,
                                                                         structDict = self.structDict)

    def dealloc(self) -> str:
        return FreeTemplate.referenceLocalTemplate.generate(name = self.name)


BasicOptimizer = NetworkOptimizer(passes = [])

includeList = ["DumpOMath.h"]


class BasicPlatform(DeploymentPlatform):

    def __init__(self,
                 BasicMapping = BasicMapping,
                 DataTypes = DataTypes,
                 BasicTypeInfer = BasicTypeInfer,
                 SimpleNetworkBuffer = SimpleNetworkBuffer,
                 SimpleGlobalBuffer = SimpleGlobalBuffer,
                 SimpleStructBuffer = SimpleStructBuffer,
                 includeList: List[str] = includeList):
        super().__init__(BasicMapping, DataTypes, BasicTypeInfer, SimpleNetworkBuffer, SimpleGlobalBuffer,
                         SimpleStructBuffer, includeList)
