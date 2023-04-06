# Deeploy: A tool to dump C-Code for Neural Network inference on heterogeneous platforms #

Deeploy is a tool designed to facilitate the deployment of ONNX-Graphs to C-programmable platforms.
The tool is designed to allow users to work with any **Quantized Neural Network (QNN)** in the **ONNX** format,
from initial optimization all the way over lowering to kernel binding.

The main mechanisms that enable this in Deeploy is a hierarchical separation of responsibilities.
This document is meant to outline the general flow used in Deeploy and also highlight where users can
insert their own extensions to the framework.

## Frontend ##

The frontend is the software responsible to train, quantize and export a neural network.
Deeploy is generally frontend agnostic - currently, only ONNX graphs are supported and the parsers that are implemented
are targeted towards the export format of _Quantlib_.

## Middleware ##

The middle ware is responsible to optimize and lower the ONNX graph produced by the frontend to a representation that can be
deployed in a node-wise manner. An instance of middle ware lowering would be the merging of convolution and requant-shift
operations as found in Deeploy. The middleware is explicitly NOT responsible for scheduling, tiling, typechecking and strict parsing of the input graph.

Currently, the middleware consists of optimization passes that currently offer fair support for sequential passes, i.e. passes that modify subgraphes where each node is sequentially connected to **one** other node.

## Backend ##

The backend is responsible for taking a node-by-node deployable ONNX graph and lowering it to C-Code. To this end, Deeploy
provides a connected series of abstractions that are used to describe matchable patterns through the following functionalities:

## Platforms ##

The main user object in Deeploy are platforms. In essence, a platform provides Deeploy the information for the following:

1. A layer mapping - given an onnx node, which type of layer do I expect?
2. A list of NodeMapper objects for each layer - given a type of layer, which kernels might this be mapped to?
3. A set of Variable, Constant and StructBuffer types - These objects tell Deeploy how to generate allocation and deallocation code
4. An optimizer - this object is Deeploy's middleware and transforms a graph to something deployable

So, for each new platform you might want to implement, you will need to implement or re-apply these four concepts.

## Deployment ##

### Layer Mapping ###

Each ONNX layer needs to first be identified by its Opname - and then mapped to an internal representation of class
ONNXLayer. Each ONNXLayer is charged with allocating and deallocating its buffers and generating its code. Besides this,
each layer has rules to broadcast its inputs to the appropriate shape.

### Parsing ###

ONNX nodes are fairly diverse and powerful representations; A convolution node, for example, representes _any_
1/2/3-D convolution, using any combinations of padding, striding, kernel size, dilation, groups and many more attributes.
While this is a great way to simplify the export of ONNX graphs, most embedded platforms do not offer support for
all these combinations of attributes.
The parsing steps in Deeploy is designed to ensure that a graph can be run on the target platform and to make sure that the
node's constant inputs (if there are any) are known at compile time. This is done using the NodeParser class in Deeploy.

#### NodeParser ####

The NodeParser has two main functionalites - it has a parseNode function that check the context-free attributes of a node,
e.g. if the platform only supported 2D-convolutions with dilation=1 and square padding, the matching parser's parseNode function would be responsible for only accepting nodes that feature combinations that pass these constraints.

The second function of the NodeParser is to check the context-bound attributes of a node. This means, that the
parseNodeCtxt function ascertains that all inputs of the node are in the context when the node is processed and the
outputs are not - otherwise the graph isn't ordered or there is a naming conflict.
Currently, this function also ensures that all required names for the kernel exist - more on that later.

Since most of these steps are very similar between different parsers, most new parsers can be implement as subclasses of
BasicParsers, so make sure to check them out before writing new ones!

Once the graph is successfully parsed, we know that our platform is able to execute the topology of the graph.
This alone is however not enough to execute the graph correctly, we still need

### Type Checking ###

Type Checking is not only best practice, but necessary to bind the correct kernels. Deeploy internally represents
every tensor, i.e. inputs and outputs of operations with a number of Levels, the logarithm of which is the lowest
number of bits required to represent the tensor. Strict type checking support in Deeploy requires that every input and output
of every kernel binding is annotated with a data type. Deeploy will then choose the first matching kernel binding
that is available and fulfills the bitwidth constraints. Deeploy will then propagate the new constraints given by the
corresponding data type on the output nodes. In this way, Deeploy ensures that not only every operation is executable but also
the full graph.

#### NodeTypeChecker ####

Implementing a new NodeTypeChecker is easy - all it has to implement is a function inferNumLevels that, given all input
numLevels and the layer configuration (parserDict), computes all output numLevels.

### Binding ###

Binding is the process of of translating a layer's representation to a C-Kernel. To facilitate this, Deeploy bundles
each template with a type checker and calls the combination a _binding_. Once every layer in the ONNX graph is _bound_,
we can deploy the network with simple code generation, using the template.
