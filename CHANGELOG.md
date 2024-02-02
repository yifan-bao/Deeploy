# Changelog

# v0.1.0

## PR PULP Support

### Added

* Initial support for Siracusa PULP Platform
* Bindings for PULP-NN Convolution, Linear, MaxPooling kernels
* New Submodule DeeployPULPLibs
* Support for global function definitions
* Support for global function definition hoisting
* Support for global function declaration code generation in `generateGlobalDefinitionCode`
* Auto Closure extension which automatically generates function definition for trampolining kernel calls
* CI for Siracusa implementation

### Fixed

* NetworkContext functions that return input or output nodes now check first whether the object
* NodeTemplate class no longer implements `__deepcopy__`, but composes a _Template class that wraps a copiable mako Template
* NodeMapper no longer implements `__deepcopy__` method since there is no use-case. Previous implementation continues to exist as a comment.

### Changed

* Generic platform uses AutoClosure for add nodes in order to achieve higher test coverage

## PR ITA Support for MemPool

### Added

- Support for ITAMax kernel (MemPool and Basic platform)
- Support for ReduceSum (Basic platform)
- Support for RQMatMul (MemPool platform)
- Support for GEMM (MemPool platform)
- Support for RQGEMM (MemPool platform)
- Support for RequantShift (MemPool platform)
- Added MatMul-RQS and GEMM-RQS merge passes
- Added `nodeOps` to parserDict
- Added tests for ReduceSum and ReduceMean kernel (Generic, CMSIS and MemPool platform)
- Added ITA_PE defines to cmake flow
- Added new tests cases to CI
- Implemented `computeOps` for `ConvLayer` and `GEMMLayer`
- Add multilib support for riscv32 compiler-rt

### Changed

- Reimplemented `DebugPrintPass` as context agnostic topology optimization
- Renamed DebugNodes to DebugPrint nodes
- Added `TOOLCHAIN` and `CMAKE_GENERATOR` variable to CI
- Update MHSA to match ICCT topology
- Removed duplicated keys (`node_name` and `node_op`) from `parserDict`\
  (Use `nodeOp` and `nodeName`)
- Extract useful base parser into separate class (`ReduceParser`, `SoftmaxParser`)
- Moved shared code in `parseNodeCtxt` function to `Conv1DParser`, `Conv2DParser` and `MaxPool2DParser` classes
- Extracted several useful base classes for parser and layer classes
- Removed duplicated code

### Fixed
- Correctly handle unsigned inputs in iLayerNorm
- Correctly propagate channel layout in `parseNodeCtxt` function
- Prevent error in `DebugPrintNode` if output tensor is a model output
- Fix potentially wrong shape calculation for `RequantShiftLayer`
- Fix wrong shape extraction in `DebugPrintLayer` for 3D vectors
- Correctly propagate n_levels in `MHSAChecker`
- Correctly calculate shapes for `RQGEMM` and `RQMatMul`
- Fixed issue if axes was negative in `ReduceMeanTemplate`

## PR CI Refactor

### Added

* Support CMAKE flow for QEMU_ARM platform
* Support CMAKE flow for Siracusa platform
* CMAKE uses CCache bindings for faster compilation

### Changed

* CI Test pipeline starts with build stage in order to assure that all dependencies are installed
* All linting and testing jobs are released at after the build_deeploy job
* All tests for each platform are released in parallel
* CCache is used to speed up build process
* CMAKE testing implements each test as a target now
* CMAKE builds in a TEST_{PLATFORM} directory to reuse common kernel libraries
* Third party dependencies now live in deeploy{PLATFORM}libs, rather than deeploytest

## PR Extension: Memory Level Annotation + Deeploy State Import/Export

### Added

* Classes representing a Memory Hierarchy and Memory Levels
* NetworkDeployer extension adding a AnnotateDefaultMemoryLevel pass after binding
* A new pass: AnnotateDefaultMemoryLevel that add a _memoryLevel attribute to every buffer in the context
* Support to import and export Deeploy's states (graph + context)
* Support to compare the equality of two contexts
* Add test for the default memory level annotation and Deeploy's states equality

## PR Optimization Pass Refactor

### Added

* contextaware and contextagnostic decorators for OptimizationPass classes

### Changed

* Optimization passes are now split into NetworkOptimizationPasses and TopologyOptimizationPasses
* NetworkOptimizationPasses work as before
* TopologyOptimizationPasses only work on the graph rather than the ctxt and graph
* The pass system now uses a mixin class system that allows to specify context-awareness
* All optimization pass replacement functions use an "_" prefix now to avoid namespace contamination
* TransposeMatMul and NHWCtoNCHW functions are reimplemented using passes
* To avoid confusion, the optimizer object of the NetworkDeployer class is now called loweringOptimizer

### Removed

* postLoweringOptimization is merged into the lower function and subsequently removed

## PR Typing System Refactor

### Added

* Introduce AbstractDataTypes, including base metaclasses for immediate, pointer and struct types
* Immediates, pointers, and structs share the typeName and typeWidth class attributes, which describe their type name and width
* Immediates may only contain an immediate value, i.e. and integer, float, or complex python primitive
* Pointers are implemented in dynamically generated PointerClasses using the Pointer function
* Structs may contain any combination of Immediates, Pointers, and Structs
* New types are user-definable and implemented in dynamically generated classes
* CMSIS-NN struct types are implemented in CMSISDataTypes.py
* Typechecking for Immediate, Structs, and Pointers is dynamically generated
* Syntactic support for type promotion from dict to StructClasses, strings to PointerClasses and int and float types to ImmediateClasses
* The DataTypeCollection metaclass implements support for concise type definitions

### Changed

* Re-implement int8\_t, int16\_t, int32\_t types as immediate
* All constant tensors are now required to be instances of a subclass of PointerClass
* Structs are now required to be full types
* Pointer-type variables can no longer be assigned zero, use None instead
* Warnings are produced when Pointer-type variables are assigned None
* Structs are no longer hoisted into the global context, but into the local context
* AlignToContext in NodeTemplate now additionally returns a list of hoisted objects

### Removed

* No types may be addressed by a string value anymore
* No types may be implemented as Enum anymore

## PR Typing System Refactor 2

## Added

* Added unsigned integer datatypes (`uint8_t`, `uint16_t`, `uint32_t`) as basic data types
* Added `SignPropTypeChecker` class with required `_inferNumLevels` and `_inferSignedness` extension.
* Added inputTypes to `NetworkDeployer` interface

## Changed

* Removed `input_signed` from `NetworkDeployer` interface
* Removed `input_n_levels` from `NetworkDeployer` interface
* Removed all `typeInfer` functions from Platform definition and type inference flow
* Renamed `inferNumLevels` to `_inferNumLevels` in `SignPropTypeChecker` class
* Renamed `inferSignedness` to `_inferSignedness` in `SignPropTypeChecker` class
* Removed `_signed` and `nLevels` field from `VariableBuffer`
* NetworkContext's `annotateType` now only annotates the type, not signedness or nLevels

## Fixed

* Expanded immediate type checking to `np.array` types for easier and faster type promotion
* Fixed bug where struct type checking assumed "keys" to be a method on passed object

## PR Future System

## Added

* Added the FutureType/FutureClass monad, which extends the PointerType/PointerClass type with dispatching and resolution code of asynchronously produced values.
* Added the FutureBinding class, a generic NodeBinding subclass that exposes the futureAssign method, which assigns a stateReference object to an output Future during late binding
* Added the AutoFutureBinding class, which automatically binds an object in the nodeRep of a template to generated output Futures, if the stateReferenceType matches.
* Added the ExecutionBlock class, which holds a list of NodeTemplates and their nodeReps
* Added a _typeDefRepr method to the StructClass, which formats a print string of the struct that can be used in a typedef statement
* Added experimental support for PULP DMA kernels
* Added experimental support for ONNX slice nodes
* Added a SignPropDeployer class which allows for end-to-end signedness inference during typeChecking

## Changed

* NodeBindings no longer generate code from NodeTemplates. NodeBindings hold an ExecutionBlock which fulfills the same purpose.
* NodeBindings now require a CodeTransformation object which implements the generation of the kernel template's execution context. A minimal CodeTransformation implements MemoryGeneration, ArgumentStructGeneration and FutureGeneration.
* The NodeBinding's bind method is now split into three phases: earlyBinding (algorithms that should run before hoisting of buffers), hoisting, and lateBinding (algorithms that should run after hoisting of buffers)
* Closure generation, Future resolution and dispatch generation, and Memory management generation, including argument struct generation are implemented as CodeTransformationPass classes
* The NodeMapper object no longer owns a nodeRep object. Ownership of the nodeRep object is transfered to the ExecutionBlock bound to a NodeBinding
* The VariableBuffer family no longer generates code directly. Instead, they own a template for initialization, allocation and freeing and the _nodeRep method to generate their binding. This avoids a lot of code duplication and enables delayed binding of memory management code.
* Refactored all signProp templates to only compute bias-pushing offsets if the current flow supports it.

## Fixed

* Fixed a bug in PULPConv templates, where the size of the hoisted transient buffer was too small

## Removed

* Removed all remaining references to the _signed and nLevels attributes in the basic Deeploy flow
* Removed allocLocal and deallocLocal methods from the NetworkContext

## PR Tiling Extension

## Added
### Deeploy
* `computeTransientBufferSize` function in the `NodeTemplate` class, which computes the size of buffers-to-be-hoisted
* BFS algorithm in `MemoryHierarchy` to compute shortest path between to memory levels
* Dependency on `ortools` package
* `GenericFlow`, a class which implements dataflow analysis for generic flow and iterator variable types

### Tiling Extensions
* `PatternConstraintFlow` which implements a memory liveness dataflow for a pattern, which is a graph of nodes
* `GraphConstraintFlow` which implements memory liveness dataflow over a graph of patterns
* `MemoryConstraint` classes which represent the liveness and location of tensors throughout the tiling Flow

* `TilerModel` class, which wraps the operation of an ortools solver
* `TileConstraintFlow` class, which propagates node-level constraints between inputs of an operator and its outputs
  * Constraints are split between geometric and policy constraints, the former of which are strictly required for the operator, the latter are down to preferences
  * Various TCF implementations, including common CNN operators for PULP
* `Tiler` class, which gathers geometric and memory constraints for the TilerModel

## Changed
* `hoistTransientBuffers` now is required to call `computeTransientBufferSize` for size calculation
* Tensor usage (`buffer._users` field) is now also indicated for `ConstantBuffers`

## Fixed
* Fixed return type of `PULPRQSGEMM` and `PULPRQSCONV` Layers
* Various type annotations

## PR Tiling Codegen Extension

Introduces basic, layerwise, single-buffered tiling code generation for L2 - L1 tiling on PULP alongside general primitives for encoding tiling solutions and implementation in `CodeTransformationPasses`.

## Added
* `TilingVariableReplacement` `CodeTransformation` which captures tiled variables from templates and replaces them with references.
* `TilingCodeGeneration` `CodeTransformation` which generates a tiling loop including dma calls, synchronization and kernel execution.
* `PULPClusterTilingGeneration` `CodeTransformation` which implements `TilingCodeGeneration` for L2 - L1 tiling on the PULP Cluster.
* `TilingCodegen` as a tiler extension which formalizes the solution of the tiling problem into a variable replacement scheme and a tiling schedule.
* `IntrospectiveCodeTransformations` can now use the `_reconstructCode` method to edit Mako templates before generating.

## Changed
* `Closure`s return a `ClosureExecutionBlock` which preserves the original `nodeRep`
* `MemoryManagementGeneration` now only captures buffers without memoryLevel annotation if regex is not set.
* Network inputs and outputs are no longer owned by the `NetworkContext`, but by the `NetworkContainer` to avoid confusion between memory arenas and memory IO


## Fixed
* Smaller issues in `MemoryLevelDeployer` and `TilerAwareDeployer`

## PR Tiling Doublebuffering Codegen Extension

Introduces code generation for double-buffered tile execution on the PULP Cluster for L2 - L1 tiling.

## Added
* `PULPSynchCoresPass` code transformation pass which adds a `pi_cl_team_barrier()` to the code
* `PULPClusterTilingDB` code transformationg which implements double-buffered tiling code generation
* Double-buffered code test in `gitlab-ci.yml`

## Changed
* `NetworkContext`s' `localObject`s and `globalObject`s are now OrderedDicts, representing the order in which buffers are added.
* All `Closure` and `IntrospectiveCodeTransformation` are changed to return a stable order of objects aligned with the `NetworkContext`.
* `TilingCodeGeneration` passes are now stackable and return an additional boolean which indicates whether tiling was applied.

## Fixed
* Some type annotations in `TilingCodegen` were fixed

## PR Tiling More Kernels

Implements support for convolutions, depthwise convolutions and maxpool operators to be run as tiles

## Added
* `serializeTilingSolution` implementations for MaxPool, Conv2D and Depthwise Conv2D operators on PULP
* `UntiledTileConstraintFlow` to implement untiled operator support in the general tiling flow
* Code generation for DMA channel allocation, assignment and release in tiling code generation for PULP
* Support for correct placement of transient buffers in L1 on PULP systems
* Several tiled end-to-end tests

## Changed
* Various PULP Conv templates now use immediates for padding, strides etc. rather than lists in code generation, which are automatically and correctly replaced by tiling code generation passes


## Fixed
* Incremental refactoring of `MemoryScheduler`

## PR Tiling Fixups

Fixes up various smaller oversights in previous PRs.

## Added
* Platform-independent performance tracking `CodeTransformationPass` in `CyclesMeasurement.py`
* Support for static arena allocation of the memory map of the outerMemoryScheduler

## Changed
* Change lots of `copy.deepcopy(DICT)` invocations on dictionary to much faster `DICT.copy()` invocations, which reduces overall runtime for big, tiled networks by ~4-5x
* Move PULP-specific `CodeTransformationPasses` into separate directory
* Moved execution context of PULP platforms directly into the Cluster

## Fixed
* Fixed hoisted tile variable type for PULP DMA Structs

## PR Tiling Transformers

Implement support for iSoftmax & end-to-end transformers on PULP Cluster

## Added
* Mapping & Bindings for iSoftmax operator on PULP Platforms
* iSoftmax kernel for PULP Platforms
* iSoftmax TileConstraintFlow
* Encoder layer and iSoftmax layer regression tests

## Fixed
* Bug in iSoftmax Parser where coeffB was parsed wrong
* Bugs in OptimizationPasses where tensors were edited in-place
* GEMMTileConstraintFlow policyConstraints adapted

## PR Autotranspose Extension

Refactor transpose operator handling in PULP; Introduce support for DMA-based transposing
This is a stopgap solution - If DMA-based transposing becomes essential for other platform (who knows...) we should refactor.

## Added
* `BindingOptimizationPasses` which can apply transformations after typechecking, but before tiling
* `AutoTransposePass`, which is a `BindingOptimizationPass`, which toggles Transpose layer code generation off if the following layer receives its input from L1
it also annotates an `in_idx_perm` attribute, which lets the code generation passes know to transpose through DMA transfers
* `PrintDebugPasses` which print tensors at runtime and format according to the shape in the `NetworkContext`
* `TransposeSplitPass` which splits Transpose layers whose output is used multiple times into multiple Transpose layers

## Changes
* Added multicore support in PULP core-based transpose template
* Refactored graph manipulation; `replaceInsertNode` now safely matches the subgraph which gets replaced

## Fixed
* Fixed issue in branching matcher where dual-input single-output nodes would not match

## PR L3 Tiling Support

Introduces end-to-end support for L3/L2 tiling on Siracusa (and other PULP platforms with minor changes).

## Added
* `PULPL3TilingGeneration` handles the generation of L3 - L2 data movement code
* `pi_cl_ram_req_t` Struct type for PULP platforms
* Allocators for L3 RAM, Code to move weights from L3 Flash to L3 RAM
* `AnnotateIOMemoryLevel` is a `NetworkOptimizationPass` which annotates only global inputs and outputs
* Several static helper functions in the `TileConstraintFlow` base class: `extractBaseAddr` calculates address offsets of tiled tensors and `sanitizeTilingSchedule` removes superfluous transfers.
* Several debugging print passes which print formatted intermediate tensors for debugging purposes.
* Utilities to run emulated L3 Flash and RAM with GVSoC in testing.
* Full `TileConstraintFlow` for simple Transposition operations.

## Changed
* `TileConstraintFlow`s `serializeTilingSolution` method not accepts `outputCubes` as its input. The result of this method is the sequence of input tiles required to produce the `outputCubes`.
* `NodeMemoryConstraint`s now save each `TensorMemoryConstraint` as input, output or intermediate. No functional change, this is used for the computation of the `outputCubes` argument in TCFs.
* The `TilerModel`s `_trySetupConstraints` now only checks for feasibility after adding all constraints, improving performance.
* `IntrospectiveCodeTransformationMixIn` now memoizes the template's parser tree to reduce recomputation.

## Fixed
* Compilers struggle with 64 Bit arguments. The `iSoftmax` kernel now uses 32 bit for the `coeffC` argument.
* Added workaround to force word-aligned accesses to L3 in the tiling constraints of depthwise convolutions

## Known Issues
* Autotranspose is known not to work with L3 - L2 - L1 tiling. Fix will follow in another PR.

## PR Tileable Tranpose Templates

Introduces 2-stage generated transpose templates to the PULP Platform. The core loops of tiling are generated in the `alignToContext` method of the `Transpose` template.

## Changed
* Transpose templates are now fully tileable; generated code is largely the same
* `TransposeTileConstraintFlow` no longer enforces untiled transpositions
* Switched hoisting position of `cluster_dev` variable to occur after binding
* `FlattenTileConstraintFlow` no longer imposes any geometrical dependency, since it's an inplace operation / nop

## PR L3 Tiling Doublebuffering

Introduces the `PULPL3TilingDB` `CodeTransformation` which, similar to `PULPClusterTilingDB`, implements double buffered L3 - L2 DMA transfers

## Added
* `PULPL3TilingDB` `CodeTransformation`, which implements L3 double-buffered tiling

## Changed
* Aligned several tests with more current, less verbose setup code
