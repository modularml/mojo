# Using the MAX C API to execute a precompiled graph

The [MAX graph API](https://max.modular.com/graph/) provides a powerful
framework for constructing computational graphs to be run on GPUs, CPUs, and
more. The process of compilation and execution can be separated. This example
demonstrates how to construct and compile a graph using the MAX Python API,
serialize the compiled graph to disk, and then load and execute the graph from
a C program.

> [!NOTE]
> The C interfaces shown here are undergoing rapid development and may change
> significantly over time. Additionally, the MEF file format is not a general
> serialization solution. It is device-specific and can only be transferred
> between two similar systems.

Within `test_capi.py`, a simple graph is constructed that performs the addition
of two vectors. A symbolic dimension of `vector_width` is specified, which
later allows variable sizes of vectors to be input to the graph, as long as
the two input vectors are the same size. The graph itself is configured to be
run on a GPU, and the input and output tensors are specified to reside on the
GPU. The graph is then compiled and serialized to a MEF file on disk.

Inside `example.c`, host tensors for the two vectors are initialized with
values and then transferred to the accelerator. The compiled graph is loaded
from the previously-saved MEF file, placed on the accelerator, and run using
the input tensors. The resulting tensor is moved from device to host, and the
results printed and compared against a reference.

A [Pixi](https://pixi.sh/latest/) command is available to run the entire
process of building the graph, compiling it, saving it to disk, and then
finally executing it on the GPU:

```sh
pixi run test
```

## Capturing a MEF with the new `max.experimental` API

`test_capi_v3.py` is the new-API counterpart to `test_capi.py`. It builds the
same vector-add graph as a `max.experimental.nn.Module`, compiles it with
`Module.compile()`, and exports the compiled artifact to a MEF file using the
public `CompiledModel.export_mef()` method. The resulting `graph.mef` is
consumed by the same `example.c` executor: both APIs name graph inputs
`input0`/`input1` and the output `output0`, so the C code is unchanged.

`export_mef()` serializes straight from the compiled artifact, so it does not
require the model to be initialized on a live device. This makes it usable in
the cross-compilation and virtual-device scenarios that production serving via
the MAX C API relies on.

## Loading models with external weights

The `weights_example.c` file demonstrates how to provide model weights at
runtime using the C API's weights registry. This is the mechanism used when
a model's weights are stored separately from the compiled graph (for example,
in safetensors or GGUF files).

The Python script `test_weights_capi.py` builds a graph that references an
external weight via `ops.constant_external()`, compiles it to a MEF file, and
then runs the C program. The C program creates a `M_WeightsRegistry` with the
weight data and passes it to `M_initModel()`.

This example runs on CPU and does not require a GPU.

## Loading weights from a Safetensors file

The `safetensors_example.c` file builds on the weights registry by loading the
weight data directly from a Safetensors file instead of an in-memory array.

The Python script `test_safetensors_capi.py` builds the same
`ops.constant_external()` graph, compiles it to a MEF file, and writes a
`weights.safetensors` file containing the `weight` tensor. The C program loads
the file with `M_loadSafetensors()`, inspects the tensors it contains, and
builds a `M_WeightsRegistry` from every tensor in one step with
`M_newWeightsRegistryFromSafetensors()` before passing it to `M_initModel()`.

The tensor names in the file must match the external weight names in the graph;
this example performs no name translation. Weights are loaded onto a host
device, and the runtime copies them to the model's device during
initialization.

## Device graph capture and replay

`graph_capture.c` demonstrates using device graph capture (e.g. CUDA graphs) to
reduce kernel launch overhead for repeated model execution. This technique
records a model execution into a replayable graph, then replays it with
near-zero launch overhead on subsequent runs.

The example shows the three key operations:

1. **Capture** (`M_captureModelSync`): Run the model once and record the
   execution as a device graph. Returns output tensors that are updated
   in-place on each replay.
2. **Replay** (`M_replayModelSync`): Re-execute the captured graph using the
   same input buffers. Results appear in the output tensors from capture.
3. **Debug verify** (`M_debugVerifyReplayModelSync`): Run eagerly and compare
   the kernel launch trace against the captured graph to verify correctness.

> [!NOTE]
> Device graph capture requires a CUDA or HIP GPU. It is not available on
> Apple GPUs or CPU-only systems.
