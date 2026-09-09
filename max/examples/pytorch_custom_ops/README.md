# Extending PyTorch with custom operations using Mojo or MAX graphs

Custom operations in PyTorch can now be written using Mojo or MAX graphs,
letting you experiment with new GPU algorithms in a familiar PyTorch
environment. Mojo custom operations are registered using the
[`CustomOpLibrary`](https://max.modular.com/api/python/generated/max.experimental.torch.CustomOpLibrary)
class in the `max.torch` package, and MAX graphs use the `@graph_op` decorator
from `max.torch`.

These examples show how to register and use custom operations in PyTorch
models, from very basic calculations to complex image processing and full model
implementations. These examples require a system with a [MAX-compatible
GPU](https://max.modular.com/faq/#gpu-requirements)

The five examples of PyTorch custom operations consist of:

- **addition.py**: A very basic example, where a Mojo custom operation that
  adds a constant value to every element of an input tensor is defined and run.
- **grayscale.py**: An image processing example that converts RGB images to
  grayscale using a custom Mojo kernel. This demonstrates how to process
  real-world data with automatic vectorization and parallelization.
- **torch-grayscale.py**: A simplified grayscale conversion example that uses
  MAX graph operations (`@graph_op`) instead of custom Mojo kernels. This shows
  how to build PyTorch custom operations using MAX's built-in graph operations
  for quick prototyping and experimentation.
- **whisper.py**: A PyTorch implementation of the
  [Whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper)
  speech model, with its attention layer replaced by a custom fused attention
  operation defined in Mojo. This shows how Mojo custom operations could be
  integrated into the context of a larger PyTorch model.
- **graph.py**: A basic example of providing a MAX graph as a custom PyTorch
  operator. In this case, the graph contains a single operation, the built-in
  matrix multiplication provided by `ops.matmul` in MAX.

You can run these examples via [Pixi](https://pixi.sh):

```sh
pixi run addition
pixi run grayscale
pixi run torch-grayscale
pixi run whisper
pixi run graph
```

or directly in a Python virtual environment where the `max` PyPI package and
PyTorch have been installed:

```sh
python addition.py
python grayscale.py
python torch-grayscale.py
python whisper.py
python graph.py
```
