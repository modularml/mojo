# Registering and serving custom model architectures with MAX

This example demonstrates how to register custom model architectures with MAX
and serve them using the MAX serving infrastructure. MAX provides an extensible
framework for defining custom models that can be served with OpenAI-compatible
endpoints, enabling you to deploy your own model architectures alongside
standard models.

For this example, we're using one of the already-registered architectures in
MAX: `Qwen2ForCausalLM`. This architecture already exists, but the local
version will override the built-in one. This demonstrates the general
structure and process for bringing a model that doesn't already exist in MAX to
the framework.

Registering your custom model architecture with MAX's model registry makes it
available for loading and serving. Once a custom model has been defined in a
directory, the flag `--custom-architectures [directory]` applied to
`max serve` or `max generate` entrypoint will
load the Python code for your custom MAX model and try to use it with weights
you link to.

A single [Pixi](https://pixi.sh/latest/) command runs the two variants of this
example:

```sh
pixi run generate
pixi run serve
```

Both the `generate` and `serve` examples use the weights and hyperparameters
from the Hugging Face model `Qwen/Qwen2.5-0.5B-Instruct`. The former directly
generates a text completion from a prompt, while the latter starts an OpenAI
API compatible serving endpoint.

A full invocation within an environment with MAX installed looks like:

```sh
max serve --custom-architectures qwen2 --model Qwen/Qwen2.5-0.5B-Instruct
```

For more information, see the
[Serve custom model architectures](https://max.modular.com/develop/serve-custom-model-architectures/)
tutorial.
