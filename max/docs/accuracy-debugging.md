# Debug MAX model accuracy

This guide walks you through debugging numerical accuracy issues in MAX
pipelines by comparing intermediate tensor outputs between the MAX model and
the corresponding PyTorch (Hugging Face) reference implementation.

## Process overview

If a MAX pipeline produces outputs that don't match the PyTorch reference
implementation, follow this guide to identify the source of divergence.
The general procedure is:

1. Run `debug_model` for the pipeline in both frameworks to dump intermediate
   tensors.
2. Compare the tensor outputs side-by-side to find a suspect layer.
3. Add fine-grained logging to the suspect layer's intermediate tensors.
4. Run `debug_model` again to export detailed tensor data.
5. Run `compare_tensors` with the exported tensor data to calculate the
   numerical differences between the PyTorch reference model and MAX model.

The `debug_model` and `compare_tensors` tools both reside in
`max/tests/integration/tools/`. See the
[tools README](/max/tests/integration/tools/README.md) for basic usage.

## Prerequisite: Add `PipelineOracle` (new models only)

If you're debugging a model that's fully supported in MAX already, you can skip
this step because it should already provide a `PipelineOracle`.

To instantiate both the MAX and PyTorch versions of your model, the
`debug_model` tool depends on a `PipelineOracle` class in
`max/tests/integration/tools/create_pipelines.py`.

To add a `PipelineOracle` for your model, add an entry to the
`PIPELINE_ORACLES` dictionary in `create_pipelines.py`. For example:

```python
PIPELINE_ORACLES: Mapping[str, PipelineOracle] = {
    # ... existing oracles ...
    "my-org/my-model": GenericOracle(
        model_path="my-org/my-model",
        device_encoding_map={"gpu": ["bfloat16"]},
        config_params={"max_length": 8192, "trust_remote_code": True},
    ),
}
```

For multimodal models or models requiring custom preprocessing, you might need
to create a custom `PipelineOracle` subclass. See existing implementations
like `InternVLPipelineOracle` or `Qwen2_5VLPipelineOracle` as examples.

## Step 1: Dump the intermediate tensors with `debug_model`

If your model requires weights from Hugging Face, make sure you've set your
[Hugging Face access
token](https://huggingface.co/docs/hub/en/security-tokens):

```bash
export HF_TOKEN="hf_..."
```

Then, use `debug_model` to dump the intermediate tensors from both the MAX and
PyTorch versions of the model. By default, the tool prints an abbreviated
tensor representation to the console. Start by capturing that into a file for
both MAX and PyTorch versions:

```bash
bazel run //max/tests/integration/tools:debug_model -- \
    --framework max \
    --pipeline google/gemma-3-1b-it > max_tensors.log
```

```bash
bazel run //max/tests/integration/tools:debug_model -- \
    --framework torch \
    --pipeline google/gemma-3-1b-it > torch_tensors.log
```

## Step 2: Compare to find the area of divergence

Open both log files and look for the first point where the tensors meaningfully
diverge. That's typically where the bug is.

**Tip:** Load these files into an LLM and ask it to find the divergences.

**GOOD—Matching outputs:**

```output
# MAX output
model.layers.0.mlp.fc1-output = tensor([[[-2.0156, -3.8125, ...

# PyTorch output (should closely match)
model.layers.0.mlp.fc1-output = tensor([[[-2.0156, -3.8125, ...
```

**BAD—Divergent outputs:**

```output
# MAX GELU output
model.layers.0.mlp.gelu-output = tensor([[[-4.32e-02, 0.00e+00, ...

# PyTorch GELU output (values differ significantly)
model.layers.0.activation-output = tensor([[[-4.42e-02, -2.61e-04, ...
```

## Step 3: Add fine-grained print logging

The above logs reveal intermediate tensors only at layer boundaries (such as
`MLP`, `Attention`, or `Linear`). To pinpoint the exact operation causing the
divergence, you need to inspect the values *within* the suspect layer (module).

For example, if the MLP layer output diverges but its input matches, you should
print the tensors at each step inside the MLP block.

### Add prints in MAX

In your MAX model code, add `TensorValue.print()` calls for each op:

```python
class MLP(Module):
    def forward(self, x: TensorValueLike) -> TensorValue:
        x_tensor = TensorValue(x)
        x_tensor.print("mlp_input")

        gate_out = self.gate_proj(x_tensor)
        gate_out.print("mlp_gate")

        activated = self.activation_function(gate_out)
        activated.print("mlp_activated")

        # ... rest of implementation
```

### Add prints in PyTorch

The `debug_model` tool uses PyTorch's hook API to capture module inputs and
outputs externally, without modifying source code. To add `torch.save()` calls
*inside* a module's `forward()` method, you need to edit the source file.

The code lives in the `transformers` package, which is read-only by default.
Make it editable with this script:

```bash
bash oss/modular/utils/local_transformers_setup/setup_local_transformers.sh
```

To find the model's source code, refer to the `torch_tensors.log` from the first
step above, where you'll see it at the top:

```output
================================================================================
Model class: transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM
Model source file: /path/to/transformers/models/gemma3/modeling_gemma3.py
================================================================================
```

Open that `.py` file and add `torch.save()` calls for each op, and save
to the `torch_debug` directory:

```python
class Gemma3MLP(nn.Module):
    def forward(self, x):
        import os

        os.makedirs("torch_debug", exist_ok=True)

        torch.save(x, "torch_debug/mlp_input.pt")

        gate = self.gate_proj(x)
        torch.save(gate, "torch_debug/mlp_gate.pt")

        activated = self.act_fn(gate)
        torch.save(activated, "torch_debug/mlp_activated.pt")

        # ... rest of implementation
```

**Important:** Name the saved tensors identically in both frameworks to enable
automatic matching with `compare_tensors` (next step). For example,
`x_tensor.print("mlp_A_input")` in MAX matches
`torch.save(x, "torch_debug/mlp_A_input.pt")` in PyTorch.

## Step 4: Export full tensors with `debug_model`

With the new print statements in place, run `debug_model` again to export the
full tensors for numerical comparison.

For MAX, add the `-o` option to save the full tensor outputs to `.max` files
in the `max_tensors` path:

```bash
bazel run //max/tests/integration/tools:debug_model -- \
    --framework max \
    --pipeline google/gemma-3-1b-it \
    -o max_tensors/ \
    > max_tensors.log
```

For PyTorch, you don't need the `-o` option because you specified the path for
the `.pt` files in the `torch.save()` code above:

```bash
bazel run //max/tests/integration/tools:debug_model -- \
    --framework torch \
    --pipeline google/gemma-3-1b-it \
    > torch_tensors.log
```

You don't need the `.log` file for the next step, but we capture it again
even if only to keep the console clean.

## Step 5: Compare the tensors with `compare_tensors`

Now you can now use `compare_tensors` to match corresponding tensor files by
name and calculate the numerical differences between them:

```bash
bazel run //max/tests/integration/tools:compare_tensors -- \
    --torch-tensor torch_tensors/ \
    --max-tensor max_tensors/
```

This reports metrics such as absolute and relative differences for each tensor
pair, helping you pinpoint exactly where and how the outputs of the two
implementations diverge.

**Example output:**

```output
Found 6 matching tensor pair(s)

Tensor: mlp_A_input vs mlp_A_input
Shapes: torch=(507, 1152), max=(507, 1152)
Greatest absolute difference: 0.354 at index (246, 941)
Greatest relative difference: 0.012 at index (100, 500)

Tensor: mlp_B_gate vs mlp_B_gate
Shapes: torch=(507, 6912), max=(507, 6912)
Greatest absolute difference: 0.102 at index (151, 2146)
Greatest relative difference: 0.008 at index (151, 2146)

...
```

That's it. You can iterate through steps 3 - 5 until you find and solve the
issue, or you have enough information to submit a detailed bug report.

When you're done, remember to restore the read-only transformers installation:

```bash
bash oss/modular/utils/local_transformers_setup/cleanup_local_transformers.sh
```

## Common issues

Here are some of the most common accuracy bugs to look for.

### Kernel bugs

A kernel implementation might have subtle numerical differences from the
reference. This is most common.

### Weight adapter issues

Check that weights are loaded and transformed correctly.

### FMA contraction differences

MAX enables FMA (fused multiply-add) contraction by default via the LLVM
`contract` fastmath flag (set in `Mojo/lib/KGENToLLVM/LLVMLoweringUtils.h`).
This allows the compiler to fuse a `fmul + fadd` into a single FMA instruction,
which rounds once instead of twice.

For `bfloat16` operations, this produces results that differ from PyTorch by up
to 1 ULP because PyTorch rounds after each operation while the FMA keeps the
intermediate product at full precision. The max error is typically within the
BF16 ULP range for the input magnitudes (for example, 0.0625 = 2^-4).

This is **by design** — the FMA result is more accurate than PyTorch's
double-rounding. If you see element-wise differences in `bfloat16`
multiply-add patterns, verify this is the cause before investigating further:

- Compare both results against a `float64` reference — the MAX result should be
  closer.
- Splitting the same ops across separate graphs (forcing intermediate
  materialization) should match PyTorch exactly, confirming FMA contraction as
  the source.

> [!NOTE]
> This does not affect model-level metrics (perplexity, KL divergence, eval
> scores). Only flag this as a bug if model-level accuracy is degraded.

### Dtype mismatches

Look for places where dtypes are cast incorrectly. A common issue is performing
an operation in `float32` when `bfloat16` is expected (or vice versa).

### Config discrepancies

Ensure MAX model configuration matches Hugging Face config exactly:

- Keyword parameters (`rope_scaling` type, activation function names)
- Numeric parameters (`head_dim`, `hidden_size`, etc.)
- Feature flags (`use_cache`, `tie_word_embeddings`, etc.)

## Additional options

Here are some additional tooling tips.

### Configure MAX debug print options

If you're writing custom scripts or want different printing behavior from MAX,
you can configure it with `InferenceSession.set_debug_print_options()`.

The style you set with `set_debug_print_options()` determines where
`TensorValue.print()` sends output:

**Console output:**

- `COMPACT`: Abbreviated output showing tensor corners and shape (default)
- `FULL`: Complete tensor contents, with configurable decimal precision

**File output:**

- `BINARY_MAX_CHECKPOINT`: Saves `.max` files with dtype/shape metadata
(recommended for `compare_tensors`)

- `BINARY`: Raw buffer files without metadata (you must track dtype/shape
  separately)

For example:

```python
from max.engine import InferenceSession
from max.engine.api import PrintStyle

session = InferenceSession(...)

# Abbreviated output to console - shows corners and shape (default)
session.set_debug_print_options(style=PrintStyle.COMPACT)

# Full tensor contents to console (with configurable decimal precision)
session.set_debug_print_options(
    style=PrintStyle.FULL,
    precision=8,  # digits of precision (default: 6)
)

# Save as MAX checkpoint files (recommended for compare_tensors)
session.set_debug_print_options(
    style=PrintStyle.BINARY_MAX_CHECKPOINT, output_directory="/tmp/max_output"
)

# Raw binary buffer (loadable with numpy.frombuffer, but requires
# you to specify dtype and shape when loading)
session.set_debug_print_options(
    style=PrintStyle.BINARY, output_directory="/tmp/max_output"
)
```

### Use print hooks directly

The `debug_model` tool handles hook setup automatically, but you may want
to use the print hook API directly when:

- Integrating tensor inspection into your own test scripts
- Filtering to print only specific layers (using the `filter` parameter)
- Debugging during model development before the pipeline is fully wired up
- Needing programmatic control over when hooks are attached and removed

Here's how to use the hooks directly in your code:

#### MAX `PrintHook`

```python
from max.nn.hooks import PrintHook

# Create hook and name layers
hook = PrintHook()
hook.name_layers(model)  # Names all layers based on their attribute path

# Build and run graph...

# Clean up
hook.remove()
```

#### PyTorch `TorchPrintHook`

```python
from test_common.torch_print_hook import TorchPrintHook

# Create hook with optional export path
hook = TorchPrintHook(export_path="/tmp/torch_tensors")
hook.name_layers(model)

# Run model - tensors are automatically saved

# Clean up
hook.remove()
```

### Debug with fewer layers

By default, `debug_model` runs with only 1 hidden layer to speed up debugging.
This is usually sufficient since bugs often appear in the first layer. If
needed, you can increase the layer count with `--num-hidden-layers`:

```bash
# Use 3 hidden layers
bazel run //max/tests/integration/tools:debug_model -- \
    --framework max \
    --pipeline google/gemma-3-1b-it \
    --num-hidden-layers 3

# Use all layers (full model)
bazel run //max/tests/integration/tools:debug_model -- \
    --framework max \
    --pipeline google/gemma-3-1b-it \
    --num-hidden-layers all
```

## FAQ

### How do I match tensors between MAX and PyTorch?

As you compare the tensors, you may notice that there are areas where the
PyTorch/HuggingFace model structure doesn’t line up perfectly with the MAX
model structure. It’s common that one model will have layers that aren’t
visible in the other model. Unfortunately, there’s no easy way to tell if this
is a bug or not without reading the code. In this situation, you have two
alternatives:

- Pause your manual comparison to read through the code directly. Figure out
why one model has more visible layers than the other model. (For example,
perhaps MAX caches `k` and `v`, so only the `q` layer appears in the model
dump.) When you see that there’s a valid reason that the architecture diverges,
you can move on.

- Skip the architectural divergence and find the next shared layer. (Using the
above example, where MAX caches `k` and `v`, you might see everything sync up
again at `layer_norm`.) Once you find a sync point, you can typically move on,
being confident that the architectural divergence was not a bug.

### What if I can't see the divergence in abbreviated output?

First try these:

- Check the details of the failure in verification. Are you running on the same
hardware with the same prompt? You can pass the —-prompt into debug_model to
specify.

- `debug_model` defaults to `--num_layers=1` for the model, but some bugs show
up on later layers. Try increasing `--num-layers` and look again.

If neither of those help, you can numerically compare layers with
`compare_tensors` to look for the divergence. This is tedious, and we recommend
to approach with a binary search. First, rerun `debug_model` to save the full
tensors with an `-o` flag for each framework:

```bash
bazel run //max/tests/integration/tools:debug_model -- \
--framework torch \
--pipeline google/gemma-3-1b-it \
--output torch_tensors/

bazel run //max/tests/integration/tools:debug_model -- \
--framework max \
--pipeline google/gemma-3-1b-it \
--output max_tensors/
```

Then start testing tensors that you expect to match, for example

```bash
bazel run //max/tests/integration/tools:compare_tensors -- \
--torch-tensor '/home/ubuntu/modular/torch_tensors/0/model.lm_head-output.pt' \
--max-tensor '/home/ubuntu/modular/max_tensors/model.language_model-output_0.max'
```

**NOTE**: Sometimes if a tensor requires a simple reshape to be compared, the
`--allow-reshape` flag on `compare_tensors` can handle it.

When you see output `rtol` and `atol` metrics start to diverge, you have a
place to start looking as described in the next section.
