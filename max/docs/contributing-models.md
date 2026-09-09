# Contributing new model architectures

MAX comes with built-in support for many popular model architectures like
`Gemma3ForCausalLM`, `Qwen2ForCausalLM`, and `LlamaForCausalLM`. You can also
contribute new model architectures you can then serve natively with MAX
using the `max serve` command.

This document walks through the process to create a new model architecture and
register it with MAX for serving. It's focused on the project setup and
developer workflow—for an API programming guide using the MAX Python API, see
the
[custom model architectures tutorial](https://max.modular.com/develop/serve-custom-model-architectures/).

## 1. Set up your development environment

See the [MAX framework developer guide](/max/docs/development.md) to
get started. (Fork the repo, clone it, and create a branch.)

Then navigate to the architectures path:

```bash
cd modular/max/python/max/pipelines/architectures
```

## 2. Create your architecture directory

Create a new directory for your architecture (e.g., `my_model/`) with the
required file structure:

```text
my_model/
├── __init__.py          # Exports your architecture for discovery
├── arch.py              # Defines the SupportedArchitecture configuration
├── model.py             # Implements the main model logic and graph building
├── model_config.py      # Handles model configuration and parameter parsing
└── weight_adapters.py   # Converts weights between different formats
```

> [!TIP]
> Copy an existing architecture folder and rename it to your custom architecture
> name, then customize it as needed.

### Requirements

Your custom architecture must:

1. **Follow the naming convention**: The architecture name in `arch.py` must
exactly match the model class name in your Hugging Face model's configuration.

2. **Implement required methods**: Your model class must inherit from
`PipelineModel` and implement the required methods.

3. **Handle weight conversion**: Provide weight adapters for supported formats
(at minimum SafeTensors).

4. **Include proper configuration**: Handle parameter mapping from Hugging Face
config to your internal format.

For more information about how to build your model see our
[custom model architectures tutorial](https://max.modular.com/develop/serve-custom-model-architectures/).

### Test your architecture

While developing the model, you can use the `--custom-architectures` flag to
run your model (before it's registered):

```bash
./bazelw run //max/python/max/_entrypoints:pipelines -- serve \
  --model your-org/your-model-name \
  --custom-architectures path/to/your/architecture
```

This `entrypoints:pipelines serve` command is the equivalent of [`max
serve`](https://max.modular.com/cli/serve) except it runs directly from
your local code.

After you register your architecture with MAX alongside the other models,
this `--custom-architectures` option is no longer needed.

## 3. Register your architecture

Add your architecture to the main
[`architectures/__init__.py`](/max/python/max/pipelines/architectures/__init__.py)
file:

```python
# Add import
from .my_model import my_model_arch

# Add to architectures list
architectures = [
    # ... existing architectures ...
    my_model_arch,
    # ... rest of architectures ...
]
```

Once registered, you can serve models using your architecture without the
`--custom-architectures` option:

```bash
./bazelw run //max/python/max/_entrypoints:pipelines -- serve \
  --model-path your-org/your-model-name
```

For models that require custom code execution (such as custom tokenizers or
model implementations on Hugging Face), add the `--trust-remote-code` flag:

```bash
./bazelw run //max/python/max/_entrypoints:pipelines -- serve \
  --model-path your-org/your-model-name --trust-remote-code
```

## 4. Verify output logits

While we generally recommend validating the end-to-end correctness of a model
using an evaluation harness (see below for more on this), it can be handy to
verify portions of the model against a reference implementation during
development.

To compare against a PyTorch reference, you can use the following logit
verification workflow:

```bash
# 1. Generate logits with MAX pipeline
./bazelw run //max/tests/integration/tools:generate_llm_logits -- \
  --device gpu \
  --framework max \
  --pipeline gemma3-1b \
  --encoding bfloat16 \
  --output /tmp/max-logits.json

# 2. Generate logits with PyTorch reference
./bazelw run //max/tests/integration/tools:generate_llm_logits -- \
  --device gpu \
  --framework torch \
  --pipeline gemma3-1b \
  --encoding bfloat16 \
  --output /tmp/torch-logits.json

# 3. Compare the logits
./bazelw run //max/tests/integration/accuracy:verify -- \
  --eval-metric cos,kl,tol \
  --relative-tolerance 1e-2 \
  --absolute-tolerance 1e-5 \
  --cos-dist-threshold 0.001 \
  --kl-div-threshold 0.01 \
  /tmp/max-logits.json /tmp/torch-logits.json

# Run verification pipeline directly (combines all steps)
./bazelw run //max/tests/integration/accuracy:verify_pipelines -- \
  --pipeline Gemma-3-1B-bfloat16 \
  --devices='gpu'
```

## 5. Validate model accuracy

After verifying your model serves correctly, validate that it produces accurate
outputs using [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness).
The exit criteria for model bringup is accuracy parity with reference
implementations like vLLM or SGLang.

The commands below run a lightweight evaluation using 320 samples, which
provides a good balance between runtime and statistical confidence. While we
run more comprehensive testing internally, passing this evaluation is
sufficient for contribution acceptance.

For help investigating inaccuracy issues, see the guide to
[debug MAX model accuracy](/max/docs/accuracy-debugging.md).

### Validate text models

Start your model server in one terminal:

```bash
./bazelw run //max/python/max/_entrypoints:pipelines -- serve \
  --model-path your-org/your-model-name
```

Then run the GSM8K evaluation in another terminal:

```bash
uvx --from 'lm-eval[api]' lm_eval \
  --model local-chat-completions \
  --tasks gsm8k_cot_llama \
  --model_args model=your-org/your-model-name,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=64,max_retries=1 \
  --apply_chat_template \
  --limit 320 \
  --seed 42 \
  --gen_kwargs seed=42,temperature=0 \
  --fewshot_as_multiturn
```

### Validate vision models

Vision models should run both `gsm8k_cot_llama` and `chartqa` evaluations:

```bash
uvx --from 'lm-eval[api]' lm_eval \
  --model local-chat-completions \
  --tasks gsm8k_cot_llama \
  --model_args model=your-org/your-model-name,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=64,max_retries=1 \
  --apply_chat_template \
  --limit 320 \
  --seed 42 \
  --gen_kwargs seed=42,temperature=0 \
  --fewshot_as_multiturn
```

```bash
uvx --from 'lm-eval[api]' lm_eval \
  --model local-chat-completions \
  --tasks chartqa \
  --model_args model=your-org/your-model-name,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=64,max_retries=1 \
  --apply_chat_template \
  --limit 320 \
  --seed 42 \
  --gen_kwargs seed=42,temperature=0 \
  --fewshot_as_multiturn
```

### Interpret the results

The evaluation outputs an accuracy score (e.g., 0.85 means 85% correct):

- **gsm8k_cot_llama**: Use `exact_match,flexible-extract`
- **chartqa**: Use `relaxed_accuracy,none`

Compare your score against the same model running on vLLM or SGLang. The pass
criteria is achieving at least 95% of the reference implementation's score.

For help investigating inaccuracy issues, see the guide to
[debug MAX model accuracy](/max/docs/accuracy-debugging.md).

### Reference scores

These scores were measured on NVIDIA B200 GPUs with MAX 25.7. The "vs Reference"
column shows the percentage relative to the best of vLLM or SGLang for that
model:

| Model                            | Task            | Accuracy | vs Reference |
|----------------------------------|-----------------|----------|--------------|
| meta-llama/Llama-3.1-8B-Instruct | gsm8k_cot_llama | 0.878    | 101.4%       |
| unsloth/gpt-oss-20b-BF16         | gsm8k_cot_llama | 0.947    | 98.1%        |
| Qwen/Qwen2.5-VL-7B-Instruct      | gsm8k_cot_llama | 0.787    | 100.3%       |
| Qwen/Qwen2.5-VL-7B-Instruct      | chartqa         | 0.812    | 100.3%       |

> [!NOTE]
> Accuracy numbers may vary across GPU types. If your model scores
> significantly below the reference, common causes are misconfigured Hugging
> Face config parsing (such as wrong defaults) or an incorrect RoPE (rotary
> position embedding) implementation. If you see different scores across
> hardware types, there may be a bug in an underlying kernel.

## Contribution guidelines

Before submitting your custom architecture to the repo:

1. **Read the [MAX contributor guide](/max/CONTRIBUTING.md)**.
2. **Test thoroughly**: Ensure your architecture works with the `max serve`
   command.
3. **Follow existing patterns**: Study similar architectures in this directory.
4. **Document your code**: Include clear docstrings and comments.
5. **Handle edge cases**: Ensure robust error handling and validation.
6. **Performance considerations**: Optimize for inference performance.

## Support

- Check out these other docs:

  - [Debug MAX model accuracy](/max/docs/accuracy-debugging.md)
  - [Op logging in MAX](/max/docs/op-logging.md)
  - [GPU profiling with Nsight Systems](https://max.modular.com/gpu-system-profiling)

- For detailed examples, explore the existing architecture implementations in
the [`max/pipelines/architectures/`](/max/python/max/pipelines/architectures/)
directory, such as:

  - **LLaMA family**: `llama3/` - Popular open-source language models.
  - **Gemma family**: `gemma3/` - Google's Gemma models.
  - **Qwen family**: `qwen3/` - Alibaba's Qwen models.

  Each subdirectory represents a different model family with its own
  implementation that you can study and adapt for your custom architecture.

- For questions or issues, please open a GitHub issue.
