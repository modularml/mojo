<!-- markdownlint-disable -->

# **MAX full container**

The [Modular](https://max.modular.com/intro) platform, powered by the MAX
framework and Mojo programming language, helps developers accelerate model
serving and scale GenAI deployments across diverse hardware architectures
without modifying a single line of code.

MAX delivers high-performance, hardware-agnostic inference for popular open
models, with seamless portability across cloud providers and devices.

The **`max-full`** container includes all the core dependencies needed to
deploy LLMs on both NVIDIA and AMD GPUs. It offers a fully integrated
environment with support for MAX models, PyTorch (GPU), ROCm, CUDA, and cuDNN,
ensuring optimal performance across GPU types. Use this container if you want a
streamlined, out-of-the-box solution that works across GPU vendors without
additional setup.

The MAX container is compatible with the OpenAI API specification and optimized
for GPU deployment. For details on container contents and hardware
compatibility, see [MAX containers](https://max.modular.com/container) in
the MAX documentation.

### **Quickstart**

You can run an LLM on GPU using the latest MAX full container with the
following commands.

Run the container on an AMD GPU:

```bash
docker run \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/max_cache:/opt/venv/share/max/.max_cache \
  --env "HF_HUB_ENABLE_HF_TRANSFER=1" \
  --env "HF_TOKEN=$HF_TOKEN" \
  --group-add keep-groups \
  --rm \
  --device /dev/kfd \
  --device /dev/dri \
  -p 8000:8000 \
  modular/max-full:<version> \
  --model <model-provider/model-id>
```

For more information on AMD-specific command configurations, see
[Running ROCm Docker containers](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.1.1/how-to/docker.html#running-rocm-docker-containers).

Run the container on an NVIDIA GPU:

```bash
docker run \
  --gpus 1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/max_cache:/opt/venv/share/max/.max_cache \
  --env "HF_HUB_ENABLE_HF_TRANSFER=1" \
  --env "HF_TOKEN=<secret>" \
  -p 8000:8000 \
  modular/max-full:<version> \
  --model <model-provider/model-id>
```

You can run a [MAX model](https://builds.modular.com/?category=models) by
referencing its HuggingFace model ID. For example,
[`google/gemma-3-1b-it`](https://builds.modular.com/models/gemma-3-it/1B).

For more information on deploying popular models with MAX, see the [model
support](https://max.modular.com/model-formats) documentation.

### **Tags**

Supported tags are updated to the latest MAX versions, which include the latest
stable release and more experimental nightly releases. The `latest` tag
provides you with the latest stable version and the `nightly` tag provides you
with the latest nightly version.

Stable

- max-full:25.X

Nightlies

- max-full:25.X.0.devYYYYMMDD

### **Documentation**

For more information on Modular and its products, visit the [Modular
documentation site⁠⁠](https://max.modular.com/).

### **Community**

To stay up to date with new releases,
[sign up for our newsletter⁠⁠](https://www.modular.com/modverse#signup),
[check out the community⁠⁠](https://www.modular.com/community), and
[join our forum⁠⁠](https://forum.modular.com/).

If you're interested in becoming a design partner to get early access and give
us feedback, please [contact us⁠⁠](https://www.modular.com/company/contact).
