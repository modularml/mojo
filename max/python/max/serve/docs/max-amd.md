<!-- markdownlint-disable -->

# MAX AMD full container

The `max-amd` container delivers high-performance inference on AMD GPUs using
the MAX framework, with full support for PyTorch GPU and ROCm. It provides a
ready-to-use environment for running LLMs with optimized performance on AMD
hardware.

The MAX container is compatible with the OpenAI API specification and optimized
for GPU deployment. For details on container contents and hardware
compatibility, see [MAX containers⁠](https://max.modular.com/container/)
in the MAX documentation.

### **Quickstart**

You can run an LLM on an AMD GPU using the latest MAX AMD full container with
the following command:

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
  modular/max-amd:<version> \
  --model <model-provider/model-id>
```

For more information on AMD-specific command configurations, see
[Running ROCm Docker containers](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.1.1/how-to/docker.html#running-rocm-docker-containers).

You can run a [MAX model](https://builds.modular.com/?category=models) by
referencing its HuggingFace model ID. For example,
[`google/gemma-3-1b-it`](https://builds.modular.com/models/gemma-3-it/1B).

For more information on deploying popular models with MAX, see the
[model support](https://max.modular.com/model-formats) documentation.

### **Tags**

Supported tags are updated to the latest MAX versions, which include the latest
stable release and more experimental nightly releases. The `latest` tag
provides you with the latest stable version and the `nightly` tag provides you
with the latest nightly version.

Stable

- max-amd:25.X

Nightlies

- max-amd:25.X.0.devYYYYMMDD

### **Documentation**

For more information on Modular and its products, visit the
[Modular documentation site⁠⁠](https://max.modular.com/).

### **Community**

To stay up to date with new releases,
[sign up for our newsletter⁠⁠](https://www.modular.com/modverse#signup),
[check out the community⁠⁠](https://www.modular.com/community), and
[join our forum⁠⁠](https://forum.modular.com/).

If you're interested in becoming a design partner to get early access and give
us feedback, please [contact us⁠⁠](https://www.modular.com/company/contact).
