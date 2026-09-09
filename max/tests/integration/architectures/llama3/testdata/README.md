# Pipelines test data

## Generate Test Data

Use these scripts to randomly generate a tiny llama checkpoint and compute
golden values.

### Tinyllama

To facilitate fast cycle times during local development, a tiny llama test
is included alongside the full weights. To re-generate the tiny llama
checkpoint, use the `gen_tiny_llama` target:

```bash
TESTDATA_DIR="$MODULAR_PATH/max/tests/integration/architectures/llama3/testdata"

# Generate float32 checkpoint
./bazelw run //ModularFramework/utils:gen_tiny_llama --\
    --output=$TESTDATA_DIR/tiny_llama.gguf \
    --quantization-encoding=float32 \
    --n-layers=1 \
    --n-heads=1 \
    --n-kv-heads=1 \
    --hidden-dim=16

# Generate bfloat16 checkpoint
./bazelw run //ModularFramework/utils:gen_tiny_llama --\
    --output=$TESTDATA_DIR/tiny_llama_bf16.gguf \
    --quantization-encoding=bfloat16 \
    --n-layers=1 \
    --n-heads=1 \
    --n-kv-heads=1 \
    --hidden-dim=16
```

Note: Hidden dim must be a multiple of 8 (required for `fused_qkv_matmul` kernel
alignment)

Then, you can use `evaluate_llama` to generate the golden values. The CLI
supports encoding (q4_k, float32, bfloat16) and model (tinyllama, llama3_1)
parameters. If either are not set they default to "all", so the typical command
simply points to the modular root so that the CLI can write the golden files for
each encoding/model pair to the test data folder.

```bash
./bazelw run //max/tests/integration/architectures/llama3:evaluate_llama --\
    --modular-path /path/to/modular \
    --encoding q4_k \ # float32, q4_k, bfloat16, or all (default)
    --model tinyllama # llama3_1, tinyllama, or all (default)
```

### Tokenizer data

`special_tokens_map.json`, `tokenizer_config.json` and `tokenizer.json` are
copied from the
[meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
HuggingFace model.

## Running the tests

The test target for CPU tests is:

```bash
./bazelw test //max/tests/integration:tests
```

For local development, it may be convenient to just run the tiny llama
tests, which you can select out using a pytest filter:

```bash
./bazelw test //max/tests/integration:tests --test_arg="-k test_llama[tiny-float32-llama3_1]"
```

Note that GPU tests have a different target:

```bash
./bazelw test //max/tests/integration:tests_gpu
```

## Registering new golden test data

We use `http_archive` to bundle and download the golden values from S3. To
add a file to this archive, you need to:

1. Download the existing archive by `cat MODULE.bazel | grep test_llama_golden`
, finding the s3 URL (at time of writing this was
`https://modular-bazel-artifacts-public.s3.amazonaws.com/artifacts/test_llama_golden/3/90811d3fff2b4d88390fb193bb545651529126729c6af2626c68341640c2d62b/test_llama_golden.tar.gz`)
and downloading to your local machine (e.g., with wget).

2. Untar the existing archive `tar -xvf test_llama_golden.tar.gz`.

3. Add any additional files you want to register alongside.

4. Run
   `./utils/upload-public-bazel-artifact.sh test_llama_golden 3 *golden.json`
to package and upload the latest version (current version number is `3`).

5. The result of ^ will be a snippet like:

   ```bash
    http_archive(
        name = "test_llama_golden",
        build_file_content = """
    filegroup(
        name = "test_llama_golden",
        srcs = glob(["**"]),
        visibility = ["//visibility:public"],
    )""",
        sha256 = "SOME_SHA",
        url = "https://modular-bazel-artifacts-public.s3.amazonaws.com/artifacts/test_llama_golden/VERSION/SOME_SHA/test_llama_golden.tar.gz",
    )
   ```

6. Find the associated section in `MODULE.bazel`, delete it, and replace
with this newly generated value.

## Registering torch golden logits

Similar to above, except the torch golden archive is named `torch_llama_golden`.
Here are the condensed instructions:

### Generating new torch goldens

Currently we only generate goldens for bfloat16 on GPU.

```bash
./bazelw run max/tests/integration/architectures/llama3/testdata:run_torch_llama_gpu \
    -- --model llama3_1 --encoding bfloat16 --verbose
```

### Uploading new goldens

1. Download the tar file from the URL reported by
   `cat MODULE.bazel | grep torch_llama_golden`
2. Untar: `tar -xvf torch_llama_golden.tar.gz`
3. Update file list with new golden files.
4. Run
   `./utils/upload-public-bazel-artifact.sh torch_llama_golden 3 torch_*golden.json`
5. Update `MODULE.bazel` with the result from above.
