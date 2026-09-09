:title: max benchmark

.. raw:: markdown

    Runs comprehensive benchmark tests on an active model server to measure
    performance metrics including throughput, latency, and resource utilization.
    For a complete walkthrough, see the tutorial to [benchmark MAX on a
    GPU](/serve/benchmark).

    Before running this command, make sure the model server is running, via [`max
    serve`](/cli/serve).

    For example, here's how to benchmark the `google/gemma-3-27b-it` model
    already running on localhost:

    ```sh
    max benchmark \
      --model google/gemma-3-27b-it \
      --backend modular \
      --endpoint /v1/chat/completions \
      --num-prompts 50 \
      --dataset-name arxiv-summarization \
      --arxiv-summarization-input-len 12000 \
      --max-output-len 1200
    ```

    When it's done, you'll see the results printed to the terminal.

    By default, it sends inference requests to `localhost:8000`, but you can change
    that with the `--host` and `--port` arguments.

    To save the results to a JSON file, set `--result-filename` to the path you
    want (the value can include a directory, which is created if needed):

    ```sh
    max benchmark ... --result-filename results/gemma-run.json
    ```

    Instead of passing all these benchmark options, you can pass a configuration
    file. See [Configuration file](#benchmark-configuration-file) below.

    :::note

    The `max benchmark` command is a convenient packaging for our open-source
    [`benchmark_serving.py`](https://github.com/modular/modular/tree/main/max/python/max/benchmark#benchmark-max)
    script and accepts all the same options.

    :::

    ## Usage

    Run `max benchmark` with one or more options:

    ```sh
    max benchmark [OPTIONS]
    ```

    ## Options

    The full option list is long. The most useful options group as follows. For
    everything else, run `max benchmark --help` or see the [benchmarking script
    source code](https://github.com/modular/modular/tree/main/max/python/max/benchmark).

    - Backend configuration:

      - `--backend`: Server type to benchmark. Choices: `modular`,
        `modular-chat`, `vllm`, `vllm-chat`, `sglang`, `sglang-chat`,
        `trtllm`, `trtllm-chat`. Default: `modular`.

      - `--model`: Hugging Face model ID or local path.

      - `--endpoint`: Specific API endpoint, such as `/v1/completions` or
        `/v1/chat/completions`. Default: `/v1/chat/completions`.

      - `--base-url`: Base URL of the API service. Overrides `--host` and
        `--port` when set.

      - `--host`: Server host. Default: `localhost`.

      - `--port`: Server port. Default: `8000`.

      - `--tokenizer`: Hugging Face tokenizer to use. Defaults to the model's
        tokenizer.

    - Load generation:

      - `--num-prompts`: Number of single-turn prompts to process. Default:
        unset. Use this or `--num-chat-sessions` (at least one is required).

      - `--num-chat-sessions`: Number of multiturn chat sessions to drive.
        Required for `chat-judge`. Use with multiturn-capable datasets instead
        of `--num-prompts`. Turns per session are dataset-specific; see
        [Datasets](#datasets) below.

      - `--request-rate`: Requests per second. Accepts a single value or a
        comma-separated sweep (such as `1,2,4,8`). Default: `inf` (no rate
        limit).

      - `--max-concurrency`: Maximum concurrent requests. Accepts a single
        integer or a comma-separated sweep.

      - `--seed`: Random seed for the workload generator (input/output lengths,
        session structure, and content). Default: `24301` (fixed for
        reproducibility). Pass `--seed none` (or `seed: null` in a workload YAML)
        to draw a fresh random seed; the drawn value is logged and recorded with
        the results.

      - `--kv-block-size`: KV cache block size in tokens for the per-turn cache
        retention metric. Default: `128`. Should match the server's
        `--kv-cache-page-size` so the retention metric is accurate; a mismatch
        does not affect the benchmark run itself.

      - `--fit-distributions`: Reshape multiturn workloads using the `random_*`
        flags and `--delay-between-chat-turns`. Requires `--num-chat-sessions`
        with `instruct-coder`, `agentic-code`, or `nemotron-opencode`. Turn
        count comes from `--random-num-turns` (see the `random` dataset below).

      - `--delay-between-chat-turns`: Delay between chat turns in
        milliseconds. Accepts a constant or a distribution string (same format as
        `--random-input-len`).

      - `--workload-config`: YAML file specifying benchmark workload options
        (hyphenated keys such as `num-prompts` and `seed`). CLI flags override
        values from this file.

    - Dataset selection:

      - `--dataset-name`: Dataset to benchmark on. Determines the dataset class
        and processing logic. Default: `sharegpt`. See [Datasets](#datasets)
        below.

      - `--dataset-path`: Path to a local dataset file. Supported only for
        datasets that accept a local override; see [Datasets](#datasets) below.

    - Output control:

      - `--max-output-len`: Maximum output length per request, in tokens.

      - `--temperature`, `--top-p`, `--top-k`: Sampling parameters forwarded to
        the server.

    - LoRA traffic:

      - `--lora`: Optional LoRA name to send with each request.

      - `--lora-paths`: Paths to existing LoRA adapters. Each entry is either
        `path` or `name=path`.

      - `--lora-uniform-traffic-ratio`: Probability (between `0.0` and `1.0`)
        that any given request targets a randomly selected LoRA instead of the
        base model. Default: `0.0`.

      - `--per-lora-traffic-ratio`: Per-adapter traffic ratios, in the same
        order as `--lora-paths`. Sum must not exceed `1.0`; the remainder goes
        to the base model. Overrides `--lora-uniform-traffic-ratio` when set.

      - `--max-concurrent-lora-ops`: Maximum concurrent LoRA load and unload
        operations. Default: `1`.

    - Result saving:

      - `--result-filename`: Path to a JSON file for benchmark results. When
        unset, no file is written. The path may include directories that the
        command creates if they don't exist.

      - `--metadata`: Key-value pairs (such as `--metadata version=0.3.3 tp=1`)
        recorded alongside the run in the result JSON.

      - `--log-dir`: Directory for log output. Default:
        `<backend>-latency-Y.m.d-H.M.S`.

    - Stats collection:

      - `--collect-gpu-stats` / `--no-collect-gpu-stats`: Report GPU utilization
        and memory consumption (NVIDIA only). Enabled by default. Only works
        when `max benchmark` runs on the same instance as the server.

      - `--collect-cpu-stats` / `--no-collect-cpu-stats`: Report CPU stats.
        Enabled by default.

      - `--collect-server-stats` / `--no-collect-server-stats`: Report server
        stats. Enabled by default.

    - Profiling:

      - `--profile`: Capture an Nsight Systems GPU trace and print a ranked
        top-N kernel summary when the run finishes. (Translates to `--trace`
        internally.) The server must already be running under `nsys launch`
        (unlike `max generate --profile`, which re-execs the client under
        `nsys profile`).

      - `--profile-output`: Path for the `.nsys-rep` file when `--profile` is
        set. Default: `$BUILD_WORKSPACE_DIRECTORY/max-profile.nsys-rep`, or
        `max-profile.nsys-rep` in the current directory.

      - `--profile-top-n`: Number of kernels to show in the summary table.
        Default: `15`.

      - `--trace`: Enable nsys tracing (lower-level alternative to
        `--profile` without the post-run kernel summary). Requires the server
        to run under `nsys launch`. NVIDIA GPUs only.

      - `--trace-file`: Path to save the `nsys` trace when using `--trace`
        directly. Default: `$BUILD_WORKSPACE_DIRECTORY/profile.nsys-rep`, or
        `./profile.nsys-rep`.

      - `--trace-session`: Optional `nsys` session name to trace.

    - Configuration file:

      - `--config-file`: Path to a YAML file containing benchmark options.
        See [Configuration file](#benchmark-configuration-file) below.

    ### Datasets

    The `--dataset-name` option supports the following datasets. For any
    dataset that has configurable flags, those flags are listed inline.

    Some datasets download from Hugging Face Hub or Hugging Face Datasets
    automatically. Others require `--dataset-path` or generate prompts in
    memory. Datasets that don't support `--dataset-path` are noted below.

    #### Text

    - `sharegpt` (default): Conversational dataset with human-AI exchanges,
      from Hugging Face Hub (`anon8231489123/ShareGPT_Vicuna_unfiltered`).

    - `axolotl`: Dataset in Axolotl format with human/assistant conversation
      segments. Uses a packaged default file; override with `--dataset-path`.

    - `chat-judge`: LLM-as-judge multiturn workload backed by a local JSONL
      session file. Each turn inlines prior context in the user message, so the
      driver sends `[system?, user]` per turn without accumulating assistant
      responses. Requires `--dataset-path` and `--num-chat-sessions` (single-turn
      mode isn't supported).

      Example JSONL (one session per line):

      ```json
      {
        "session_id": "s1",
        "turns": [
          {"text": "You are a safety judge.", "role": "system"},
          {"text": "Rate this content: ..."}
        ]
      }
      ```

    - `obfuscated-conversations`: Local obfuscated conversation dataset.
      Requires `--dataset-path` pointing at a local JSONL file.

      - `--obfuscated-conversations-average-output-len`: Average output length
        when per-request output lengths are not provided. Default: `175`.
      - `--obfuscated-conversations-coefficient-of-variation`: Coefficient of
        variation for output length. Default: `0.1`.
      - `--obfuscated-conversations-shuffle` /
        `--no-obfuscated-conversations-shuffle`: Shuffle the dataset.
        Disabled by default.

    - `arxiv-summarization`: Research paper summarization dataset, from Hugging
      Face Datasets.

      - `--arxiv-summarization-input-len`: Input tokens per request.
        Default: `15000`.

    - `sonnet`: Poetry dataset using poem lines from a packaged text file.
      Override with `--dataset-path`.

      - `--sonnet-input-len`: Input tokens per request. Default: `550`.
      - `--sonnet-prefix-len`: Shared prefix tokens per request. Default: `200`.

    - `random`: Synthetically generated dataset with configurable token
      distributions.

      - `--random-input-len`: Input tokens per request. Accepts a constant or a
        distribution string: `N(mean,std)`, `U(lower,upper)`, `DU(lower,upper)`,
        `NB(n,p)`, `G(shape,scale)`, or `LN(mean,std)`. Use `;` to set
        separate distributions for the first and subsequent turns (for example,
        `N(2048,200);N(512,50)`). Default: `1024`.
      - `--random-output-len`: Output tokens per request. Same format as
        `--random-input-len`. Default: `128`.
      - `--random-num-turns`: Turns per session. Same format as
        `--random-input-len`. Default: `1`. Used by `random` and `synthetic`
        multiturn workloads, and by `--fit-distributions` on `instruct-coder`,
        `agentic-code`, and `nemotron-opencode`.
      - `--random-sys-prompt-ratio`: Fraction of the input length to use as a
        system prompt. Range: `0.0`–`1.0`. Default: `0.0`.
      - `--random-max-num-unique-sys-prompt`: Maximum number of distinct system
        prompts to generate. Default: `1`.
      - `--warm-shared-prefix` / `--no-warm-shared-prefix`: Send each unique
        shared prefix as a single-token request before the run to prime
        prefix-cache KV entries. Requires `--random-sys-prompt-ratio > 0`.
        Disabled by default.
      - `--random-image-count`: Images to attach per request (enables vision
        mode on this dataset). Default: `0`.
      - `--random-image-size`: Pixel dimensions of generated images (for
        example, `512x512`). Used with `--random-image-count`.

    - `synthetic`: Synthetic text generation workload that uses the same
      distribution flags as `random`, but generates synthetic token IDs instead
      of vocabulary text. Supports multiturn via `--num-chat-sessions` and the
      `random_*` flags listed above. Also supports `--warm-shared-prefix`.

    #### Code

    - `instruct-coder`: Instruction-following coding dataset from Hugging Face
      Hub (`likaixin/InstructCoder`). Supports single-turn (`--num-prompts`) and
      multiturn (`--num-chat-sessions`) modes.

      If using with multiturn, it groups editing tasks at their
      natural token lengths, by default (up to 5 turns per session). With
      `--fit-distributions`, turn count comes from `--random-num-turns` instead,
      and per-turn input/output lengths and inter-turn delays follow the same
      distributions as the `random` dataset, via `random_*` flags and
      `--delay-between-chat-turns`; prompts are padded or truncated to match
      those targets.

    - `agentic-code`: Multiturn agentic coding workload with tool-call turns,
      from Hugging Face Hub (`novita/agentic_code_dataset_22`). Supports
      single-turn (`--num-prompts`) and multiturn (`--num-chat-sessions`) modes.
      By default, each session replays a full recorded conversation (variable
      turn count). `--fit-distributions` behaves the same as for
      `instruct-coder`.

      - `--tool-calls` / `--no-tool-calls`: Include or strip tool-call turns and
        forward tool definitions. Default: enabled.

    - `nemotron-opencode`: Large-scale agentic coding traces from Hugging Face
      (`nvidia/Nemotron-SFT-OpenCode-v1`), streamed on demand. Includes tool
      schemas translated to OpenAI function-tool format. Doesn't support
      `--dataset-path`. Supports single-turn (`--num-prompts`) and multiturn
      (`--num-chat-sessions`) modes. By default, each session replays a full
      recorded conversation (variable turn count). `--fit-distributions`
      behaves the same as for `instruct-coder`.

      - `--tool-calls` / `--no-tool-calls`: Include or strip tool-call turns and
        forward tool definitions. Default: enabled.

    - `code_debug`: Long-context code debugging dataset with multiple-choice
      questions, from Hugging Face Hub (`xinrongzhang2022/InfiniteBench`).
      Single-turn via `--num-prompts`. Also supports a fixed two-turn long-context
      template via `--num-chat-sessions`.

    #### Vision

    - `batch-job`: Batch image workload in OpenAI Batch API format. Requires
      `--dataset-path` (tar archive or extracted directory with `jobs.jsonl`).

      - `--batch-job-image-dir`: Directory where the server can access images
        (file reference mode). When unset, images are embedded as base64.

    - `local-image`: Local images for vision benchmarks. Requires
      `--dataset-path` (JSONL with `prompt` and `image_path` per line).

    - `vision-arena`: Vision-language benchmark dataset with images and
      associated questions for multimodal model evaluation, from Hugging Face
      Datasets.

    - `synthetic-pixel`: Synthetic pixel-generation workload for image-output
      backends.

    ### Configuration file {#benchmark-configuration-file}

    The `--config-file` option loads benchmark settings from YAML instead of
    spelling out every flag on the command line. Define options under a top-level
    `benchmark_config` key. CLI flags override values from the file when both
    are supplied.

    :::caution

    In the YAML file, the properties **must use `snake_case` names** instead of
    the hyphenated names from the command line. For example, `--num-prompts`
    becomes `num_prompts`.

    :::

    For example, instead of specifying configurations on the command line like
    this:

    ```sh
    max benchmark \
      --model google/gemma-3-27b-it \
      --backend modular \
      --endpoint /v1/chat/completions \
      --host localhost \
      --port 8000 \
      --num-prompts 50 \
      --dataset-name arxiv-summarization \
      --arxiv-summarization-input-len 12000 \
      --max-output-len 1200
    ```

    Create this configuration file:

    ```yaml title="gemma-benchmark.yaml"
    benchmark_config:
      model: google/gemma-3-27b-it
      backend: modular
      endpoint: /v1/chat/completions
      host: localhost
      port: 8000
      num_prompts: 50
      dataset_name: arxiv-summarization
      arxiv_summarization_input_len: 12000
      max_output_len: 1200
    ```

    Then run the benchmark by passing that file:

    ```sh
    max benchmark --config-file gemma-benchmark.yaml
    ```

    For more config file examples, see our [benchmark configs on
    GitHub](https://github.com/modular/modular/tree/main/max/python/max/benchmark/configs).

    For a walkthrough of setting up an endpoint and running a benchmark, see
    the [quickstart guide](/get-started).

    ## Output

    Each run prints the following metrics on completion:

    - **Request throughput**: number of complete requests processed per second.
    - **Input token throughput**: number of input tokens processed per second.
    - **Output token throughput**: number of tokens generated per second.
    - **TTFT** (time to first token): time from request start to first token
      generation.
    - **TPOT** (time per output token): average time taken to generate each
      output token.
    - **ITL** (inter-token latency): average time between consecutive token or
      token-chunk generations.

    For multiturn workloads, the run also reports:

    - **Per-turn cached token rate**: percentage of each turn's prompt tokens
      served from prefix cache (when the server reports token statistics).
    - **Per-turn KV cache retention**: for each turn after the first, the
      percentage of the previous turn's block-aligned prefix that remains cached.
      Surfaces when the server drops cached tokens across turns (distinct from
      cached token rate, whose denominator includes new and uncacheable tokens).
      Configure block alignment with `--kv-block-size`.

    When `--collect-gpu-stats` is enabled, the run also reports:

    - **GPU utilization**: percentage of time during which at least one GPU
      kernel is executing.
    - **Peak GPU memory used**: peak memory usage during the benchmark run.
