---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

- Added a dedicated [metrics](/serve/metrics) reference page with all
available Prometheus metrics, categorized by subsystem. The metrics section in
the [container](/container) page now links to the new page.
- Added an [audio generation](/serve/audio-generation) guide, covering serving
  a text-to-music model over `/v1/audio/speech` and `/v1/responses`, the
  request fields and their defaults, the lyric tag syntax, and the length a
  single render is capped at.
- Added a music generation example that renders songs past that per-render
  cap by rendering sections and joining them, and checks the joins for
  audible seams.

## MAX models

- Added the `audio_generation` pipeline task, for models that render audio
  rather than tokens or pixels. Its request options (lyrics, duration,
  denoising steps, guidance scale, output format) arrive as the `audio`
  provider options of an OpenResponses request, and an architecture on the
  task serves over `/v1/audio/speech` and `/v1/responses`. Responses report
  `usage` the way image generation does: token counts stay at 0 and a
  `usage.audio_generation_details` block carries `duration_seconds`,
  `sample_rate`, `channels`, `num_samples`, and `steps`, measured from the
  audio actually produced rather than the duration that was asked for.
- Added MiniMax-Music3 (`MiniMaxMusic3ModularPipeline`) support, the first
  architecture on the `audio_generation` task: a text-to-music model that
  renders a style caption plus lyrics into 44.1 kHz stereo audio. The five
  component networks exceed a 24 GB card together, so the pipeline builds and
  releases each stage in turn within a request; the first request after a cold
  start pays a multi-minute compile that later ones replay from the
  compilation cache.
- Startup no longer prints one `unknown dtype found in safetensors file`
  warning for each tensor with a dtype that is not a weight encoding. Each
  scan of the weight files now prints one warning for each unknown dtype.
- Fixed unbounded host-memory usage in Gemma 4 video pre-processing: the
  server now decodes only the sampled frames of a video instead of
  materializing every frame before sampling, bounding peak memory at the
  sampled frame count (previously a long clip could transiently allocate
  ~100 GB in the API server process).
- Added GLM-5.2 (`GlmMoeDsaForCausalLM`) support, extending the GLM-5.1
  sparse-attention architecture with cross-layer index sharing.
  - Added multi-token prediction (MTP) speculative decoding for GLM-5.2
    (`UnifiedMTPGlm5_2ForCausalLM`), serving the baked-in NextN layer as a
    single-layer sparse-MLA draft; enabled automatically for GLM checkpoints
    that ship a NextN layer with `--speculative-method mtp`.
  - Added tool-calling, reasoning, and structured-output (`response_format`)
    support to GLM-5.1 / GLM-5.2, enabled with `--tool-parser glm45
    --reasoning-parser glm45 --enable-structured-output`.
  - Fixed a GLM-5.1-FP8 crash caused by a shared-experts dtype mismatch.
  - The GLM-5.2 B200 recipe now serves the checkpoint's full 1M-token
    context window (`max_length: 1048576`, previously pinned to 163840).
    The pin existed because the wider window cost ~33% decode throughput on
    long-context workloads; the sparse-attention indexer now does work
    proportional to actual sequence lengths (per-layer kernel cost measured
    flat across frozen bounds), and a weekly long-context serving benchmark
    tracks the end-to-end throughput at this configuration.
- Added multi-token prediction (MTP) speculative decoding for Inkling
  (`UnifiedMTPInklingForConditionalGeneration`), serving the checkpoint's
  chained dense draft depths; enabled automatically for Inkling checkpoints
  that ship `mtp_config` with `--speculative-method mtp`.
- Added Laguna (`LagunaForCausalLM`) support for
  `poolside/Laguna-M.1-NVFP4`, including tool calling.
- Added DiffusionGemma (`DiffusionGemmaForBlockDiffusion`) support for
  `google/diffusiongemma-26B-A4B-it` (bfloat16) and
  `nvidia/diffusiongemma-26B-A4B-it-NVFP4`; text-only for now.
- Added Nemotron-H (`NemotronHForCausalLM`) support, NVIDIA's hybrid
  Mamba-2 + attention decoder, with modelopt per-tensor FP8 and a new
  Mamba-2 SSD chunked-scan varlen kernel.
  - Extended Nemotron-H with the Nemotron-3-Nano-30B-A3B hybrid MoE variant
    and enabled the architecture on Apple silicon GPUs in bfloat16.
  - Enabled NVIDIA's official FP8 Nemotron-H checkpoints on Apple silicon
    (previously crashing or producing all-zero logits) and sped up
    Nemotron-H decode on Apple M5 by ~41-81%.
  - Added support for serving `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` on
    Apple silicon via a tiled simdgroup-MMA grouped-FP8 (W8A16) MoE matmul,
    decoding faster than bf16 at concurrency with half the weight memory.
- Fixed the `max_batch_size` handling for Nemotron-H.
- Added support for the `detail` parameter on image and video content
  parts in chat requests.
- Added Ideogram 4 (`Ideogram4Pipeline`) support, a text-to-image
  flow-matching diffusion transformer; serve via `/v1/responses`.
  - FP8 checkpoint weights run hot projections on native FP8 GEMMs (~24%
    faster end-to-end on MI355).
- Added support for `amd/Kimi-K2.7-Code-MXFP4` on AMD GPUs.
- Expanded Gemma 4 support:
  - Added DSpark speculative decoding for Gemma 4 12B
    (`UnifiedDSparkGemma4ForCausalLM`), DeepSeek's block-drafting method:
    a small draft transformer drafts a 7-token block per step. Enabled with
    `--draft-model-path deepseek-ai/dspark_gemma4_12b_block7
    --speculative-method dflash --num-speculative-tokens 7`.
    - Fixed the draft applying full rope instead of the checkpoint's
      partial rotary factor (0.25), which was costing roughly 10% of the
      draft acceptance rate.
  - Added DSpark speculative decoding for Gemma 4 31B
    (`UnifiedDSparkGemma4_31BForCausalLM`), serving `google/gemma-4-31B-it`
    with the vLLM speculators-format draft
    `RedHatAI/gemma-4-31B-it-speculator.dspark` (llama-style causal draft
    block, pruned 32k draft vocabulary mapped through the checkpoint's d2t
    table). Enabled with the `gemma4_31b_dspark.yaml` recipe or
    `--draft-model-path RedHatAI/gemma-4-31B-it-speculator.dspark
    --speculative-method dflash`. An explicit `--num-speculative-tokens` is
    honored: values below the trained 7 truncate the causal draft block
    prefix-stably, values above run as extrapolation with a warning and
    degrading acceptance; unset defaults to the trained 7.
  - Added DFlash speculative decoding for Gemma 4 31B
    (`UnifiedDflashGemma4_31BForCausalLM`), serving `google/gemma-4-31B-it`
    with the z-lab block-diffusion drafter `z-lab/gemma-4-31B-it-DFlash`: a
    5-layer noncausal draft block drafts 15 tokens per step from six target
    hidden-state taps. Enabled with the `gemma4_31b_dflash.yaml` recipe or
    `--draft-model-path z-lab/gemma-4-31B-it-DFlash --speculative-method
    dflash`. The draft width is pinned to the drafter's trained
    `block_size - 1`; a mismatching `--num-speculative-tokens` is overridden
    with a warning. NVFP4 target checkpoints (`nvidia/Gemma-4-31B-IT-NVFP4`)
    are supported via the `gemma4_31b_dflash_nvfp4.yaml` recipe.
  - Gemma 4 31B DSpark now supports structured output (JSON schemas and
    tool-call grammars, enforced on the target verify pass; a
    grammar-violating draft is rejected at its position) and Gemma 4
    thinking: reasoning content is split out of responses, and relaxed
    acceptance during the thinking phase can be enabled with
    `use_relaxed_acceptance_for_thinking`.
  - Renamed the Gemma 4 12B DSpark architecture to
    `UnifiedDSparkGemma4_12BForCausalLM` (module
    `max.pipelines.architectures.unified_dspark_gemma4_12b`), so the two
    Gemma 4 DSpark architectures are named by model line.
  - Sped up Gemma4-12B DSpark decode by up to ~1.3x via a packed wide-N
    shallow-K GEMV, a single-pass streaming argmax kernel, and device graph
    capture.
  - Gemma 4 with MTP speculative decoding (`UnifiedMTPGemma4ForCausalLM`)
    now supports image and video input; previously the vision encoder output
    never reached the language model, so image prompts were answered as if
    the model were blind.
  - MTP speculative decoding now samples recovered tokens from the residual
    distribution when stochastic acceptance rejects a draft token,
    preserving the target distribution for argmax draft proposals.
  - Added structured-output and tool-calling support via the xgrammar
    backend, covering Gemma 4's special tool-call format.
  - Added float16 support, with the logit softcap and vision pooler run in
    fp32.
  - Added tensor-parallel support for the MoE variant.
  - Video inputs now route through the shared `VisionEncoderCache`, so a
    repeated clip is served from cache with no re-encode.
  - Video decoding now runs on a worker thread, so concurrent requests
    overlap video decode.
  - Improved vision-batch serving latency by concatenating embeddings
    on-device instead of round-tripping through host numpy.
  - Fixed the MoE expert-router softmax being computed in `bfloat16`
    instead of `float32`, which degraded MoE quality.
  - Fixed image/video position and scatter indexing desyncs under chunked
    prefill, which could corrupt vision embeddings on multimodal prompts
    split across chunks.
  - Fixed crashes in multi-device serving and multi-image batches by making
    `merge_per_device_buffers` rank-agnostic.
  - Fixed reasoning being dropped after tool results.
  - Fixed a vision-batch crash caused by constructing a `Device()` instead
    of `CPU()` for host tensors.
- Expanded DeepSeek-V3 ModuleV3 support:
  - Added NVFP4 (modelopt) weight support, running experts, dense MLPs,
    and the attention output projection on SM100 block-scaled FP4 matmul
    kernels.
  - Added data-parallel + expert-parallel (DP-EP) and multi-GPU
    tensor-parallel + expert-parallel (TP+EP) serving. Note: `Tensor.to`
    no longer implicitly calls `F.distributed_broadcast`; call it
    explicitly where needed.
  - Fixed the FP8 adapter by casting f32 normalization gammas, resolving a
    dtype mismatch.
- Expanded Kimi K2.5 support:
  - Kimi with DFlash speculative decoding
    (`UnifiedDflashKimiK25ForCausalLM`) now supports image input;
    previously the vision encoder was not compiled, so image prompts were
    answered as if the model were blind.
  - Added support for combining Kimi tool calling with
    `response_format=json_schema` on the xgrammar constrained-decoding
    backend.
- Expanded FLUX.2 support:
  - FLUX.2-klein bf16 checkpoints on Apple M5 GPUs now default to int8
    W8A8 quantization, ~1.45x faster end-to-end than bf16 on
    FLUX.2-klein-4B at near-lossless quality; set
    `APPLE_FLUX2_INT8_W8A8=0` to opt out.
  - NVFP4 checkpoints can now opt into an int8 W8A8 requant at load on
    Apple M5 with `APPLE_FLUX2_INT8_W8A8=1`, ~2.56x faster end-to-end
    than the default W4A16 path on FLUX.2-dev.
  - Diffusion pipelines now support two denoising-cache backends to skip
    redundant transformer passes: `--taylorseer` (recommended default,
    with `balanced` and `fast` presets) and `--first-block-caching`; the
    two are mutually exclusive and both off by default.
- Expanded Qwen support:
  - Added tool-calling and reasoning support to Qwen 3.5 / 3.6.
  - Added `Qwen/Qwen3.8-27B` support in bfloat16 on the existing
    `Qwen3_5ForConditionalGeneration` architecture, covered by logit
    verification against the torch reference.
  - `Qwen3_5ForConditionalGeneration` now serves across multiple GPUs.
    Tensor parallelism splits the attention heads, the gated-DeltaNet key and
    value heads, and the per-device linear-attention state pools; both mixers
    reject a device count that would not divide their head counts evenly.
  - `Qwen3_5ForConditionalGeneration` now supports device graph capture.
  - Added multi-token prediction (MTP) speculative decoding for Qwen3.8
    (`UnifiedMTPQwen3_5ForConditionalGeneration`), fusing the target, the
    baked-in MTP head and a recurrent-state rollback into one graph, selected
    for Qwen3.5-family checkpoints that ship an MTP head with
    `--speculative-method mtp`. Rejecting a speculated token cannot be undone
    by rewinding a KV length pointer when the layer is recurrent, so the graph
    snapshots the gated-DeltaNet conv and recurrent pools before verifying and
    replays the two state kernels over the accepted rows. The graph is served
    through the Mach engine; MAX compiles and exports it but does not run it.
  - Fixed a `Qwen3EmbeddingModel` crash.
- Added `--state-pool-dtype`, which overrides the storage dtype of a hybrid
  model's recurrent state pools (SSM and linear-attention conv and recurrent
  state). It defaults to the model's compute dtype. `float32` makes a
  speculated generation follow the same state trajectory as an unspeculated
  one -- the recurrence rounds to the pool dtype at each call boundary, so a
  lossy pool makes the trajectory depend on how speculation chunked the
  sequence -- at roughly double the per-request state memory (Qwen3.8-27B:
  74.8 to 149.6 MiB per seated request).
- Added per-request LoRA adapter support: `LoRALinear` and
  `StackedLinearLoRA` extend LoRA to standalone and fused-QKV projections,
  with `LoRAManager.apply` swapping target layers in a model.
- Improved Eagle3 speculative-decoding performance by removing a redundant
  concatenate in the draft path.
- Fixed Step-3.5-Flash accuracy and performance.
- Fixed the EAGLE3 MHA draft `lm_head` all-gather in pure tensor-parallel
  mode.

## MAX framework

- Added `Device.is_host_unified` (`max.driver`) and
  `DeviceContext.is_host_unified()` (Mojo): whether a device and the host draw
  from one physical memory pool. Reports hardware topology, so it does not imply
  any given buffer is host-readable. Driver plugins answer it through the new
  optional `host_unified` device property.
- Host-side profiling spans (`max.profiler.Tracer`, `@traced`, and Mojo
  `Trace` scopes) now annotate external profiler tools on release builds:
  with `MODULAR_ENABLE_PROFILING` set, spans appear as NVTX ranges in
  NVIDIA Nsight Systems captures and as roctx ranges in rocprofv3 captures,
  with no build flags required. Previously these spans were only emitted in
  special profiling builds.
- Greedy speculative acceptance (`greedy_acceptance_sampler`,
  `AcceptanceSampler` in greedy mode) now applies the structured-output
  grammar bitmask to the target logits (with a `-inf` fill) before the
  argmax, so a grammar-invalid draft is always rejected and recovered and
  bonus tokens always satisfy the constraint — matching the stochastic
  path. Unconstrained batches are unchanged.
- `stochastic_acceptance_sampler` and `AcceptanceSampler` also accept a
  rank-1 `[batch_size]` per-row seed tensor in stochastic argmax mode: each
  row's acceptance sampling is then keyed off its own seed instead of row
  0's, so a row samples independently of its co-residents. A single-row
  batch is bit-identical to the scalar-seed behavior. The gemma4 and
  qwen3.5 unified MTP graphs now pass their per-row seed tensors through.
- Fixed `response_format` schema normalization skipping containers the grammar
  backends compile: an untyped object-shaped subschema under
  `additionalProperties`, `unevaluatedProperties`, `unevaluatedItems` or
  `dependentSchemas` is now anchored to an object, as one under `properties`
  already was. Such a subschema previously compiled to a grammar admitting an
  unbounded value, letting a looping model run to `max_length`.
- Added the experimental `--experimental-device-graph-synthesis` flag
  (`PipelineRuntimeConfig.experimental_device_graph_synthesis`): compiles
  model graphs with device-graph synthesis, so the compiled model records its
  kernels into a device graph and replays it on execute. Honored only by
  architectures that opt in (currently Gemma 4's language graph), and
  mutually exclusive with `device_graph_capture`.
- Added `max.pipelines.lib.MemoryPlan`, the result of memory planning when a
  pipeline is loaded: the effective `planned_max_length`, `max_batch_size`,
  `max_batch_total_tokens`, KV-cache budget, and device specs the pipeline
  and its schedulers consume.
- Renamed `MemoryEstimator.estimate_memory_footprint` to
  `MemoryEstimator.plan_from_sizes`, after the `MemoryPlan` it returns. Use
  `MemoryEstimator.plan` instead to plan from a `PipelineConfig` alone;
  `plan_from_sizes` is for callers that have already computed the weight,
  activation, and signal-buffer sizes.
- The sequence-length rule now runs once, when the config is built:
  `config.model.max_length` holds the resolved length and
  `PipelineArgs.max_length` keeps what the user asked for.
  `ArchConfig.initialize` receives that length instead of deriving it
  (`max_seq_len` is now a required keyword argument), and memory planning
  may only lower it, on the plan.
  `PipelineModel.calculate_max_seq_len`,
  `ArchConfigWithAttentionKVCache.user_provided_max_length` and
  `model_max_seq_len` are removed; architectures own the rule, so Mistral,
  Mistral3 and Pixtral now bound `max_length` on their configs.

- Memory planning no longer writes its planned `max_length` and
  `max_batch_total_tokens` back onto the pipeline config. After startup,
  `PipelineConfig.model.max_length` keeps the construction-resolved value
  and `PipelineConfig.runtime.max_batch_total_tokens` keeps the
  user-provided value (`None` when unset); the effective
  values live on `MemoryPlan`.
- `PipelineModel` now requires the `memory_plan` constructor argument
  (keyword-only; constructing a pipeline model without a plan raises a
  `TypeError`), and `PipelineModel.max_seq_len` is a read-only view of the
  plan's `planned_max_length` rather than a stored copy with a config
  fallback.
- Made `MemoryEstimator.free_memory`, `static_memory_size`,
  `available_kv_cache_memory`, and `max_supported_sequence_length` private.
  They are steps within a memory plan rather than useful on their own, and
  the values they produced are now available on `MemoryPlan`.
- The block-based vision encoder cache now shards its storage across
  devices instead of replicating every entry on each one. The same
  `--vision-cache-utilization` fraction buys the same cache capacity while
  reserving only `1/n_devices` of it per device; the remainder stays with
  the KV cache. Cache hits gather rows to each device in one batched
  submission.
- Added opt-in token-balanced CE scheduling across data-parallel replicas.
  With `--dp-ce-balance-timeout-ms` >= 0 (default -1 = off), new context
  encoding requests wait in an unbound pool and are placed by a per-step
  planner that prices them at their post-prefix-cache length (a read-only
  probe of each replica's device cache and the shared host/disk tiers) and
  binds them to the least-loaded replica when first scheduled. Unbalanced CE
  work may be deferred up to the timeout while its replica runs decode
  instead, until per-step occupancy reaches `--dp-ce-balance-threshold`
  (default 0.8). A below-threshold step with CE work on two or more replicas
  still runs immediately with each replica's chunk size reduced to the
  balance level, so only the excess defers
  (`--dp-ce-balance-enable-dynamic-chunk-size`, default on; skipped when the
  balance level is under half the CE chunk target, where the extra chunks
  would cost more than the imbalance).
- Added `--chunked-prefill-min-chunk-size` (config key
  `runtime.chunked_prefill_min_chunk_size`, default 0 = off) to set a floor,
  in tokens, on any chunk created by chunked prefill. When splitting a
  request against the CE token budget, the cut is moved earlier so that
  neither the chunk nor its remainder is smaller than the floor; if no legal
  cut point exists within the remaining budget, the request is left unsplit
  for a later step. This avoids degenerate slivers (for example an 8-token
  tail chunk after an 8192-token budget cut) that pay a full step's overhead
  and re-read the request's entire context in attention for almost no
  progress.
- Fixed non-streaming chat completions leaking a literal structural tool-call
  marker (for example `<tool_call>`) into `message.content` when a
  `max_tokens` truncation landed mid tool-call block. The response now
  surfaces only the content before the marker, with
  `finish_reason == "length"`.
- Added an experimental `--fold-sampler-into-graph` option (default off) that
  folds greedy token selection (argmax) into the captured forward graph, so a
  single device-graph replay materializes the sampled token instead of a
  separate sampler submission with a blocking readback. Applies to all-greedy
  decode batches on architectures that emit the folded token output (currently
  Nemotron-H); non-greedy requests fall back to the separate sampler.
- Added a `max-pending-futures` config (default 1, the classic
  overlap-scheduler depth of one forward in flight per request). Request
  bookkeeping now tracks unrealized future-token placeholders with a counted
  model instead of a single-sentinel check, and setting the value to 2 enables
  experimental schedule-ahead decoding: two forwards in flight per request,
  with the next step's input token realized on-device from the folded sampler
  output. Behavior at the default depth is unchanged.
- Fixed the serve CLI dropping the `fold-sampler-into-graph`,
  `max-pending-futures`, and greedy-sampling gate settings on their way to the
  model worker, which silently disabled the folded greedy sampler. With the
  flags threaded through, `--fold-sampler-into-graph` removes the per-token
  blocking sampler submission and substantially improves decode latency on
  architectures that support it.
- Added `max.engine.read` for loading a compiled-model artifact (a `.mef`
  file) without an `InferenceSession`. The resulting `CompiledModel` can
  be initialized on any session via `InferenceSession.init`. It replaces
  `InferenceSession.read`, which has been removed.
- Image generation responses on the Open Responses endpoint now report
  `usage`: token counts stay at 0 and a new `usage.image_generation_details`
  block carries `width`, `height`, `megapixels`, `steps`, and `image_count`,
  measured from the actual generated images rather than the requested
  dimensions. Previously `usage` was always `null`. (An interim nightly
  reported the raw pixel count as `output_tokens`; that encoding is replaced
  by `image_generation_details`.)
- Added `InferenceSession.read` for loading a compiled-model artifact (a
  `.mef` file) previously saved with `CompiledModel.export_mef`. It accepts a
  path or a binary file-like object (such as `io.BytesIO`), deserializes
  without invoking the graph compiler, and returns a `CompiledModel` ready to
  pass to `InferenceSession.init`.
- Added `--no-enable-tool-call-constrained-decode` (config key
  `sampling.enable_tool_call_constrained_decode`, default enabled) to decouple
  tool-call parsing from constrained decoding. When disabled, a configured
  `--tool-parser` still parses tool calls out of the generated text, but no
  server-generated grammar is produced and the bitmask constrained-decode path
  is skipped for tool calls. Note that with it disabled, `tool_choice=required`
  or a named function can no longer force a tool call. This is independent of
  `--enable-structured-output`, which continues to gate user-supplied
  `response_format` JSON schemas.
- Fixed the `code` label on the `maxserve_request_count` metric so it reports
  the HTTP status code actually returned to the client. The count is now
  recorded from the HTTP layer, so failures rejected before generation (for
  example a request with an unreachable image URL) are counted with their real
  status code instead of being labeled `200` or dropped entirely. Liveness and
  observability endpoints (`/health`, `/version`, `/ping`, `/metrics`) are not
  counted.
- Failed request submissions in the OpenAI-compatible serving endpoints now
  surface as HTTP error responses instead of a `200 OK` streaming response that
  carries an error payload. Request tokenization and the handoff to the model
  worker now complete before the streaming response headers are sent, so a
  failure at submission time (for example, a dead model worker) maps to an HTTP
  5xx (or 4xx for input errors). Errors that occur mid-stream, after the first
  chunk has been sent, are still serialized as an error event within the stream.
- Added request-queue backpressure to MAX serve via two cooperating caps. The
  `--max-queue-size` flag (env var `MAX_SERVE_MAX_QUEUE_SIZE`, cap *N*) bounds
  the request queue to the model worker; once it is full, new requests are
  rejected immediately with HTTP 429 instead of being enqueued. The
  `--max-pending-requests` flag (env var `MAX_SERVE_MAX_PENDING_REQUESTS`, cap
  *M*) stops the worker from draining the request queue once its pending
  (prefill) queue is *M* deep, so the request queue actually backs up under
  load. Together they form a self-calibrating mechanism that sheds load to keep
  latency within SLAs and naturally accounts for long requests holding batch
  space. Both default to unbounded. Rejections are observable via the existing
  `maxserve.request_count` metric with `code="429"`.
- Added `MAX_SERVE_GRACEFUL_SHUTDOWN_TIMEOUT_S` to control how long the server
  waits for in-flight requests to finish after receiving `SIGTERM` before
  exiting (default 5 seconds). Raise it so long-running requests are drained
  rather than dropped during a rolling restart.
- Added a request body size limit. `MAX_SERVE_MAX_REQUEST_BYTES` (default
  100 MiB) caps the size of an accepted HTTP request body; a larger request is
  rejected with HTTP 413 before the body is buffered, so a client cannot
  exhaust host memory with an oversized payload. The cap is enforced both from
  an oversized `Content-Length` and by counting the bytes actually received, so
  a chunked or mislabeled body cannot evade it. Raise it for larger inline
  (base64) multimodal payloads, or set it to 0 to disable the limit.
- Data-parallel (DP) serving now shares the prefix cache across replicas, so a
  multi-turn conversation gets cache hits even when a later turn is scheduled on
  a different replica than the previous one. GPU prefix-cache hits are served by
  a cheap device-to-device copy of the cached pages onto the assigned replica,
  and the CPU/disk offload tiers are now a single pool shared by every replica
  (a block offloaded by one replica can be loaded by another). As a result,
  `host_offload_max_gb` now sizes one shared host pool of that size for
  the whole deployment, rather than allocating a separate pool of that size per
  replica.
- `--kv-connector-config '{"type": "rust_tiered", "disk_offload_max_gb": 0}'`
  now runs the tiered connector with no disk last level: offloaded blocks stop
  at the pinned host tier and no offload directory is created. Leaving
  `disk_offload_max_gb` unset still sizes the disk tier from the device page
  pool, and a negative budget is now rejected instead of silently accepted.
- The dKV external KV-cache connector (`--kv-connector-config '{"type":
  "dkv"}'`) now supports
  data-parallel (DP) serving and shares its prefix cache across DP replicas on
  the default single-tenant path, matching the `local` and `tiered` connectors.
  Every replica resolves to the same replica-agnostic store, and the stored
  block key carries no replica component, so a block offloaded through one
  replica is served to any other.
- The dKV external KV-cache connector now supports tensor parallelism
  (TP greater than 1) on the multi-tenant path for head-sharded (MHA/GQA), MLA
  (replicated-KV), and GQA head-replicated (`allow_kv_head_replication`) models.
  Each GPU handshakes its own per-shard store, and every KV load/offload fans
  out across the processing replica's shard clients with identical block ids and
  hashes; a block counts as loaded only once every shard has it. The store key
  reflects the KV-head slice each GPU holds: the TP rank when head-sharded, a
  single shared shard for MLA, and the head-group index under head replication.
- On the dKV multi-tenant tensor-parallel path, a KV load that returns
  differing block counts across a replica's per-GPU shard clients now drains
  the over-loading shards' in-flight device reads before returning the minimum
  count. This keeps a stray in-flight host-to-device copy (into a block the
  block manager frees because it did not land on every shard) from later
  clobbering a reallocated block. The drain host-completes the reads on the
  remote (NIXL) transport and enqueues a cross-stream ordering on the
  co-located same-host (CUDA) transport, so it closes the window on both. The
  common equal-count path is unchanged and pays no extra synchronization.
- The dKV external KV-cache connector (`--kv-connector-config '{"type":
  "dkv"}'`) now requires a
  non-empty tenant identity (`MODULAR_DKV_TENANT_ID`, set by the deployment
  operator); the empty-tenant "default" path is removed. Both the connector and
  the dKV server now reject an unset/empty tenant rather than keying an unfenced
  shared store, so every deployment (single-tenant included) routes through the
  per-tenant region-sharded store — DP replicas of one tenant still share one
  store. Multi-cache models (speculative draft+target, quantized values+scales)
  now resolve on this path, folded into the handshake's `kv_config_hash`. A
  single-tenant node spanning more than one GPU must set the dKV server's
  `--fair-share-partitions` to its GPU count.
- The dKV external KV-cache connector now waits out a busy node instead of
  failing model load on it. dKV refuses a handshake when it has no room for
  another share, which is a transient condition that clears once a departing
  share's memory is released, so the refusal is now retriable and the
  connector's admission budget (`MODULAR_DKV_ADMISSION_TIMEOUT_S`, default
  raised from 120s to 600s) retries it. A budget too small to cover several
  attempts is raised to that floor with a warning rather than rejected, so a
  deployment that pinned the old default keeps starting.
- A request's `dkv_cache_hint` now reaches the dKV external KV-cache connector,
  which reads it to load a cached prefix from the instance that holds it rather
  than only from the co-located one. The serving layer forwards the field
  without interpreting it, so the hint schema is versioned in one place and a
  hint this build cannot use costs a cache miss rather than a failed request.
  Previously the field was parsed into a form nothing read, and every hinted
  load went to the co-located dKV.
- The dKV external KV-cache connector now accepts a KV cache tree that mixes
  TP-replicated and head-sharded caches, instead of failing model load. Only
  an all-replicated tree produces a block that is byte-identical across TP
  shards, so a mixed tree offloads over the ordinary per-shard path. On that
  path a replicated cache is stored once per TP shard rather than once, so
  size the dKV share above what the `rust_tiered` connector needs for the
  same model.
- Added `MODULAR_MAX_RELEASE_FREE_HOST_MEMORY`, an opt-in serving knob that
  returns free host-allocator pages to the OS once model compilation finishes,
  before graph capture. Graph compilation leaves tens of GiB free-but-unreturned
  in glibc's per-thread arenas, which glibc never reclaims on its own; setting
  this variable to any non-empty value calls `malloc_trim(0)` at that point.
  On Gemma 4 31B this returns ~24 GiB of anonymous RSS per model worker in
  ~1.4s. Unset by default, and a no-op on platforms without `malloc_trim`.
- Setting the `MODULAR_MAX_RELEASE_HOST_WEIGHTS` environment variable to
  `1` frees the host copies of checkpoint weights once the GPU holds
  them, returning the full checkpoint size in host RSS. GPU deployments
  of graph-API architectures only; weights that execute on CPU must not
  be released.
- Chat completions now honor `reasoning_effort`; previously only an explicit
  `chat_template_kwargs.reasoning_effort` had any effect and the standard
  fields were silently ignored. An effort of `none` disables thinking, and
  values set directly in `chat_template_kwargs` still win.
- `--num-speculative-tokens` is now unset by default, and each speculative
  method resolves its own default: `eagle` and `mtp` keep drafting 2 tokens
  per step, while `dflash`-style block drafters (DFlash, DSpark) derive the
  draft checkpoint's trained block width. Explicit values are honored as
  before. Previously the flag defaulted to 2 for every method and block
  drafters overrode it at load time with a warning; a bare DFlash run now
  also sizes its KV cache draft headroom at the trained width instead of the
  old default.
- The vision encoder cache now stores embeddings in fixed-size blocks.
  Capacity is a byte budget carved into 128-token blocks — a video spans
  many blocks and an image a few — so a video-capable model no longer
  collapses the cache to a handful of worst-case-video slots that starve
  image workloads. The budget is set with the new
  `--vision-cache-utilization` flag, a fraction of the KV cache pool
  budget (default `0.05`; `0` disables caching). The previous
  entry-count cache and its `--max-vision-cache-entries` flag are
  removed.
- Vision embedding assembly during chunked prefill is now bounded by the
  active window: each step copies only the embedding rows whose
  placeholder tokens fall inside the chunk, with dense scatter indices,
  instead of rebuilding every image's rows with out-of-bounds sentinels.
  Per-chunk copy cost now scales with the chunk size rather than the
  request's total image tokens.
- Added `DeviceBuffer.unsafe_host_ptr()` to the Mojo `max.gpu.host` API. On
  devices with unified memory (Apple silicon), it returns a CPU-addressable
  pointer to the buffer, so the host can read a kernel's output after
  `DeviceContext.synchronize()` without an `enqueue_copy` round trip. Reads
  through it are uncached, so it suits small control records rather than bulk
  readback. A CPU device returns the buffer's own pointer, since its
  allocations are host memory already; devices whose memory is not
  CPU-addressable raise.
- `DeviceContext.create_event()` and `DeviceEvent` are now supported on Apple
  GPUs, backed by `MTLSharedEvent`. Event queries and waits track actual GPU
  completion instead of command-buffer submission order, and waiting on an
  event from another context's queue no longer blocks the host thread.
- `DeviceContext.create_event()` on NVIDIA GPUs now honors the default
  `disable_timing` flag (previously inverted) and recycles events through the
  driver's event cache instead of growing it on every create/destroy cycle.
- Device-to-device copies on Apple GPUs no longer race when the source was
  written on another `DeviceStream`.
- `MODULAR_DEBUG=device-sync-mode` now works on Apple GPUs, where it
  previously did nothing.
- Capturing `DeviceContext.enqueue_function()` now encodes the closure
  through `DevicePassable` before launch, matching explicit kernel
  arguments. Host handles such as `DevicePointer` reach the device as
  device addresses rather than raw host bytes.
- Added `max.nn.state_space.kda_decode`, a wrapper over the Kimi Delta
  Attention recurrence op.

### Inference server

- `/v1/responses` now fetches client-supplied `input_image` URLs through the
  same media resolver as `/v1/chat/completions`, so the two paths share one
  byte cap and one error mapping. Previously the responses path had its own
  downloader with no size limit, meaning an arbitrarily large image could be
  fetched and base64-expanded in memory, and its failures echoed the
  underlying network error back to the client. The inlined `data:` URI's MIME
  type is now sniffed from the fetched bytes instead of guessed from the URL,
  and content that is not a decodable image is rejected with a 400 rather than
  inlined as an image.

- Structured-output grammars are now compiled once, in the model worker.
  The API server used to compile a `response_format` schema or tool-call
  grammar just to validate it, throw the result away, and leave the worker
  to compile the same grammar again against a cache it does not share.
  Removing the duplicate lowers time to first token for structured
  requests by 12-22% (Gemma 4 31B, concurrency 32); decode latency and
  requests without structured output are unchanged. An uncompilable
  grammar is still rejected with the same HTTP 400, streaming requests
  included, and a disaggregated prefill node now reports the failure to
  the decode node instead of leaving the request to time out.

- Speculative decoding can now verify only some of the draft tokens it
  generates, varying that count with the decode batch size via the new
  `num_speculative_tokens_per_batch_size` speculative-config field. Each entry
  names an inclusive batch-size range and a count through the keys
  `batch_start`, `batch_end`, and `num_tokens`, so a two-range schedule is
  `[{"batch_start": 1, "batch_end": 16, "num_tokens": 3}, {"batch_start": 17,
  "batch_end": 64, "num_tokens": 1}]`. The first range must start at batch size
  1 so every batch size resolves to a count; gaps and the tail carry the
  previous count forward. Drafting is cheap, but every draft the target verifies
  is another query position in its forward pass, so at high concurrency those
  positions compete with real tokens for the same compute and a rejected draft
  is compute spent for nothing. Whether narrowing pays off therefore depends on
  how well the drafts are being accepted, which is a property of the workload
  rather than of the batch size. Measure your own workload before adopting a
  schedule. The field is off by default, and unset behavior is unchanged. It
  applies to every speculative method. A block drafter (`dflash`) still drafts
  its whole checkpoint-fixed block every step, so a schedule narrows only how
  much of that block the target verifies; the saving comes from the target's
  verify pass, never from drafting less.

  It is most useful for a block drafter, whose draft depth is fixed by its
  checkpoint, making the verified count the only runtime lever on step cost.
  Where the draft depth is itself configurable, as it is for `eagle` and
  `mtp`, lowering `num_speculative_tokens` is the better tool: it removes the
  draft passes as well as the verify positions, while a schedule pays for
  drafts it then discards. A count of `0` is accepted and disables
  verification for that batch-size range.

- GLM models now map `reasoning_effort` onto the two thinking levels their
  chat template can express, instead of forwarding it verbatim. The template
  reads only `high` as a distinct level and treats every other value as
  maximum effort, so passing the value through inverted the scale: `low` and
  `medium` requested maximum reasoning while `high` requested less than they
  did. Every effort other than `none` (which disables thinking), `max` (the
  template's own top level, still addressable directly) and `xhigh`
  (OpenRouter's name for that same top level) now selects the lower level, so
  an unrecognized value degrades to less reasoning instead of silently maxing
  out. Requests that set no effort are unaffected.

- Structured-output JSON grammars can be made whitespace-tolerant, per
  architecture via `default_structured_output_any_whitespace`.
  - GLM 5 models default to whitespace-tolerant `response_format` grammars.

- Structured-output grammar compilation now runs off both serving hot
  paths. A new request's grammar matcher (from `response_format` JSON
  schemas or tool-call grammars) is built on a worker thread while the
  request waits for admission instead of on the scheduler's decode
  thread, and the API server's admission-time schema validation runs off
  the event loop instead of freezing in-flight streaming responses. A
  cold multi-second compile of a complex schema now delays only that
  request instead of stalling inter-token latency for every active
  request.
- A JSON schema that composes with `allOf` is now enforced instead of
  refused. `response_format` and tool-call schemas previously returned 400
  for any `allOf` with more than one member, or with a sibling object
  keyword. The members now fold into one schema before compilation,
  including members nested in another member's `allOf` and members that are
  a bare local `$ref`, so the common "shared definition plus an extension"
  shape compiles. A conjunction that cannot be folded exactly still returns
  400 naming the keyword pair at fault, rather than compiling to a looser
  grammar.
- A JSON schema using `oneOf` is now enforced when its branches can be
  proven pairwise disjoint, instead of being refused outright. Disjoint
  branches make the union exactly-one, which is what `oneOf` means. Branch
  types and `const`/`enum` value sets carry the proof, covering nullable
  values, scalar unions, enum partitions and unions discriminated by a
  constant property. A union that cannot be proven disjoint still returns
  400, as does a `const`/`enum` branch carrying a keyword the lowering
  drops. The refusals apply when unsupported-schema rejection
  (`reject_unsupported`) is enabled.
- Fixed a union (`anyOf`/`oneOf`) folding its sibling keywords into each
  branch too widely, which could accept values the schema forbids. These
  shapes now return 400 instead, when
  unsupported-schema rejection (`reject_unsupported`) is enabled: a closing
  `additionalProperties`, `items` or `unevaluatedProperties` beside a union,
  a base constraint beside `$ref`, `const` or `enum` — whether folded in
  from a union or written in the same object — and `$ref` beside a
  sibling union.

- Hardened the server-side fetch of client-supplied `image_url` / `video_url`
  references against SSRF: the host is now validated and hosts that resolve to
  internal or reserved addresses are rejected before the fetch. On by default
  (`MAX_SERVE_MEDIA_URL_SSRF_PROTECTION_ENABLED`); a per-host allowlist
  (`MAX_SERVE_MEDIA_URL_ALLOWED_HOSTS`, hostnames or CIDRs) permits trusted
  internal hosts.

- Compiling deeply nested JSON schemas is substantially faster and uses
  less memory by avoiding repeated subtree copies while constructing cache
  keys. Emitted grammars are unchanged.

- Fixed strict JSON Schema compilation silently dropping string length
  bounds when a pattern or format is present. Redundant bounds now compile,
  while unsatisfiable or partially overlapping constraints return 400.
  Equivalent direct, `allOf`, and union-folded schemas receive the same
  result. Regex length analysis has a per-schema work limit, so oversized
  patterns return 400 promptly.

- Fixed JSON Schema compilation resolving a local `$ref` against the wrong
  resource when the document declares a resource identifier (`$id`, or `id` in
  Draft 4) below its root. A fragment names a place inside the resource it is
  resolved against, and every fragment was resolved against the whole document,
  so a definition name that appeared in both an embedded resource and at the
  root bound the root's copy in silence. When unsupported-schema rejection
  (`reject_unsupported`) is enabled, such a document now returns 400 naming the
  declaration, rather than compiling a grammar the author never wrote. A
  document whose only resource identifier sits at the root, or that has none at
  all, is one resource and is unaffected.

- JSON Schema compilation now recognizes the `$schema` dialects it models:
  Draft 4, whose resource identifier is `id`, and Drafts 6, 7, 2019-09 and
  2020-12, whose identifier is `$id`. When unsupported-schema rejection
  (`reject_unsupported`) is enabled, any other `$schema` returns 400 rather
  than being read as a modern document, because assuming the wrong dialect
  walks past the resources a document declares and resolves its fragments
  against the wrong one. Omitting `$schema`, as most tool schemas do, still
  means the current draft and is unaffected.

- Speculative decoding takes `--draft-proposal sampled` (default `argmax`,
  unchanged). The draft model samples its proposal under the request's
  temperature/top-k/top-p and keeps the distribution it drew from, so
  verification runs true speculative sampling — accept on the
  `p_target/q_draft` ratio, recover from `max(p_target - q_draft, 0)` — rather
  than the typical-acceptance approximation, and the emitted tokens follow the
  target model's distribution.

### Server metrics

- `maxserve_cache_hits_tokens_total` now carries a `tier` label naming what
  served each token: `g0` for the on-device prefix cache (including
  cross-replica device-to-device copies), `external` for the KV connector.
  The per-tier series sum to the untagged total, so an existing single-series
  query that doesn't group by `tier` returns the same numbers as before. Misses
  stay unlabeled, which means a PromQL binary operation pairing hits against
  misses (a hit-rate expression) now matches on mismatched label sets and
  returns empty: add `ignoring(tier)`, or wrap the hits side in
  `sum without(tier) (...)`. The in-tree Datadog dashboard aggregates the tag
  away and is unaffected; external Prometheus consumers are the exposure.
  Previously the on-device share could only be derived by subtracting the
  external tier's own server-side counters, which measure what that tier holds
  rather than what a request could use and so overstate reuse. Note that the
  untagged series is replaced rather than extended, so a `rate()` window
  spanning the upgrade sees the old series go stale and the labeled ones start
  from zero.
- Added `maxserve_dkv_read_blocks_total`, the count of KV blocks that landed in
  device memory from the dKV tier. Only confirmed-complete transfers count, so
  it measures delivered reuse. It is emitted only on dKV deployments, while
  `maxserve_cache_hits_tokens_total{tier="external"}` is stamped for any KV
  connector, so a missing counter means "not dKV" rather than "nothing landed".
  On a dKV deployment the two track each other for every load that lands, and
  comparing them needs the server's `--kv-cache-page-size`, since one is in
  blocks and the other in tokens.
- Fixed the speculative-decoding per-position acceptance-rate histogram
  (`maxserve_spec_decode_acceptance_rate_per_position`) understating
  acceptance: decode batches that performed zero verifications published a
  full row of 0% observations, diluting every position's average. Such
  batches now contribute nothing, matching the acceptance-length histogram's
  population. The batch log line also shows the acceptance length including
  the bonus token next to the accepted-drafts-per-step value, since the two
  conventions are easy to confuse.

### `max` CLI

- `max warm-interpreter-cache` now shows a live progress row per op family.

- Fixed `max warm-interpreter-cache` failing with a `ValueError` on a
  machine where an op family supports none of the available devices (for
  example, a GPU-only op family on a CPU-only machine). Such a family now
  warms as a no-op instead of aborting the whole command.

- Fixed LoRA and denoising-cache CLI flags replacing, rather than
  overriding, the matching `--config-file` section; `--enable-lora=false`
  now also disables LoRA that a recipe enabled, instead of being ignored.

### Python API

- `max.experimental.nn.Module.compile` reuses precompiled MEFs when the session
  has them, so a ModuleV3 model can be compiled where no accelerator is attached
  and initialized where one is. `max.experimental.support.set_export_mefs`
  records each compiled graph into a directory, and
  `max.experimental.support.set_precompiled_mefs` initializes those artifacts
  instead of compiling. `InferenceSession.compile_reusing_mefs` is the same
  half-step for callers that trace a graph and initialize it themselves.

- Eager mode tensors will use the JIT by default. This unlocks fusion and
  shape specialization optimizations even for eager code, beating PyTorch
  performance in eager in the common case.

- `max.experimental.sharding.NamedMapping` takes its mesh from the enclosing
  `mesh_context()` when none is passed, so a layer can name the axis it shards
  along without being handed a mesh. Its `original_spec` and
  `original_unreduced` properties are removed.

- Added `max.experimental.tree_utils`, pytree utilities over nested `list` /
  `tuple` / `namedtuple` / `dict` and any class declaring the tree protocol:
  `__tree_flatten__` with either `__tree_unflatten__` or `__tree_empty__`, and
  an optional `__tree_setattr__`. There is no registry and no decorator, so a
  type opts in by declaring the methods. `flatten` and `unflatten` carry a
  value across a flat boundary, `leaves`, `paths` and `nodes` read it, `map`
  builds a new tree, and `update` writes path-keyed values into an existing one
  in place. Every walk takes `leaf`, saying where it stops, and `shared`,
  saying whether a value reachable by two paths is one object or two. Import
  the module as a namespace: `from max.experimental import tree_utils as tree`.

- Added `max.experimental.compilation`, three transforms over plain
  callables. `stage(fn)(*args, **kwargs)` traces `fn` into a `max.graph`
  that can be printed and inspected as MLIR. The arguments are `fn`'s own,
  except that each tensor is given as a `TensorType`. This partially
  evaluates `fn`: the tensor types become graph inputs, and every other
  argument is evaluated during tracing. `compile(fn, weights=...)(*args,
  **kwargs)` stages the same way and compiles the graph; the result is
  callable on real tensors. Weights and device memory load only on the
  first call, so `export_mef` can save the compiled graph to a file without
  loading either. `as_subgraph(fn)` returns a drop-in replacement for `fn`
  that, during tracing, calls one shared subgraph instead of inlining its
  body, so a stack of identical layers compiles once.

- `max.graph.ops.reduce_scatter_rms_norm` takes an optional `group_size`
  argument, matching `max.graph.ops.reducescatter.sum`: the devices split into
  contiguous groups of that many, each reducing independently, so the fused op
  also works under tensor-parallel-within-data-parallel topologies. It was
  previously full-world only and silently disabled itself whenever the
  tensor-parallel degree was smaller than the device count.

- `max.graph.ops.allgather_rms_norm` takes an optional `group_size` argument,
  matching `max.graph.ops.allgather`: the devices split into contiguous groups
  of that many, each gathering independently, so the fused op also works under
  tensor-parallel-within-data-parallel topologies. It was previously full-world
  only.

- `max.driver.Buffer` now implements `__str__`, so `str(buffer)` and
  `print(buffer)` show the buffer's data formatted like a numpy array, followed
  by its `dtype`, `shape`, and `device`. `repr(buffer)` still returns the
  metadata-only representation.

- Added `max.driver.Usage`, an allocation-intent flag for `Buffer`.
  `Buffer(..., usage=Usage.STAGING)` requests host memory for staging
  transfers to and from the given device, which may be page-locked
  depending on the backend. `Buffer.usage` reports the intent;
  `Buffer.pinned` reports whether the memory is page-locked.

- **Breaking:** the `pinned=` argument to `Buffer(...)` and
  `Buffer.zeros(...)` is removed. Use `usage=Usage.STAGING` instead.

- DLPack export of a staging buffer (`__dlpack__`, and `to_numpy()` in
  turn) does not synchronize pending device work. Synchronize explicitly
  before reading one after a device operation.

- `max.nn.sampling.AcceptanceSampler` and
  `max.nn.sampling.stochastic_acceptance_sampler` take a `draft_proposal`
  argument. The default, `"argmax"`, is unchanged: the draft proposes
  deterministically and verification runs typical acceptance. With
  `"sampled"`, the caller passes the distribution the draft sampled from, so
  verification runs the real `p_target / q_draft` ratio test and recovers
  rejected positions from `max(p_target - q_draft, 0)`; temperature, top-k and
  top-p then all apply to the draft-verification distribution, where
  `"argmax"` applies only temperature. Sampled mode is GPU-only, needs a
  static `vocab_size`, and cannot be combined with relaxed thinking-phase
  acceptance, whose rule assumes the drafted token is the draft's argmax.

### C API

## MAX kernels

- SM100 matmuls with an elementwise epilogue no longer leave output columns
  unwritten when `N` is not a multiple of 16, such as `N=136` or `N=776`.

- SM100 bf16 and fp8-input matmuls whose N leaves the output row stride short
  of TMA's 16-byte alignment, such as a 258-wide MoE router projection, now
  take the split-K GEMV at up to 64 rows instead of falling back to vendor
  BLAS.

- The SM100 MLA decode dispatch now enumerates 12, 24 and 48 query heads
  alongside the powers of two it already covered, so a model whose per-device
  head count is not a power of two can bind its dispatch metadata.

- KDA prefill now runs on the chunk-parallel pipeline. The pipeline existed as
  a Mojo kernel with no graph-op registration, so every prefill fell back to
  the token-sequential decode recurrence: O(total_seq_len) sequential steps per
  sequence, with no parallelism to spend on a long prompt. Registering
  `kda_chunk` as its own graph op takes that to O(total_seq_len / CHUNK_SIZE).

- Added `MODULAR_APPLE_M5_ALLOW_LOSSY_F32_ATTENTION`. Set it to `0` to keep
  fp32 attention off the Apple M5 MMA, which truncates operands to fp19. It
  defaults to the fast (lossy) path, matching
  `MODULAR_APPLE_M5_ALLOW_LOSSY_F32_MATMUL`.

- Improved MXFP8 block-scaled matmul decode latency for attention
  output-projection shapes at M=4, M=32, M=64, and M=128 on MI355.

- Improved MXFP8 block-scaled fused QKV projection decode latency at M=4 on
  MI355.

- The MLA sparse-attention indexer (DeepSeek V3.2, GLM 5.x) now does work
  proportional to each row's actual key count instead of the batch's
  `max_cache_length` metadata. Inside captured decode device graphs that
  metadata is baked at capture time — with a 1M-token maximum sequence length
  it sits orders of magnitude above the tokens a batch actually holds — and
  the indexer paid a full-width `-inf` score fill, a full-width top-k scan,
  and a key-tile-per-CTA scorer grid per layer per step at that frozen
  bound. The bitonic top-k kernels now clamp each row's scan to its live
  causal range, the score-buffer fill is skipped on the SM100 scorer path
  (which writes every live slot itself), and the SM100 scorer's key-split
  route now covers the tensor-parallel head counts (4 and 8) with its part
  count capped at a fixed number of waves, so the grid is sized to the
  hardware rather than to the metadata bound while per-CTA loop bounds come
  from the runtime cache lengths. At the GLM 5.2 MTP decode shape (batch 8,
  width 6, 76k-token context, 4 heads per rank) with metadata frozen at 1M,
  one indexer layer drops from 0.89 ms to 0.10 ms on B200, matching its
  cost at a bound sized to the runtime lengths; shapes without a metadata
  gap are unchanged except a small fixed per-call cost for the row-bounds
  clamp (~4% on a batch-256, 4k-context decode).
- Sped up GPU token sampling by about 4% per output token when the largest
  `top_k` in the batch is below 10, by removing a device synchronize from
  `fused_token_sampling_gpu`. The synchronize backed a check that raised on an
  all-NaN logits row. Such a row now yields an arbitrary in-range token rather
  than an error. Set `max-debug.assert-level` to `all` to restore the check, or
  use `max-debug.nan-check` to locate NaN logits.
- Fixed expert-parallel dispatch dropping half of every token belonging to an
  expert that only one communication SM serves, which surfaced as NaN logits.
  The block-scaled wire formats (NVFP4 and MXFP8) copy a token tile as two
  column halves claimed separately, and the claim loop stopped as soon as a
  claim covered the last token, so the remaining half was never copied unless
  a second SM happened to be on the same expert. Since experts are assigned
  round-robin over the communication SMs, this began once a device held more
  experts than half that count — 74 per device on a B200, so a 896-expert MoE
  over eight devices returned NaN while 512 experts stayed correct.

- The SM100 grouped block-scaled matmul accepts MXFP4 weights against MXFP8
  activations (W4A8), so a quantized MoE can feed its packed 4-bit experts
  straight to the tensor cores rather than dequantizing them to bfloat16
  first. This removes MAX's per-forward `mxfp4_dequant` over the routed expert
  stack, and it keeps the weights at their 4-bit footprint in global memory,
  which matters most at expert counts where a bfloat16 copy of the stack does
  not fit. A new `unpack_fp4` option on the NVIDIA TMA descriptor helpers,
  backed by the `TensorMapDataType.PACKED_FP4_ALIGN16B` tensor-map type, pads
  the weights into the byte-addressed form the tensor cores read as the copy
  engine lands them in shared memory.

- The joint top-k/top-p sampling kernel can now also return the masked,
  renormalized distribution it drew from, exposed as
  `max.nn.kernels.topk_fused_sampling_with_dist`. Speculative decoding needs
  that distribution to build a rejection residual, and reads the sampled
  token's own probability out of it -- a value that has to agree with the
  sampler's accept decision, so it comes from the sampling kernel rather than
  a separate softmax. When top-k, top-p, and min-p are disabled, the
  distribution-producing path now skips its cutoff search. The existing
  single-output path is unchanged. On AMD GPUs, the distribution output also
  serves as temporary storage for exponentiated logits during sampling.
- MiniMax-M3 sampled MTP now samples only the accepted initial draft row
  instead of every possible acceptance position.
- Added `max.nn.kernels.topk_topp_masked_probs`, which computes a row's
  top-k/top-p masked renormalized softmax without sampling and without a
  sort. Speculative decoding verification reads the target's masked
  probability of each drafted token and builds its rejection residual from
  this one tensor, in the same form the draft sampler emits its proposal
  distribution. When top-k and top-p are disabled, the kernel now skips the
  cutoff search because every positive-probability token already survives.
  On AMD GPUs, it also caches exponentiated logits in the output buffer so
  cutoff-search passes do not recompute them. Rows with top-k disabled also
  omit positive-value counting from the initial mass reduction and cutoff
  search.
- Top-p-only distribution kernels bias cutoff-search pivots toward lower
  weights when the retained-mass budget is large relative to the mass still
  above the search's low bound, so the gain follows the bracket state rather
  than the requested `top_p`.
- The fused gumbel-argmax sampling kernel takes a `from_probs` parameter,
  exposed as `max.nn.kernels.gumbel_argmax_from_probs`: each row's score is
  `ln(p) + gumbel` over unnormalized probabilities, drawn with noise the
  kernel generates from a per-row seed. This enables sampling a speculative
  decoding rejection residual `max(p_target - q_draft, 0)` that the caller
  builds in graph ops. GPU-only, non-Apple.
- Improved wide-row FP32 Gumbel sampling performance on AMD GPUs.
- Retuned the MI355X dispatch table for a grouped block-scaled MoE
  matmul (gate-up and down projections) at the estimated-total-M > 2048
  band that real serving traffic hits, plus the down projection's
  estimated-total-M <= 2048 band. Gate-up projection speeds up 7.4-10.1%
  and down projection 18.2-19.6% (etm > 2048) and 6.9-23.3% (etm <= 2048)
  across real ragged-M, skewed routing scenarios.

## Breaking changes

- Removed the `NPU` device class from `max.driver` and the corresponding
  `DeviceRef.NPU()`, `DeviceRef.is_npu()`, and `DeviceKind.NPU` from
  `max.graph`, along with the `M_newNPUDevice()` C API entry point. `NPU` was
  a thin subclass of `Accelerator` that differed only in the device label it
  stamped on the graph; it had no callers, and accelerator backends reached
  through a driver plugin are already served by `Accelerator`. Construct
  `Accelerator()` (or `DeviceRef.GPU()`) for any non-CPU device, and read the
  `Accelerator.api` property to tell the concrete backends apart.
- The tile-tensor storage policy is renamed to an engine, and the
  `layout.tensor_storage` module is renamed `layout.tensor_engine`. The
  `TensorStorage` trait becomes `TensorEngine`, `TileTensor`'s `Storage`
  parameter becomes `Engine`, and the conforming policies `PointerStorage`,
  `DevicePointerStorage`, and `StaticOffsetStorage` become `DefaultEngine`,
  `DevicePointerEngine`, and `StaticOffsetEngine`. The trait describes the
  operations a tile tensor performs on its handle (load, store, bitcast,
  elementwise) rather than the memory it points at, so the old name described
  the wrong thing. Update `Storage=` keyword arguments to `Engine=` and any
  `tensor.Storage` accesses to `tensor.Engine`. The `TensorOps` trait and the
  associated `StorageType` handle keep their names, since they still describe
  the borrowed memory itself.

  Kernel signatures follow. Every comptime parameter bound to `TensorEngine`
  or `TensorOps` now ends in `Engine`, replacing the three spellings that
  were in use: `OutputStorage` and `XStorage` become `OutputEngine` and
  `XEngine`, `QStorageType` and `SeedStorageType` become `QEngine` and
  `SeedEngine`, and the snake_case `q_storage` and `x_store` become
  `q_engine` and `x_engine`. Callers passing any of these by keyword need to
  update the name.
- The KV connector's external host and disk tiers now report occupancy and
  transfer volume in bytes rather than in blocks. Those tiers are byte budgets
  the operator sizes in bytes (`host_offload_max_gb`, `disk_offload_max_gb`),
  their block width need not match the device's, and bytes rate directly
  against PCIe and disk bandwidth. The device (G0) cache is unchanged and
  still reports blocks.

  `KVConnector` replaces `host_block_count` / `disk_block_count` with
  `host_byte_count` / `disk_byte_count`, returning a new `ByteCount` (the same
  `free` / `total` / `used` / `used_pct` / `free_pct` surface as `BlockCount`,
  measured in bytes). The KV cache managers make the same swap;
  `block_count()` is untouched. `KVCacheMetrics` renames `h2d_blocks_copied`,
  `d2h_blocks_copied`, `disk_blocks_read`, and `disk_blocks_written` to
  `h2d_bytes_copied`, `d2h_bytes_copied`, `disk_bytes_read`, and
  `disk_bytes_written`.

  The exported metrics follow: `maxserve.cache.h2d_blocks_copied`,
  `maxserve.cache.d2h_blocks_copied`, `maxserve.cache.disk_blocks_read`, and
  `maxserve.cache.disk_blocks_written` become `h2d_bytes_copied`,
  `d2h_bytes_copied`, `disk_bytes_read`, and `disk_bytes_written`, with unit
  `bytes`. `maxserve.cache.used_host_kv_pct` and
  `maxserve.cache.used_disk_kv_pct` keep their names and are now computed over
  bytes. Dashboards and alerts on the old tier counter names need updating.

- The pipeline configs are now immutable: `PipelineArgs`,
  `PipelineConfig`, `PipelineRuntimeConfig`, `SamplingConfig`,
  `MAXModelConfig`, `KVCacheConfig` and its nested `KVConnectorConfig`,
  `LoRAConfig`, and `ProfilingConfig`. Assigning to a field after
  construction raises a pydantic `ValidationError`. Construct them with
  the values you need.

- `ModelManifest` is now immutable from construction: mutating the mapping
  (item assignment, `update`, `pop`, and so on) raises a `TypeError`, and
  `ModelManifest.resolve()` is removed — a manifest is complete when built.
  Construct it with the component configs you need. The unused
  `total_weights_size` property is also removed.
- `SpeculativeConfig` is now immutable: assigning to a field after
  construction raises a pydantic `ValidationError`. Construct it with the
  values you need. A failed speculative target-architecture rewrite now
  raises from `PipelineConfig.from_args()` instead of being logged and
  ignored.

- An architecture can set `checkpoint_draft_width` on its
  registration to supply the draft width its checkpoint was trained for,
  so users of those models do not have to pass `--num-speculative-tokens`.
  A width that disagrees with the checkpoint is replaced, with a warning.

- Constructing a `MAXModelConfig` directly now only validates the fields
  you pass. It no longer fills in the weight and model paths or loads the
  HuggingFace config. Configs the pipeline builds are unchanged.

- `ArchConfig.calculate_max_seq_len()` no longer takes `pipeline_config`,
  and `model_config` is now required.

- `KVCacheConfig.allow_kv_head_replication`, the architecture registration
  field `requires_kv_head_replication`, and the
  `--allow-kv-head-replication` flag are removed. An architecture now asks
  for KV head replication in its `construct_kv_params()`.

- The KV cache connector is now configured as a single object: its type moved
  onto `--kv-connector-config` as a `type` field, and the separate
  `--kv-connector` flag is removed. Replace `--kv-connector rust_tiered` with
  `--kv-connector-config '{"type": "rust_tiered"}'`, and in a recipe set
  `model.kv_cache.kv_connector_config.type`. `host_kvcache_swap_space_gb` is
  renamed `host_offload_max_gb` to match `disk_offload_max_gb`, and both now
  default to sizing their tier from the device page pool (1.5 times it on host,
  twice on disk) rather than to a fixed 50 GiB. Dict-valued `kv_cache`
  flags now merge field-wise over a config file's value instead of replacing
  it, so overriding one connector field on the command line keeps the rest --
  previously a partial override reset the connector type and silently disabled
  offloading.

- Renamed `max.driver.DeviceStream` to `DeviceQueue` and
  `Device.default_stream` to `Device.default_queue`; the old names were
  removed. The driver models work submission as a command queue; a stream
  is one backend's implementation of that queue. Method, property, and
  argument names (`Buffer.stream`, `stream=`, `native_stream_handle`) are
  unchanged.

- Reworked `max.pipelines.PipelineArgs` and `PipelineConfig` construction
  around a single path and a single (nested) shape:
  - `PipelineArgs` now nests its runtime, sampling, and profiling fields in
    `runtime`, `sampling`, and `profiling` sub-configs
    (`PipelineRuntimeConfig`, `SamplingConfig`, and `ProfilingConfig`),
    matching the nested shape already used by recipes and `PipelineConfig`.
    Flat constructor kwargs for those fields (for example `max_batch_size=1`)
    are rejected; pass `runtime=PipelineRuntimeConfig(max_batch_size=1)`
    instead, and use the nested keys in config files validated into
    `PipelineArgs`. `PipelineArgs.from_flat_kwargs` (the CLI path) still
    accepts the flat spellings and routes them to the sub-configs.
  - Removed `PipelineConfig.from_flat_kwargs` and
    `PipelineArgs.from_pipeline_config`; `PipelineConfig.from_args` is the
    single way to construct a `PipelineConfig` from user input. Replace
    `PipelineConfig.from_flat_kwargs(...)` with
    `PipelineConfig.from_args(PipelineArgs.from_flat_kwargs(...))`.
  - `PipelineConfig.from_args` now also applies the model generation config's
    sampling defaults, applies `--model-override` entries, and resolves the
    speculative draft architecture, so programmatically constructed
    `PipelineArgs` behave the same as CLI invocations.
  - `PipelineRuntimeConfig` is now exported from `max.pipelines`.
- `--max-vision-cache-entries` is replaced by `--vision-cache-utilization`,
  a fraction of the KV cache pool budget for the vision encoder cache
  (default `0.05`; `0` disables caching). The cache is block-based, so an
  entry count no longer describes its capacity; configs setting the old
  flag must convert to a pool fraction.

- The legacy alias-buffer LoRA path has been removed. ModuleV3 LoRA (adapters
  passed as graph inputs) is now the only supported LoRA implementation.
  Serving a non-ModuleV3 architecture with `--lora-paths` now raises a clear
  error at startup instead of building a manager that never applies the
  adapters; serve the model's ModuleV3 variant (for example,
  `--prefer-module-v3`) to use LoRA adapters.

- Removed the parametric `max.benchmark.bencher_iter_custom[fn](bencher, ctx)`
  overloads and unused `bencher_iter_custom_multicontext()`. Pass the launch
  closure as a value: `bencher_iter_custom(bencher, fn, ctx)`.

- Removed `max.algorithm.reduce_boolean()`, which took its `reduce_fn` and
  `continue_fn` as `capturing` compile-time parameters and had no callers. Use
  `max.algorithm.reduce()` with a boolean accumulator, or write the early-exit
  loop directly.

- Removed the parametric `max.algorithm.parallelize[func](num_work_items, ...)`
  and `max.algorithm.parallelize_over_rows[func](shape, axis, grain_size, ...)`
  overloads that took a `capturing` closure as a compile-time parameter. Pass
  the body as a unified closure in the first runtime argument instead:
  `parallelize(func, num_work_items, ...)` and
  `parallelize_over_rows(func, shape, axis, grain_size, ...)`. Closure bodies
  drop `@__parameter` / `@__copy_capture` in favor of an explicit capture list,
  for example `def body(start: Int, end: Int) {imm}:`.

- Removed the parametric
  `max.algorithm.sync_parallelize[func](num_work_items, ...)` overload that
  took a `capturing` closure as a compile-time parameter. Pass the body as a
  unified closure in the first runtime argument instead:
  `sync_parallelize(func, num_work_items, ...)`. The remaining overload
  accepts `def(Int) raises -> None`, so both raising and non-raising
  closures bind. Closure bodies drop `@__parameter` / `@__copy_capture` in
  favor of an explicit capture list, for example `def body(i: Int) {imm}:`.

- Removed the parametric
  `max.benchmark.bench_multicontext[fn](bench, ctxs, ...)` overload. Pass the
  body as a unified closure in the second runtime argument:
  `bench_multicontext(bench, fn, ctxs, ...)`. Nested closures passed this way
  drop `@__parameter` in favor of an explicit capture list such as `{imm}` or
  `{mut buf, imm}`.

- Removed the parametric `capturing` overloads of
  `DeviceContext.execution_time[fn](num_iters)`,
  `DeviceContext.execution_time_iter[fn](num_iters)`, and
  `DeviceContext.enqueue_cpu_function[fn]()`. Pass the closure as a runtime
  argument instead: `execution_time(fn, num_iters)`,
  `execution_time_iter(fn, num_iters)`, and `enqueue_cpu_function(fn)`. Nested
  closures passed this way are unified closures, so replace `@__parameter` and
  `@__copy_capture(x)` with an explicit capture list such as `{imm}` or
  `{var x, imm}`.

- Removed the parametric capturing `layout.int_tuple.apply[func](t)`,
  `reduce[reducer](t, initializer)`, and capturing `apply_zip[func](...)`
  overloads. Pass the closure as a runtime value: `apply(t, func)`,
  `reduce(t, initializer, reducer)`, and `apply_zip(t1, t2, func)` (or
  `apply_zip(t1, t2, t3, func)`). Nested closures passed this way are
  unified closures, so replace `@__parameter` with an explicit capture list
  such as `{}` or `{imm}`. Thin `apply_zip[func](t1, t2)` function-pointer
  overloads are unchanged.

- `DeviceGraphBuilder.add_function[kernel](*args, ...)` takes a thin
  function pointer (`func: def(...) thin -> None`), the same identity as
  `DeviceContext.compile_function[kernel]()`.

- `PipelineRegistry.retrieve_factory` now returns a `RetrievedPipeline`
  dataclass with `tokenizer`, `factory`, and `memory_plan` fields instead of
  a `(tokenizer, factory)` tuple, so callers can reach the memory plan
  computed during retrieval. Replace tuple unpacking with attribute access.
  `PipelineRegistry.retrieve` is unchanged.

- The serving surface now reads the planned sequence length and batch token
  budget from the memory plan instead of re-reading them from the pipeline
  config. `TokenGenerationSchedulerConfig.from_pipeline_config`,
  `start_model_worker`, the scheduler loaders, and the startup log helpers
  (`log_basic_config`, `log_pipeline_info`) take the memory plan as a
  parameter. Resolved values are unchanged.

- Renamed `MemoryPlan.max_length` to `MemoryPlan.planned_max_length` to
  distinguish the plan's value from the user intent on
  `PipelineArgs.max_length` and the construction-resolved
  `PipelineConfig.model.max_length`, which keep their names.

- Denoising-cache input is now a frozen `DenoisingCacheSettings` on
  `PipelineArgs` (`denoising_cache`; in config files this section moves
  from `runtime.denoising_cache` to the top level). Construction fills
  unset fields from the architecture's TaylorSeer defaults into a frozen
  `DenoisingCacheConfig`. Enabling TaylorSeer without resolvable tuning
  fails at construction, as does enabling TaylorSeer and first-block
  caching together.

## Fixes

- Fixed a pre-tokenized prompt longer than `--max-length` killing the model
  worker instead of being rejected. Only a string prompt was length-checked,
  so a token-array prompt — an OpenAI `/v1/completions` token array, or the
  pre-tokenized prompt an orchestrator supplies for KV cache-aware routing —
  was admitted at any length and produced a request whose length exceeded the
  model's context window. Under speculative decoding the response path then
  raised rather than capping, taking the worker down and wiping its prefix
  cache; such a request now returns HTTP 400 like an over-length string
  prompt.

- Fixed constrained decoding producing invalid output when combined with
  speculative decoding on AMD GPUs. The in-graph wait that gates the grammar
  bitmask copy was not recorded into captured device graphs, so replays read a
  stale mask.

- Fixed tool calls being returned as raw markup in the assistant's `content`
  when a request did not declare a `tools` array. A tool established only by
  the conversation history, such as retrying a call that previously failed,
  now comes back as a structured `tool_calls` entry. Parsing runs whenever the
  model has a tool parser configured; `tool_choice="none"` still opts out.

- Fixed `max-debug.source-tracebacks` (for example
  `MODULAR_DEBUG=source-tracebacks` or
  `Graph.debug.source_tracebacks = True`) being silently ignored when it
  was enabled after `max.graph` was first imported. The flag was cached at
  import time, so runtime error messages lacked the `Source Traceback`
  section pointing back at the Python code that built the failing op, even
  though the config reported the feature as enabled.

- Fixed the `disk_bytes_written` KV cache metric counting blocks the tiered
  connector's disk tier declined to write because they were already saved or
  had a write pending. Re-offloading a block that had been evicted from the
  host tier but was still on disk inflated the count, and with it any
  disk-throughput figure derived from it.

- Fixed `generate_async` raising `KeyError: Request ID not found in replica
  batch` when requests in one batch finish on different steps, which happens
  whenever they are given different `max_new_tokens`.

- Fixed the offline `generate()` and `generate_async()` APIs releasing only a
  finished request's KV cache blocks, and never the pipeline itself. The
  pipeline-level release is what frees a recurrent state pool slot and drops a
  request's vision encoder cache references, so architectures that carry
  recurrent state — such as Nemotron-H, Mamba, and LFM2 — leaked one state slot
  per request, and a second run in the same process could inherit the first
  run's slot and return different greedy tokens.

- Fixed `DeviceExternalFunction` crashing on Metal instead of launching, so
  separately compiled kernels now load and launch there as they already did on
  other GPU backends.

- Fixed device buffer allocation no longer being pooled on GPUs without
  GPUDirect RDMA support, such as GeForce cards. A
  `DeviceContext.enqueue_create_buffer()` create and destroy round trip on an
  affected device took roughly 67 us instead of 375 ns.

- Fixed abandoned image, video, and audio generation requests still being
  rendered. The scheduler these tasks share never read its cancellation
  queue, so a request whose client had disconnected was executed in full
  once it reached the front of the queue, and the cancellations themselves
  accumulated. A request cancelled before it starts is now dropped and
  answered as cancelled; one already in flight still runs to completion,
  since a render is a single uninterruptible call.

- Fixed reductions over a zero-extent axis — for example `ops.sum(x, axis=1)`
  where that axis has length `0` — leaving their output unwritten, along with
  anything fused into the reduction's epilogue. Each now writes its identity:
  `0` for `sum`, `1` for `prod`, the dtype's minimum for `max` and its maximum
  for `min`, index `0` for `argmax` and `argmin`, and NaN for floating-point
  `mean` (as `numpy.mean` reports). Integer `mean` returns `0`. Note that
  `max`, `min`, `argmax`, and `argmin` return an identity here rather than
  raising the way numpy does.

- Fixed `max benchmark --base-url` failing before the first request against
  remote OpenAI-compatible endpoints: the server-readiness probe and the
  prefix-cache flush now target the `--base-url` endpoint (instead of
  `http://<host>:<port>`) and send `Authorization: Bearer $OPENAI_API_KEY`,
  matching the benchmark requests themselves.

- Fixed run-to-run nondeterminism of `layer_norm`, `rms_norm`, and other
  Row-API rowwise reductions on Apple Silicon GPUs: a block that reduced
  several rows re-used its shared-memory strip across row iterations
  without ordering the trailing broadcast read against the next combine's
  first store. Model outputs on Metal (for example FLUX.2 image
  generation) are now byte-identical across runs; NVIDIA and AMD codegen
  is unchanged.

- On Apple Silicon, a missing Metal Toolchain (a separate download since
  Xcode 16) now surfaces `xcrun`'s own error, which names the fix
  (`xcodebuild -downloadComponent MetalToolchain`), instead of the opaque
  "Please submit a bug report." message.

- Fixed GPU discovery inside a container granted only MIG compute instances,
  which made MAX and Mojo unusable on MIG-sliced clusters. Discovery reported
  `GPU is not present`, and a container holding several instances carved from
  the same GPU saw only one of them. Where NVML answers for the parent GPU,
  discovery now defers to CUDA, which describes the instance: for a device's
  memory when NVML rejects the query, and for the device count when MIG is
  enabled.
  ([Issue #6896](https://github.com/modular/modular/issues/6896))

- Fixed tool-call requests failing with HTTP 400 (`anyOf branch and base
  schema both set "description"`) on models whose grammar compiles in strict
  mode (GLM-5.x, Gemma 4). The xgrammar JSON-schema converter's `anyOf`
  base-merge now skips annotation-only keywords (`description`, `title`,
  `default`, `examples`, `$comment`, `deprecated`, `readOnly`, `writeOnly`)
  instead of rejecting them as branch/base conflicts; they carry no grammar
  constraint.

- Fixed a race that enforced structured-output grammars during a reasoning
  model's thinking span.

- Fixed structured output and constrained tool calling being silently ignored
  on the Kimi K2.5-family pipelines when serving with DFlash speculative
  decoding (`--speculative-method dflash`). The unified DFlash graph compiled
  without the constrained-decoding bitmask inputs. The graph now binds the
  bitmask inputs and applies the grammar mask across every speculative position,
  matching the EAGLE speculative-decoding pipelines.

- Fixed GPT-OSS, OLMo 3, and OLMo 2 ignoring `--max-length`. The server
  accepted prompts up to the checkpoint's own length limit, and sized the KV
  cache for that limit, whatever you asked for and whatever memory allowed.
  Both now use the length you set. Runs that pass no `--max-length` are
  unaffected.

- Fixed DeepSeek-V3.2 and GLM-5.x pipelines ignoring `--max-length`: the
  resolved maximum sequence length was silently pinned to the DeepSeek
  default (163840) regardless of the flag or the checkpoint's advertised
  limit. These models also now size their rotary-embedding tables from the
  resolved maximum sequence length instead of the checkpoint's
  `max_position_embeddings`.

- Fixed `ops.group_norm()` raising `NotImplementedError` in eager mode on
  CPU. `group_norm` previously had a GPU-only kernel; it now has a CPU
  compute path too, so eager `group_norm` runs on CPU the same way
  `layer_norm`/`rms_norm` already do.

- Fixed the BF16 Expert Parallelism (EP) dispatch path failing to compile.
  The `ep.dispatch_async` kernel requires a `dispatch_scale_dtype` comptime
  parameter, but the BF16 branch of `call_ep_dispatch_async` only set
  `dispatch_fmt_str` and omitted the scale dtype, so any model using BF16 EP
  dispatch (for example, a non-quantized MoE) hit a graph-compile error. The
  BF16 branch now sets `dispatch_scale_dtype = float32` to match the kernel
  signature.

- Fixed CPU `argmax`/`argmin` reductions returning a wrong index for reduce
  axes of 256K+ elements, for example an argmax over a `[1, 2097152]` tensor,
  where the row's reduction fans out across multiple CPU workers.

- Fixed the distribution the top-k/top-p sampler emits for speculative
  decoding (`emit_dist`) being under-normalized when a `min_p` mask removes
  weight and the row passes top-p at the first trial: the row was scaled by
  the unmasked softmax mass instead of the masked kept mass, so it summed to
  less than one and skewed the rejection residual. The sampled token stream
  was and remains unchanged.

- Fixed a model worker crash when constrained decoding and speculative
  decoding were enabled together. A batch at the prefill-to-decode boundary
  verifies no drafts, which the grammar bitmask fill rejected.

## Mojo language

For all the updates to the Mojo language, standard library, and tools,
see the [Mojo release notes](https://mojolang.org/releases/).
