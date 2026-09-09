# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Benchmark configuration classes with inheritance structure for MAX benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from cyclopts import Parameter
from max.config import ConfigFileModel
from pydantic import ConfigDict, Field, field_validator, model_validator

from .backend_names import Backend
from .datasets import DatasetMode, DistributionParameter
from .utils import int_or_none, parse_comma_separated

# Fixed default seed for the workload generator and request sampling. Scheduled
# and repeated benchmark runs pin this so run-to-run deltas reflect the change
# under test rather than workload-draw variance. Pass ``--seed none`` (or
# ``seed: null`` in YAML) to draw a fresh random seed instead.
DEFAULT_BENCHMARK_SEED = 0x5EED  # spells "SEED" (= 24301)

Endpoint = Literal[
    "/v1/completions",
    "/v1/chat/completions",
    "/v2/models/ensemble/generate_stream",
    "/v1/responses",
    "/v1/images/generations",
    "/v1/videos/sync",
    "/v1/videos",
]

CACHE_RESET_ENDPOINT_MAP: Mapping[Backend, str] = {
    "atom": "/reset_prefix_cache",
    "atom-chat": "/reset_prefix_cache",
    "mach": "/reset_prefix_cache",
    "modular": "/reset_prefix_cache",
    "modular-chat": "/reset_prefix_cache",
    "vllm": "/reset_prefix_cache",
    "vllm-chat": "/reset_prefix_cache",
    "sglang": "/flush_cache",
    "sglang-chat": "/flush_cache",
}

BenchmarkTask = Literal[
    "text-generation",
    "text-to-image",
    "image-to-image",
    "text-to-video",
    "image-to-video",
]

PIXEL_GENERATION_TASKS: tuple[BenchmarkTask, ...] = (
    "text-to-image",
    "image-to-image",
    "text-to-video",
    "image-to-video",
)

# Pixel-generation tasks that emit video. These route to the dedicated video
# endpoints on backends that have one (see VIDEO_GEN_DEFAULT_ENDPOINT).
VIDEO_GENERATION_TASKS: tuple[BenchmarkTask, ...] = (
    "text-to-video",
    "image-to-video",
)

# Default endpoint per backend for pixel generation tasks.
# Backends not listed here do not support pixel generation.
# NOTE: Each endpoint value here is coupled to a specific request driver in
# get_request_driver_class() in request.py. Adding a new backend that reuses
# an existing endpoint will route to that endpoint's existing driver.
PIXEL_GEN_DEFAULT_ENDPOINT: Mapping[Backend, Endpoint] = {
    "modular": "/v1/responses",
    "modular-chat": "/v1/responses",
    "mcloud": "/v1/responses",
    "sglang": "/v1/images/generations",
    "sglang-chat": "/v1/images/generations",
    "vllm": "/v1/chat/completions",
    "vllm-chat": "/v1/chat/completions",
}

# Override endpoint per backend for video tasks (text-to-video and
# image-to-video). For vllm-omni use the dedicated /v1/videos/sync endpoint.
VIDEO_GEN_DEFAULT_ENDPOINT: Mapping[Backend, Endpoint] = {
    "vllm": "/v1/videos/sync",
    "vllm-chat": "/v1/videos/sync",
    "sglang": "/v1/videos",
    "sglang-chat": "/v1/videos",
}

# Valid endpoints for pixel generation tasks (union of all backend defaults).
PIXEL_GENERATION_ENDPOINTS: frozenset[Endpoint] = frozenset(
    set(PIXEL_GEN_DEFAULT_ENDPOINT.values())
    | set(VIDEO_GEN_DEFAULT_ENDPOINT.values())
)


def get_pixel_gen_endpoint(backend: Backend, task: BenchmarkTask) -> Endpoint:
    """Return the pixel-generation endpoint for a given backend and task."""
    if task in VIDEO_GENERATION_TASKS and backend in VIDEO_GEN_DEFAULT_ENDPOINT:
        return VIDEO_GEN_DEFAULT_ENDPOINT[backend]
    if backend in PIXEL_GEN_DEFAULT_ENDPOINT:
        return PIXEL_GEN_DEFAULT_ENDPOINT[backend]
    raise ValueError(
        f"Backend {backend!r} does not have a default"
        " pixel-generation endpoint."
    )


class SamplingConfig(ConfigFileModel):
    """Configuration class for sampling options."""

    temperature: float | None = Field(default=None)
    """Sampling temperature. Default: None (use model / pipeline defaults)."""

    thinking_temperature: float | None = Field(default=None)
    """Sampling temperature override for tokens inside a ``<think>...</think>``
    block. MAX-only OpenAI extension; other backends will ignore or reject
    the field. Default: None (request omits the field)."""

    top_p: float | None = Field(default=None)
    """Nucleus sampling cumulative probability threshold. Default: None (use defaults)."""

    top_k: int | None = Field(default=None)
    """Limits the sampling to the K most probable tokens. Default: None (no sampling)."""


class BaseBenchmarkConfig(ConfigFileModel):
    """Base configuration class containing parameters common to all benchmark types.

    This class contains the core parameters that are shared across all benchmark types:
    - Model and tokenizer configuration
    - Basic dataset configuration
    - Common workload parameters
    - Basic output control
    - Result saving configuration
    - Common control flags
    """

    section_name: str | None = Field(default="benchmark_config", exclude=True)
    """Default section name for benchmark config files.

    Overrides ConfigFileModel's default to automatically extract the
    ``benchmark_config`` section when loading YAML config files."""

    # Model and tokenizer configuration (common to all benchmarks)
    model: str | None = Field(
        default=None,
        description="Name of the model. Required when running benchmark.",
    )

    tokenizer: str | None = Field(
        default=None,
        description="Name or path of the tokenizer, if not using the default tokenizer.",
    )

    tokenizer_local_files_only: bool = Field(
        default=False,
        description=(
            "Load the tokenizer from the local HF cache only. Set this for a"
            " checkpoint the Hub will not serve this caller, where the remote"
            " chat-template probe fails instead of degrading."
        ),
    )

    tokenizer_revision: str | None = Field(
        default=None,
        description=(
            "Commit revision to load the tokenizer at. Set this for a"
            " checkpoint whose revision cannot be looked up on the Hub (a"
            " private repo, or one staged into the cache with no branch ref),"
            " where the default lookup returns nothing and the load falls back"
            " to ``main``."
        ),
    )

    model_max_length: int | None = Field(
        default=None,
        description="Override for tokenizer max length. Needed if server has a lower max length than the tokenizer.",
    )

    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code from huggingface.",
    )

    # Dataset configuration (common across all benchmark types)
    dataset_name: str = Field(
        default="sharegpt",
        description="Name of the dataset to benchmark on.",
    )

    dataset_path: str | None = Field(
        default=None,
        description="Path to the dataset.",
    )

    dataset_mode: DatasetMode = Field(
        default="huggingface",
        description="Mode for loading the dataset: LOCAL (from local path/env var) or HUGGINGFACE (HuggingFace Hub).",
    )

    # Basic workload parameters
    num_prompts: int | None = Field(
        default=None,
        description="Number of prompts to process.",
    )

    seed: int | None = Field(
        default=DEFAULT_BENCHMARK_SEED,
        description=(
            "Random seed for the workload generator and request sampling. "
            "Defaults to a fixed value so repeated and scheduled runs are "
            "reproducible and run-to-run deltas reflect the change under test "
            "rather than workload-draw variance. Pass ``--seed none`` (or "
            "``seed: null`` in YAML) to draw a fresh random seed instead; the "
            "drawn seed is logged and recorded with the results so the run "
            "stays reproducible after the fact."
        ),
    )

    # Control flags
    disable_tqdm: bool = Field(
        default=False,
        description="Specify to disable tqdm progress bar.",
    )

    print_inputs_and_outputs: bool = Field(
        default=False,
        description="Print all input and outputs to console.",
    )

    verbose: bool = Field(
        default=False,
        description="Enable detailed DEBUG logging.",
    )

    @field_validator("seed", mode="before")
    @classmethod
    def _parse_seed_cli_string(cls, value: object) -> object:
        """Map the CLI/YAML string ``none`` to ``None`` (draw a random seed)."""
        if isinstance(value, str):
            return int_or_none(value)
        return value


class BaseServingBenchmarkConfig(BaseBenchmarkConfig):
    """Fields shared by every serving-style benchmark.

    Sits between :class:`BaseBenchmarkConfig` and the concrete
    :class:`ServingBenchmarkConfig` class. Holds fields whose type and default
    align across serving codepaths so downstream configs can opt into shared
    behavior without per-codepath overrides.
    """

    burstiness: float = Field(
        default=1.0,
        description="Burstiness factor (1.0 = Poisson process).",
        json_schema_extra={"group": "Traffic Control"},
    )

    skip_test_prompt: bool = Field(
        default=False,
        description="Skip the test prompt. Useful when doing external profiling.",
        json_schema_extra={"group": "Control Flags"},
    )

    collect_gpu_stats: bool = Field(
        default=True,
        description="Enable GPU stats collection (NVIDIA only).",
        json_schema_extra={"group": "Control Flags"},
    )

    lora_paths: list[str] = Field(
        default_factory=list,
        description="Paths to existing LoRA adapters. Format: 'path' or 'name=path'.",
        json_schema_extra={"group": "LoRA Configuration"},
    )

    lora_uniform_traffic_ratio: float = Field(
        default=0.0,
        description=(
            "Probability of selecting any LoRA uniformly at random (vs base model). "
            "Only used when per_lora_traffic_ratio is not specified. Range: 0.0-1.0."
        ),
        json_schema_extra={"group": "LoRA Configuration"},
    )

    per_lora_traffic_ratio: list[float] = Field(
        default_factory=list,
        description=(
            "Traffic percentages for each LoRA adapter in the benchmark. "
            "Must have same length as lora_paths. Sum must not exceed 1.0. "
            "Remainder goes to base model requests. "
            "If specified, overrides lora_uniform_traffic_ratio."
        ),
        json_schema_extra={"group": "LoRA Configuration"},
    )


class ServingBenchmarkConfig(BaseServingBenchmarkConfig):
    """Configuration class for serving benchmarks (benchmark_serving.py).

    Inherits shared serving fields (burstiness, LoRA traffic, GPU stats,
    skip_test_prompt) from :class:`BaseServingBenchmarkConfig` and adds
    serving-specific parameters:

    - Backend and API configuration
    - Request configuration (concurrency, sweeps)
    - Traffic control (request rate, TTFT)
    - Chat session configuration
    - Serving-specific dataset parameters
    - CPU and server stats collection
    """

    # TODO(MXTOOLS-166): The validate_assignment here is only because workload
    # YAMLs are not themselves parsed with Pydantic, and when we set fields via
    # _apply_workload_to_config, we rely on re-running the validators.
    # Workload YAMLs should probably be parsed with Pydantic too, and then
    # values need not be re-validated, and validate_assignment can be removed.
    model_config = ConfigDict(strict=False, validate_assignment=True)

    # Backend and API configuration (serving-specific)
    backend: Backend = Field(
        default="modular",
        description="Backend to use for benchmarking. Choices: atom, atom-chat, mach, mcloud, modular, modular-chat, sglang, sglang-chat, trtllm, trtllm-chat, vllm, vllm-chat",
        json_schema_extra={
            "group": "Backend and API Configuration",
            "group_description": "Configuration for backend selection and API endpoints",
        },
    )

    base_url: str | None = Field(
        default=None,
        description="Server or API base url if not using http host and port.",
        json_schema_extra={"group": "Backend and API Configuration"},
    )

    host: str = Field(
        default="localhost",
        description="Server host.",
        json_schema_extra={"group": "Backend and API Configuration"},
    )

    port: int = Field(
        default=8000,
        description="Server port.",
        json_schema_extra={"group": "Backend and API Configuration"},
    )

    endpoint: Endpoint = Field(
        default="/v1/chat/completions",
        description="API endpoint. Choices: /v1/completions, /v1/chat/completions, /v1/responses, /v1/images/generations, /v2/models/ensemble/generate_stream. For pixel generation tasks, auto-selected from backend if not specified.",
        json_schema_extra={"group": "Backend and API Configuration"},
    )

    benchmark_task: BenchmarkTask = Field(
        default="text-generation",
        description="Benchmark task type. Choices: text-generation, text-to-image, image-to-image, text-to-video, image-to-video",
        json_schema_extra={"group": "Backend and API Configuration"},
    )

    # Request configuration (serving-specific)
    max_concurrency: Sequence[int | None] = Field(
        default=[None],
        description=(
            "Maximum concurrent requests per sweep step. Parsed from a single "
            "value or comma-separated string (e.g. ``1,none,8``); ``none`` means "
            "unbounded."
        ),
        json_schema_extra={
            "group": "Request Configuration",
            "group_description": "Parameters controlling request concurrency and processing",
        },
    )

    lora: str | None = Field(
        default=None,
        description="Optional LoRA name.",
        json_schema_extra={"group": "Request Configuration"},
    )

    max_concurrent_conversations: int | None = Field(
        default=None,
        description=(
            "Maximum conversation workers active at once for KV-cache stress "
            "benchmarking. When set, runs run_kv_cache_stress_benchmark "
            "instead of run_multiturn_benchmark: each worker drives one chat "
            "session to completion before picking up the next, keeping all "
            "session KV caches resident simultaneously. "
            "--max-concurrency caps in-flight turn requests and must be <= "
            "--max-concurrent-conversations to stress the server's KV-cache: "
            "more open sessions than active turns grows the footprint and "
            "increases the likelihood of offloading or dropping pre-computed "
            "historical KV data."
        ),
        json_schema_extra={"group": "Request Configuration"},
    )

    kv_block_size: int = Field(
        default=128,
        description=(
            "KV cache block (page) size in tokens, used to block-align the "
            "per-turn cache retention metric. Should match the server's "
            "--kv-cache-page-size so retention reflects true block-aligned "
            "cache reuse."
        ),
        json_schema_extra={"group": "Request Configuration"},
    )

    # Workload configuration (serving-specific)
    max_benchmark_duration_s: int | None = Field(
        default=None,
        description="Maximum benchmark duration in seconds.",
        json_schema_extra={
            "group": "Workload Configuration",
            "group_description": "Parameters controlling benchmark duration and workload characteristics",
        },
    )

    num_chat_sessions: int | None = Field(
        default=None,
        description="Number of multiturn chat sessions.",
        json_schema_extra={"group": "Workload Configuration"},
    )

    delay_between_chat_turns: DistributionParameter | None = Field(
        default=None,
        description=(
            "Delay between chat turns in milliseconds. Accepts a float or int for a constant delay, "
            "or a distribution string: 'N(mean,std)' for normal, 'U(lower,upper)' for continuous uniform, "
            "'DU(lower,upper)' for discrete uniform, 'G(shape,scale)' for gamma, or 'LN(mean,std)' for log-normal."
        ),
        json_schema_extra={"group": "Workload Configuration"},
    )

    force_unique_runs: bool = Field(
        default=False,
        description=(
            "Prepend a single run-level UUID prefix to every prompt in this run. "
            "All requests in the same run share the same prefix, preserving "
            "within-run system-prompt prefix caching while preventing KV-cache "
            "reuse across benchmark runs."
        ),
        json_schema_extra={"group": "Workload Configuration"},
    )

    # Output control (serving-specific extensions)
    output_lengths: str | int | None = Field(
        default=None,
        description="Path to YAML file with output lengths or int.",
        json_schema_extra={
            "group": "Output Control",
            "group_description": "Parameters controlling output generation and sampling",
        },
    )

    max_output_len: int | None = Field(
        default=None,
        description="Maximum output length per request.",
        json_schema_extra={"group": "Output Control"},
    )

    temperature: float | None = Field(
        default=None,
        description="Temperature for sampling.",
        json_schema_extra={"group": "Output Control"},
    )

    thinking_temperature: float | None = Field(
        default=None,
        description=(
            "Temperature override for tokens inside a ``<think>...</think>`` "
            "block. MAX-only OpenAI extension; other backends will ignore or "
            "reject the field."
        ),
        json_schema_extra={"group": "Output Control"},
    )

    top_p: float | None = Field(
        default=None,
        description="Top-p for sampling.",
        json_schema_extra={"group": "Output Control"},
    )

    top_k: int | None = Field(
        default=None,
        description="Top-k for sampling.",
        json_schema_extra={"group": "Output Control"},
    )

    response_format: str | None = Field(
        default=None,
        description=(
            "JSON response format for structured output. Can be a JSON string "
            "or '@path/to/schema.json' to load from file. "
            'Example: \'{"type": "json_schema", "json_schema": {...}}\''
        ),
        json_schema_extra={"group": "Output Control"},
    )

    # cyclopts wants scalar key=value pairs by default and can't take a single
    # nested-JSON object or file path as one value. accepts_keys=False passes
    # the raw token through for the validator to parse.
    extra_body: Annotated[dict[str, Any], Parameter(accepts_keys=False)] = (
        Field(
            default_factory=dict,
            description=(
                "Extra top-level fields to add to every text-generation request "
                "body, for fields without a dedicated flag (e.g. stop, "
                "chat_template_kwargs). On the CLI, pass an inline JSON object "
                '(e.g. \'{"stop": ["}"], "chat_template_kwargs": '
                '{"reasoning_effort": "low"}}\') or a path to a YAML/JSON file; '
                "it can also be set as a field in a --config-file YAML. Nested "
                "objects and "
                "arrays are kept verbatim. Applied after the dedicated flags "
                "(--temperature, --max-output-len, --response-format, etc.), so a "
                "key that collides with one overrides it (last-writer-wins, and "
                "the collision is logged). Currently applied only to "
                "text-generation requests (the chat/completions, completions, "
                "and TensorRT-LLM generate_stream endpoints); ignored for "
                "image/video generation tasks."
            ),
            json_schema_extra={"group": "Output Control"},
        )
    )

    # Image generation options (serving-specific)
    image_width: int | None = Field(
        default=None,
        description="Output image width in pixels for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    image_height: int | None = Field(
        default=None,
        description="Output image height in pixels for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    image_steps: int | None = Field(
        default=None,
        description="Number of denoising steps for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    image_guidance_scale: float | None = Field(
        default=None,
        description="Guidance scale for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    image_negative_prompt: str | None = Field(
        default=None,
        description="Negative prompt for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    image_seed: int | None = Field(
        default=None,
        description="Deterministic seed for pixel generation.",
        json_schema_extra={"group": "Output Control"},
    )

    num_frames: int | None = Field(
        default=None,
        description="Number of frames to generate. Required for text-to-video.",
        json_schema_extra={"group": "Output Control"},
    )

    # Traffic control (serving-specific)
    request_rate: Sequence[float] = Field(
        default=[float("inf")],
        description=(
            "Requests per second per sweep step. Parsed from a single value or "
            "comma-separated string (use ``inf`` for unlimited)."
        ),
        json_schema_extra={
            "group": "Traffic Control",
            "group_description": "Parameters controlling request rate and traffic patterns",
        },
    )

    skip_first_n_requests: int | None = Field(
        default=None,
        description="Skip first N requests for measurements. Omit to auto-set to max_concurrency; pass 0 to disable.",
        json_schema_extra={"group": "Traffic Control"},
    )

    skip_last_n_requests: int | None = Field(
        default=None,
        description="Skip last N requests for measurements. Omit to auto-set to max_concurrency; pass 0 to disable.",
        json_schema_extra={"group": "Traffic Control"},
    )

    chat_warmup_delay_ms: float = Field(
        default=0.0,
        description="Delay between starting chat sessions.",
        json_schema_extra={"group": "Traffic Control"},
    )

    ignore_first_turn_stats: bool = Field(
        default=False,
        description="Ignore the first turn statistics in multiturn chat sessions.",
        json_schema_extra={"group": "Traffic Control"},
    )

    use_session_id_as_cache_salt: bool = Field(
        default=False,
        description="Send each multi-turn chat session's id as the X-Cache-Salt header on every request in that session, so MAX Serve's per-session KV-cache isolation can be measured. Same salt across a session's turns; distinct salt per session. No-op against servers that ignore the header.",
        json_schema_extra={"group": "Traffic Control"},
    )

    warmup_to_steady_state: bool = Field(
        default=True,
        description="Attempt to start the benchmark in steady state by starting with a later turn distribution. Disable to start every session at turn 0.",
        json_schema_extra={"group": "Traffic Control"},
    )

    warmup_oversample_factor: int = Field(
        default=8,
        description="Warmup-candidate pool multiplier (sessions per warmup slot). 0 disables length-biased warmup; 1 = uniform random turn warmup; >=2 = length-biased warmup.",
        json_schema_extra={"group": "Traffic Control"},
    )

    warmup_delay_biased: bool = Field(
        default=False,
        description="Bias warmup session/turn selection by inter-turn think time (sum of delays) instead of turn count. Requires delays to be configured; falls back to turn-based when none are present.",
        json_schema_extra={"group": "Traffic Control"},
    )

    warmup_delay_estimated_ttft_ms: float = Field(
        default=0.0,
        description="Estimated time-to-first-token in milliseconds per turn. When set (with --warmup-delay-biased), warmup weights each turn by its occupancy (estimated generation time + inter-turn delay) instead of delay alone, better matching steady state for high-TPOT / long-output workloads. 0 = off.",
        json_schema_extra={"group": "Traffic Control"},
    )

    warmup_delay_estimated_tpot_ms: float = Field(
        default=0.0,
        description="Estimated time-per-output-token in milliseconds, used with --warmup-delay-estimated-ttft-ms to weight warmup by per-turn generation time (ttft + tpot * output_len). Only valid with --warmup-delay-biased. 0 = off.",
        json_schema_extra={"group": "Traffic Control"},
    )

    warmup_concurrency: int = Field(
        default=128,
        description="Maximum in-flight requests during prefix-cache priming (--warm-shared-prefix) and steady-state warmup (--warmup-to-steady-state).",
        json_schema_extra={"group": "Traffic Control"},
    )

    # Dataset-specific parameters (serving workloads)
    arxiv_summarization_input_len: int = Field(
        default=15000,
        description="Number of input tokens per request, used only for arxiv-summarization dataset.",
        json_schema_extra={
            "group": "Dataset-Specific Parameters",
            "group_description": "Parameters specific to different dataset types and workloads",
        },
    )
    batch_job_image_dir: str | None = Field(
        default=None,
        description="Directory where server can access images for batch-job dataset (file reference mode). If not specified, uses embedded base64 mode.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    obfuscated_conversations_average_output_len: int = Field(
        default=175,
        description="Average output length for obfuscated-conversations dataset when output_lengths is not provided.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    obfuscated_conversations_coefficient_of_variation: float = Field(
        default=0.1,
        description="Coefficient of variation for output length for obfuscated-conversations dataset when output_lengths is not provided.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    obfuscated_conversations_shuffle: bool = Field(
        default=False,
        description="Shuffle the obfuscated-conversations dataset.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    tool_calls: bool = Field(
        default=True,
        description="Include turns with tool calls for datasets that support it. When disabled, only system+user turns are used.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_image_count: int = Field(
        default=0,
        description="Number of random images to generate.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_image_size: str = Field(
        default="",
        description="Size of random images to generate.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_input_len: DistributionParameter = Field(
        default=1024,
        description="Number of input tokens per request, used by the random and artificial-analysis datasets. Use ';' to separate first-turn and remaining-turn distributions for multiturn.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_max_num_unique_sys_prompt: int = Field(
        default=1,
        description="Maximum number of unique system prompts, used only for random sampling.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_num_turns: DistributionParameter = Field(
        default=1,
        description="Number of turns per session, used only for random sampling and --num-chat-sessions.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_output_len: DistributionParameter = Field(
        default=128,
        description="Number of output tokens per request, used by the random and artificial-analysis datasets. Use ';' to separate first-turn and remaining-turn distributions for multiturn.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    random_sys_prompt_ratio: float = Field(
        default=0.0,
        description="Ratio to determine the system prompt length, used only for random sampling.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    fit_distributions: bool = Field(
        default=False,
        description=(
            "With --num-chat-sessions on instruct-coder or agentic-code, reshape "
            "workloads to match random_* and delay_between_chat_turns (same "
            "semantics as random multiturn). Unsupported for code-debug multiturn. "
            "Random/synthetic datasets already follow these distributions."
        ),
        json_schema_extra={"group": "Workload Configuration"},
    )
    sonnet_input_len: int = Field(
        default=550,
        description="Number of input tokens per request, used only for sonnet dataset.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )
    sonnet_prefix_len: int = Field(
        default=200,
        description="Number of prefix tokens per request, used only for sonnet dataset.",
        json_schema_extra={"group": "Dataset-Specific Parameters"},
    )

    # Control flags (serving-specific)
    warm_shared_prefix: bool = Field(
        default=False,
        description=(
            "Send each unique shared prefix once (max_tokens=1) before the"
            " benchmark run to prime prefix-cache KV entries. Supported for"
            " random/synthetic datasets, or instruct-coder/agentic-code with"
            " --fit-distributions; in all cases requires"
            " --random-sys-prompt-ratio > 0."
        ),
        json_schema_extra={
            "group": "Control Flags",
            "group_description": "Boolean flags controlling benchmark behavior",
        },
    )
    collect_cpu_stats: bool = Field(
        default=True,
        description="Enable CPU stats collection for serving benchmarks.",
        json_schema_extra={"group": "Control Flags"},
    )

    server_pids: list[int] = Field(
        default_factory=list,
        description=(
            "Explicit PIDs to monitor for CPU stats, including their children."
            " Bypasses automatic port-based PID detection. Useful when the"
            " server runs inside a container."
            " Example: --server-pids $(docker inspect -f '{{.State.Pid}}'"
            " <container_name>)"
        ),
        json_schema_extra={"group": "Control Flags"},
    )

    collect_server_stats: bool = Field(
        default=True,
        description="Enable server stats collection for serving benchmarks.",
        json_schema_extra={"group": "Control Flags"},
    )

    # `dict[str, str]` (not `Mapping`) so cyclopts 3.24 accepts the
    # `--metrics-urls.<label>=<url>` syntax for nested keys.
    metrics_urls: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Explicit Prometheus metrics endpoint URLs, keyed by label "
            "(e.g. '--metrics-urls.orchestrator=http://host:8001/metrics "
            "--metrics-urls.engine-0=http://host2:8001/metrics'). "
            "When empty, a single endpoint is auto-derived from --host/--port."
        ),
        json_schema_extra={"group": "Control Flags"},
    )

    print_workload_stats: bool = Field(
        default=False,
        description="Print workload distribution statistics (input/output lengths, num turns, delays).",
        json_schema_extra={"group": "Control Flags"},
    )

    dry_run: bool = Field(
        default=False,
        description="Build the dataset and print workload stats + warmup-sampling preview, then exit without contacting the server.",
        json_schema_extra={"group": "Control Flags"},
    )

    trace: bool = Field(
        default=False,
        description="Enable nsys tracing. Requires server run under 'nsys launch'. Using '--gpu-profiling detailed' is recommended. Currently only supported on NVIDIA GPUs.",
        json_schema_extra={"group": "Control Flags"},
    )

    trace_file: str | None = Field(
        default=None,
        description="Path to save nsys trace. Default: $MODULAR_PATH/profile.nsys-rep or ./profile.nsys-rep.",
        json_schema_extra={"group": "Control Flags"},
    )

    trace_session: str | None = Field(
        default=None,
        description="Optional session name to trace. If not specified, nsys traces the default session.",
        json_schema_extra={"group": "Control Flags"},
    )

    # Result saving (serving-specific extensions)
    record_output_lengths: str | None = Field(
        default=None,
        description="Path to save output lengths in YAML format.",
        json_schema_extra={"group": "Result Saving"},
    )

    record_request_text: bool = Field(
        default=False,
        description=(
            "Include each request's generated text in the per-request"
            " records. Needed to compare two runs' outputs; off by default"
            " because it grows the result file by the size of the run's own"
            " output."
        ),
        json_schema_extra={"group": "Result Saving"},
    )

    result_filename: str | None = Field(
        default=None,
        description="JSON filename for results. If None, no results are saved. Can include directory path.",
        json_schema_extra={"group": "Result Saving"},
    )

    always_save_result: bool = Field(
        default=False,
        description="Save result files even when benchmark validation fails (e.g. some requests errored). The process still exits with code 1 on failure.",
        json_schema_extra={"group": "Result Saving"},
    )

    metadata: list[str] = Field(
        default_factory=list,
        description="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) for metadata of this run to be saved in the result JSON file for record keeping purposes.",
        json_schema_extra={"group": "Result Saving"},
    )

    server_ready_timeout_s: int = Field(
        default=0,
        description="Maximum seconds to wait for the server to become ready (HTTP-poll) after sample generation finishes.",
    )

    log_dir: str | None = Field(
        default=None,
        description="Path to save logs. Default: <backend>-latency-Y.m.d-H.M.S",
        json_schema_extra={"group": "Result Saving"},
    )

    latency_percentiles: str = Field(
        default="50,90,95,99",
        description="Comma separated list of latency percentiles to include in CSV output. Only P50, P90, P95, and P99 are supported.",
        json_schema_extra={"group": "Result Saving"},
    )

    # Workload config file
    workload_config: str | None = Field(
        default=None,
        description="YAML file specifying the workload to benchmark.",
        json_schema_extra={"group": "Workload Configuration"},
    )

    # Result upload configuration
    upload_results: bool = Field(
        default=False,
        description="Upload results to BigQuery.",
        json_schema_extra={"group": "Result Upload Configuration"},
    )

    benchmark_sha: str | None = Field(
        default=None,
        description="Commit hash of the docker image used for load generation.",
        json_schema_extra={"group": "Result Upload Configuration"},
    )

    cluster_information_path: str | None = Field(
        default=None,
        description="Path to the cluster information file. Usually a json file with metadata about the cluster setup if you're benchmarking more than a single node.",
        json_schema_extra={"group": "Result Upload Configuration"},
    )

    benchmark_config_name: str | None = Field(
        default=None,
        description="(For serving benchmarks) config name for tracking.",
        json_schema_extra={"group": "Result Upload Configuration"},
    )

    # Sweep configuration
    num_iters: int = Field(
        default=1,
        description="Number of iterations to run per configuration.",
        json_schema_extra={"group": "Sweep Configuration"},
    )

    flush_prefix_cache: bool = Field(
        default=True,
        description="Flush the prefix cache between iterations.",
        json_schema_extra={"group": "Sweep Configuration"},
    )

    num_prompts_multiplier: int | None = Field(
        default=None,
        description=(
            "When set, num_prompts is computed as num_prompts_multiplier * max_concurrency "
            "for each concurrency level, replacing the default 300s duration timeout."
        ),
        json_schema_extra={"group": "Sweep Configuration"},
    )

    max_concurrent_lora_ops: int = Field(
        default=1,
        description="Maximum concurrent LoRA loading/unloading operations.",
        json_schema_extra={"group": "LoRA Configuration"},
    )

    @field_validator("max_concurrency", mode="before")
    @classmethod
    def _parse_max_concurrency_cli_strings(cls, value: object) -> object:
        """Expand comma-separated CLI/env strings (and cyclopts ``['sweep']``)."""
        if isinstance(value, int):
            return [value]
        if (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], str)
        ):
            value = value[0]
        if isinstance(value, str):
            return parse_comma_separated(value, int_or_none)
        return value

    @field_validator("request_rate", mode="before")
    @classmethod
    def _parse_request_rate_cli_strings(cls, value: object) -> object:
        """Expand comma-separated CLI/env strings (and cyclopts ``['sweep']``)."""
        if isinstance(value, (int, float)):
            return [value]
        if (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], str)
        ):
            value = value[0]
        if isinstance(value, str):
            return parse_comma_separated(value, float)
        return value

    @field_validator("extra_body", mode="before")
    @classmethod
    def _parse_extra_body(cls, value: object) -> dict[str, Any]:
        """Parse ``extra_body`` into a dict from one of three inputs:

        - a mapping (e.g. from a ``--config-file`` YAML), used directly;
        - an inline JSON object string (leading ``{``), parsed with ``yaml.safe_load``;
        - any other non-empty string, treated as a YAML/JSON file path.

        Raises:
            ValueError: If the string is neither an inline object nor a readable
                file, the content fails to parse, or it does not resolve to an
                object.
        """
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            if text.startswith("{"):
                source, origin = text, "inline JSON"
            else:
                path = Path(text)
                if not path.is_file():
                    raise ValueError(
                        f"extra_body {text!r} is not a readable file path; "
                        "an inline value must be a JSON object starting "
                        "with '{'."
                    )
                try:
                    source = path.read_text()
                except OSError as e:
                    raise ValueError(
                        f"extra_body {text!r} is not a readable file path: {e}"
                    ) from e
                origin = f"file {text!r}"
            try:
                parsed = yaml.safe_load(source)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Failed to parse extra_body ({origin}): {e}"
                ) from e
            if not isinstance(parsed, Mapping):
                raise ValueError(
                    f"extra_body ({origin}) must be a mapping/object, got "
                    f"{type(parsed).__name__}."
                )
            return dict(parsed)
        raise ValueError(
            "extra_body must be a mapping, an inline JSON string, or a path "
            f"to a YAML/JSON file; got {type(value).__name__}."
        )

    @model_validator(mode="after")
    def _check_warmup_runtime_estimates(self) -> ServingBenchmarkConfig:
        """Runtime estimates only refine delay-biased warmup."""
        if (
            self.warmup_delay_estimated_ttft_ms > 0.0
            or self.warmup_delay_estimated_tpot_ms > 0.0
        ) and not self.warmup_delay_biased:
            raise ValueError(
                "--warmup-delay-estimated-ttft-ms /"
                " --warmup-delay-estimated-tpot-ms require"
                " --warmup-delay-biased to be set."
            )
        return self

    @property
    def sampling(self) -> SamplingConfig:
        """OpenAI-style completion sampling from flat ``temperature`` / ``top_p`` / ``top_k`` / ``thinking_temperature``."""
        # TODO: We should just embed SamplingConfig directly.  This may change
        # the CLI interface, so we'd need to find all callers to update them.
        return SamplingConfig(
            temperature=self.temperature,
            thinking_temperature=self.thinking_temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
