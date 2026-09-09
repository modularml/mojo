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

"""
Placeholder file for any configs (runtime, models, pipelines, etc)
"""

from __future__ import annotations

import functools
import logging
import os
from enum import Enum
from pathlib import Path

from max.support.human_readable_formatter import to_human_readable_bytes
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("max.serve")


class APIType(Enum):
    KSERVE = "kserve"
    OPENAI = "openai"
    SAGEMAKER = "sagemaker"
    OPENRESPONSES = "responses"


class RunnerType(Enum):
    PYTORCH = "pytorch"
    TOKEN_GEN = "token_gen"


class MetricRecordingMethod(Enum):
    """How should metrics be recorded?"""

    # Do not record metrics
    NOOP = "NOOP"
    # Synchronously record metrics
    SYNC = "SYNC"
    # Record metrics asynchronously using asyncio
    ASYNCIO = "ASYNCIO"
    # Send metric observations to a separate process for recording
    PROCESS = "PROCESS"


@functools.total_ordering
class KernelTraceLevel(Enum):
    """Controls GPU kernel-trace capture depth.

    Members are declared in increasing capture depth and compare in that
    order, so gates can be written as e.g. ``level >= BATCH``. Each level
    includes everything at the levels below it. All levels above ``off`` add
    overhead to the model worker process. Use the minimum level that
    satisfies your observability needs.
    """

    OFF = "off"
    """No ``max.batch`` spans and no libkineto capture (default). Request and
    phase spans are governed by the tracing exporter config, not this flag."""

    BATCH = "batch"
    """Emit a ``max.batch`` OTel span per forward pass (requires tracing to
    be configured, see ``disable_telemetry``). No per-kernel GPU detail.
    Minimal overhead."""

    OP = "op"
    """Op-level NVTX annotation. Enables Nsight / libkineto user-annotation
    ranges around each model op. Moderate overhead."""

    KERNEL = "kernel"
    """Full GPU kernel timeline via libkineto. Records every CUDA kernel
    launch and NVTX range. Highest overhead; use for deep profiling only."""

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, KernelTraceLevel):
            return NotImplemented
        members = list(KernelTraceLevel)
        return members.index(self) < members.index(other)


class Settings(BaseSettings):
    # env files, direct initialization, and aliases interact in some confusing
    # ways.  this is the way:
    #   1. extra="allow"
    #      This allows .env files to include entries for non-modular use cases.  eg HF_TOKEN
    #   2. populate_by_name=True
    #      Allow both field names and aliases to be used for initialization, but aliases are preferred for clarity.
    #   3. initialize with alias names `Settings(MAX_SERVE_HOST="host")`
    #
    # Known sharp edges:
    #   1. .env files can use both the Settings attr name (eg host) as well as the alias MAX_SERVE_HOST.
    #   2. Environment variables can only use the alias (MAX_SERVE_...)
    #   3. Both name and alias can be used for direct initialization, but alias is preferred. Both Settings(MAX_SERVE_HOST=...) and Settings(host=...) work.

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MAX_SERVE_",
        extra="allow",
        populate_by_name=True,
    )

    # Server configuration
    api_types: list[APIType] = Field(
        description="List of exposed API types.",
        default=[APIType.OPENAI, APIType.SAGEMAKER],
        alias="MAX_SERVE_API_TYPES",
    )
    offline_inference: bool = Field(
        description="If True, the server is run in offline inference mode. While it will still spin up workers, it will not spin up the API endpoint or use an HTTP port.",
        default=False,
        alias="MAX_SERVE_OFFLINE_INFERENCE",
    )
    headless: bool = Field(
        default=False,
        description="If True, runs a model worker and dispatch worker without starting an API server.",
        alias="MAX_SERVE_HEADLESS",
    )
    host: str = Field(
        description="Hostname to use", default="0.0.0.0", alias="MAX_SERVE_HOST"
    )
    port: int = Field(
        description="Port to use", default=8000, alias="MAX_SERVE_PORT"
    )

    metrics_port: int = Field(
        description="Port to use for the metrics endpoint",
        default=8001,
        alias="MAX_SERVE_METRICS_ENDPOINT_PORT",
    )

    http_keepalive_timeout_s: int = Field(
        description=(
            "Seconds an idle HTTP connection is held open before the server "
            "closes it. Keep this above the idle-connection timeout of every "
            "client that pools connections to MAX Serve. Whichever side closes "
            "first wins the race, and a server-side close landing just as a "
            "pooled client writes its next request reaches that client as a "
            "TCP reset instead of a response -- the client cannot replay a "
            "POST body, so it surfaces as a user-visible error rather than a "
            "retry. Go's default client-side idle timeout is 90 seconds."
        ),
        default=120,
        alias="MAX_SERVE_HTTP_KEEPALIVE_TIMEOUT_S",
    )

    max_queue_size: int | None = Field(
        description=(
            "Cap (N) on the request queue to the model worker. The queue to "
            "the worker is bounded to roughly this many in-transit requests; "
            "once full, new requests are rejected immediately with HTTP 429 "
            "instead of being enqueued, giving a self-calibrating backpressure "
            "mechanism to keep latency within SLAs. Enforced approximately via "
            "the ZeroMQ high-water mark. Pair with 'max_pending_requests' so "
            "the worker stops draining the queue under load and it actually "
            "backs up. Defaults to None (unbounded)."
        ),
        default=None,
        ge=0,
        alias="MAX_SERVE_MAX_QUEUE_SIZE",
    )
    max_pending_requests: int | None = Field(
        description=(
            "Cap (M) on the scheduler's pending (context-encoding / prefill) "
            "queue depth. When set, the model worker stops pulling new requests "
            "from the request queue once it already holds this many "
            "not-yet-running requests, so excess backlog stays in the bounded "
            "request queue and exerts backpressure rather than growing the "
            "worker's unbounded pending pool. Long requests that hold batch/KV "
            "space keep this queue full and shed new admissions sooner. Should "
            "be at least 'max_batch_size' to keep the scheduler fed. Defaults "
            "to None (unbounded)."
        ),
        default=None,
        ge=1,
        alias="MAX_SERVE_MAX_PENDING_REQUESTS",
    )

    max_request_bytes: int = Field(
        description=(
            "Maximum size in bytes of an accepted HTTP request body. Requests "
            "whose body exceeds this are rejected with HTTP 413 before the body "
            "is buffered, bounding per-request memory so a client cannot "
            "exhaust host memory with an oversized payload. The default (100 "
            "MiB) leaves ample headroom for multimodal requests that inline "
            "base64 media; raise it for larger inline payloads, or set 0 to "
            "disable the limit."
        ),
        default=100 * 1024 * 1024,  # 100 MiB
        ge=0,
        alias="MAX_SERVE_MAX_REQUEST_BYTES",
    )

    # File URI configuration
    allowed_image_roots: list[str] = Field(
        description="List of allowed root directories for file:// URI access",
        default_factory=list,
        alias="MAX_SERVE_ALLOWED_IMAGE_ROOTS",
    )
    max_local_image_bytes: int = Field(
        description="Maximum size in bytes for local image files accessed via file:// URIs",
        default=20 * 1024 * 1024,  # 20MiB
        alias="MAX_SERVE_MAX_LOCAL_IMAGE_BYTES",
    )
    # Media (image/video) resolution configuration for http(s):// and data:
    # URIs. ``max_bytes`` is a server-level cap applied on top of any
    # per-model cap (the smaller of the two wins); 0 disables the
    # server-level cap. ``media_kind`` is only the default label used in
    # size-limit error messages when a resolver caller does not pass one.
    max_bytes: int = Field(
        description=(
            "Server-level maximum size in bytes for media resolved from "
            "http(s):// or data: URIs. Applied on top of any per-model cap "
            "(the smaller wins). 0 disables the server-level cap."
        ),
        default=0,
        alias="MAX_SERVE_MAX_BYTES",
    )
    media_kind: str = Field(
        description=(
            "Default media kind ('image' or 'video') used in size-limit error "
            "messages when a resolver caller does not specify one."
        ),
        default="image",
        alias="MAX_SERVE_MEDIA_KIND",
    )
    media_url_ssrf_protection_enabled: bool = Field(
        description=(
            "Guard client-supplied http(s):// media URLs against SSRF. On by"
            " default. Break-glass switch: disable only to restore the legacy"
            " unvalidated fetch, and prefer MAX_SERVE_MEDIA_URL_ALLOWED_HOSTS to"
            " permit specific internal hosts instead."
        ),
        default=True,
        alias="MAX_SERVE_MEDIA_URL_SSRF_PROTECTION_ENABLED",
    )
    media_url_allowed_hosts: list[str] = Field(
        description=(
            "Allowlist permitting otherwise-blocked internal hosts to be fetched"
            " while SSRF protection stays on. Each entry is an exact hostname"
            " (case-insensitive) or an IP/CIDR (e.g. '10.0.0.0/8',"
            " '192.168.1.10')."
        ),
        default_factory=list,
        alias="MAX_SERVE_MEDIA_URL_ALLOWED_HOSTS",
    )
    generated_media_storage_mb: int = Field(
        description="Maximum amount of local disk space in MiB to use for generated image/video artifacts served via /content routes.",
        default=512,
        alias="MAX_SERVE_GENERATED_MEDIA_STORAGE_MB",
    )

    use_client_cache_salt: bool = Field(
        description="If True, honor cache_salt from clients (header or body). Off by default",
        default=False,
        alias="MAX_SERVE_USE_CLIENT_CACHE_SALT",
    )

    # Telemetry and logging configuration
    logs_console_level: str | None = Field(
        default="INFO",
        description="Logging level",
        alias="MAX_SERVE_LOGS_CONSOLE_LEVEL",
    )
    logs_otlp_level: str | None = Field(
        default=None,
        description="OTLP log level",
        alias="MAX_SERVE_LOGS_OTLP_LEVEL",
    )
    logs_file_level: str | None = Field(
        default=None,
        description="File log level",
        alias="MAX_SERVE_LOGS_FILE_LEVEL",
    )
    logs_file_path: str | None = Field(
        default=None,
        description="Logs file path",
        alias="MAX_SERVE_LOGS_FILE_PATH",
    )
    structured_logging: bool = Field(
        default=False,
        description="Structured logging for deployed services",
        alias="MODULAR_STRUCTURED_LOGGING",
    )
    logs_enable_components: str | None = Field(
        default=None,
        description="Comma separated list of additional components to enable for logging",
        alias="MAX_SERVE_LOGS_ENABLE_COMPONENTS",
    )

    disable_telemetry: bool = Field(
        default=False,
        description="Disable remote telemetry",
        alias="MAX_SERVE_DISABLE_TELEMETRY",
    )

    otlp_metrics_endpoint: str | None = Field(
        default=None,
        description=(
            "Optional OTLP endpoint (e.g. a Datadog Agent OTLP receiver or "
            "any OTel collector) to push histogram metrics to. When set, "
            "histogram instruments switch from hand-tuned explicit bucket "
            "boundaries to a self-calibrating exponential-histogram "
            "aggregation, exported to this endpoint with delta "
            "temporality. The local Prometheus endpoint continues serving "
            "counters and gauges as before, but histograms stop appearing "
            "there (the classic Prometheus text format cannot carry "
            "exponential histograms). Leave unset to keep today's "
            "explicit-bucket histograms on the local Prometheus endpoint, "
            "unchanged."
        ),
        alias="MAX_SERVE_OTLP_METRICS_ENDPOINT",
    )

    kernel_trace_level: KernelTraceLevel = Field(
        default=KernelTraceLevel.OFF,
        description=(
            "GPU kernel-trace capture depth. 'off' (default) adds zero "
            "overhead. 'batch' enables per-forward-pass max.batch OTel "
            "spans (when tracing is enabled) with no GPU capture. 'op' "
            "adds NVTX op-level ranges. 'kernel' enables full libkineto "
            "GPU kernel timeline capture (highest overhead)."
        ),
        alias="MAX_SERVE_KERNEL_TRACE_LEVEL",
    )

    # Model worker configuration
    use_heartbeat: bool = Field(
        default=False,
        description="When True, uses a periodic heart beat to confirm model worker liveness. This can result in false negatives if a single batch takes longer than the heartbeat interval to process (as may be the case for large context prefill)",
        alias="MAX_SERVE_USE_HEARTBEAT",
    )
    mw_timeout_s: float | None = Field(
        default=None,
        description="Amount of time in seconds to wait for the model worker to warm up and become ready to serve",
        alias="MAX_SERVE_MW_TIMEOUT",
    )
    mw_health_fail_s: float = Field(
        # TODO: we temporarily set it to 1 minute to handle long context input
        default=60.0,
        description="Maximum time to wait for a heartbeat & remain healthy.  This should be longer than ITL",
        alias="MAX_SERVE_MW_HEALTH_FAIL",
    )
    eplb_profile: bool = Field(
        default=False,
        description=(
            "When True, enables expert-parallel load balancing (EPLB) MoE routing "
            "histogram profiling in the model worker. The accumulator "
            "is opt-in and unused unless this flag is set."
        ),
        alias="MAX_SERVE_EPLB_PROFILE",
    )

    gc_debug: bool = Field(
        default=False,
        description=(
            "When True, attaches a CPython garbage-collection callback in the "
            "model worker that times every GC pass and logs metrics."
        ),
        alias="MAX_SERVE_GC_DEBUG",
    )
    gc_debug_top_objects: int = Field(
        default=0,
        description=(
            "When gc_debug is enabled and this is greater than zero, log the N "
            "most common live object types in the collected generation on each "
            "GC pause. Walks the heap and is expensive; leave at 0 unless "
            "actively investigating what is filling the heap."
        ),
        alias="MAX_SERVE_GC_DEBUG_TOP_OBJECTS",
    )

    telemetry_worker_spawn_timeout: float | None = Field(
        default=None,
        description="Amount of time in seconds to wait for the telemetry worker to spawn and turn healthy",
        alias="MAX_SERVE_TELEMETRY_WORKER_SPAWN_TIMEOUT",
    )

    graceful_shutdown_timeout_s: int = Field(
        default=5,
        description=(
            "Seconds to wait for in-flight requests to finish after SIGTERM "
            "before cancelling them and exiting."
        ),
        alias="MAX_SERVE_GRACEFUL_SHUTDOWN_TIMEOUT_S",
    )

    metric_recording: MetricRecordingMethod = Field(
        default=MetricRecordingMethod.PROCESS,
        description="How metrics should be recorded?",
        alias="MAX_SERVE_METRIC_RECORDING_METHOD",
    )

    stream_min_chunk_tokens: int = Field(
        default=1,
        ge=1,
        description=(
            "Minimum number of generated tokens to coalesce into a single "
            "streaming (SSE) chunk. 1 (default) emits each scheduler response "
            "as its own chunk with no buffering. Higher values buffer decode "
            "tokens into larger chunks and suppress empty deltas; the first "
            "chunk is always flushed early so time-to-first-token is "
            "unaffected."
        ),
        alias="MAX_SERVE_STREAM_MIN_CHUNK_TOKENS",
    )

    transaction_recording_file: Path | None = Field(
        default=None,
        description="File to record all HTTP transactions to",
        alias="MAX_SERVE_TRANSACTION_RECORDING_FILE",
    )

    @field_validator("transaction_recording_file", mode="after")
    def validate_transaction_recording_file(
        cls, path: Path | None
    ) -> Path | None:
        if path is None:
            return None
        if not path.name.endswith(".rec.jsonl"):
            raise ValueError(
                "Transaction recording files must have a '.rec.jsonl' file extension."
            )
        return path

    transaction_recording_include_responses: bool = Field(
        default=False,
        description="When recording HTTP transactions, whether to include responses",
        alias="MAX_SERVE_TRANSACTION_RECORDING_INCLUDE_RESPONSES",
    )

    di_bind_address: str = Field(
        default="tcp://127.0.0.1:5555",
        description=(
            "Bind address for the disaggregated inference dispatcher. "
            "This address is used for communication between the decode and prefill workers."
        ),
        alias="MAX_SERVE_DI_BIND_ADDRESS",
    )

    @field_validator("di_bind_address", mode="before")
    def validate_di_bind_address(cls, value: str) -> str:
        """Validate that deprecated MAX_SERVE_DISPATCHER_CONFIG is not being used.

        This validator checks if the deprecated environment variable
        MAX_SERVE_DISPATCHER_CONFIG is set and fails loudly if it is,
        directing users to use MAX_SERVE_DI_BIND_ADDRESS instead.
        """
        deprecated_var = "MAX_SERVE_DISPATCHER_CONFIG"
        if deprecated_var in os.environ:
            raise ValueError(
                f"The environment variable '{deprecated_var}' is deprecated and no longer supported. "
                f"Please use 'MAX_SERVE_DI_BIND_ADDRESS' instead. "
                f"For more information, see: https://linear.app/modularml/issue/CLIN-608"
            )
        return value

    log_prefix: str | None = Field(
        default=None,
        description="Prefix to prepend to all log messages for this service instance.",
        alias="MAX_SERVE_LOG_PREFIX",
    )

    def log_server_info(self) -> None:
        """Log comprehensive server configuration information.

        Displays all server settings in a consistent visual format similar to
        pipeline configuration logging.
        """
        # Build API types string
        api_types_str = ", ".join(api_type.value for api_type in self.api_types)

        # Build operation mode string
        mode_flags = []
        if self.offline_inference:
            mode_flags.append("offline_inference")
        if self.headless:
            mode_flags.append("headless")
        mode_str = ", ".join(mode_flags) if mode_flags else "standard"

        # Build allowed roots string
        allowed_roots_str = (
            ", ".join(self.allowed_image_roots)
            if self.allowed_image_roots
            else "None"
        )

        # Log Server Configuration
        logger.info("")
        logger.info("Server Config")
        logger.info("=" * 60)
        logger.info(f"    host                   : {self.host}")
        logger.info(f"    port                   : {self.port}")
        logger.info(f"    metrics_port           : {self.metrics_port}")
        logger.info(f"    api_types              : {api_types_str}")
        logger.info(f"    operation_mode         : {mode_str}")
        logger.info(
            f"    max_queue_size         : "
            f"{self.max_queue_size if self.max_queue_size is not None else 'unbounded'}"
        )
        logger.info(
            f"    max_pending_requests   : "
            f"{self.max_pending_requests if self.max_pending_requests is not None else 'unbounded'}"
        )
        max_request_str = (
            to_human_readable_bytes(self.max_request_bytes)
            if self.max_request_bytes
            else "unbounded"
        )
        logger.info(f"    max_request_bytes      : {max_request_str}")
        logger.info("")

        # File System Configuration
        logger.info("File System Config")
        logger.info("=" * 60)
        logger.info(f"    allowed_image_roots    : {allowed_roots_str}")
        logger.info(
            f"    max_local_image_bytes  : {to_human_readable_bytes(self.max_local_image_bytes)}"
        )
        max_bytes_str = (
            to_human_readable_bytes(self.max_bytes)
            if self.max_bytes
            else "None"
        )
        logger.info(f"    max_bytes              : {max_bytes_str}")
        logger.info(f"    media_kind             : {self.media_kind}")
        media_url_allowed_hosts_str = (
            ", ".join(self.media_url_allowed_hosts)
            if self.media_url_allowed_hosts
            else "None"
        )
        logger.info(
            f"    media_url_ssrf_guard   : {'enabled' if self.media_url_ssrf_protection_enabled else 'DISABLED'}"
        )
        logger.info(
            f"    media_url_allowed_hosts: {media_url_allowed_hosts_str}"
        )
        logger.info("")

        # Metrics and Telemetry Configuration
        logger.info("Metrics and Telemetry Config")
        logger.info("=" * 60)
        logger.info(
            f"    metric_recording       : {self.metric_recording.value}"
        )
        logger.info(f"    disable_telemetry      : {self.disable_telemetry}")
        logger.info(
            f"    kernel_trace_level     : {self.kernel_trace_level.value}"
        )

        # Transaction recording (part of telemetry)
        if self.transaction_recording_file:
            logger.info(
                f"    transaction_recording  : {self.transaction_recording_file}"
            )
            logger.info(
                f"    include_responses      : {self.transaction_recording_include_responses}"
            )
        else:
            logger.info("    transaction_recording  : None")
        logger.info("")

        # Model Worker Configuration
        logger.info("Model Worker Config")
        logger.info("=" * 60)
        logger.info(f"    use_heartbeat          : {self.use_heartbeat}")
        if self.mw_timeout_s is not None:
            logger.info(
                f"    timeout                : {self.mw_timeout_s:.1f}s"
            )
        logger.info(
            f"    health_fail_timeout    : {self.mw_health_fail_s:.1f}s"
        )
        if self.telemetry_worker_spawn_timeout is not None:
            logger.info(
                f"    telemetry_spawn_timeout: {self.telemetry_worker_spawn_timeout:.1f}s"
            )
        logger.info("")


def parse_api_and_target_arch(compile_spec: str) -> tuple[str, str]:
    """Parse the compile-only specification into API and target architecture.

    Supports two formats:
    1. <api> - Uses default target architecture for the API
    2. <api>:<target_arch> - Uses explicit target architecture

    Args:
        compile_spec: The compile-only specification string

    Returns:
        A tuple of (api, target_arch)

    Raises:
        ValueError: If the API is invalid

    Example:
        >>> parse_api_and_target_arch("cuda")
        ('cuda', 'sm_80')
        >>> parse_api_and_target_arch("cuda:sm_90")
        ('cuda', 'sm_90')
    """
    # Default target architectures for each API
    default_target_archs = {
        "cuda": "sm_80",  # Ampere (A100, RTX 30xx)
        "hip": "gfx942",  # MI300X
        "metal": "apple-m1",  # Apple Silicon
    }

    # Parse the compile-only value as <api> or <api>:<target_arch>
    if ":" in compile_spec:
        api, target_arch = compile_spec.split(":", 1)
    else:
        api = compile_spec
        target_arch = default_target_archs.get(api, "")

    # Validate API
    valid_apis = ["cuda", "hip", "metal"]
    if api not in valid_apis:
        raise ValueError(
            f"Invalid API in --target: '{api}'. "
            f"Valid APIs are: {', '.join(valid_apis)}"
        )

    return api, target_arch
