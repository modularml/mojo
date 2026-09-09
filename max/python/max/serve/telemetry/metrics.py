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

from __future__ import annotations

import abc
import functools
import logging
import time
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass, field
from typing import get_args

from max.serve.config import Settings
from opentelemetry import context
from opentelemetry.metrics import get_meter_provider
from opentelemetry.metrics._internal import instrument as api_instrument
from opentelemetry.sdk.metrics._internal import instrument as sdk_instrument
from opentelemetry.sdk.metrics._internal import measurement

"""!! Jank alert !!

We want to use OTEL for propagating telemetry. It is the best vendor-agnostic
metrics system, but that doesn't mean that it is _good_.  OTEL is _slow_. If we
use it directly, it significally degrades the perf of Max Serve. Consequently,
we have all this machinery to observe some metric (MaxMeasurement) and record
the observation async.

OTEL actively obscures its machinery, uses bunch of proxy classes, has an baroque inheritance tree, and is generally awful.
To record an observation at a specific point in time you do the following:
`meter.create_{foo}._real_instrument._measurement_consumer(Measurement(value, timestamp, instrument, ...))`

Here is how you work with metrics (Instruments) observations (Measurements) and recording them (Consumers):
Lets unpack:
1. meter.create_{foo} gives you a proxy instrument with an obscured type eg _internal.instrument._ProxyCounter.
2. `._real_instrument` The proxy can't do anything, you need to grab the _real_ instrument to record.
3. `._measurement_consumer` The _real_ instrument doesn't expose a way to set the time of the observation, so you have to directly talk to the consumer.
4. `Measurement(...)` now we can create a measurement with a timestamp & pass it down.
"""
logger = logging.getLogger("max.serve")
_meter = get_meter_provider().get_meter("modular")


NumberType = float | int
OtelAttributes = dict[str, str] | None

# API_PROXIES the "types" of measurements we make from a meter
# SDK instruments are the "types" that actually do recording
API_PROXIES = (
    api_instrument._ProxyCounter
    | api_instrument._ProxyGauge
    | api_instrument._ProxyHistogram
    | api_instrument._ProxyUpDownCounter
)

SDK_INSTRUMENTS = (
    sdk_instrument._Counter
    | sdk_instrument._Gauge
    | sdk_instrument._Histogram
    | sdk_instrument._UpDownCounter
)

SupportedInstruments = API_PROXIES | SDK_INSTRUMENTS


# Sorry for the type ignores, OTEL goes out of its way to obscure its types.
# We need to use the fact that these are actually _Proxy{Type} objects  rather
# than {Type} objects
SERVE_METRICS: dict[str, SupportedInstruments] = {
    "maxserve.request_count": _meter.create_counter(
        "maxserve.request_count", description="Http request count"
    ),  # type: ignore
    "maxserve.request_time": _meter.create_histogram(
        "maxserve.request_time", unit="ms", description="Time spent in requests"
    ),  # type: ignore
    "maxserve.input_processing_time": _meter.create_histogram(
        "maxserve.input_processing_time",
        unit="ms",
        description="Input processing time",
    ),  # type: ignore
    "maxserve.output_processing_time": _meter.create_histogram(
        "maxserve.output_processing_time",
        unit="ms",
        description="Output processing time",
    ),  # type: ignore
    "maxserve.time_to_first_token": _meter.create_histogram(
        "maxserve.time_to_first_token",
        unit="ms",
        description=(
            "Time to first token, measured from when the API server received "
            "the request (same origin as 'maxserve.request_time'), so it "
            "includes request parsing, validation and media resolution as "
            "well as tokenization, queueing and prefill."
        ),
    ),  # type: ignore
    "maxserve.num_input_tokens": _meter.create_counter(
        "maxserve.num_input_tokens", description="Count of input tokens"
    ),  # type: ignore
    "maxserve.num_input_characters": _meter.create_counter(
        "maxserve.num_input_characters", description="Count of input characters"
    ),  # type: ignore
    "maxserve.num_output_tokens": _meter.create_counter(
        "maxserve.num_output_tokens", description="Count of generated tokens"
    ),  # type: ignore
    "maxserve.num_requests_queued": _meter.create_gauge(
        "maxserve.num_requests_queued",
        description=(
            "Current depth of the scheduler's CE / prefill queue, "
            "sampled once per scheduler iteration. Mirrors the "
            "'Pending: N reqs' value in scheduler logs."
        ),
    ),  # type: ignore
    "maxserve.num_requests_running": _meter.create_up_down_counter(
        "maxserve.num_requests_running",
        description="Count of requests currently being processed",
    ),  # type: ignore
    "maxserve.num_requests_awaiting_admission": _meter.create_up_down_counter(
        "maxserve.num_requests_awaiting_admission",
        description=(
            "Count of requests received by the API server but not yet handed "
            "off to the model worker (i.e. still in tokenization / pre-submit "
            "on the API side). Incremented on arrival and decremented just "
            "before the request is enqueued to the model worker, so a "
            "persistently high value indicates a backlog stuck in the API "
            "server rather than in the scheduler."
        ),
    ),  # type: ignore
    "maxserve.requests_awaiting_admission": _meter.create_histogram(
        "maxserve.requests_awaiting_admission",
        description=(
            "Distribution of the ingress backlog (requests accepted by the "
            "API server but not yet handed off to the model worker), sampled "
            "periodically. Companion to the "
            "'maxserve.num_requests_awaiting_admission' up/down counter: the "
            "counter is the live value for dashboards, while this histogram "
            "captures the distribution / tail (p50/p99) over time."
        ),
    ),  # type: ignore
    "maxserve.num_responses_buffered": _meter.create_gauge(
        "maxserve.num_responses_buffered",
        description=(
            "Egress backlog: model-worker responses received by the API "
            "server but not yet consumed by the streaming layer (sum of the "
            "per-request output-queue depths), sampled periodically. A "
            "persistently high value means the API server is shipping tokens "
            "back to clients (detokenize + serialize + network) slower than "
            "the model produces them, and the unbounded output queues are "
            "accumulating in API-process memory."
        ),
    ),  # type: ignore
    "maxserve.responses_buffered": _meter.create_histogram(
        "maxserve.responses_buffered",
        description=(
            "Distribution of the egress backlog (responses received by the "
            "API server but not yet streamed to clients), sampled "
            "periodically. Companion to the 'maxserve.num_responses_buffered' "
            "gauge: the gauge shows the latest value for live dashboards, "
            "while this histogram captures the distribution / tail (p50/p99) "
            "over time, which a scrape-interval gauge sample would miss."
        ),
    ),  # type: ignore
    "maxserve.response_queue_time": _meter.create_histogram(
        "maxserve.response_queue_time",
        unit="ms",
        description=(
            "Time a model-worker response waits in the API server's "
            "per-request output queue before the streaming layer consumes it. "
            "Sampled at the head of line (once per consumer wake), so it "
            "tracks the egress-side delay the user experiences when the API "
            "server falls behind the model on decode."
        ),
    ),  # type: ignore
    "maxserve.model_load_time": _meter.create_histogram(
        "maxserve.model_load_time",
        unit="ms",
        description=(
            "Time to load a model. Recorded once per model-worker startup, "
            "both as an untagged aggregate and split by the 'component' tag "
            "(build, compile, init, graph_capture, pinned_memory, spawn, "
            "total), mirroring the per-phase breakdown in the model worker's "
            "startup log lines."
        ),
    ),  # type: ignore
    "maxserve.itl": _meter.create_histogram(
        "maxserve.itl", unit="ms", description="inter token latency"
    ),  # type: ignore
    "maxserve.time_per_output_token": _meter.create_histogram(
        "maxserve.time_per_output_token",
        unit="ms",
        description=(
            "Mean decode-phase latency per generated token, emitted once per "
            "request: decode_time / (num_generated_tokens - 1). Excludes the "
            "first token and prefill/TTFT; accounts for speculative decoding."
        ),
    ),  # type: ignore
    "maxserve.pipeline_load": _meter.create_counter(
        "maxserve.pipeline_load",
        description="Count of pipelines loaded for each model",
    ),  # type: ignore
    "maxserve.batch_size": _meter.create_histogram(
        "maxserve.batch_size",
        description=(
            "Distribution of batch sizes (number of requests), labeled by "
            "'batch_type' (CE prefill or TG decode). For TG this is the "
            "decode batch size; for CE see 'batch_input_tokens' for the "
            "token-count view."
        ),
    ),  # type: ignore
    "maxserve.batch_execution_time": _meter.create_histogram(
        "maxserve.batch_execution_time",
        unit="ms",
        description="Distribution of batch execution time",
    ),  # type: ignore
    "maxserve.di.decode_admission_queue_wait_time": _meter.create_histogram(
        "maxserve.di.decode_admission_queue_wait_time",
        unit="ms",
        description=(
            "Decode-local time a request spends in decode's own admission "
            "queue (self.pending_reqs) before being admitted into the "
            "in-flight set and dispatched to prefill. Decode-clock only, "
            "single process. Added after the dispatch-latency breakdown "
            "(di.dispatch_rtt/prefill_span/reply_rtt) showed the "
            "decode-prefill pipeline staying flat while TTFT still grew "
            "sharply past mc=40 -- this covers the one remaining segment "
            "upstream of dispatch that had no wait-time instrumentation."
        ),
    ),  # type: ignore
    "maxserve.di.decode_send_time": _meter.create_histogram(
        "maxserve.di.decode_send_time",
        unit="ms",
        description=(
            "Decode-side time to build and send a PrefillRequest (struct "
            "construction, msgspec serialization, ZMQ enqueue). Decode-local; "
            "does not include network transit. Part of the DI dispatch "
            "latency breakdown."
        ),
    ),  # type: ignore
    "maxserve.di.dispatch_rtt": _meter.create_histogram(
        "maxserve.di.dispatch_rtt",
        unit="ms",
        description=(
            "Decode-clock round trip from sending a PrefillRequest to "
            "receiving prefill's 'arrived' ack ping. Covers the outbound ZMQ "
            "hop, prefill's negligible ack-send overhead, and the ack's "
            "return hop combined -- a one-way network leg is not measurable "
            "without synchronized clocks, so this is reported as a round "
            "trip rather than split. Part of the DI dispatch latency "
            "breakdown."
        ),
    ),  # type: ignore
    "maxserve.di.prefill_span": _meter.create_histogram(
        "maxserve.di.prefill_span",
        unit="ms",
        description=(
            "Decode-clock time between prefill's 'arrived' and 'ce_done' "
            "ack pings for a request: prefill's queue wait + CE execution + "
            "the second ping's return hop. Cross-check this against "
            "'maxserve.di.prefill_queue_wait_time' + "
            "'maxserve.batch_execution_time' (batch_type=CE), which measure "
            "the same work on prefill's own clock without the ping transit. "
            "Part of the DI dispatch latency breakdown."
        ),
    ),  # type: ignore
    "maxserve.di.reply_rtt": _meter.create_histogram(
        "maxserve.di.reply_rtt",
        unit="ms",
        description=(
            "Decode-clock time from prefill's 'ce_done' ack ping to the "
            "real PrefillResponse arriving: KV-transfer initiation, reply "
            "construction/serialization on prefill, and its network transit. "
            "Part of the DI dispatch latency breakdown."
        ),
    ),  # type: ignore
    "maxserve.di.prefill_queue_wait_time": _meter.create_histogram(
        "maxserve.di.prefill_queue_wait_time",
        unit="ms",
        description=(
            "Prefill-local time a request spends enqueued before its first "
            "CE batch executes. Prefill-clock only, single process -- tests "
            "whether TTFT growth at high concurrency is queue saturation "
            "rather than dispatch/network overhead."
        ),
    ),  # type: ignore
    "maxserve.di.decode_postprocess_time": _meter.create_histogram(
        "maxserve.di.decode_postprocess_time",
        unit="ms",
        description=(
            "Decode-local time from receiving a PrefillResponse to handing "
            "the result off to the response queue. Decode-clock only; does "
            "not include the API-process hop after the response queue (see "
            "'maxserve.response_queue_time' for the API-side tail). Part of "
            "the DI dispatch latency breakdown."
        ),
    ),  # type: ignore
    "maxserve.di.early_sync_time": _meter.create_histogram(
        "maxserve.di.early_sync_time",
        unit="ms",
        description=(
            "Duration of the early-sync guard's blocking commit of the "
            "previous batch's structured-output FSM state, when it fired "
            "(see '_should_early_sync_prev_batch')."
        ),
    ),  # type: ignore
    "maxserve.di.ce_preempted_tg_iteration_count": _meter.create_counter(
        "maxserve.di.ce_preempted_tg_iteration_count",
        description=(
            "Counter: incremented once per scheduling iteration in which a "
            "CE batch was admitted while TG requests were pending, so those "
            "TG requests got zero progress this iteration -- how often this "
            "preemption pattern occurs."
        ),
    ),  # type: ignore
    "maxserve.di.ce_preempted_tg_pending_count": _meter.create_histogram(
        "maxserve.di.ce_preempted_tg_pending_count",
        description=(
            "Histogram: the number of TG requests pending at the moment of "
            "one such preemption (paired with "
            "'maxserve.di.ce_preempted_tg_iteration_count', which counts how "
            "often it happens; this measures how large each occurrence is)."
        ),
    ),  # type: ignore
    "maxserve.di.handoff_to_first_token_time": _meter.create_histogram(
        "maxserve.di.handoff_to_first_token_time",
        unit="ms",
        description=(
            "Wall-clock time from a prefill-to-decode handoff landing (its "
            "token discarded, re-queued as a one-token chunked-CE "
            "continuation) to decode's own re-forward actually sending "
            "token 1 to the API process. A TTFT contribution specific to "
            "the handoff path; the ordinary path's token already arrives "
            "with the 'PrefillResponse', so it has no equivalent wait."
        ),
    ),  # type: ignore
    "maxserve.cache.num_used_blocks": _meter.create_gauge(
        "maxserve.cache.num_used_blocks",
        unit="blocks",
        description="Number of used blocks or pages, measured at the scheduler after batch work.",
    ),  # type: ignore
    "maxserve.cache.num_total_blocks": _meter.create_gauge(
        "maxserve.cache.num_total_blocks",
        unit="blocks",
        description="Total number of blocks or pages, measured at the scheduler after batch work.",
    ),  # type: ignore
    "maxserve.cache.request_prefix_coverage": _meter.create_histogram(
        "maxserve.cache.request_prefix_coverage",
        unit="percent",
        description=(
            "Per-request prefix cache coverage (cached prefix tokens / "
            "prompt tokens), emitted once per admitted request and "
            "unweighted by request size. For the token-weighted cache hit "
            "rate, derive it from maxserve.cache.hits and "
            "maxserve.cache.misses instead."
        ),
    ),  # type: ignore
    "maxserve.cache.preemption_count": _meter.create_counter(
        "maxserve.cache.preemption_count",
        description="Total number of preemptions",
    ),  # type: ignore
    "maxserve.cache.hits": _meter.create_counter(
        "maxserve.cache.hits",
        unit="tokens",
        description=(
            "Cumulative KV cache hit tokens across all CE batches "
            "(prompt tokens served from prefix cache). Tagged with the tier "
            "that served each token: g0 for the on-device prefix cache, "
            "external for the KV connector. The tiers sum to the untagged "
            "total. g0 includes cross-replica device-to-device copies, and "
            "counts first admissions only, so it does not match "
            "maxserve.cache.device_blocks_served scaled by the page size."
        ),
    ),  # type: ignore
    "maxserve.cache.misses": _meter.create_counter(
        "maxserve.cache.misses",
        unit="tokens",
        description=(
            "Cumulative KV cache miss tokens across all CE batches "
            "(prompt tokens actually prefilled by the model)."
        ),
    ),  # type: ignore
    "maxserve.input_tokens_per_request": _meter.create_histogram(
        "maxserve.input_tokens_per_request",
        unit="tokens",
        description="Distribution of input tokens per request",
    ),  # type: ignore
    "maxserve.output_tokens_per_request": _meter.create_histogram(
        "maxserve.output_tokens_per_request",
        unit="tokens",
        description="Distribution of output tokens per request",
    ),  # type: ignore
    "maxserve.dkv.nixl_read_latency": _meter.create_histogram(
        "maxserve.dkv.nixl_read_latency",
        unit="ms",
        description="NIXL READ transfer latency",
    ),  # type: ignore
    "maxserve.dkv.nixl_write_latency": _meter.create_histogram(
        "maxserve.dkv.nixl_write_latency",
        unit="ms",
        description="NIXL WRITE transfer latency",
    ),  # type: ignore
    "maxserve.dkv.rpc_acquire_latency": _meter.create_histogram(
        "maxserve.dkv.rpc_acquire_latency",
        unit="ms",
        description="dKV acquire_blocks RPC latency",
    ),  # type: ignore
    "maxserve.dkv.rpc_read_latency": _meter.create_histogram(
        "maxserve.dkv.rpc_read_latency",
        unit="ms",
        description="dKV read_blocks RPC latency",
    ),  # type: ignore
    "maxserve.dkv.read_blocks": _meter.create_counter(
        "maxserve.dkv.read_blocks",
        unit="blocks",
        description="Cache blocks LANDED in device memory from the dKV tier, over both the remote NIXL path and same-host device copies. Only confirmed-complete transfers count, so this is delivered reuse and not an upper bound on it: blocks dKV holds but cannot serve as a contiguous prefix are dropped before a transfer is ever built, and never reach this counter. For any load that lands it is proportional to maxserve.cache.hits{tier=external}; a persistent shortfall means transfers are failing.",
    ),  # type: ignore
    "maxserve.dkv.connected_clients": _meter.create_gauge(
        "maxserve.dkv.connected_clients",
        unit="clients",
        description="Number of dKV connector clients currently connected to the external KV tier.",
    ),  # type: ignore
    "maxserve.dkv.total_clients": _meter.create_gauge(
        "maxserve.dkv.total_clients",
        unit="clients",
        description="Total number of dKV connector clients, one per data-parallel replica.",
    ),  # type: ignore
    "maxserve.dkv.reconnect_attempts": _meter.create_gauge(
        "maxserve.dkv.reconnect_attempts",
        unit="attempts",
        description="Cumulative dKV reconnect attempts across all clients, reported as a gauge of the current lifetime total.",
    ),  # type: ignore
    "maxserve.spec_decode.acceptance_rate_per_position": _meter.create_histogram(
        "maxserve.spec_decode.acceptance_rate_per_position",
        unit="percent",
        description="Draft token acceptance rate per position (0-100%)",
    ),  # type: ignore
    "maxserve.batch_input_tokens": _meter.create_histogram(
        "maxserve.batch_input_tokens",
        unit="tokens",
        description="Distribution of input tokens per scheduler batch (CE prefill or TG decode).",
    ),  # type: ignore
    "maxserve.batch_context_tokens": _meter.create_histogram(
        "maxserve.batch_context_tokens",
        unit="tokens",
        description="Distribution of accumulated context tokens per scheduler batch.",
    ),  # type: ignore
    "maxserve.batch_creation_time": _meter.create_histogram(
        "maxserve.batch_creation_time",
        unit="ms",
        description="Distribution of scheduler batch creation time.",
    ),  # type: ignore
    "maxserve.batch_prompt_throughput": _meter.create_histogram(
        "maxserve.batch_prompt_throughput",
        unit="tokens/s",
        description="Per-batch prompt-side throughput in tokens/second.",
    ),  # type: ignore
    "maxserve.batch_generation_throughput": _meter.create_histogram(
        "maxserve.batch_generation_throughput",
        unit="tokens/s",
        description="Per-batch generation-side throughput in tokens/second.",
    ),  # type: ignore
    "maxserve.dp_active_token_occupancy": _meter.create_histogram(
        "maxserve.dp_active_token_occupancy",
        unit="%",
        description=(
            "Per-batch data-parallel balance: mean/max of per-rank "
            "active-token load as a percentage. 100 = perfectly balanced "
            "ranks; the floor is 100/DP-degree (all load on one rank). "
            "Excludes DP padding dummies. Recorded only when "
            "data_parallel_degree > 1."
        ),
    ),  # type: ignore
    "maxserve.dp_context_token_occupancy": _meter.create_histogram(
        "maxserve.dp_context_token_occupancy",
        unit="%",
        description=(
            "Per-batch data-parallel balance: mean/max of per-rank "
            "context-token (KV / attention) load as a percentage. 100 = "
            "perfectly balanced ranks; the floor is 100/DP-degree (all "
            "load on one rank). Excludes DP padding dummies. Recorded only "
            "when data_parallel_degree > 1 and at least one rank has "
            "processed tokens (fresh prefill batches are skipped)."
        ),
    ),  # type: ignore
    "maxserve.dp_active_tokens": _meter.create_counter(
        "maxserve.dp_active_tokens",
        unit="tokens",
        description=(
            "Cumulative active tokens scheduled across all DP replicas, "
            "excluding padding dummies. Divided by "
            "maxserve.dp_step_capacity_tokens over the same window, this "
            "gives the token-weighted DP occupancy (each batch weighted by "
            "its step cost rather than counted once). Recorded only when "
            "data_parallel_degree > 1."
        ),
    ),  # type: ignore
    "maxserve.dp_step_capacity_tokens": _meter.create_counter(
        "maxserve.dp_step_capacity_tokens",
        unit="tokens",
        description=(
            "Cumulative synchronized step capacity in tokens: for each "
            "batch, DP-degree times the heaviest rank's active tokens "
            "(ranks step together, so the heaviest rank sets the step "
            "cost). Denominator for token-weighted DP occupancy. Recorded "
            "only when data_parallel_degree > 1."
        ),
    ),  # type: ignore
    "maxserve.batch_terminated_reqs": _meter.create_histogram(
        "maxserve.batch_terminated_reqs",
        unit="reqs",
        description="Distribution of requests terminated per scheduler batch.",
    ),  # type: ignore
    "maxserve.batch_pending_reqs": _meter.create_histogram(
        "maxserve.batch_pending_reqs",
        unit="reqs",
        description="Distribution of requests pending in the queue, sampled once per scheduler batch.",
    ),  # type: ignore
    "maxserve.cache.used_kv_pct": _meter.create_histogram(
        "maxserve.cache.used_kv_pct",
        unit="percent",
        description="Percentage of device KV cache blocks in use (0-100%), sampled once per scheduler batch.",
    ),  # type: ignore
    "maxserve.cache.used_host_kv_pct": _meter.create_histogram(
        "maxserve.cache.used_host_kv_pct",
        unit="percent",
        description="Percentage of the host KV tier's bytes in use (0-100%), sampled once per scheduler batch when host paging is enabled.",
    ),  # type: ignore
    "maxserve.cache.device_blocks_served": _meter.create_counter(
        "maxserve.cache.device_blocks_served",
        unit="blocks",
        description=(
            "Cumulative KV blocks served directly from the local device "
            "prefix cache, with no host/disk promotion or cross-replica "
            "copy needed."
        ),
    ),  # type: ignore
    "maxserve.cache.h2d_bytes_copied": _meter.create_counter(
        "maxserve.cache.h2d_bytes_copied",
        unit="bytes",
        description=(
            "Cumulative KV bytes copied from the connector's host tier to "
            "device. Rate this to see the PCIe bandwidth host paging consumes."
        ),
    ),  # type: ignore
    "maxserve.cache.cross_replica_blocks_copied": _meter.create_counter(
        "maxserve.cache.cross_replica_blocks_copied",
        unit="blocks",
        description=(
            "Cumulative KV blocks copied device-to-device across "
            "data-parallel replicas to reuse a prefix-cache hit resident "
            "on another replica."
        ),
    ),  # type: ignore
    "maxserve.cache.cross_replica_bytes_copied": _meter.create_counter(
        "maxserve.cache.cross_replica_bytes_copied",
        unit="bytes",
        description=(
            "Cumulative bytes moved by device-to-device KV copies across "
            "data-parallel replicas. Rate this to see NVLink/interconnect "
            "bandwidth consumed by cross-replica prefix reuse."
        ),
    ),  # type: ignore
    "maxserve.cache.d2h_bytes_copied": _meter.create_counter(
        "maxserve.cache.d2h_bytes_copied",
        unit="bytes",
        description=(
            "Cumulative KV bytes copied from device to the connector's host "
            "tier."
        ),
    ),  # type: ignore
    "maxserve.cache.disk_bytes_read": _meter.create_counter(
        "maxserve.cache.disk_bytes_read",
        unit="bytes",
        description="Cumulative KV bytes read from the disk cache tier.",
    ),  # type: ignore
    "maxserve.cache.disk_bytes_written": _meter.create_counter(
        "maxserve.cache.disk_bytes_written",
        unit="bytes",
        description="Cumulative KV bytes written to the disk cache tier.",
    ),  # type: ignore
    "maxserve.spec_decode.avg_acceptance_length": _meter.create_histogram(
        "maxserve.spec_decode.avg_acceptance_length",
        unit="tokens",
        description="Mean draft-token acceptance length per spec-decode batch.",
    ),  # type: ignore
    "maxserve.dkv.nixl_read_gib_per_s": _meter.create_histogram(
        "maxserve.dkv.nixl_read_gib_per_s",
        unit="GiB/s",
        description="NIXL READ throughput.",
    ),  # type: ignore
    "maxserve.dkv.nixl_write_gib_per_s": _meter.create_histogram(
        "maxserve.dkv.nixl_write_gib_per_s",
        unit="GiB/s",
        description="NIXL WRITE throughput.",
    ),  # type: ignore
    "maxserve.cache.used_disk_kv_pct": _meter.create_histogram(
        "maxserve.cache.used_disk_kv_pct",
        unit="percent",
        description="Percentage of the disk KV tier's bytes in use (0-100%), sampled once per scheduler batch when disk paging is enabled.",
    ),  # type: ignore
    "maxserve.vision.images_encoded": _meter.create_counter(
        "maxserve.vision.images_encoded",
        unit="images",
        description="Cumulative images run through the vision encoder (cache misses).",
    ),  # type: ignore
    "maxserve.vision.images_cached": _meter.create_counter(
        "maxserve.vision.images_cached",
        unit="images",
        description="Cumulative images served from the vision encoder cache (cache hits).",
    ),  # type: ignore
    "maxserve.vision.patches_encoded": _meter.create_counter(
        "maxserve.vision.patches_encoded",
        unit="patches",
        description="Cumulative image patches fed to the vision encoder.",
    ),  # type: ignore
    "maxserve.vision.tokens_encoded": _meter.create_counter(
        "maxserve.vision.tokens_encoded",
        unit="tokens",
        description="Cumulative merged vision tokens produced by the vision encoder.",
    ),  # type: ignore
    "maxserve.vision.cache_hit_rate": _meter.create_histogram(
        "maxserve.vision.cache_hit_rate",
        unit="percent",
        description="Per-batch vision encoder cache hit rate (0-100%).",
    ),  # type: ignore
    "maxserve.video.clips_encoded": _meter.create_counter(
        "maxserve.video.clips_encoded",
        unit="clips",
        description="Cumulative video clips run through the video encoder (cache misses).",
    ),  # type: ignore
    "maxserve.video.tokens_encoded": _meter.create_counter(
        "maxserve.video.tokens_encoded",
        unit="tokens",
        description="Cumulative merged video tokens produced by the video encoder.",
    ),  # type: ignore
    "maxserve.video.encoding_time_milliseconds": _meter.create_histogram(
        "maxserve.video.encoding_time_milliseconds",
        unit="ms",
        description="Per-batch video encoder wall-clock time.",
    ),  # type: ignore
    "maxserve.video.frames_per_clip": _meter.create_histogram(
        "maxserve.video.frames_per_clip",
        unit="frames",
        description="Sampled frame count per newly-encoded video clip.",
    ),  # type: ignore
    "maxserve.tool_call.conformance_errors": _meter.create_counter(
        "maxserve.tool_call.conformance_errors",
        description=(
            "Count of generated tool calls that failed the observability-only "
            "schema-conformance check, split by the 'outcome' tag "
            "(invalid_json, unknown_tool, schema_mismatch). Mirrors the "
            "'tool_call_conformance' warning log; the function name and failing "
            "JSON paths stay in the log to keep label cardinality bounded."
        ),
    ),  # type: ignore
    "maxserve.structured_output.grammar_rejections": _meter.create_counter(
        "maxserve.structured_output.grammar_rejections",
        description=(
            "Count of structured-output requests rejected at admission "
            "(HTTP 400) because the active grammar backend could not compile "
            "the schema, split by the 'kind' tag (tool_grammar, json_schema)."
        ),
    ),  # type: ignore
    "maxserve.structured_output.grammar_build_time": _meter.create_histogram(
        "maxserve.structured_output.grammar_build_time",
        unit="ms",
        description=(
            "Wall-clock time of an off-thread grammar-matcher build in "
            "AsyncGrammarGate, from the start of the backend's compile/create "
            "call to its completion. For a DI handoff, this build overlaps "
            "prefill's round trip when submitted at admission time; a build "
            "that outlasts that round trip can still gate the handoff's "
            "chunked-CE continuation."
        ),
    ),  # type: ignore
    "maxserve.response_format.conformance_errors": _meter.create_counter(
        "maxserve.response_format.conformance_errors",
        description=(
            "Count of response_format (json_schema/json_object) responses "
            "whose final content failed the observability-only "
            "schema-conformance check, split by the 'outcome' tag "
            "(invalid_json, schema_mismatch). Mirrors the "
            "'response_format_conformance' warning log; the failing JSON "
            "paths stay in the log to keep label cardinality bounded."
        ),
    ),  # type: ignore
}


class UnknownMetric(Exception):
    pass


# Suffix identifying the exponential-histogram shadow of a histogram metric.
# When histogram_shadow_emission is enabled (MAX_SERVE_OTLP_METRICS_ENDPOINT
# is set; see common.configure_metrics), every histogram commit is mirrored
# to "<name><HISTOGRAM_SHADOW_SUFFIX>", recorded with
# ExponentialBucketHistogramAggregation instead of the explicit-bucket
# aggregation used by "<name>". Both reach the OTLP endpoint side by side for
# MXSERV-258 comparison; the shadow name is matched by a View instrument_name
# glob in common.py, so it never needs enumerating per metric.
HISTOGRAM_SHADOW_SUFFIX = ".exponential"

_histogram_shadow_emission_enabled = False


def configure_histogram_shadow_emission(enabled: bool) -> None:
    """Enable or disable mirroring histogram commits to their shadow metric.

    Called from common.configure_metrics with whether
    MAX_SERVE_OTLP_METRICS_ENDPOINT is set.
    """
    global _histogram_shadow_emission_enabled
    _histogram_shadow_emission_enabled = enabled


def _is_histogram(inst: object) -> bool:
    # SERVE_METRICS entries are created at import time, before any real
    # meter provider is configured, so they are always proxy instruments
    # here (never the concrete SDK ones) — check the proxy type directly.
    return isinstance(inst, api_instrument._ProxyHistogram)


# Build the exponential-histogram shadow instruments once, alongside the
# primary ones above: one per histogram, sharing its unit, named
# "<name><HISTOGRAM_SHADOW_SUFFIX>". Created unconditionally (instrument
# creation is cheap and inert until something records to it) so
# configure_histogram_shadow_emission can be toggled at runtime without
# touching instrument setup.
for _name, _inst in list(SERVE_METRICS.items()):
    if _is_histogram(_inst):
        _shadow_name = _name + HISTOGRAM_SHADOW_SUFFIX
        SERVE_METRICS[_shadow_name] = _meter.create_histogram(
            _shadow_name,
            description=f"Exponential-histogram shadow of {_name} (MXSERV-258)",
        )  # type: ignore
del _name, _inst


def _commit_measurement(
    instrument_name: str,
    value: NumberType,
    attributes: OtelAttributes,
    time_unix_nano: int,
) -> None:
    # find the instrument
    try:
        instrument = SERVE_METRICS[instrument_name]
    except KeyError as e:
        raise UnknownMetric(instrument_name) from e

    # Sometimes the instrument is a proxy.  Unrap it.
    if isinstance(instrument, get_args(API_PROXIES)):
        instrument = instrument._real_instrument
        # bail if there is no underlying instrument
        if instrument is None:
            logger.error(f"instrument is None for {instrument_name}")
            return

    # instrument should be one of the supported sdk types now
    if not isinstance(instrument, get_args(SDK_INSTRUMENTS)):
        # If you're hitting this, metrics were likely not configured properly.
        logger.error(
            f"instrument {instrument_name} is not one of the supported sdk types"
        )
        return

    # convert to an otel measurement
    m = measurement.Measurement(
        value,
        time_unix_nano,
        instrument,
        context.get_current(),
        attributes,
    )

    # record the measurement
    consumer = instrument._measurement_consumer
    consumer.consume_measurement(m)
    logger.debug(f"consumed measurement for {instrument_name}")


@dataclass
class MaxMeasurement:
    """Shim around the recording of a metric observation

    Simplifies decoupling the observation of a metric from its recording.
    """

    instrument_name: str
    value: NumberType
    attributes: OtelAttributes | None = None
    time_unix_nano: int = field(default_factory=time.time_ns)

    def commit(self) -> None:
        _commit_measurement(
            self.instrument_name,
            self.value,
            self.attributes,
            self.time_unix_nano,
        )

        if _histogram_shadow_emission_enabled:
            shadow_name = self.instrument_name + HISTOGRAM_SHADOW_SUFFIX
            if shadow_name in SERVE_METRICS:
                _commit_measurement(
                    shadow_name,
                    self.value,
                    self.attributes,
                    self.time_unix_nano,
                )


TelemetryFn = Callable[[MaxMeasurement], None]


class MetricClient(abc.ABC):
    @abc.abstractmethod
    def send_measurement(self, metric: MaxMeasurement) -> None: ...

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group a burst of measurements into a single flush.

        A caller that knows it is about to emit many measurements at once
        (e.g. the per-iteration scheduler metrics) can wrap them in a
        transaction so a client that crosses a process boundary batches them
        into one packet instead of one send per measurement. Measurements
        emitted outside a transaction are sent immediately.

        Clients that record in-process ignore this and always emit
        immediately; the default implementation is a no-op.
        """
        yield

    @abc.abstractmethod
    def cross_process_factory(
        self,
        settings: Settings,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        """Get a copier for use of this client in another process.

        To use a MetricClient across processes, call cross_process_factory in
        the parent process and pass the result across the process boundary.
        Then in the child process, use an 'async with' to get a semantically
        identical MetricClient that can be used.

        This is needed because some metric clients require reinitialization on
        the other side of a process boundary before they can be safely used.
        """
        ...


@asynccontextmanager
async def _trivially_picklable_xprocess_factory(
    client: MetricClient,
) -> AsyncGenerator[MetricClient, None]:
    yield client


class NoopClient(MetricClient):
    def send_measurement(self, m: MaxMeasurement) -> None:
        pass

    def cross_process_factory(
        self,
        settings: Settings,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        return functools.partial(_trivially_picklable_xprocess_factory, self)


class SyncClient(MetricClient):
    def send_measurement(self, m: MaxMeasurement) -> None:
        m.commit()

    def cross_process_factory(
        self,
        settings: Settings,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        return functools.partial(_trivially_picklable_xprocess_factory, self)


class _AsyncMetrics:
    """Centralizes metrics to encapsulate the OTEL dependency and avoid breaking schema changes

    Produce metric measurements to be consumed elsewhere
    """

    def __init__(self) -> None:
        self.client: MetricClient = NoopClient()
        self.extra_attributes: dict[str, str] = {}

    def configure(
        self,
        client: MetricClient,
        extra_attributes: dict[str, str] | None = None,
    ) -> None:
        self.client = client
        self.extra_attributes = extra_attributes or {}

    def transaction(self) -> AbstractContextManager[None]:
        """Group the measurements emitted in the ``with`` block into one flush.

        See :meth:`MetricClient.transaction`. Use this to wrap a burst of
        measurements that are always produced together so a client that
        crosses the telemetry process boundary sends them as a single packet::

            with METRICS.transaction():
                METRICS.batch_size(...)
                METRICS.batch_execution_time(...)
                ...

        Only cross-process clients batch; in-process clients treat this as a
        no-op and emit each measurement immediately.
        """
        return self.client.transaction()

    def request_count(self, responseCode: int, urlPath: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.request_count",
                1,
                {
                    **self.extra_attributes,
                    "code": f"{responseCode:d}",
                    "path": urlPath,
                },
            ),
        )

    def request_time(self, value: float, urlPath: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.request_time",
                value,
                {**self.extra_attributes, "path": urlPath},
            ),
        )

    def input_time(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.input_processing_time", value, self.extra_attributes
            ),
        )

    def output_time(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.output_processing_time", value, self.extra_attributes
            ),
        )

    def ttft(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.time_to_first_token", value, self.extra_attributes
            ),
        )

    def input_tokens(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_input_tokens", value, self.extra_attributes
            ),
        )

    def input_characters(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_input_characters", value, self.extra_attributes
            ),
        )

    def output_tokens(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_output_tokens", value, self.extra_attributes
            ),
        )

    def reqs_queued(self, value: int) -> None:
        """Publish the current depth of the scheduler's CE / prefill queue.

        ``maxserve.num_requests_queued`` is a synchronous gauge: every call
        replaces the previously reported value rather than accumulating.
        Schedulers should call this once per iteration with
        ``len(all_ce_reqs)`` (or the equivalent for their queue layout) at
        the same point that the "Pending: N reqs" log line is computed.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_requests_queued", value, self.extra_attributes
            ),
        )

    def reqs_running(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_requests_running", value, self.extra_attributes
            ),
        )

    def reqs_awaiting_admission(self, value: int) -> None:
        """Adjust the count of API-side requests not yet handed to the worker.

        ``maxserve.num_requests_awaiting_admission`` is an up/down counter:
        call with ``1`` when a request is accepted by the API server (before
        tokenization) and ``-1`` just before it is enqueued to the model
        worker. A persistently high value means requests are backing up in the
        API server (e.g. tokenization) rather than in the scheduler queue.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_requests_awaiting_admission",
                value,
                self.extra_attributes,
            ),
        )

    def requests_awaiting_admission_dist(self, value: int) -> None:
        """Record a sample of the ingress backlog for distribution analysis.

        Companion to :meth:`reqs_awaiting_admission` (the live up/down
        counter): a periodic sample of the same running count is fed into the
        ``maxserve.requests_awaiting_admission`` histogram so p50/p99 ingress
        backlog over time can be recovered.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.requests_awaiting_admission",
                value,
                self.extra_attributes,
            ),
        )

    def responses_buffered(self, value: int) -> None:
        """Publish the current egress backlog (sum of output-queue depths).

        ``maxserve.num_responses_buffered`` is a synchronous gauge: every call
        replaces the previously reported value. The API server should sample
        it periodically with the total number of model-worker responses
        received but not yet consumed by the streaming layer.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_responses_buffered",
                value,
                self.extra_attributes,
            ),
        )

    def responses_buffered_dist(self, value: int) -> None:
        """Record a sample of the egress backlog for distribution analysis.

        Companion to :meth:`responses_buffered` (the live gauge): the same
        periodic sample is also fed into the ``maxserve.responses_buffered``
        histogram so p50/p99 backlog over time can be recovered, which a
        scrape-interval gauge sample alone cannot provide.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.responses_buffered", value, self.extra_attributes
            ),
        )

    def response_queue_time(self, ms: float) -> None:
        """Record how long a response waited in the API output queue (ms)."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.response_queue_time", ms, self.extra_attributes
            ),
        )

    def model_load_time(self, ms: float, component: str | None = None) -> None:
        """Record a model-worker startup duration in milliseconds.

        Args:
            ms: The duration in milliseconds.
            component: Optional phase name (e.g. ``"build"``, ``"compile"``,
                ``"init"``, ``"graph_capture"``, ``"pinned_memory"``,
                ``"spawn"``, ``"total"``). Recorded as the ``component`` tag
                so a single metric can be split by startup phase. When
                omitted, records the untagged model-load aggregate.
        """
        attributes = self.extra_attributes
        if component is not None:
            attributes = {**attributes, "component": component}
        self.client.send_measurement(
            MaxMeasurement("maxserve.model_load_time", ms, attributes),
        )

    def itl(self, ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.itl", ms, self.extra_attributes),
        )

    def time_per_output_token(self, ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.time_per_output_token", ms, self.extra_attributes
            ),
        )

    def pipeline_load(self, name: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.pipeline_load",
                1,
                {**self.extra_attributes, "model": name},
            ),
        )

    def batch_size(self, size: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_size",
                size,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_execution_time(
        self, execution_time: float, batch_type: str
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_execution_time",
                execution_time,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def di_decode_admission_queue_wait_time(self, ms: float) -> None:
        """Decode-local wait in decode's own admission queue before dispatch."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.di.decode_admission_queue_wait_time",
                ms,
                self.extra_attributes,
            ),
        )

    def di_decode_send_time(self, ms: float) -> None:
        """Decode-side PrefillRequest build+serialize+send duration."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.di.decode_send_time", ms, self.extra_attributes
            ),
        )

    def di_dispatch_rtt(self, ms: float) -> None:
        """Decode-clock round trip to prefill's 'arrived' ack ping."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.di.dispatch_rtt", ms, self.extra_attributes
            ),
        )

    def di_prefill_span(self, ms: float) -> None:
        """Decode-clock span between prefill's 'arrived' and 'ce_done' pings."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.di.prefill_span", ms, self.extra_attributes
            ),
        )

    def di_reply_rtt(self, ms: float) -> None:
        """Decode-clock span from the 'ce_done' ping to the real reply."""
        self.client.send_measurement(
            MaxMeasurement("maxserve.di.reply_rtt", ms, self.extra_attributes),
        )

    def di_prefill_queue_wait_time(self, ms: float) -> None:
        """Prefill-local queue wait before a request's first CE batch."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.di.prefill_queue_wait_time",
                ms,
                self.extra_attributes,
            ),
        )

    def di_decode_postprocess_time(self, ms: float) -> None:
        """Decode-local time from PrefillResponse receipt to response queue."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.di.decode_postprocess_time",
                ms,
                self.extra_attributes,
            ),
        )

    def di_early_sync_time(self, ms: float) -> None:
        """Duration of the early-sync guard's blocking commit of the
        previous batch's structured-output FSM state, when it fired (see
        ``_should_early_sync_prev_batch``)."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.di.early_sync_time", ms, self.extra_attributes
            ),
        )

    def di_ce_preempted_tg_iteration_count(self) -> None:
        """Counter: how often a CE batch admits work while TG requests are
        pending, giving those TG requests zero progress that iteration."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.di.ce_preempted_tg_iteration_count",
                1,
                self.extra_attributes,
            ),
        )

    def di_ce_preempted_tg_pending_count(self, count: int) -> None:
        """Histogram: how large each such preemption was -- the number of TG
        requests pending at the moment (paired with
        ``di_ce_preempted_tg_iteration_count``, which counts how often)."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.di.ce_preempted_tg_pending_count",
                count,
                self.extra_attributes,
            ),
        )

    def di_handoff_to_first_token_time(self, ms: float) -> None:
        """Wall-clock time from a prefill-to-decode handoff landing (its
        token discarded, re-queued as a one-token chunked-CE continuation)
        to decode's own re-forward actually sending token 1 to the API
        process. A TTFT contribution specific to the handoff path; the
        ordinary path's token already arrives with the ``PrefillResponse``,
        so it has no equivalent wait."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.di.handoff_to_first_token_time",
                ms,
                self.extra_attributes,
            ),
        )

    def cache_num_used_blocks(self, num_used_blocks: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.num_used_blocks",
                num_used_blocks,
                self.extra_attributes,
            ),
        )

    def cache_num_total_blocks(self, total_blocks: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.num_total_blocks",
                total_blocks,
                self.extra_attributes,
            ),
        )

    def cache_request_prefix_coverage(self, coverage: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.request_prefix_coverage",
                coverage,
                self.extra_attributes,
            ),
        )

    def cache_hits(self, hits: int, *, tier: str) -> None:
        """Records prefix-cache hit tokens served by ``tier``.

        Args:
            hits: Prompt tokens served from the cache.
            tier: Which tier served them, one of ``g0`` (device prefix cache) or
                ``external`` (the KV connector, tier unresolved). Shared with
                Mach's ``mach.cache.hits`` so one dashboard panel covers both
                engines; Mach additionally resolves ``g1`` and ``g2``.

        Every hit point carries a ``tier``, so the tiers sum to the counter's
        untagged total and a query that does not group by ``tier`` reads exactly
        what it read before the attribute existed.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.hits",
                hits,
                {**self.extra_attributes, "tier": tier},
            ),
        )

    def cache_misses(self, cache_misses: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.misses", cache_misses, self.extra_attributes
            ),
        )

    def preemption(self) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.preemption_count", 1, self.extra_attributes
            ),
        )

    def vision_images_encoded(self, images: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.vision.images_encoded",
                images,
                self.extra_attributes,
            ),
        )

    def vision_images_cached(self, images: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.vision.images_cached",
                images,
                self.extra_attributes,
            ),
        )

    def vision_patches_encoded(self, patches: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.vision.patches_encoded",
                patches,
                self.extra_attributes,
            ),
        )

    def vision_tokens_encoded(self, tokens: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.vision.tokens_encoded",
                tokens,
                self.extra_attributes,
            ),
        )

    def vision_cache_hit_rate(self, hit_rate: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.vision.cache_hit_rate",
                hit_rate,
                self.extra_attributes,
            ),
        )

    def video_clips_encoded(self, clips: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.video.clips_encoded",
                clips,
                self.extra_attributes,
            ),
        )

    def video_tokens_encoded(self, tokens: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.video.tokens_encoded",
                tokens,
                self.extra_attributes,
            ),
        )

    def video_encoding_time_milliseconds(self, time_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.video.encoding_time_milliseconds",
                time_ms,
                self.extra_attributes,
            ),
        )

    def video_frames_per_clip(self, frames: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.video.frames_per_clip",
                frames,
                self.extra_attributes,
            ),
        )

    def input_tokens_per_request(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.input_tokens_per_request",
                value,
                self.extra_attributes,
            ),
        )

    def output_tokens_per_request(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.output_tokens_per_request",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_nixl_read_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.nixl_read_latency",
                latency_ms,
                self.extra_attributes,
            ),
        )

    def dkv_nixl_write_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.nixl_write_latency",
                latency_ms,
                self.extra_attributes,
            ),
        )

    def dkv_rpc_acquire_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.rpc_acquire_latency",
                latency_ms,
                self.extra_attributes,
            ),
        )

    def dkv_rpc_read_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.rpc_read_latency",
                latency_ms,
                self.extra_attributes,
            ),
        )

    def dkv_connected_clients(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.connected_clients",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_total_clients(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.total_clients",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_reconnect_attempts(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.reconnect_attempts",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_read_blocks(self, blocks: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.read_blocks",
                blocks,
                self.extra_attributes,
            ),
        )

    def spec_decode_acceptance_rate_per_position(
        self, position: int, acceptance_rate: float
    ) -> None:
        """Emit draft token acceptance rate for a specific position.

        Args:
            position: The draft token position (0-indexed).
            acceptance_rate: The acceptance rate as a percentage (0-100).
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.spec_decode.acceptance_rate_per_position",
                acceptance_rate,
                {**self.extra_attributes, "position": str(position)},
            ),
        )

    def batch_input_tokens(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_input_tokens",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_context_tokens(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_context_tokens",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_creation_time(self, ms: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_creation_time",
                ms,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_prompt_throughput(self, tps: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_prompt_throughput",
                tps,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_generation_throughput(self, tps: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_generation_throughput",
                tps,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def dp_active_token_occupancy(self, pct: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dp_active_token_occupancy",
                pct,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def dp_context_token_occupancy(self, pct: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dp_context_token_occupancy",
                pct,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def dp_active_tokens(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dp_active_tokens",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def dp_step_capacity_tokens(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dp_step_capacity_tokens",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_terminated_reqs(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_terminated_reqs",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_pending_reqs(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_pending_reqs",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def cache_used_kv_pct(self, ratio: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.used_kv_pct", ratio, self.extra_attributes
            ),
        )

    def cache_used_host_kv_pct(self, ratio: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.used_host_kv_pct",
                ratio,
                self.extra_attributes,
            ),
        )

    def cache_device_blocks_served(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.device_blocks_served",
                count,
                self.extra_attributes,
            ),
        )

    def cache_h2d_bytes_copied(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.h2d_bytes_copied",
                count,
                self.extra_attributes,
            ),
        )

    def cache_cross_replica_blocks_copied(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.cross_replica_blocks_copied",
                count,
                self.extra_attributes,
            ),
        )

    def cache_cross_replica_bytes_copied(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.cross_replica_bytes_copied",
                count,
                self.extra_attributes,
            ),
        )

    def cache_d2h_bytes_copied(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.d2h_bytes_copied",
                count,
                self.extra_attributes,
            ),
        )

    def cache_disk_bytes_read(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.disk_bytes_read",
                count,
                self.extra_attributes,
            ),
        )

    def cache_disk_bytes_written(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.disk_bytes_written",
                count,
                self.extra_attributes,
            ),
        )

    def cache_used_disk_kv_pct(self, ratio: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.used_disk_kv_pct",
                ratio,
                self.extra_attributes,
            ),
        )

    def spec_decode_avg_acceptance_length(self, length: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.spec_decode.avg_acceptance_length",
                length,
                self.extra_attributes,
            ),
        )

    def dkv_nixl_read_gib_per_s(self, gib_per_s: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.nixl_read_gib_per_s",
                gib_per_s,
                self.extra_attributes,
            ),
        )

    def dkv_nixl_write_gib_per_s(self, gib_per_s: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.nixl_write_gib_per_s",
                gib_per_s,
                self.extra_attributes,
            ),
        )

    def tool_call_conformance_error(self, outcome: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.tool_call.conformance_errors",
                1,
                {**self.extra_attributes, "outcome": outcome},
            ),
        )

    def structured_output_grammar_rejection(self, kind: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.structured_output.grammar_rejections",
                1,
                {**self.extra_attributes, "kind": kind},
            ),
        )

    def structured_output_grammar_build_time(self, ms: float) -> None:
        """Wall-clock time of an off-thread grammar-matcher build."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.structured_output.grammar_build_time",
                ms,
                self.extra_attributes,
            ),
        )

    def response_format_conformance_error(self, outcome: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.response_format.conformance_errors",
                1,
                {**self.extra_attributes, "outcome": outcome},
            ),
        )


METRICS = _AsyncMetrics()
