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

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from max.driver import Buffer
from max.nn.kv_cache.metrics import dkv_tier_degraded
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import PagedKVCacheManagerInterface
from max.pipelines.lib.vision_encoder_cache import (
    VideoEncoderMetrics,
    VisionEncoderMetrics,
)
from max.pipelines.modeling.types import (
    BatchType,
    CompletedBatchStats,
    RequestID,
    TextGenerationInputs,
)
from max.pipelines.speculative.utils import (
    _SpeculativeDecodingMetrics,
)
from max.serve.queue import MAXPullQueue, drain_queue
from max.serve.telemetry.metrics import METRICS
from max.support.human_readable_formatter import (
    to_human_readable_bytes,
    to_human_readable_latency,
)

from .config import TokenGenerationSchedulerConfig

logger = logging.getLogger("max.serve")


def _to_human_readable_throughput(tps: float) -> str:
    if tps >= 1_000:
        return f"{tps / 1e3:.1f}K tok/s"
    return f"{tps:.1f} tok/s"


def _dp_token_occupancy_pct(
    per_rank_tokens: Sequence[int],
    num_replicas: int,
) -> float | None:
    """Mean/max of per-rank token sums as a percentage.

    Ranks step together, so the heaviest rank sets the step cost; mean/max
    is the fraction of that synchronized capacity doing useful work (100 =
    perfectly balanced; the floor is 100/DP-degree). Replicas missing from
    ``per_rank_tokens`` scheduled zero tokens. ``None`` when every rank sums
    to zero.
    """
    per_rank = list(per_rank_tokens)
    per_rank.extend([0] * (num_replicas - len(per_rank)))
    max_rank_tokens = max(per_rank, default=0)
    if max_rank_tokens == 0:
        return None
    return 100.0 * sum(per_rank) / (num_replicas * max_rank_tokens)


def _format_spec_decode_clause(
    draft_tokens_generated: int,
    draft_tokens_accepted: int,
    avg_acceptance_length: float,
    max_acceptance_length: int,
    acceptance_rate_per_position: list[float],
) -> str:
    """Formats the "Draft Tokens: ..." log clause; empty when no drafts."""
    if draft_tokens_generated <= 0:
        return ""
    acceptance_rate = draft_tokens_accepted / draft_tokens_generated
    if acceptance_rate_per_position:
        pos_rates_str = ", ".join(
            f"p{i}={rate:.0%}"
            for i, rate in enumerate(acceptance_rate_per_position)
        )
        per_pos_str = f", Per-Pos: [{pos_rates_str}]"
    else:
        per_pos_str = ""
    # "Acceptance Len: <avg> / <max> toks" is parsed by
    # analyze_batch_logs.py, so only append after "toks". The parenthetical
    # names both conventions: <avg> counts accepted drafts per verify step,
    # while "acceptance length" elsewhere often includes the bonus token (one
    # per verification).
    return (
        f"Draft Tokens: {draft_tokens_accepted}/{draft_tokens_generated} "
        f"({acceptance_rate:.2%}) accepted, "
        f"Acceptance Len: {avg_acceptance_length:.2f} / "
        f"{max_acceptance_length} toks "
        f"(accepted drafts/step; {avg_acceptance_length + 1:.2f} toks/step "
        f"incl bonus)"
        f"{per_pos_str} | "
    )


@dataclass
class BatchMetrics:
    batch_type: BatchType
    batch_size: int
    max_batch_size: int
    terminated_reqs: int
    num_pending_reqs: int
    num_input_tokens: int
    max_batch_input_tokens: int
    num_context_tokens: int
    max_batch_total_tokens: int
    batch_creation_time_s: float
    batch_execution_time_s: float
    prompt_throughput: float
    generation_throughput: float
    total_preemption_count: int

    used_kv_pct: float
    total_kv_blocks: int
    cache_hit_rate: float
    cache_hit_tokens: int
    cache_miss_tokens: int
    device_blocks_served: int

    used_host_kv_pct: float
    total_host_kv_bytes: int
    h2d_bytes_copied: int
    d2h_bytes_copied: int
    disk_bytes_read: int
    disk_bytes_written: int
    inflight_disk_ops: int

    used_disk_kv_pct: float
    total_disk_kv_bytes: int

    draft_tokens_generated: int
    draft_tokens_accepted: int
    avg_acceptance_length: float
    max_acceptance_length: int
    acceptance_rate_per_position: list[float]

    nixl_read_latency_avg_ms: float
    nixl_write_latency_avg_ms: float
    rpc_acquire_latency_avg_ms: float
    rpc_read_latency_avg_ms: float
    nixl_read_gib_per_s: float = 0.0
    nixl_write_gib_per_s: float = 0.0

    # dKV external-tier health, summed across the per-replica connector
    # clients. The connected and total counts support a degraded alert when
    # connected is below total and a dead-tier alert when connected is zero,
    # and reconnect_attempts is a lifetime cumulative counter. All three are
    # levels or cumulative rather than per-batch deltas, so they publish as
    # gauges. Zero when no dKV tier is attached.
    dkv_connected_clients: int = 0
    dkv_total_clients: int = 0
    dkv_reconnect_attempts: int = 0

    # Cache blocks dKV served this batch, an upper bound on delivered reuse:
    # a block behind a hole in the request's hash chain is served and then
    # dropped untransferred. Pairs with cache_hit_external_tokens to surface
    # the served-versus-landed gap. Zero when no dKV tier is attached.
    dkv_read_blocks: int = 0

    # How many of ``cache_hit_tokens`` the KV connector served. The remainder
    # came from the device prefix cache, which is how ``cache_hits`` splits per
    # ``tier``. Always 0 without a connector.
    cache_hit_external_tokens: int = 0

    # When True, ``batch_execution_time_s`` and the throughputs describe the
    # previously enqueued batch, so ``completed`` is reported instead.
    overlap_active: bool = False

    # The batch whose outputs were synchronized this iteration. ``None`` when
    # nothing completed, or when overlap is off and this record covers both.
    completed: CompletedBatchStats | None = None

    # Data-parallel balance of this batch's active-token load: mean/max of
    # per-rank active-token sums as a percentage (100 = perfectly balanced;
    # the floor is 100/DP-degree). ``None`` when data_parallel_degree == 1 or
    # the batch is empty. DP padding dummies are excluded so this reflects the
    # batch constructor's placement decisions, not the padded shapes.
    dp_active_token_occupancy_pct: float | None = None

    # Data-parallel balance of this batch's context-token (KV / attention)
    # load: mean/max of per-rank processed-length sums as a percentage. Same
    # convention as ``dp_active_token_occupancy_pct``; ``None`` when
    # data_parallel_degree == 1 or no rank has processed tokens yet (e.g. a
    # fresh prefill batch).
    dp_context_token_occupancy_pct: float | None = None

    # Raw numerator/denominator behind ``dp_active_token_occupancy_pct``:
    # total active tokens across replicas, and the synchronized step capacity
    # (DP-degree x the heaviest rank's active tokens). Emitted as counters so
    # token-weighted occupancy = sum(tokens) / sum(capacity) over any window.
    # Both 0 when data_parallel_degree == 1 or the batch is empty.
    dp_active_tokens: int = 0
    dp_step_capacity_tokens: int = 0

    # Device-to-device KV block copies across DP replicas (a prefix-cache hit
    # resident on another replica, materialized locally). Both 0 when
    # data_parallel_degree == 1 or no such copy happened this batch.
    cross_replica_blocks_copied: int = 0
    cross_replica_bytes_copied: int = 0

    # Per-request prefix cache coverage for requests admitted in this batch
    # (cached_prefix_length / prompt_length). Empty for non-CE batches and
    # for CE batches that admit no new requests (e.g. follow-up prefill
    # chunks of an already-admitted long prefill).
    per_request_prefix_coverage: list[float] = field(default_factory=list)

    # Number of requests newly admitted in this batch. Zero for TG batches
    # and for CE batches that contain only chunked-prefill continuations of
    # already-admitted requests. The cache-hit log clause and cumulative
    # hit/miss counters are gated on this being non-zero.
    num_new_admissions: int = 0

    # Per-iteration vision encoder statistics for multimodal models. None
    # when the batch did no vision encoding (text-only model, or a batch /
    # decode step with no images). The vision log clause and vision metrics
    # are gated on this being non-None.
    vision_metrics: VisionEncoderMetrics | None = None

    # Per-iteration video encoder statistics for multimodal models. None
    # when the batch did no video encoding. The video log clause and video
    # metrics are gated on this being non-None.
    video_metrics: VideoEncoderMetrics | None = None

    @classmethod
    def create(
        cls,
        sch_config: TokenGenerationSchedulerConfig,
        inputs: TextGenerationInputs[TextContext],
        kv_cache: PagedKVCacheManagerInterface | None,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
        num_pending_reqs: int,
        num_terminated_reqs: int,
        total_preemption_count: int,
        batch_spec_decode_metrics: _SpeculativeDecodingMetrics | None = None,
        batch_vision_metrics: VisionEncoderMetrics | None = None,
        batch_video_metrics: VideoEncoderMetrics | None = None,
        overlap_active: bool = False,
        completed_batch_stats: CompletedBatchStats | None = None,
    ) -> BatchMetrics:
        num_input_tokens = inputs.input_tokens
        batch_size = inputs.batch_size
        prompt_throughput = num_input_tokens / batch_execution_time_s
        if (
            batch_spec_decode_metrics is not None
            and inputs.batch_type == BatchType.TG
        ):
            generation_throughput = (
                batch_spec_decode_metrics.output_tokens / batch_execution_time_s
            )
        else:
            generation_throughput = batch_size / batch_execution_time_s

        total_kv_blocks = 0
        used_kv_pct = 0.0
        device_blocks_served = 0
        used_host_kv_pct = 0.0
        total_host_kv_bytes = 0
        h2d_bytes_copied = 0
        d2h_bytes_copied = 0
        cross_replica_blocks_copied = 0
        cross_replica_bytes_copied = 0
        disk_bytes_read = 0
        disk_bytes_written = 0
        inflight_disk_ops = 0
        used_disk_kv_pct = 0.0
        total_disk_kv_bytes = 0
        nixl_read_latency_avg_ms = 0.0
        nixl_write_latency_avg_ms = 0.0
        rpc_acquire_latency_avg_ms = 0.0
        rpc_read_latency_avg_ms = 0.0
        nixl_read_gib_per_s = 0.0
        nixl_write_gib_per_s = 0.0
        dkv_connected_clients = 0
        dkv_total_clients = 0
        dkv_reconnect_attempts = 0
        dkv_read_blocks = 0
        num_replicas = sch_config.data_parallel_degree

        # Data-parallel balance, along two axes: active tokens (compute load
        # this step) and context tokens (KV / attention load), computed from
        # the per-replica sums frozen at batch construction. See
        # ``_dp_token_occupancy_pct`` for the mean/max convention. Alongside
        # the per-batch percentage, accumulate the raw numerator/denominator
        # (active tokens vs. synchronized step capacity) so a token-weighted
        # occupancy can be computed over any window from counter deltas.
        dp_active_token_occupancy_pct: float | None = None
        dp_context_token_occupancy_pct: float | None = None
        dp_active_tokens = 0
        dp_step_capacity_tokens = 0
        if num_replicas > 1:
            per_rank_active = list(inputs.per_replica_input_tokens)
            per_rank_active.extend([0] * (num_replicas - len(per_rank_active)))
            max_rank_active = max(per_rank_active, default=0)
            if max_rank_active > 0:
                dp_active_tokens = sum(per_rank_active)
                dp_step_capacity_tokens = num_replicas * max_rank_active
                dp_active_token_occupancy_pct = (
                    100.0 * dp_active_tokens / dp_step_capacity_tokens
                )
            dp_context_token_occupancy_pct = _dp_token_occupancy_pct(
                inputs.per_replica_context_tokens, num_replicas
            )

        if kv_cache is not None:
            # TODO SERVOPT-939: Add some sugar
            block_counts = [
                kv_cache.block_count(replica_idx)
                for replica_idx in range(num_replicas)
            ]
            total_kv_blocks = sum(bc.total for bc in block_counts)
            used_kv_blocks = sum(bc.used for bc in block_counts)
            assert total_kv_blocks > 0
            used_kv_pct = used_kv_blocks / total_kv_blocks

            host_byte_counts = [
                kv_cache.host_byte_count(replica_idx)
                for replica_idx in range(num_replicas)
            ]
            total_host_kv_bytes = sum(bc.total for bc in host_byte_counts)

            metrics_agg = kv_cache.get_metrics_aggregated()

            if total_host_kv_bytes > 0:
                used_host_kv_bytes = sum(bc.used for bc in host_byte_counts)
                used_host_kv_pct = used_host_kv_bytes / total_host_kv_bytes

            device_blocks_served = metrics_agg.device_blocks_served
            h2d_bytes_copied = metrics_agg.h2d_bytes_copied
            d2h_bytes_copied = metrics_agg.d2h_bytes_copied
            cross_replica_blocks_copied = (
                metrics_agg.cross_replica_blocks_copied
            )
            cross_replica_bytes_copied = metrics_agg.cross_replica_bytes_copied
            disk_bytes_written = metrics_agg.disk_bytes_written
            disk_bytes_read = metrics_agg.disk_bytes_read
            inflight_disk_ops = metrics_agg.inflight_disk_ops

            disk_byte_counts = [
                kv_cache.disk_byte_count(replica_idx)
                for replica_idx in range(num_replicas)
            ]
            total_disk_kv_bytes = sum(bc.total for bc in disk_byte_counts)
            if total_disk_kv_bytes > 0:
                used_disk_kv_bytes = sum(bc.used for bc in disk_byte_counts)
                used_disk_kv_pct = used_disk_kv_bytes / total_disk_kv_bytes

            # dKV latency metrics: sum across replicas then average.
            nixl_read_latency_avg_ms = metrics_agg.nixl_read_latency_avg_ms
            nixl_write_latency_avg_ms = metrics_agg.nixl_write_latency_avg_ms
            rpc_acquire_latency_avg_ms = metrics_agg.rpc_acquire_latency_avg_ms
            rpc_read_latency_avg_ms = metrics_agg.rpc_read_latency_avg_ms
            nixl_read_gib_per_s = metrics_agg.nixl_read_gib_per_s
            nixl_write_gib_per_s = metrics_agg.nixl_write_gib_per_s

            # dKV external-tier health. Read before reset_metrics like the
            # metrics above, though the connector reports these live and does
            # not clear them on reset.
            dkv_connected_clients = metrics_agg.dkv_connected_clients
            dkv_total_clients = metrics_agg.dkv_total_clients
            dkv_reconnect_attempts = metrics_agg.dkv_reconnect_attempts
            dkv_read_blocks = metrics_agg.nixl_read_blocks

            kv_cache.reset_metrics()

        # Capture per-request KV cache hit rates for newly admitted requests.
        # The block manager set ``cached_prefix_length`` on each context's
        # first admission; consume it here so chunked-prefill follow-ups do
        # not re-emit observations for the same request. Admission only
        # happens on CE batches, so skip the scan on TG entirely.
        # The same admission data feeds the batch-level cache hit/miss
        # numbers, so a continuation-only CE batch contributes nothing to
        # them and the log line drops the cache-hit clause entirely.
        per_request_prefix_coverage: list[float] = []
        admission_hit_tokens = 0
        admission_prompt_tokens = 0
        admission_external_tokens = 0
        if inputs.batch_type == BatchType.CE:
            for ctx in inputs.flat_batch:
                if (
                    ctx._cache_metrics_emitted
                    or ctx.cached_prefix_length is None
                ):
                    continue
                ctx._cache_metrics_emitted = True
                cached = ctx.cached_prefix_length
                prompt_length = ctx.tokens.prompt_length
                if prompt_length > 0:
                    per_request_prefix_coverage.append(cached / prompt_length)
                    admission_hit_tokens += cached
                    admission_prompt_tokens += prompt_length
                    # The block manager caps this at ``cached``, so the device
                    # remainder below can never go negative.
                    admission_external_tokens += (
                        ctx.cached_prefix_external_length
                    )

        cache_hit_tokens = admission_hit_tokens
        cache_miss_tokens = admission_prompt_tokens - admission_hit_tokens
        cache_hit_rate = (
            cache_hit_tokens / admission_prompt_tokens
            if admission_prompt_tokens > 0
            else 0.0
        )

        draft_tokens_generated = 0
        draft_tokens_accepted = 0
        avg_acceptance_length = 0.0
        max_acceptance_length = 0
        acceptance_rate_per_position: list[float] = []
        # Acceptance metrics describe the decode/verify step. Under the overlap
        # pipeline a CE iteration observes the previous TG batch's metrics, so
        # gate on batch type to avoid mis-attributing them to CE batches.
        if (
            batch_spec_decode_metrics is not None
            and inputs.batch_type == BatchType.TG
        ):
            draft_tokens_generated = (
                batch_spec_decode_metrics.draft_tokens_generated
            )
            draft_tokens_accepted = (
                batch_spec_decode_metrics.draft_tokens_accepted
            )
            avg_acceptance_length = (
                batch_spec_decode_metrics.avg_acceptance_length
            )
            max_acceptance_length = (
                batch_spec_decode_metrics.num_speculative_tokens
            )
            acceptance_rate_per_position = (
                batch_spec_decode_metrics.acceptance_rate_per_position
            )

        return cls(
            batch_type=inputs.batch_type,
            batch_size=batch_size,
            max_batch_size=sch_config.max_batch_size,
            terminated_reqs=num_terminated_reqs,
            num_pending_reqs=num_pending_reqs,
            num_input_tokens=num_input_tokens,
            max_batch_input_tokens=sch_config.target_tokens_per_batch_ce,
            num_context_tokens=inputs.context_tokens,
            max_batch_total_tokens=sch_config.max_batch_total_tokens or 0,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            prompt_throughput=prompt_throughput,
            generation_throughput=generation_throughput,
            total_preemption_count=total_preemption_count,
            used_kv_pct=used_kv_pct,
            total_kv_blocks=total_kv_blocks,
            cache_hit_rate=cache_hit_rate,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
            cache_hit_external_tokens=admission_external_tokens,
            device_blocks_served=device_blocks_served,
            used_host_kv_pct=used_host_kv_pct,
            total_host_kv_bytes=total_host_kv_bytes,
            h2d_bytes_copied=h2d_bytes_copied,
            d2h_bytes_copied=d2h_bytes_copied,
            cross_replica_blocks_copied=cross_replica_blocks_copied,
            cross_replica_bytes_copied=cross_replica_bytes_copied,
            disk_bytes_read=disk_bytes_read,
            disk_bytes_written=disk_bytes_written,
            used_disk_kv_pct=used_disk_kv_pct,
            total_disk_kv_bytes=total_disk_kv_bytes,
            inflight_disk_ops=inflight_disk_ops,
            draft_tokens_generated=draft_tokens_generated,
            draft_tokens_accepted=draft_tokens_accepted,
            avg_acceptance_length=avg_acceptance_length,
            max_acceptance_length=max_acceptance_length,
            acceptance_rate_per_position=acceptance_rate_per_position,
            nixl_read_latency_avg_ms=nixl_read_latency_avg_ms,
            nixl_write_latency_avg_ms=nixl_write_latency_avg_ms,
            rpc_acquire_latency_avg_ms=rpc_acquire_latency_avg_ms,
            rpc_read_latency_avg_ms=rpc_read_latency_avg_ms,
            nixl_read_gib_per_s=nixl_read_gib_per_s,
            nixl_write_gib_per_s=nixl_write_gib_per_s,
            dkv_connected_clients=dkv_connected_clients,
            dkv_total_clients=dkv_total_clients,
            dkv_reconnect_attempts=dkv_reconnect_attempts,
            dkv_read_blocks=dkv_read_blocks,
            overlap_active=overlap_active,
            completed=completed_batch_stats,
            dp_active_token_occupancy_pct=dp_active_token_occupancy_pct,
            dp_context_token_occupancy_pct=dp_context_token_occupancy_pct,
            dp_active_tokens=dp_active_tokens,
            dp_step_capacity_tokens=dp_step_capacity_tokens,
            per_request_prefix_coverage=per_request_prefix_coverage,
            num_new_admissions=len(per_request_prefix_coverage),
            vision_metrics=batch_vision_metrics,
            video_metrics=batch_video_metrics,
        )

    def pretty_format(self) -> str:
        context_tokens_str = ""
        if self.max_batch_total_tokens != 0:
            context_tokens_str = f"Context Tokens: {self.num_context_tokens}/{self.max_batch_total_tokens} toks | "

        kv_str = ""
        if self.total_kv_blocks != 0:
            usage_str = f"KVCache usage: {self.used_kv_pct:.1%} of {self.total_kv_blocks} blocks"
            # Only show the cache-hit clause when this batch newly admitted
            # at least one request. CE batches that are pure chunked-prefill
            # continuations report 0 admissions and would otherwise display
            # a misleading 0.0% hit rate over their continuation tokens.
            if self.num_new_admissions > 0:
                kv_str = (
                    f"{usage_str}, "
                    f"Cache hit rate: {self.cache_hit_rate:.1%} "
                    f"({self.cache_hit_tokens} hit, {self.cache_miss_tokens} miss) | "
                )
            else:
                kv_str = f"{usage_str} | "

        host_kv_str = ""
        if self.total_host_kv_bytes != 0:
            disk_str = ""
            if self.disk_bytes_read > 0 or self.disk_bytes_written > 0:
                disk_str = (
                    f", Disk: {to_human_readable_bytes(self.disk_bytes_read)} "
                    f"read, "
                    f"{to_human_readable_bytes(self.disk_bytes_written)} written"
                )
            host_kv_str = (
                f"Host KVCache Usage: {self.used_host_kv_pct:.1%} of "
                f"{to_human_readable_bytes(self.total_host_kv_bytes)}, "
                f"Copied: {to_human_readable_bytes(self.h2d_bytes_copied)} H2D, "
                f"{to_human_readable_bytes(self.d2h_bytes_copied)} D2H{disk_str} | "
            )

        disk_kv_str = ""
        if self.total_disk_kv_bytes != 0:
            disk_kv_str = (
                f"Disk KVCache Usage: {self.used_disk_kv_pct:.1%} of "
                f"{to_human_readable_bytes(self.total_disk_kv_bytes)}, "
                f"Inflight Disk Ops: {self.inflight_disk_ops} | "
            )

        spec_decode_str = _format_spec_decode_clause(
            self.draft_tokens_generated,
            self.draft_tokens_accepted,
            self.avg_acceptance_length,
            self.max_acceptance_length,
            self.acceptance_rate_per_position,
        )

        dkv_str = ""
        has_dkv = (
            self.nixl_read_latency_avg_ms > 0
            or self.nixl_write_latency_avg_ms > 0
            or self.rpc_acquire_latency_avg_ms > 0
            or self.rpc_read_latency_avg_ms > 0
        )
        if has_dkv:
            dkv_str = (
                f"dKV: read {self.nixl_read_latency_avg_ms:.1f}ms"
                f" ({self.nixl_read_gib_per_s:.2f} GiB/s), "
                f"write {self.nixl_write_latency_avg_ms:.1f}ms"
                f" ({self.nixl_write_gib_per_s:.2f} GiB/s), "
                f"acquire {self.rpc_acquire_latency_avg_ms:.1f}ms, "
                f"pin {self.rpc_read_latency_avg_ms:.1f}ms | "
            )

        # A separate clause from dkv_str above, because a degraded tier does no
        # transfers and so would show nothing there. Emitted only while
        # degraded to keep the healthy log line quiet.
        dkv_health_str = ""
        if dkv_tier_degraded(
            self.dkv_connected_clients, self.dkv_total_clients
        ):
            dkv_health_str = (
                f"dKV degraded: {self.dkv_connected_clients}/"
                f"{self.dkv_total_clients} connected, "
                f"{self.dkv_reconnect_attempts} reconnect attempts | "
            )

        vision_str = ""
        vm = self.vision_metrics
        if vm is not None and vm.num_images_total > 0:
            vision_str = (
                f"Vision Encoder: {vm.num_images_encoded} imgs, "
                f"{vm.num_patches_encoded} patches, "
                f"{vm.num_tokens_encoded} toks encoded, "
                f"cache hit rate {vm.cache_hit_rate:.1%} "
                f"({vm.num_images_cached} hit, {vm.num_images_encoded} miss) | "
            )

        video_str = ""
        vid = self.video_metrics
        if vid is not None and vid.num_clips_total > 0:
            video_str = (
                f"Video Encoder: {vid.num_clips_encoded} clips, "
                f"{sum(vid.frame_counts)} frames, "
                f"{vid.num_tokens_encoded} toks encoded, "
                f"{vid.encoding_time_ms:.1f}ms, "
                f"cache hit rate {vid.cache_hit_rate:.1%} "
                f"({vid.num_clips_cached} hit, {vid.num_clips_encoded} miss) | "
            )

        # One occupancy clause, matched to the batch's dominant load: active
        # tokens for CE (compute) and context tokens for TG (KV / attention).
        # The published metrics keep the full active/context split.
        dp_occupancy_pct = (
            self.dp_active_token_occupancy_pct
            if self.batch_type == BatchType.CE
            else self.dp_context_token_occupancy_pct
        )
        dp_str = ""
        if dp_occupancy_pct is not None:
            dp_str = f"DP Occupancy: {dp_occupancy_pct:.1f}% | "

        # Scheduler/cache state rather than a specific batch's execution, so
        # these stay valid under either path below.
        state_str = (
            f"{dp_str}{kv_str}{host_kv_str}{disk_kv_str}{dkv_str}"
            f"{dkv_health_str}"
        )
        encoder_str = f"{vision_str}{video_str}"

        if not self.overlap_active:
            return (
                f"Executed {self.batch_type.value} batch with {self.batch_size} reqs | "
                f"Terminated: {self.terminated_reqs} reqs, "
                f"Pending: {self.num_pending_reqs} reqs | "
                f"Input Tokens: {self.num_input_tokens}/{self.max_batch_input_tokens} toks | "
                f"{context_tokens_str}"
                f"Prompt Tput: {_to_human_readable_throughput(self.prompt_throughput)}, "
                f"Generation Tput: {_to_human_readable_throughput(self.generation_throughput)} | "
                f"Batch creation: {to_human_readable_latency(self.batch_creation_time_s)}, "
                f"Execution: {to_human_readable_latency(self.batch_execution_time_s)} | "
                f"{state_str}"
                f"{spec_decode_str}"
                f"{encoder_str}"
                f"All Preemptions: {self.total_preemption_count} reqs"
            )

        # The enqueued and completed batches are different, so each gets its
        # own clause rather than sharing one misleading identity.
        clauses = ""
        if self.batch_size > 0 or self.completed is None:
            clauses += (
                f"Enqueued {self.batch_type.value} batch with {self.batch_size} reqs | "
                f"Pending: {self.num_pending_reqs} reqs | "
                f"Input Tokens: {self.num_input_tokens}/{self.max_batch_input_tokens} toks | "
                f"{context_tokens_str}"
                f"Batch creation: {to_human_readable_latency(self.batch_creation_time_s)} | "
            )
        if self.completed is not None:
            c = self.completed
            completed_spec_str = _format_spec_decode_clause(
                c.draft_tokens_generated,
                c.draft_tokens_accepted,
                c.avg_acceptance_length,
                c.max_acceptance_length,
                c.acceptance_rate_per_position,
            )
            # "Completed Input Tokens" is deliberately distinct from the
            # enqueued clause's "Input Tokens" so both stay parseable.
            clauses += (
                f"Completed {c.batch_type.value} batch with {c.batch_size} reqs | "
                f"Terminated: {self.terminated_reqs} reqs | "
                f"Completed Input Tokens: {c.num_input_tokens} toks | "
                f"Prompt Tput: {_to_human_readable_throughput(c.prompt_throughput)}, "
                f"Generation Tput: {_to_human_readable_throughput(c.generation_throughput)} | "
                f"Execution: {to_human_readable_latency(c.execution_time_s)} | "
                f"{completed_spec_str}"
            )
        return (
            f"{clauses}"
            f"{state_str}"
            f"{encoder_str}"
            f"All Preemptions: {self.total_preemption_count} reqs"
        )

    def to_log_extra(self) -> dict[str, object]:
        """Curated flat-scalar dict for ``logger.info(..., extra=...)``.

        ``configure_logging``'s ``JsonFormatter`` merges these keys into the
        JSON payload when ``MODULAR_STRUCTURED_LOGGING=True``; the plaintext
        formatter ignores them. Conditional clauses mirror :meth:`pretty_format`
        so it doesn't emit zeros from subsystems that didn't run.

        Every quantity :meth:`pretty_format` prints appears here as its own
        key, including the denominators (``max_batch_input_tokens``,
        ``max_acceptance_length``) and the per-position acceptance rates. Log
        backends such as Datadog facet on scalar attributes only, so a value
        that lives solely in the message text is not queryable; lists are
        flattened to indexed keys rather than emitted as arrays.
        """
        extra: dict[str, object] = {
            "event": "batch_metrics",
            "batch_type": self.batch_type.value,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "num_pending_reqs": self.num_pending_reqs,
            "num_input_tokens": self.num_input_tokens,
            "max_batch_input_tokens": self.max_batch_input_tokens,
            "num_context_tokens": self.num_context_tokens,
            "batch_creation_time_ms": self.batch_creation_time_s * 1000,
            "overlap_active": self.overlap_active,
            "total_preemption_count": self.total_preemption_count,
        }

        if self.max_batch_total_tokens != 0:
            extra["max_batch_total_tokens"] = self.max_batch_total_tokens

        if self.dp_active_token_occupancy_pct is not None:
            extra["dp_active_token_occupancy_pct"] = (
                self.dp_active_token_occupancy_pct
            )

        if self.dp_context_token_occupancy_pct is not None:
            extra["dp_context_token_occupancy_pct"] = (
                self.dp_context_token_occupancy_pct
            )

        # Raw DP numerator/denominator and the cross-replica copy counters,
        # under the same gate ``publish_metrics`` uses for them.
        if self.dp_step_capacity_tokens > 0:
            extra["dp_active_tokens"] = self.dp_active_tokens
            extra["dp_step_capacity_tokens"] = self.dp_step_capacity_tokens
            extra["cross_replica_blocks_copied"] = (
                self.cross_replica_blocks_copied
            )
            extra["cross_replica_bytes_copied"] = (
                self.cross_replica_bytes_copied
            )

        if not self.overlap_active:
            extra["terminated_reqs"] = self.terminated_reqs
            extra["prompt_throughput"] = self.prompt_throughput
            extra["generation_throughput"] = self.generation_throughput
            extra["batch_execution_time_ms"] = (
                self.batch_execution_time_s * 1000
            )
        elif self.completed is not None:
            # completed_* names so consumers cannot mistake these for the
            # enqueued batch. terminated_reqs already describes this batch.
            c = self.completed
            extra["completed_batch_type"] = c.batch_type.value
            extra["completed_batch_size"] = c.batch_size
            extra["completed_num_input_tokens"] = c.num_input_tokens
            extra["completed_terminated_reqs"] = self.terminated_reqs
            extra["completed_prompt_throughput"] = c.prompt_throughput
            extra["completed_generation_throughput"] = c.generation_throughput
            extra["completed_execution_time_ms"] = c.execution_time_s * 1000

        if self.total_kv_blocks != 0:
            extra["used_kv_pct"] = self.used_kv_pct
            extra["total_kv_blocks"] = self.total_kv_blocks

        if self.num_new_admissions > 0:
            extra["num_new_admissions"] = self.num_new_admissions
            extra["cache_hit_rate"] = self.cache_hit_rate
            extra["cache_hit_tokens"] = self.cache_hit_tokens
            extra["cache_miss_tokens"] = self.cache_miss_tokens
            extra["cache_hit_external_tokens"] = self.cache_hit_external_tokens
            extra["device_blocks_served"] = self.device_blocks_served

        if self.total_host_kv_bytes != 0:
            extra["total_host_kv_bytes"] = self.total_host_kv_bytes
            extra["used_host_kv_pct"] = self.used_host_kv_pct
            extra["h2d_bytes_copied"] = self.h2d_bytes_copied
            extra["d2h_bytes_copied"] = self.d2h_bytes_copied

        if self.total_disk_kv_bytes != 0:
            extra["total_disk_kv_bytes"] = self.total_disk_kv_bytes
            extra["used_disk_kv_pct"] = self.used_disk_kv_pct
            extra["disk_bytes_read"] = self.disk_bytes_read
            extra["disk_bytes_written"] = self.disk_bytes_written
            extra["inflight_disk_ops"] = self.inflight_disk_ops

        if not self.overlap_active:
            if self.draft_tokens_generated > 0:
                extra["draft_tokens_generated"] = self.draft_tokens_generated
                extra["draft_tokens_accepted"] = self.draft_tokens_accepted
                extra["draft_token_acceptance_rate"] = (
                    self.draft_tokens_accepted / self.draft_tokens_generated
                )
                extra["avg_acceptance_length"] = self.avg_acceptance_length
                extra["max_acceptance_length"] = self.max_acceptance_length
                for position, rate in enumerate(
                    self.acceptance_rate_per_position
                ):
                    extra[f"acceptance_rate_p{position}"] = rate
        elif (
            self.completed is not None
            and self.completed.draft_tokens_generated > 0
        ):
            cs = self.completed
            extra["completed_draft_tokens_generated"] = (
                cs.draft_tokens_generated
            )
            extra["completed_draft_tokens_accepted"] = cs.draft_tokens_accepted
            extra["completed_draft_token_acceptance_rate"] = (
                cs.draft_tokens_accepted / cs.draft_tokens_generated
            )
            extra["completed_avg_acceptance_length"] = cs.avg_acceptance_length
            extra["completed_max_acceptance_length"] = cs.max_acceptance_length
            for position, rate in enumerate(cs.acceptance_rate_per_position):
                extra[f"completed_acceptance_rate_p{position}"] = rate

        vm = self.vision_metrics
        if vm is not None and vm.num_images_total > 0:
            extra["vision_images_total"] = vm.num_images_total
            extra["vision_images_encoded"] = vm.num_images_encoded
            extra["vision_images_cached"] = vm.num_images_cached
            extra["vision_patches_encoded"] = vm.num_patches_encoded
            extra["vision_tokens_encoded"] = vm.num_tokens_encoded
            extra["vision_cache_hit_rate"] = vm.cache_hit_rate

        vid = self.video_metrics
        if vid is not None and vid.num_clips_total > 0:
            extra["video_clips_total"] = vid.num_clips_total
            extra["video_clips_encoded"] = vid.num_clips_encoded
            extra["video_clips_cached"] = vid.num_clips_cached
            extra["video_frames_encoded"] = sum(vid.frame_counts)
            extra["video_tokens_encoded"] = vid.num_tokens_encoded
            extra["video_encoding_time_ms"] = vid.encoding_time_ms
            extra["video_cache_hit_rate"] = vid.cache_hit_rate

        if (
            self.nixl_read_latency_avg_ms > 0
            or self.nixl_write_latency_avg_ms > 0
            or self.rpc_acquire_latency_avg_ms > 0
            or self.rpc_read_latency_avg_ms > 0
        ):
            extra["nixl_read_latency_avg_ms"] = self.nixl_read_latency_avg_ms
            extra["nixl_write_latency_avg_ms"] = self.nixl_write_latency_avg_ms
            extra["nixl_read_gib_per_s"] = self.nixl_read_gib_per_s
            extra["nixl_write_gib_per_s"] = self.nixl_write_gib_per_s
            extra["rpc_acquire_latency_avg_ms"] = (
                self.rpc_acquire_latency_avg_ms
            )
            extra["rpc_read_latency_avg_ms"] = self.rpc_read_latency_avg_ms

        # Emitted whenever a dKV tier is attached, not only while transferring,
        # so a dead tier that does no transfers still records its health.
        if self.dkv_total_clients > 0:
            extra["dkv_connected_clients"] = self.dkv_connected_clients
            extra["dkv_total_clients"] = self.dkv_total_clients
            extra["dkv_reconnect_attempts"] = self.dkv_reconnect_attempts

        if self.dkv_read_blocks > 0:
            extra["dkv_read_blocks"] = self.dkv_read_blocks

        return extra

    def publish_metrics(self, *, defer_execution_metrics: bool = False) -> None:
        """Publishes batch-level telemetry.

        Args:
            defer_execution_metrics: When True (overlap scheduling), skip the
                execution-time, throughput and termination metrics: they
                describe the batch synchronized this iteration rather than the
                one enqueued, so they are instead published from the
                pipeline's ``CompletedBatchStats`` via
                :func:`publish_completed_batch_metrics`, labeled with the
                completed batch's type.
        """
        bt = self.batch_type.value  # "CE" (prefill) or "TG" (decode)
        # This runs once per scheduler iteration and emits the whole batch of
        # measurements below together. Wrap them in a transaction so the
        # telemetry client flushes them as a single cross-process packet
        # instead of one send per measurement.
        with METRICS.transaction():
            self._publish_metrics(bt, defer_execution_metrics)

    def _publish_metrics(self, bt: str, defer_execution_metrics: bool) -> None:
        METRICS.batch_size(self.batch_size, batch_type=bt)
        METRICS.batch_input_tokens(self.num_input_tokens, batch_type=bt)
        METRICS.batch_context_tokens(self.num_context_tokens, batch_type=bt)

        # Terminations come from the synchronized outputs, so under overlap
        # they belong to the completed batch and are published there.
        if not defer_execution_metrics:
            METRICS.batch_terminated_reqs(self.terminated_reqs, batch_type=bt)
        METRICS.batch_pending_reqs(self.num_pending_reqs, batch_type=bt)
        # Publish the current scheduler queue depth as a synchronous gauge
        # (mirrors the "Pending: N reqs" value emitted in scheduler logs).
        METRICS.reqs_queued(self.num_pending_reqs)

        if not defer_execution_metrics:
            METRICS.batch_prompt_throughput(
                self.prompt_throughput, batch_type=bt
            )
            METRICS.batch_generation_throughput(
                self.generation_throughput, batch_type=bt
            )
            METRICS.batch_execution_time(
                self.batch_execution_time_s * 1000, batch_type=bt
            )

        METRICS.batch_creation_time(
            self.batch_creation_time_s * 1000, batch_type=bt
        )

        # DP balance describes the enqueued batch's composition, so it is
        # published every iteration regardless of overlap deferral.
        if self.dp_active_token_occupancy_pct is not None:
            METRICS.dp_active_token_occupancy(
                self.dp_active_token_occupancy_pct, batch_type=bt
            )
        if self.dp_context_token_occupancy_pct is not None:
            METRICS.dp_context_token_occupancy(
                self.dp_context_token_occupancy_pct, batch_type=bt
            )
        if self.dp_step_capacity_tokens > 0:
            METRICS.dp_active_tokens(self.dp_active_tokens, batch_type=bt)
            METRICS.dp_step_capacity_tokens(
                self.dp_step_capacity_tokens, batch_type=bt
            )
            METRICS.cache_cross_replica_blocks_copied(
                self.cross_replica_blocks_copied
            )
            METRICS.cache_cross_replica_bytes_copied(
                self.cross_replica_bytes_copied
            )

        METRICS.cache_num_used_blocks(
            int(self.total_kv_blocks * self.used_kv_pct)
        )
        METRICS.cache_num_total_blocks(self.total_kv_blocks)
        if self.total_kv_blocks != 0:
            METRICS.cache_used_kv_pct(self.used_kv_pct * 100)

        if self.batch_type == BatchType.CE and self.num_new_admissions > 0:
            # Tag each hit with the tier that served it. The two add up to
            # ``cache_hit_tokens``, so a query that does not group by ``tier``
            # still reads the same total it did before the attribute existed.
            #
            # ``g0`` fires even at zero: before the attribute an all-cold CE
            # batch recorded 0, so an ungrouped query read 0. Skipping it would
            # leave the series absent on a cold server, and rate() over an
            # absent series returns no data rather than a flat zero.
            #
            # ``external`` follows the same rule, but keyed on whether a tier is
            # ATTACHED rather than on whether it delivered: a connector that has
            # served nothing all process (dKV down at startup, or degraded
            # before its first hit) is exactly the state worth alerting on, and
            # skipping it there would publish nothing to alert on. Same
            # reasoning as the dKV health gauges below. Without a connector
            # there is no series to mint.
            device_tokens = (
                self.cache_hit_tokens - self.cache_hit_external_tokens
            )
            METRICS.cache_hits(device_tokens, tier="g0")
            if self.cache_hit_external_tokens > 0 or self.dkv_total_clients > 0:
                METRICS.cache_hits(
                    self.cache_hit_external_tokens, tier="external"
                )
            METRICS.cache_misses(self.cache_miss_tokens)
            METRICS.cache_device_blocks_served(self.device_blocks_served)
            for coverage in self.per_request_prefix_coverage:
                METRICS.cache_request_prefix_coverage(coverage * 100)

        if self.total_host_kv_bytes != 0:
            METRICS.cache_used_host_kv_pct(self.used_host_kv_pct * 100)
            METRICS.cache_h2d_bytes_copied(self.h2d_bytes_copied)
            METRICS.cache_d2h_bytes_copied(self.d2h_bytes_copied)

        if self.total_disk_kv_bytes != 0:
            METRICS.cache_used_disk_kv_pct(self.used_disk_kv_pct * 100)
            METRICS.cache_disk_bytes_read(self.disk_bytes_read)
            METRICS.cache_disk_bytes_written(self.disk_bytes_written)

        if self.nixl_read_latency_avg_ms > 0:
            METRICS.dkv_nixl_read_latency(self.nixl_read_latency_avg_ms)
            METRICS.dkv_nixl_read_gib_per_s(self.nixl_read_gib_per_s)
        if self.nixl_write_latency_avg_ms > 0:
            METRICS.dkv_nixl_write_latency(self.nixl_write_latency_avg_ms)
            METRICS.dkv_nixl_write_gib_per_s(self.nixl_write_gib_per_s)
        if self.rpc_acquire_latency_avg_ms > 0:
            METRICS.dkv_rpc_acquire_latency(self.rpc_acquire_latency_avg_ms)
        if self.rpc_read_latency_avg_ms > 0:
            METRICS.dkv_rpc_read_latency(self.rpc_read_latency_avg_ms)
        # Its own guard, not the cache-hit clause: this is a per-window delta
        # that ``reset_metrics`` clears after every batch, so a window published
        # under a different batch type would lose its count for good. Keyed on a
        # tier being attached rather than on it moving blocks, so a dead tier
        # reads a flat zero instead of nothing.
        if self.dkv_read_blocks > 0 or self.dkv_total_clients > 0:
            METRICS.dkv_read_blocks(self.dkv_read_blocks)

        # Publish dKV health whenever a dKV tier is attached, independent of
        # transfer activity, because a dead tier does no transfers and yet is
        # exactly the state an operator needs to alert on.
        if self.dkv_total_clients > 0:
            METRICS.dkv_connected_clients(self.dkv_connected_clients)
            METRICS.dkv_total_clients(self.dkv_total_clients)
            METRICS.dkv_reconnect_attempts(self.dkv_reconnect_attempts)

        if self.draft_tokens_generated > 0:
            METRICS.spec_decode_avg_acceptance_length(
                self.avg_acceptance_length
            )

        # Emit per-position acceptance rate metrics for speculative decoding
        for position, rate in enumerate(self.acceptance_rate_per_position):
            METRICS.spec_decode_acceptance_rate_per_position(
                position=position,
                acceptance_rate=rate * 100,  # Convert to percentage
            )

        vm = self.vision_metrics
        if vm is not None and vm.num_images_total > 0:
            METRICS.vision_images_encoded(vm.num_images_encoded)
            METRICS.vision_images_cached(vm.num_images_cached)
            METRICS.vision_patches_encoded(vm.num_patches_encoded)
            METRICS.vision_tokens_encoded(vm.num_tokens_encoded)
            METRICS.vision_cache_hit_rate(vm.cache_hit_rate * 100)

        vid = self.video_metrics
        if vid is not None and vid.num_clips_total > 0:
            METRICS.video_clips_encoded(vid.num_clips_encoded)
            METRICS.video_tokens_encoded(vid.num_tokens_encoded)
            METRICS.video_encoding_time_milliseconds(vid.encoding_time_ms)
            for frame_count in vid.frame_counts:
                METRICS.video_frames_per_clip(frame_count)


def publish_completed_batch_metrics(
    stats: CompletedBatchStats, terminated_reqs: int
) -> None:
    """Publishes execution-time, throughput and termination telemetry for a
    completed batch.

    Used with the overlap pipeline, where a batch's completion is observed one
    scheduler iteration after it was enqueued: these metrics must be labeled
    with the completed batch's type and computed from that same batch's token
    counts, not the current iteration's.

    Args:
        stats: Stats for the batch whose outputs were synchronized this
            iteration.
        terminated_reqs: Requests the scheduler released this iteration.
            Derived by the scheduler rather than the pipeline, but it
            describes ``stats``' batch.
    """
    if stats.execution_time_s <= 0.0:
        return
    bt = stats.batch_type.value
    METRICS.batch_terminated_reqs(terminated_reqs, batch_type=bt)
    prompt_throughput = stats.num_input_tokens / stats.execution_time_s
    if stats.num_output_tokens is not None and stats.batch_type == BatchType.TG:
        generation_throughput = stats.num_output_tokens / stats.execution_time_s
    else:
        generation_throughput = stats.batch_size / stats.execution_time_s
    METRICS.batch_prompt_throughput(prompt_throughput, batch_type=bt)
    METRICS.batch_generation_throughput(generation_throughput, batch_type=bt)
    METRICS.batch_execution_time(stats.execution_time_s * 1000, batch_type=bt)
    if stats.early_sync_duration_s is not None:
        METRICS.di_early_sync_time(stats.early_sync_duration_s * 1000)


class SchedulerLogger:
    """Class to periodically log batch-level metrics to console."""

    def __init__(self, log_interval_s: float | None = None):
        """Initializes the SchedulerLogger.

        Args:
            log_interval_s: How frequently to log CE and TG batches, in seconds.
        """

        if log_interval_s is None:
            log_interval_s = float(
                os.getenv("MAX_SERVE_SCHEDULER_STATS_LOG_INTERVAL_S", "3")
            )
        logger.debug(
            f"Enabled scheduler batch statistic logging at interval of {log_interval_s:.2f}s"
        )

        # How frequently to log CE and TG batches.
        # We restrict logs to at most once every few seconds to avoid spam.
        self.log_interval_s = log_interval_s

        # The last time we last logged a CE or TG batch.
        self.time_of_last_log = 0.0

    def log_metrics(
        self,
        sch_config: TokenGenerationSchedulerConfig,
        inputs: TextGenerationInputs[TextContext],
        kv_cache: PagedKVCacheManagerInterface | None,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
        num_pending_reqs: int,
        num_terminated_reqs: int,
        total_preemption_count: int,
        batch_spec_decode_metrics: _SpeculativeDecodingMetrics | None = None,
        batch_vision_metrics: VisionEncoderMetrics | None = None,
        batch_video_metrics: VideoEncoderMetrics | None = None,
        overlap_active: bool = False,
        completed_batch_stats: CompletedBatchStats | None = None,
    ) -> None:
        """Periodically logs batch-level metrics to console.

        Args:
            sch_config: The scheduler configuration.
            inputs: The pipeline input / batch.
            kv_cache: The PagedKVCacheManagerInterface, if any.
            batch_creation_time_s: The time it took to create the batch.
            batch_execution_time_s: The time it took to execute the batch.
            num_pending_reqs: The number of pending requests.
            total_preemption_count: The total number of preemptions.
            batch_spec_decode_metrics: Per-batch speculative decoding metrics
                for the most recent batch.
            batch_vision_metrics: Per-batch vision encoder metrics for the
                most recent batch, or None when no vision encoding ran.
            batch_video_metrics: Per-batch video encoder metrics for the
                most recent batch, or None when no video encoding ran.
            overlap_active: When True, the overlap scheduler is active:
                ``batch_execution_time_s`` measured this iteration belongs to
                the previously enqueued batch, so execution-time, throughput
                and termination telemetry for ``inputs`` is suppressed in
                favor of ``completed_batch_stats``, and the log line describes
                the enqueued and completed batches as separate clauses.
            completed_batch_stats: Stats for the batch whose outputs were
                synchronized this iteration (overlap scheduling), used with
                ``num_terminated_reqs`` to publish execution-time, throughput
                and termination telemetry and to render the log line's
                ``Completed`` clause. ``None`` when no batch completed this
                iteration or overlap is inactive.

        Returns:
            None
        """
        # Compute the batch level metrics.
        metrics = BatchMetrics.create(
            sch_config=sch_config,
            inputs=inputs,
            kv_cache=kv_cache,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            num_pending_reqs=num_pending_reqs,
            num_terminated_reqs=num_terminated_reqs,
            total_preemption_count=total_preemption_count,
            batch_spec_decode_metrics=batch_spec_decode_metrics,
            batch_vision_metrics=batch_vision_metrics,
            batch_video_metrics=batch_video_metrics,
            overlap_active=overlap_active,
            completed_batch_stats=completed_batch_stats,
        )

        # Always publish metrics. Under overlap scheduling the wall-clock
        # execution time measured this iteration belongs to the previously
        # enqueued batch, so the execution-time and throughput metrics are
        # published from ``completed_batch_stats`` (labeled with the completed
        # batch's type) instead of from ``inputs``. Wrap both emitters in one
        # transaction so the whole per-iteration burst — including the deferred
        # overlap metrics — coalesces into a single cross-process packet (the
        # inner transaction opened by ``publish_metrics`` nests harmlessly).
        with METRICS.transaction():
            metrics.publish_metrics(defer_execution_metrics=overlap_active)
            if completed_batch_stats is not None:
                publish_completed_batch_metrics(
                    completed_batch_stats, num_terminated_reqs
                )

        # Only periodically log batch info to the console to avoid log spam.
        now = time.monotonic()
        time_since_last_log = now - self.time_of_last_log
        if self.log_interval_s < time_since_last_log:
            # Reset the time of the last log.
            self.time_of_last_log = now
            logger.info(metrics.pretty_format(), extra=metrics.to_log_extra())


def get_cancelled_reqs(
    cancel_q: MAXPullQueue[list[RequestID]],
) -> list[RequestID]:
    """Drains the cancel queue and returns all cancelled request IDs.

    Args:
        cancel_q: The queue containing lists of cancelled request IDs.

    Returns:
        A list of all cancelled request IDs.
    """
    cancelled_reqs = []
    for req_ids in drain_queue(cancel_q):
        for req_id in req_ids:
            cancelled_reqs.append(req_id)
    return cancelled_reqs


def reshape_flat_kv_blocks_to_grid(
    flat_blocks: list[Buffer], dp: int, group_name: str
) -> list[list[Buffer]]:
    """Reshape a flat per-device buffer list into ``[dp][tp]`` row-major.

    Matches the primary tensor grid that ``KVTransferEngine`` expects
    when registering an extra tensor group. ``flat_blocks`` is
    ``[r0t0, r0t1, ..., r0t(tp-1), r1t0, ...]`` as produced by
    ``PagedKVCacheManager.runtime_inputs``.
    """
    tp = len(flat_blocks) // dp
    if dp * tp != len(flat_blocks):
        raise ValueError(
            f"{group_name} KV tensor group has {len(flat_blocks)} "
            f"buffers, not divisible by DP={dp}."
        )
    return [list(flat_blocks[r * tp : (r + 1) * tp]) for r in range(dp)]
