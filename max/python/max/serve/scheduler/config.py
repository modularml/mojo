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

from dataclasses import dataclass

from max.pipelines.lib import MemoryPlan, PipelineConfig


@dataclass
class TokenGenerationSchedulerConfig:
    """Scheduler configuration."""

    max_batch_size: int
    """The maximum number of requests that can be in the token generation batch."""

    """The number of tokens to generate for each request in the token generation iteration."""

    target_tokens_per_batch_ce: int
    """The target total number of tokens to encode in the context encoding batch."""

    max_seq_len: int | None = None
    """The maximum sequence length of the model."""

    max_batch_total_tokens: int | None = None
    """Ensures the sum of page-aligned context lengths in a batch does not
    exceed max_batch_total_tokens. Alignment uses the KV cache page size."""

    enable_chunked_prefill: bool = True
    """Enables chunked prefill, where the scheduler splits requests into chunks to ensure
    each batch contains exactly `target_tokens_per_batch_ce` tokens."""

    chunked_prefill_min_chunk_size: int = 0
    """Floor, in tokens, on any chunk created by chunked prefill: a split
    never creates a piece (chunk or remainder) smaller than this; contexts
    with no legal cut point within the remaining budget are left unsplit for
    a later step. 0 disables the floor."""

    enable_in_flight_batching: bool = False
    """When enabled, prioritizes token generation by batching it with context encoding requests."""

    data_parallel_degree: int = 1
    """Data-parallelism parameter. The degree to which the model is replicated
    is dependent on the model type."""

    num_speculative_tokens: int = 0
    """The number of speculative tokens to generate per step.

    If speculative decoding is disabled, this should be 0.
    """

    decode_stall_timeout_s: float | None = None
    """Seconds of no-batch-activity after which the decode worker exits to trigger a pod restart. None disables the watchdog."""

    decode_request_ttl_s: float | None = None
    """Per-request TTL (seconds) for entries in the decode scheduler's
    ``prefill_reqs`` and ``inflight_transfers`` dicts. Expired entries are
    evicted individually (KV blocks released, failure surfaced to the
    client) so a stuck PD pipeline does not force the stall watchdog to
    kill the whole engine. ``None`` disables eviction."""

    dp_ce_balance_timeout_ms: float = -1.0
    """Max time (ms) a CE request's work may be deferred, from arrival,
    while awaiting token-balanced scheduling across DP replicas. -1 disables
    the balancer entirely (requests bind to a replica on arrival; current
    behavior). 0 enables post-cache-weighted placement with late binding but
    never defers work. > 0 additionally defers unbalanced CE work (unbound
    requests and mid-prefill tails) until ``dp_ce_balance_threshold`` is
    met, the deadline expires, or there is nothing else to run."""

    dp_ce_balance_threshold: float = 0.8
    """Per-step CE active-token occupancy across DP replicas (mean/max,
    0-1) at or above which CE work is scheduled without further deferral.
    Only consulted when ``dp_ce_balance_timeout_ms`` > 0."""

    dp_ce_balance_enable_dynamic_chunk_size: bool = False
    """Whether a below-threshold CE step with work on 2+ replicas runs
    immediately with each replica's chunk size reduced to the balance level
    (deferring only the excess). When False, such steps are held whole until
    the threshold is met, a deadline expires, or there is nothing else to
    run."""

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError(
                f"`max_batch_size` must be greater than 0, found {self.max_batch_size}"
            )
        if self.target_tokens_per_batch_ce <= 0:
            raise ValueError(
                f"`target_tokens_per_batch_ce` must be greater than 0, found {self.target_tokens_per_batch_ce}"
            )
        if (
            self.enable_chunked_prefill
            and self.target_tokens_per_batch_ce is None
        ):
            raise ValueError(
                "Need set `target_tokens_per_batch_ce` for the scheduler to enable chunked prefill."
            )
        if self.chunked_prefill_min_chunk_size < 0:
            raise ValueError(
                "`chunked_prefill_min_chunk_size` must be non-negative, found"
                f" {self.chunked_prefill_min_chunk_size}"
            )
        if (
            self.max_batch_total_tokens is not None
            and self.max_seq_len is not None
            and self.max_batch_total_tokens < self.max_seq_len
        ):
            raise ValueError(
                f"`max_batch_total_tokens` must be greater than or equal to `max_seq_len`, found {self.max_batch_total_tokens} < {self.max_seq_len}"
            )
        if self.max_batch_size > self.target_tokens_per_batch_ce:
            raise ValueError(
                f"`max_batch_size` must be less than or equal to `target_tokens_per_batch_ce`, found {self.max_batch_size} > {self.target_tokens_per_batch_ce}"
            )
        if not 0.0 <= self.dp_ce_balance_threshold <= 1.0:
            raise ValueError(
                "`dp_ce_balance_threshold` must be in [0, 1], found"
                f" {self.dp_ce_balance_threshold}"
            )

    @classmethod
    def from_pipeline_config(
        cls,
        pipeline_config: PipelineConfig,
        max_batch_size: int,
        memory_plan: MemoryPlan | None,
    ) -> TokenGenerationSchedulerConfig:
        """Builds the scheduler config from the pipeline config and memory plan.

        ``memory_plan`` carries the planned sequence length and batch token
        budget; ``None`` only for pipelines sized without a plan (test echoes).
        """
        assert pipeline_config.model is not None

        return cls(
            max_batch_size=max_batch_size,
            target_tokens_per_batch_ce=pipeline_config.runtime.max_batch_input_tokens,
            max_seq_len=(
                memory_plan.planned_max_length
                if memory_plan is not None
                else None
            ),
            max_batch_total_tokens=(
                memory_plan.planned_max_batch_total_tokens
                if memory_plan is not None
                else None
            ),
            enable_chunked_prefill=pipeline_config.runtime.enable_chunked_prefill,
            chunked_prefill_min_chunk_size=pipeline_config.runtime.chunked_prefill_min_chunk_size,
            enable_in_flight_batching=pipeline_config.runtime.enable_in_flight_batching,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            decode_stall_timeout_s=pipeline_config.runtime.decode_stall_timeout_s,
            decode_request_ttl_s=pipeline_config.runtime.decode_request_ttl_s,
            dp_ce_balance_timeout_ms=pipeline_config.runtime.dp_ce_balance_timeout_ms,
            dp_ce_balance_threshold=pipeline_config.runtime.dp_ce_balance_threshold,
            dp_ce_balance_enable_dynamic_chunk_size=pipeline_config.runtime.dp_ce_balance_enable_dynamic_chunk_size,
            num_speculative_tokens=(
                pipeline_config.speculative.num_speculative_tokens or 0
            )
            if pipeline_config.speculative is not None
            else 0,
        )
