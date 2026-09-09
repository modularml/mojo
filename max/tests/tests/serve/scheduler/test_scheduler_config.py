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

from unittest.mock import MagicMock

import pytest
from max.pipelines.lib import MemoryPlan
from max.serve.scheduler import TokenGenerationSchedulerConfig


def test_scheduler_max_batch_size_less_than_target_tokens_per_batch_ce() -> (
    None
):
    # ok
    TokenGenerationSchedulerConfig(
        max_batch_size=100,
        target_tokens_per_batch_ce=100,
    )

    # not ok because max_batch_size > target_tokens_per_batch_ce
    with pytest.raises(
        ValueError,
        match=r"`max_batch_size` must be less than or equal to `target_tokens_per_batch_ce`, found 101 > 100",
    ):
        TokenGenerationSchedulerConfig(
            max_batch_size=101,
            target_tokens_per_batch_ce=100,
        )


def test_from_pipeline_config_reads_the_memory_plan() -> None:
    """The planned sequence length and batch token budget come from the
    memory plan, not from the (possibly divergent) config fields."""
    pipeline_config = MagicMock()
    pipeline_config.runtime.max_batch_input_tokens = 8192
    pipeline_config.runtime.max_batch_total_tokens = 1
    pipeline_config.runtime.enable_chunked_prefill = True
    pipeline_config.runtime.chunked_prefill_min_chunk_size = 0
    pipeline_config.runtime.enable_in_flight_batching = False
    pipeline_config.runtime.dp_ce_balance_threshold = 0.8
    pipeline_config.runtime.decode_stall_timeout_s = None
    pipeline_config.runtime.decode_request_ttl_s = None
    pipeline_config.model.max_length = 1
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.speculative = None
    memory_plan = MemoryPlan(
        planned_max_batch_size=1,
        footprint=0,
        planned_max_length=2048,
        planned_max_batch_total_tokens=8192,
    )

    config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config, max_batch_size=1, memory_plan=memory_plan
    )

    assert config.max_seq_len == 2048
    assert config.max_batch_total_tokens == 8192


def test_from_pipeline_config_without_a_memory_plan() -> None:
    """``None`` (pipelines sized without a plan) leaves both bounds unset."""
    pipeline_config = MagicMock()
    pipeline_config.runtime.max_batch_input_tokens = 8192
    pipeline_config.runtime.enable_chunked_prefill = True
    pipeline_config.runtime.chunked_prefill_min_chunk_size = 0
    pipeline_config.runtime.enable_in_flight_batching = False
    pipeline_config.runtime.dp_ce_balance_threshold = 0.8
    pipeline_config.runtime.decode_stall_timeout_s = None
    pipeline_config.runtime.decode_request_ttl_s = None
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.speculative = None

    config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config, max_batch_size=1, memory_plan=None
    )

    assert config.max_seq_len is None
    assert config.max_batch_total_tokens is None
