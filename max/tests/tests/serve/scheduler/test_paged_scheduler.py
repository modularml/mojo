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

import numpy as np
import pytest
from max.driver import CPU
from max.nn.kv_cache import KVConnectorType
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.config import KVConnectorConfig
from max.pipelines.modeling.types import BatchType
from max.serve.scheduler.batch_constructor.text_batch_constructor import (
    TG_PRIORITY_KV_PERCENTAGE_ENV_VAR,
)
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
)
from max.support.math import ceildiv
from tests.serve.scheduler.common import (
    CE,
    TG,
    BatchInfo,
    assert_batch_info_equal,
    create_batch_and_execute,
    create_paged_scheduler,
    enqueue_request,
    rand,
    run_until_completion,
)


def test_paged_scheduler_tg_request_exceed_max_seq_len() -> None:
    num_reqs = 3
    max_seq_len = 2048
    page_size = 128
    num_blocks = int(max_seq_len / page_size * num_reqs)
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=max_seq_len,
        max_batch_size=100,
        num_blocks=num_blocks,
        page_size=page_size,
    )

    # Check that we would exceed max_seq_len during TG step
    prompt_len = 2045

    # Create a few requests with 2040 tokens
    for _ in range(num_reqs):
        enqueue_request(request_queue, prompt_len, max_seq_len=max_seq_len)

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=6135, cached_toks=0),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=6135),
        BatchInfo(TG, batch_size=3, terminated=3, steps=1, preempted=0, input_toks=3, cached_toks=6138),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on

    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def create_scheduler_under_kv_pressure() -> TokenGenerationScheduler:
    """Prefill one long request so the device cache sits above 90% used.

    The returned scheduler holds one in-flight TG request and one pending CE
    request, which is the tie ``_identify_priority`` has to break.
    """
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=100,
        num_blocks=100,
        page_size=1,
        enable_chunked_prefill=False,
    )

    enqueue_request(request_queue, prompt_len=92, max_seq_len=95)
    assert create_batch_and_execute(scheduler).batch_type == CE

    kv_cache = scheduler.batch_constructor.kv_cache
    assert kv_cache.block_count().used_pct > 90

    enqueue_request(request_queue, prompt_len=2, max_seq_len=4)
    return scheduler


def test_paged_scheduler_prioritizes_tg_under_kv_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TG_PRIORITY_KV_PERCENTAGE_ENV_VAR, raising=False)
    scheduler = create_scheduler_under_kv_pressure()

    assert create_batch_and_execute(scheduler).batch_type == TG


def test_paged_scheduler_tg_priority_kv_percentage_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A threshold above 100% is unreachable, restoring CE-first scheduling."""
    monkeypatch.setenv(TG_PRIORITY_KV_PERCENTAGE_ENV_VAR, "101")
    scheduler = create_scheduler_under_kv_pressure()

    assert create_batch_and_execute(scheduler).batch_type == CE


def test_paged_scheduler_basic_chunked_prefill() -> None:
    max_seq_len = 99999  # unbounded length
    target_tokens_per_batch_ce = 1000
    page_size = 128
    prompt_len = 9123
    output_tokens = 5
    num_blocks = ceildiv(prompt_len + output_tokens, page_size)
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=max_seq_len,
        num_blocks=num_blocks,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        page_size=page_size,
        enable_chunked_prefill=True,
    )

    enqueue_request(
        request_queue,
        prompt_len=prompt_len,
        max_seq_len=prompt_len + output_tokens,
    )

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=0),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=1000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=2000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=3000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=4000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=5000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=6000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=7000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=8000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=123, cached_toks=9000),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=9123),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=9124),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=9125),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=9126),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_max_pending_requests_caps_drain() -> None:
    """The M cap (max_pending_requests) bounds how many requests the scheduler
    pulls into its pending CE queue, leaving the rest in the request queue so it
    backs up and exerts backpressure."""
    scheduler, request_queue = create_paged_scheduler(
        max_batch_size=999,
        max_pending_requests=2,
    )

    for _ in range(5):
        enqueue_request(request_queue, prompt_len=10, max_seq_len=15)

    # First drain pulls at most M=2 into the pending CE queue; the other 3 stay
    # queued in the request queue.
    scheduler._retrieve_pending_requests()
    assert len(scheduler.batch_constructor.all_ce_reqs) == 2

    # Still at capacity: a second drain pulls nothing more.
    scheduler._retrieve_pending_requests()
    assert len(scheduler.batch_constructor.all_ce_reqs) == 2

    # Raising the cap lets the 3 still-queued requests drain in, proving they
    # had been held in the request queue rather than dropped.
    scheduler.max_pending_requests = 5
    scheduler._retrieve_pending_requests()
    assert len(scheduler.batch_constructor.all_ce_reqs) == 5


def test_basic_ce_scheduling() -> None:
    num_prompts = 3
    prompt_len = 10
    output_tokens = 5
    page_size = 15
    num_blocks = 3  # Budget of 45 tokens total
    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        num_blocks=num_blocks,
        max_batch_size=999,
        page_size=page_size,
    )

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=30, cached_toks=0),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=30),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=33),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=36),
        BatchInfo(TG, batch_size=3, terminated=3, steps=1, preempted=0, input_toks=3, cached_toks=39),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_basic_small_batch_size() -> None:
    prompt_len = 100
    output_tokens = 3
    max_batch_size = 13
    num_requests = 40
    scheduler, request_queue = create_paged_scheduler(
        max_batch_size=max_batch_size,
    )

    for _ in range(num_requests):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=13, terminated=0, steps=1, preempted=0, input_toks=1300, cached_toks=0),
        BatchInfo(TG, batch_size=13, terminated=0, steps=1, preempted=0, input_toks=13, cached_toks=1300),
        BatchInfo(TG, batch_size=13, terminated=13, steps=1, preempted=0, input_toks=13, cached_toks=1313),
        BatchInfo(CE, batch_size=13, terminated=0, steps=1, preempted=0, input_toks=1300, cached_toks=0),
        BatchInfo(TG, batch_size=13, terminated=0, steps=1, preempted=0, input_toks=13, cached_toks=1300),
        BatchInfo(TG, batch_size=13, terminated=13, steps=1, preempted=0, input_toks=13, cached_toks=1313),
        BatchInfo(CE, batch_size=13, terminated=0, steps=1, preempted=0, input_toks=1300, cached_toks=0),
        BatchInfo(TG, batch_size=13, terminated=0, steps=1, preempted=0, input_toks=13, cached_toks=1300),
        BatchInfo(TG, batch_size=13, terminated=13, steps=1, preempted=0, input_toks=13, cached_toks=1313),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=0),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=100),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=101),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_basic_small_batch_size_with_chunked_prefill() -> None:
    prompt_len = 1500
    output_tokens = 2
    max_batch_size = 13
    num_requests = 40
    scheduler, request_queue = create_paged_scheduler(
        max_batch_size=max_batch_size,
        enable_chunked_prefill=True,
    )

    for _ in range(num_requests):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=692),
        BatchInfo(CE, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3116, cached_toks=1384),
        BatchInfo(TG, batch_size=13, terminated=13, steps=1, preempted=0, input_toks=13, cached_toks=19500),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=692),
        BatchInfo(CE, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3116, cached_toks=1384),
        BatchInfo(TG, batch_size=13, terminated=13, steps=1, preempted=0, input_toks=13, cached_toks=19500),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=692),
        BatchInfo(CE, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3116, cached_toks=1384),
        BatchInfo(TG, batch_size=13, terminated=13, steps=1, preempted=0, input_toks=13, cached_toks=19500),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1500, cached_toks=0),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=1500),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16() -> (
    None
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16

    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
    )

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # We will schedule 8192 / 500 = 16.38 CE req per batch due to target_tokens_per_batch_ce.
    # This is rounded up to 17 due to chunked prefill.
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=192),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=384),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=76),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=268),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=460),
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=848, cached_toks=152),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50000),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50100),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50200),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50300),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50400),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50500),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50600),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50700),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50800),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50900),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51000),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51100),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51200),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51300),
        BatchInfo(TG, batch_size=100, terminated=100, steps=1, preempted=0, input_toks=100, cached_toks=51400),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16_prefix_len_384() -> (
    None
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 384

    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    # We predict approx 384 tokens to be cache hit.
    # This means we encode 500 - 384 = 116 tokens per CE batch.
    # Hence, we will schedule approx 8192 / 116 = 70.62 CE req per batch.
    # This is rounded up to 71 due to chunked prefill.
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=71, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=27264),
        BatchInfo(CE, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=1552, cached_toks=5448),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50000),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50100),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50200),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50300),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50400),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50500),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50600),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50700),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50800),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50900),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51000),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51100),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51200),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51300),
        BatchInfo(TG, batch_size=100, terminated=100, steps=1, preempted=0, input_toks=100, cached_toks=51400),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16_prefix_len_200() -> (
    None
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 200

    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    # We predict 200 tokens to be cache hit.
    # This means we encode 500 - 200 = 300 tokens per CE request.
    # Hence, we will schedule approx 8192 / 300 = 27.31 CE req per batch.
    # This is rounded up to 28 due to chunked prefill.
    # The first batch doesn't get cache hits so it is smaller.
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=23, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=3008),
        BatchInfo(CE, batch_size=23, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=3016),
        BatchInfo(CE, batch_size=23, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=3024),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=6608, cached_toks=2392),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50000),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50100),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50200),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50300),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50400),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50500),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50600),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50700),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50800),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50900),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51000),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51100),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51200),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51300),
        BatchInfo(TG, batch_size=100, terminated=100, steps=1, preempted=0, input_toks=100, cached_toks=51400),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16_prefix_len_64() -> (
    None
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 64

    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    # We predict 64 tokens to be cache hit.
    # This means we encode 500 - 64 = 436 tokens per CE request.
    # Hence, we will schedule approx 8192 / 436 = 18.79 CE req per batch.
    # This is rounded up to 19 due to chunked prefill.
    # fmt: off
    expected = [
       BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=192),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=384),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=76),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=268),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=460),
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=848, cached_toks=152),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50000),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50100),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50200),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50300),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50400),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50500),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50600),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50700),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50800),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50900),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51000),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51100),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51200),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51300),
        BatchInfo(TG, batch_size=100, terminated=100, steps=1, preempted=0, input_toks=100, cached_toks=51400),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler__num_prompts_10_prompt_len_100_output_tokens_100_prefix_len_64_low_mem_basic() -> (
    None
):
    num_prompts = 10
    prompt_len = 100
    output_tokens = 100
    prefix_len = 64

    page_size = 10
    num_blocks = 50  # this is enough for 500 tokens

    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=num_blocks * page_size,
        page_size=page_size,
        num_blocks=num_blocks,
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        enable_prefix_caching=False,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=500, cached_toks=0),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=1, input_toks=4, cached_toks=400),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=404),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=408),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=412),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=416),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=420),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=424),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=428),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=432),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=436),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=440),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=444),
    ]
    # fmt: on
    actual = run_until_completion(scheduler, max_num_iters=len(expected))
    assert_batch_info_equal(actual, expected)


def test_num_prompts_10_prompt_len_100_output_tokens_100_prefix_len_64_low_mem_prefix_caching() -> (
    None
):
    num_prompts = 10
    prompt_len = 100
    output_tokens = 100
    prefix_len = 64

    page_size = 10
    num_blocks = 50  # this is enough for 500 tokens

    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=num_blocks * page_size,
        page_size=page_size,
        num_blocks=num_blocks,
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=500, cached_toks=0),
        BatchInfo(CE, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=200, cached_toks=300),
        BatchInfo(TG, batch_size=8, terminated=0, steps=1, preempted=1, input_toks=8, cached_toks=800),
        BatchInfo(TG, batch_size=8, terminated=0, steps=1, preempted=0, input_toks=8, cached_toks=808),
        BatchInfo(TG, batch_size=8, terminated=0, steps=1, preempted=0, input_toks=8, cached_toks=816),
        BatchInfo(TG, batch_size=8, terminated=0, steps=1, preempted=0, input_toks=8, cached_toks=824),
        BatchInfo(TG, batch_size=8, terminated=0, steps=1, preempted=0, input_toks=8, cached_toks=832),
        BatchInfo(TG, batch_size=8, terminated=0, steps=1, preempted=0, input_toks=8, cached_toks=840),
        BatchInfo(TG, batch_size=8, terminated=0, steps=1, preempted=0, input_toks=8, cached_toks=848),
        BatchInfo(TG, batch_size=8, terminated=0, steps=1, preempted=0, input_toks=8, cached_toks=856),
        BatchInfo(TG, batch_size=8, terminated=0, steps=1, preempted=0, input_toks=8, cached_toks=864),
        BatchInfo(TG, batch_size=8, terminated=0, steps=1, preempted=0, input_toks=8, cached_toks=872),
        BatchInfo(TG, batch_size=7, terminated=0, steps=1, preempted=2, input_toks=7, cached_toks=770),
        BatchInfo(TG, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=7, cached_toks=777),
        BatchInfo(TG, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=7, cached_toks=784),
        BatchInfo(TG, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=7, cached_toks=791),
        BatchInfo(TG, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=7, cached_toks=798),
        BatchInfo(TG, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=7, cached_toks=805),
        BatchInfo(TG, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=7, cached_toks=812),
        BatchInfo(TG, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=7, cached_toks=819),
        BatchInfo(TG, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=7, cached_toks=826),
        BatchInfo(TG, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=7, cached_toks=833),
        BatchInfo(TG, batch_size=6, terminated=0, steps=1, preempted=1, input_toks=6, cached_toks=720),
        BatchInfo(TG, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=6, cached_toks=726),
        BatchInfo(TG, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=6, cached_toks=732),
        BatchInfo(TG, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=6, cached_toks=738),
        BatchInfo(TG, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=6, cached_toks=744),
        BatchInfo(TG, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=6, cached_toks=750),
        BatchInfo(TG, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=6, cached_toks=756),
        BatchInfo(TG, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=6, cached_toks=762),
        BatchInfo(TG, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=6, cached_toks=768),
        BatchInfo(TG, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=6, cached_toks=774),
        BatchInfo(TG, batch_size=5, terminated=0, steps=1, preempted=1, input_toks=5, cached_toks=650),
        BatchInfo(TG, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=5, cached_toks=655),
        BatchInfo(TG, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=5, cached_toks=660),
        BatchInfo(TG, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=5, cached_toks=665),
        BatchInfo(TG, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=5, cached_toks=670),
        BatchInfo(TG, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=5, cached_toks=675),
        BatchInfo(TG, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=5, cached_toks=680),
        BatchInfo(TG, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=5, cached_toks=685),
        BatchInfo(TG, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=5, cached_toks=690),
        BatchInfo(TG, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=5, cached_toks=695),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=560),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=564),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=568),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=572),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=576),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=580),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=584),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=588),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16_in_flight_batching() -> (
    None
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16

    scheduler, request_queue = create_paged_scheduler(
        enable_in_flight_batching=True,
    )

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # With inflight batching, the CE batches become bigger and bigger since they
    # now include TG requests.
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=33, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=8192),
        BatchInfo(CE, batch_size=50, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=16384),
        BatchInfo(CE, batch_size=66, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=24576),
        BatchInfo(CE, batch_size=82, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=32768),
        BatchInfo(CE, batch_size=98, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=40960),
        BatchInfo(CE, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=1188, cached_toks=49152),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50340),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50440),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50540),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50640),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50740),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50840),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=50940),
        BatchInfo(TG, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=51040),
        BatchInfo(TG, batch_size=100, terminated=16, steps=1, preempted=0, input_toks=100, cached_toks=51140),
        BatchInfo(TG, batch_size=84, terminated=16, steps=1, preempted=0, input_toks=84, cached_toks=43000),
        BatchInfo(TG, batch_size=68, terminated=17, steps=1, preempted=0, input_toks=68, cached_toks=34844),
        BatchInfo(TG, batch_size=51, terminated=16, steps=1, preempted=0, input_toks=51, cached_toks=26157),
        BatchInfo(TG, batch_size=35, terminated=16, steps=1, preempted=0, input_toks=35, cached_toks=17968),
        BatchInfo(TG, batch_size=19, terminated=16, steps=1, preempted=0, input_toks=19, cached_toks=9763),
        BatchInfo(TG, batch_size=3, terminated=3, steps=1, preempted=0, input_toks=3, cached_toks=1542),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_tg_preemption_basic() -> None:
    num_prompts = 2
    prompt_len = 3
    output_tokens = 7
    page_size = 2
    num_blocks = 5  # enough for 10 tokens or exactly 1 request
    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        num_blocks=num_blocks,
        max_batch_size=999,
        page_size=page_size,
        max_seq_len=110,
    )

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=6, cached_toks=0),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=6),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=4),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=5),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=1, input_toks=1, cached_toks=6),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=7),
        # Preempt the request
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=8),
        # Resume the preempted request
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=5, cached_toks=0),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=5),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=6),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=7),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=8),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_oom_ce() -> None:
    prompt_len = 200
    output_tokens = 1
    page_size = 10
    num_blocks = 10
    scheduler, request_queue = create_paged_scheduler(
        num_blocks=num_blocks,
        page_size=page_size,
    )

    enqueue_request(
        request_queue,
        prompt_len=prompt_len,
        max_seq_len=prompt_len + output_tokens,
    )

    actual: list[BatchInfo] = []
    with pytest.raises(InsufficientBlocksError) as e:
        run_until_completion(scheduler, output_list=actual)

    # The error message should be informative:
    assert (
        "Insufficient KV pages for a single request with 200 tokens.\n"
        "The KVCache has 10 pages with page size 10. This is only enough to support 100 tokens.\n"
        "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
        in str(e.value)
    )


def test_paged_scheduler_oom_tg() -> None:
    num_prompts = 2
    # one req is 100 tokens
    prompt_len = 10
    output_tokens = 990
    # this can hold 15 tokens, but is not enough for even 1 request
    page_size = 4
    num_blocks = 7
    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        num_blocks=num_blocks,
        max_batch_size=999,
        page_size=page_size,
    )

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    actual: list[BatchInfo] = []
    with pytest.raises(InsufficientBlocksError):
        run_until_completion(scheduler, output_list=actual)

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=20, cached_toks=0),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=20),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=22),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=12),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=13),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=14),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=15),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=1, input_toks=1, cached_toks=16),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=17),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=18),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=19),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=20),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=21),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=22),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=23),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=24),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=25),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=26),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=27),
    ]
    # fmt: on
    # The error message should be informative:
    # assert (
    #     "Insufficient KV pages for a single request with 101 tokens.\n"
    #     "The KVCache has 10 pages with page size 10. This is only enough to support 100 tokens.\n"
    #     "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
    #     in str(e.value)
    # )
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_max_batch_total_tokens_ce() -> None:
    max_batch_total_tokens = 1000
    prompt_len = 60
    page_size = 128
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=max_batch_total_tokens,
        max_batch_total_tokens=max_batch_total_tokens,
        target_tokens_per_batch_ce=max_batch_total_tokens,
        enable_chunked_prefill=True,
        page_size=page_size,
    )

    for _ in range(20):
        enqueue_request(request_queue, prompt_len=prompt_len, max_seq_len=63)

    actual = run_until_completion(scheduler)
    # fmt: off
    expected = [
        # CE batch is limited by the page-aligned total-context budget.
        BatchInfo(CE, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=420, cached_toks=0),
        BatchInfo(CE, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=420, cached_toks=0),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=360, cached_toks=0),
        BatchInfo(TG, batch_size=20, terminated=0, steps=1, preempted=0, input_toks=20, cached_toks=1200),
        BatchInfo(TG, batch_size=20, terminated=20, steps=1, preempted=0, input_toks=20, cached_toks=1220),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(actual, expected)
    for batch in actual:
        if batch.batch_type == BatchType.CE:
            aligned_len = ceildiv(prompt_len, page_size) * page_size
            assert batch.batch_size * aligned_len <= max_batch_total_tokens


def test_paged_scheduler_max_batch_total_tokens_tg() -> None:
    max_batch_total_tokens = 1000
    prompt_len = 30
    page_size = 128
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=max_batch_total_tokens,
        max_batch_total_tokens=max_batch_total_tokens,
        target_tokens_per_batch_ce=max_batch_total_tokens,
        enable_chunked_prefill=True,
        page_size=page_size,
    )

    for _ in range(30):
        enqueue_request(request_queue, prompt_len=prompt_len, max_seq_len=40)

    actual = run_until_completion(scheduler)
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=210, cached_toks=0),
        BatchInfo(CE, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=210, cached_toks=0),
        BatchInfo(CE, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=210, cached_toks=0),
        BatchInfo(CE, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=210, cached_toks=0),
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=60, cached_toks=0),
        BatchInfo(TG, batch_size=30, terminated=0, steps=1, preempted=0, input_toks=30, cached_toks=900),
        BatchInfo(TG, batch_size=30, terminated=0, steps=1, preempted=0, input_toks=30, cached_toks=930),
        BatchInfo(TG, batch_size=30, terminated=0, steps=1, preempted=0, input_toks=30, cached_toks=960),
        BatchInfo(TG, batch_size=30, terminated=0, steps=1, preempted=0, input_toks=30, cached_toks=990),
        BatchInfo(TG, batch_size=30, terminated=0, steps=1, preempted=0, input_toks=30, cached_toks=1020),
        BatchInfo(TG, batch_size=30, terminated=0, steps=1, preempted=0, input_toks=30, cached_toks=1050),
        BatchInfo(TG, batch_size=30, terminated=0, steps=1, preempted=0, input_toks=30, cached_toks=1080),
        BatchInfo(TG, batch_size=30, terminated=0, steps=1, preempted=0, input_toks=30, cached_toks=1110),
        BatchInfo(TG, batch_size=30, terminated=30, steps=1, preempted=0, input_toks=30, cached_toks=1140),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(actual, expected)
    for batch in actual:
        steps = batch.steps
        batch_size = batch.batch_size
        per_req_aligned_len = ceildiv(prompt_len + steps, page_size) * page_size
        if batch.batch_type == BatchType.CE:
            assert batch_size * per_req_aligned_len <= max_batch_total_tokens


def test_paged_scheduler_dp8() -> None:
    # Each replica has a max batch size of 4
    # Across all replicas the aggregate max batch size is 8 * 4 = 32
    scheduler, request_queue = create_paged_scheduler(dp=8, max_batch_size=4)

    for _ in range(50):
        enqueue_request(request_queue, prompt_len=12, max_seq_len=16)

    actual = run_until_completion(scheduler)
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=32, terminated=0, steps=1, preempted=0, input_toks=384, cached_toks=0),
        BatchInfo(TG, batch_size=32, terminated=0, steps=1, preempted=0, input_toks=32, cached_toks=384),
        BatchInfo(TG, batch_size=32, terminated=0, steps=1, preempted=0, input_toks=32, cached_toks=416),
        BatchInfo(TG, batch_size=32, terminated=32, steps=1, preempted=0, input_toks=32, cached_toks=448),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=216, cached_toks=0),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=216),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=234),
        BatchInfo(TG, batch_size=18, terminated=18, steps=1, preempted=0, input_toks=18, cached_toks=252),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_paging_to_host_on_cpu_raises() -> None:
    with pytest.raises(ValueError) as e:
        create_paged_scheduler(
            kv_connector_config=KVConnectorConfig(type=KVConnectorType.tiered),
            enable_prefix_caching=True,
            device=CPU(),
        )
    assert "KVCacheMemory is on the CPU; cannot offload" in str(e.value)


def test_paged_scheduler_speculative_tokens_allocates_extra_pages() -> None:
    """Verifies that num_speculative_tokens causes extra KV page allocation.

    With num_speculative_tokens=7, each alloc reserves extra pages to
    accommodate draft tokens. Under tight memory (6 pages of size 10 = 60
    token capacity), this causes a preemption that would not happen without
    speculative decoding.
    """
    prompt_len = 10
    output_tokens = 10
    page_size = 10
    num_blocks = 6
    num_speculative_tokens = 7
    max_seq_len = prompt_len + output_tokens

    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=max_seq_len,
        num_blocks=num_blocks,
        page_size=page_size,
        num_speculative_tokens=num_speculative_tokens,
    )

    for _ in range(3):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=max_seq_len,
        )

    # fmt: off
    expected = [
        # CE: seq_len = 10 + 2*7 + 1 - 1 = 24 -> 3 pages/req. Only 2 fit in 6 pages.
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=20, cached_toks=0),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=20),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=22),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=24),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=26),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=28),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=30),
        # TG needs more pages, preempt 1 of the 2 encoded requests.
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=1, input_toks=1, cached_toks=16),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=17),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=18),
        # Encode the preempted request (7 toks) + the waiting request (10 toks).
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=17, cached_toks=0),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=17),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=18),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=10, cached_toks=0),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=10),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=11),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=12),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=13),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=14),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=15),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=16),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=17),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=18),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on

    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


@pytest.mark.parametrize(
    "num_prompts, input_tokens, output_tokens, target_tokens_per_batch_ce, enable_chunked_prefill, enable_prefix_caching",
    [
        (1, 1, 1, 1, True, True),
        (1, 60, 95, 30, True, False),
        (2, 511, 1, 500, False, True),
        (2, 512, 1, 1000, False, False),
        (30, 256, 16, 33, False, True),
        (30, 256, 16, 33, True, True),
        (100, 256, 1024, 8192, True, False),
    ],
)
def test_paged_scheduler_misc_sch_configs(
    num_prompts: int,
    input_tokens: int,
    output_tokens: int,
    target_tokens_per_batch_ce: int,
    enable_chunked_prefill: bool,
    enable_prefix_caching: bool,
) -> None:
    max_seq_len = input_tokens + output_tokens
    page_size = 128
    num_blocks = ceildiv(max_seq_len, page_size) * max(16, num_prompts)
    max_batch_size = ceildiv(num_prompts, 3)
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=max_seq_len,
        page_size=page_size,
        max_batch_size=max_batch_size,
        num_blocks=num_blocks,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_prefix_caching=enable_prefix_caching,
    )

    prefix_len = ceildiv(input_tokens, 2)
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            input_tokens,
            max_seq_len,
            shared_prefix=shared_prefix,
        )

    # make sure that we terminated within 10000 iterations
    actual = run_until_completion(scheduler, max_num_iters=10000)
    assert actual[-1] == BatchInfo.empty()
