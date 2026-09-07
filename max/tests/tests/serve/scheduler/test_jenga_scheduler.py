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

"""Scheduler coverage for a hybrid Jenga KV cache."""

from __future__ import annotations

import queue

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import (
    PagedKVCacheManager,
    PagedKVCacheManagerInterface,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_cache_manager import (
    JengaKVCacheManager,
)
from max.serve.scheduler.config import TokenGenerationSchedulerConfig
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
)
from max.support.math import ceildiv
from tests.serve.scheduler.common import (
    CE,
    TG,
    BatchInfo,
    FakeTokenGeneratorPipeline,
    assert_batch_info_equal,
    create_batch_and_execute,
    create_text_context,
    enqueue_request,
    rand,
    run_until_completion,
)

PAGE_SIZE = 10
WINDOW_SIZE = 25


def create_scheduler(
    is_jenga: bool,
    *,
    num_huge_pages: int = 500,
    max_batch_size: int = 512,
    max_seq_len: int = 1000,
    target_tokens_per_batch_ce: int = 8192,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = False,
) -> tuple[TokenGenerationScheduler, queue.Queue[TextContext]]:
    session = InferenceSession(devices=[CPU()])
    page_size = PAGE_SIZE
    params = MultiKVCacheParams.from_params(
        {
            "sliding": MHAKVCacheParams(
                dtype=DType.float32,
                num_layers=1,
                n_kv_heads=1,
                head_dim=16,
                page_size=page_size,
                window_size=WINDOW_SIZE,
                enable_prefix_caching=enable_prefix_caching,
                devices=[DeviceRef.CPU()],
            ),
            "full": MHAKVCacheParams(
                dtype=DType.float32,
                num_layers=1,
                n_kv_heads=1,
                head_dim=1,
                page_size=page_size,
                enable_prefix_caching=enable_prefix_caching,
                devices=[DeviceRef.CPU()],
            ),
        }
    )
    bytes_per_leaf = {leaf.bytes_per_page for leaf in params.leaves().values()}
    huge_page_bytes = max(bytes_per_leaf)
    avail_bytes = num_huge_pages * huge_page_bytes
    if is_jenga:
        kv_cache: PagedKVCacheManagerInterface = JengaKVCacheManager.create(
            params=params,
            available_bytes=avail_bytes,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
    else:
        kv_cache = PagedKVCacheManager(
            params=params,
            total_num_pages=avail_bytes // sum(bytes_per_leaf),
            session=session,
            enable_runtime_checks=True,
            max_batch_size=max_batch_size,
        )
    request_queue: queue.Queue[TextContext] = queue.Queue()
    scheduler = TokenGenerationScheduler(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=max_batch_size,
            target_tokens_per_batch_ce=target_tokens_per_batch_ce,
            max_seq_len=max_seq_len,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_in_flight_batching=False,
        ),
        pipeline=FakeTokenGeneratorPipeline(kv_cache, max_seq_len),
        kv_cache=kv_cache,
        request_queue=request_queue,
        response_queue=queue.Queue(),
        cancel_queue=queue.Queue(),
    )
    return scheduler, request_queue


@pytest.mark.parametrize("is_jenga", [True, False])
def test_scheduler_is_jenga_isl_250_osl_6(is_jenga: bool) -> None:
    scheduler, request_queue = create_scheduler(is_jenga)
    num_requests = 50
    isl = 250
    osl = 6
    for _ in range(num_requests):
        enqueue_request(request_queue, prompt_len=isl, max_seq_len=isl + osl)

    # fmt: off
    jenga_expected = [
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=4500, cached_toks=0),
        BatchInfo(CE, batch_size=15, terminated=0, steps=1, preempted=0, input_toks=3750, cached_toks=0),
        BatchInfo(CE, batch_size=13, terminated=0, steps=1, preempted=0, input_toks=3250, cached_toks=0),
        BatchInfo(CE, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=0),
        BatchInfo(TG, batch_size=50, terminated=0, steps=1, preempted=0, input_toks=50, cached_toks=12500),
        BatchInfo(TG, batch_size=50, terminated=0, steps=1, preempted=0, input_toks=50, cached_toks=12550),
        BatchInfo(TG, batch_size=50, terminated=0, steps=1, preempted=0, input_toks=50, cached_toks=12600),
        BatchInfo(TG, batch_size=50, terminated=0, steps=1, preempted=0, input_toks=50, cached_toks=12650),
        BatchInfo(TG, batch_size=50, terminated=50, steps=1, preempted=0, input_toks=50, cached_toks=12700),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    legacy_expected = [
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=4500, cached_toks=0),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4500),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4518),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4536),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4554),
        BatchInfo(TG, batch_size=18, terminated=18, steps=1, preempted=0, input_toks=18, cached_toks=4572),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=4500, cached_toks=0),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4500),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4518),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4536),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4554),
        BatchInfo(TG, batch_size=18, terminated=18, steps=1, preempted=0, input_toks=18, cached_toks=4572),
        BatchInfo(CE, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=3500, cached_toks=0),
        BatchInfo(TG, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=14, cached_toks=3500),
        BatchInfo(TG, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=14, cached_toks=3514),
        BatchInfo(TG, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=14, cached_toks=3528),
        BatchInfo(TG, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=14, cached_toks=3542),
        BatchInfo(TG, batch_size=14, terminated=14, steps=1, preempted=0, input_toks=14, cached_toks=3556),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(
        run_until_completion(scheduler),
        jenga_expected if is_jenga else legacy_expected,
    )


@pytest.mark.parametrize("is_jenga", [True, False])
def test_scheduler_is_jenga_chunked_prefill_one_long_request(
    is_jenga: bool,
) -> None:
    """One request far longer than the CE token target, chunk by chunk.

    Nothing here is memory-constrained, so both managers must produce the
    same trace: chunked prefill is scheduler bookkeeping, and swapping the
    pool underneath it may not change how a prompt is split.
    """
    isl = 3123
    osl = 3
    scheduler, request_queue = create_scheduler(
        is_jenga,
        max_seq_len=isl + osl,
        target_tokens_per_batch_ce=1000,
        max_batch_size=100,
    )
    enqueue_request(request_queue, prompt_len=isl, max_seq_len=isl + osl)

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=0),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=1000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=2000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=123, cached_toks=3000),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=3123),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=3124),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(run_until_completion(scheduler), expected)


@pytest.mark.parametrize("is_jenga", [True, False])
def test_scheduler_is_jenga_tg_preemption_under_kv_pressure(
    is_jenga: bool,
) -> None:
    """A pool too small to decode every admitted request to max length.

    CE admits all five requests -- at prefill each holds two pages per leaf --
    but they keep growing, so TG has to preempt and later re-prefill one of
    them. How many survive differs: the legacy manager gives each leaf its own
    12 pages, while Jenga's leaves compete for 12 huge blocks and the sliding
    leaf's pages are 16x the size of the full leaf's, so it pays a whole huge
    block per page.
    """
    scheduler, request_queue = create_scheduler(
        is_jenga,
        num_huge_pages=13,
        max_seq_len=30,
        max_batch_size=999,
        enable_chunked_prefill=False,
    )
    for _ in range(5):
        enqueue_request(request_queue, prompt_len=20, max_seq_len=30)

    # fmt: off
    jenga_expected = [
        BatchInfo(CE, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=0),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=1, input_toks=3, cached_toks=60),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=63),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=66),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=69),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=72),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=75),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=78),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=81),
        BatchInfo(TG, batch_size=3, terminated=3, steps=1, preempted=0, input_toks=3, cached_toks=84),
        # Re-prefill the preempted request: its 20 prompt tokens plus the one
        # token it had already generated before losing its pages.
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=21, cached_toks=0),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=41),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=43),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=45),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=47),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=49),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=51),
        BatchInfo(TG, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=2, cached_toks=53),
        BatchInfo(TG, batch_size=2, terminated=1, steps=1, preempted=0, input_toks=2, cached_toks=55),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=28),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    legacy_expected = [
        BatchInfo(CE, batch_size=5, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=0),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=1, input_toks=4, cached_toks=80),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=84),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=88),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=92),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=96),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=100),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=104),
        BatchInfo(TG, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=4, cached_toks=108),
        BatchInfo(TG, batch_size=4, terminated=4, steps=1, preempted=0, input_toks=4, cached_toks=112),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=21, cached_toks=0),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=21),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=22),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=23),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=24),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=25),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=26),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=27),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=28),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(
        run_until_completion(scheduler),
        jenga_expected if is_jenga else legacy_expected,
    )


@pytest.mark.parametrize("is_jenga", [True, False])
def test_scheduler_is_jenga_prefix_cache_hit_shortens_prefill(
    is_jenga: bool,
) -> None:
    """A repeated prompt prefills only its last page the second time.

    ``max_batch_size=1`` serializes the two identical requests so the second
    one prefills against what the first committed. Both leaves have to serve
    the hit for the scheduler to see it, and the sliding leaf commits under a
    different coordinator than the full one -- yet the trace matches the
    legacy manager's, hit for hit.
    """
    isl = 100
    osl = 4
    scheduler, request_queue = create_scheduler(
        is_jenga,
        max_batch_size=1,
        max_seq_len=isl + osl,
        enable_prefix_caching=True,
    )

    np.random.seed(42)
    prompt = rand(isl)
    for _ in range(2):
        enqueue_request(
            request_queue,
            prompt_len=isl,
            max_seq_len=isl + osl,
            shared_prefix=prompt,
        )

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=0),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=100),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=101),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=102),
        # Nine of the ten prompt pages are served from the prefix cache; the
        # last page is always recomputed so the forward has a query to run.
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=10, cached_toks=90),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=100),
        BatchInfo(TG, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1, cached_toks=101),
        BatchInfo(TG, batch_size=1, terminated=1, steps=1, preempted=0, input_toks=1, cached_toks=102),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(run_until_completion(scheduler), expected)


@pytest.mark.parametrize("is_jenga", [True, False])
def test_scheduler_is_jenga_terminates_at_max_length(is_jenga: bool) -> None:
    """Requests that reach ``max_seq_len`` mid-TG terminate there."""
    max_seq_len = 300
    scheduler, request_queue = create_scheduler(
        is_jenga, max_seq_len=max_seq_len, max_batch_size=100
    )
    for _ in range(3):
        enqueue_request(request_queue, prompt_len=297, max_seq_len=max_seq_len)

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=891, cached_toks=0),
        BatchInfo(TG, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3, cached_toks=891),
        BatchInfo(TG, batch_size=3, terminated=3, steps=1, preempted=0, input_toks=3, cached_toks=894),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(run_until_completion(scheduler), expected)


def test_jenga_sliding_leaf_holds_only_its_window() -> None:
    """The sliding leaf's page count plateaus while the full leaf's grows.

    This is what the hybrid pool buys, so assert it at the block level rather
    than inferring it from a batch trace: after prefilling a prompt ten times
    the window, the sliding leaf holds only the pages the window covers and
    points the rest of its lookup table at the null page, while the full leaf
    holds one page per prompt page.
    """
    isl = 250
    scheduler, request_queue = create_scheduler(True, max_seq_len=isl + 6)
    ctx = create_text_context(prompt_len=isl, max_seq_len=isl + 6)
    request_queue.put_nowait(ctx)

    assert create_batch_and_execute(scheduler).batch_type == CE

    kv_cache = scheduler.batch_constructor.kv_cache
    assert isinstance(kv_cache, JengaKVCacheManager)
    blocks = kv_cache.get_req_blocks_per_leaf(ctx)
    sliding, full = (
        next(v for k, v in blocks.items() if k.startswith(leaf))
        for leaf in ("sliding", "full")
    )

    # Both leaves cover the whole sequence in the lookup table the kernel
    # reads; only the sliding one repeats the null page (block 0) across the
    # positions that have fallen out of its window.
    prompt_pages = ceildiv(isl, PAGE_SIZE)
    assert len(sliding) == len(full) == prompt_pages
    assert len(set(sliding) - {0}) == ceildiv(WINDOW_SIZE, PAGE_SIZE)
    assert len(set(full)) == prompt_pages

    # Huge blocks charge for what each leaf actually holds: the sliding leaf's
    # page fills a whole huge block, the full leaf's is a fraction of one.
    huge = kv_cache.block_count()
    full_pages_per_huge = (
        kv_cache.little_block_count()[
            next(k for k in blocks if k.startswith("full"))
        ].total
        // huge.total
    )
    assert huge.total - huge.free == ceildiv(WINDOW_SIZE, PAGE_SIZE) + ceildiv(
        prompt_pages, full_pages_per_huge
    )


def test_jenga_rejects_a_max_seq_len_that_cannot_fit() -> None:
    """Jenga refuses to start when one max-length request cannot fit.

    A request that big would exhaust the pool with nothing left to preempt
    and take the model worker down mid-serve, so this fails at startup
    instead. The legacy manager has no equivalent constructor check -- its
    page count is sized for ``max_seq_len`` before it is built.
    """
    with pytest.raises(RuntimeError) as e:
        create_scheduler(True, num_huge_pages=8, max_seq_len=1000)

    # 7 allocatable huge blocks: 3 go to the sliding leaf's window, leaving 4
    # for the full leaf at 16 pages of 10 tokens each.
    assert "Insufficient cache memory" in str(e.value)
    assert "Reduce --max-length to at most 640" in str(e.value)
