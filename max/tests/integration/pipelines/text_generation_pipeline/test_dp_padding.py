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
"""Tests for DPBatchPadder, DPPaddingInfo, and TextBatchConstructor DP integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.context import TextAndVisionContext, TextContext
from max.pipelines.kv_cache import PagedKVCacheManager
from max.pipelines.modeling.types import (
    BatchType,
    RequestID,
    TextGenerationInputs,
)
from max.serve.scheduler.batch_constructor.text_batch_constructor import (
    TextBatchConstructor,
)
from max.serve.scheduler.config import TokenGenerationSchedulerConfig
from max.serve.scheduler.dp_padding import DPBatchPadder, DPPaddingInfo
from test_common.context_utils import create_text_context


@dataclass(kw_only=True)
class _KimiLikeContext(TextAndVisionContext):
    """Mimics KimiK2_5TextAndVisionContext: extra fields all have defaults."""

    grid_thws: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int64)
    )


@dataclass(kw_only=True)
class _Gemma4LikeContext(TextAndVisionContext):
    """Mimics Gemma4Context: a required constructor field with no default."""

    mm_token_type_ids: npt.NDArray[np.int64]

    @classmethod
    def _padding_context_required_fields(cls) -> dict[str, Any]:
        return {
            **super()._padding_context_required_fields(),
            "mm_token_type_ids": np.zeros(1, dtype=np.int64),
        }


PadderKV = tuple[DPBatchPadder, PagedKVCacheManager, MagicMock]
UnevenTGBatch = tuple[
    DPBatchPadder,
    PagedKVCacheManager,
    MagicMock,
    list[TextContext],
    list[TextContext],
    TextGenerationInputs[TextContext],
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_padder(
    dp_size: int,
    total_num_pages: int = 128,
    page_size: int = 1,
    max_batch_size: int = 128,
    max_length: int = 100,
    pipeline: MagicMock | None = None,
    context_type: type[TextContext] = TextContext,
) -> tuple[DPBatchPadder, PagedKVCacheManager, MagicMock]:
    """Creates a DPBatchPadder, PagedKVCacheManager, and pipeline mock."""
    if pipeline is None:
        pipeline = MagicMock()
        pipeline.release = MagicMock()
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        num_layers=1,
        n_kv_heads=1,
        head_dim=1,
        page_size=page_size,
        devices=[DeviceRef.CPU()] * dp_size,
        data_parallel_degree=dp_size,
    )
    session = InferenceSession(devices=[CPU()] * dp_size)
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=total_num_pages,
        session=session,
        max_batch_size=max_batch_size,
    )
    padder = DPBatchPadder(
        dp_size=dp_size,
        kv_manager=kv_manager,
        max_length=max_length,
        model_name="test-model",
        pipeline=pipeline,
        context_type=context_type,
    )
    return padder, kv_manager, pipeline


def _make_contexts(n: int) -> list[TextContext]:
    """Creates *n* text contexts with distinct request IDs."""
    return [
        create_text_context(np.array([i + 1], dtype=np.int64)) for i in range(n)
    ]


def _claim_and_alloc(
    kv_manager: PagedKVCacheManager,
    contexts: list[TextContext],
    replica_idx: int,
) -> None:
    """Claims and allocates KV cache entries for *contexts* on a replica."""
    for ctx in contexts:
        kv_manager.claim(ctx, replica_idx=replica_idx)
        kv_manager.alloc(ctx)


def _simulate_execute(contexts: list[TextContext]) -> None:
    """Simulates a pipeline execute step by advancing each context's generated_length.

    After CE, the pipeline calls ``ctx.update(token)`` so that
    ``generated_length > 0`` and the next ``construct_batch()`` produces a
    TG batch (``__post_init__`` checks ``generated_length``).
    """
    for ctx in contexts:
        ctx.update(0)


def _get_dummy_ids(
    padded_batches: list[list[TextContext]],
    original_batches: list[list[TextContext]],
) -> set[RequestID]:
    """Returns request IDs present in padded batches but not in originals."""
    original_ids = {
        ctx.request_id for batch in original_batches for ctx in batch
    }
    return {
        ctx.request_id
        for batch in padded_batches
        for ctx in batch
        if ctx.request_id not in original_ids
    }


def _make_inputs(
    batches: list[list[TextContext]],
    num_steps: int = 1,
    batch_type: BatchType = BatchType.CE,
) -> TextGenerationInputs[TextContext]:
    inputs = TextGenerationInputs(batches=batches)
    inputs.batch_type = batch_type
    return inputs


def _release_info(
    info: DPPaddingInfo,
    kv_manager: PagedKVCacheManager,
    pipeline: MagicMock,
) -> None:
    """Releases dummy KV entries and pipeline resources for a DPPaddingInfo."""
    for ctx in info.dummies:
        if kv_manager.contains(ctx):
            kv_manager.release(ctx)
        pipeline.release(ctx.request_id)


def _make_batch_constructor(
    dp_size: int = 2,
    total_num_pages: int = 256,
    max_batch_size: int = 128,
) -> tuple[TextBatchConstructor, DPBatchPadder, PagedKVCacheManager, MagicMock]:
    """Creates a TextBatchConstructor wired to a DPBatchPadder."""
    pipeline = MagicMock()
    pipeline.release = MagicMock()
    # Prevent get_lora_manager() from detecting a LoRA manager
    # on the mock pipeline (MagicMock responds to hasattr checks).
    del pipeline._pipeline_model
    del pipeline.speech_lm_pipeline
    del pipeline.pipeline_model
    padder, kv_manager, _pipeline = _make_padder(
        dp_size=dp_size,
        total_num_pages=total_num_pages,
        max_batch_size=max_batch_size,
        pipeline=pipeline,
    )
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=max_batch_size,
        target_tokens_per_batch_ce=4096,
        data_parallel_degree=dp_size,
    )
    bc = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_manager,
        dp_padder=padder,
    )
    return bc, padder, kv_manager, pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def padder_kv() -> PadderKV:
    return _make_padder(dp_size=2)


@pytest.fixture
def uneven_tg_batch(padder_kv: PadderKV) -> UnevenTGBatch:
    """Standard [2, 1] TG batch with KV entries claimed and allocated."""
    padder, kv_manager, pipeline = padder_kv
    ctxs_r0 = _make_contexts(2)
    ctxs_r1 = _make_contexts(1)
    _claim_and_alloc(kv_manager, ctxs_r0, replica_idx=0)
    _claim_and_alloc(kv_manager, ctxs_r1, replica_idx=1)
    inputs = _make_inputs([ctxs_r0, ctxs_r1], batch_type=BatchType.TG)
    return padder, kv_manager, pipeline, ctxs_r0, ctxs_r1, inputs


# ---------------------------------------------------------------------------
# Padding behavior
# ---------------------------------------------------------------------------


def test_no_padding_when_replicas_equal(padder_kv: PadderKV) -> None:
    """Equal-sized replicas produce info=None and return inputs unchanged."""
    padder, kv_manager, _pipeline = padder_kv
    ctxs_r0 = _make_contexts(2)
    ctxs_r1 = _make_contexts(2)
    _claim_and_alloc(kv_manager, ctxs_r0, replica_idx=0)
    _claim_and_alloc(kv_manager, ctxs_r1, replica_idx=1)

    inputs = _make_inputs([ctxs_r0, ctxs_r1], batch_type=BatchType.TG)
    padded_inputs, info = padder.pad_batch(inputs)

    assert info is None
    assert padded_inputs is inputs


def test_no_padding_for_ce_batches(padder_kv: PadderKV) -> None:
    """CE batches are never padded — device graph capture is TG-only."""
    padder, kv_manager, _pipeline = padder_kv
    ctxs_r0 = _make_contexts(2)
    ctxs_r1 = _make_contexts(1)
    _claim_and_alloc(kv_manager, ctxs_r0, replica_idx=0)
    _claim_and_alloc(kv_manager, ctxs_r1, replica_idx=1)

    inputs = _make_inputs([ctxs_r0, ctxs_r1])
    padded_inputs, info = padder.pad_batch(inputs)

    assert info is None
    assert padded_inputs is inputs


def test_pads_short_replica(uneven_tg_batch: UnevenTGBatch) -> None:
    """Replicas [2, 1] are padded to [2, 2] with a dummy KV entry."""
    padder, kv_manager, pipeline, ctxs_r0, ctxs_r1, inputs = uneven_tg_batch
    padded_inputs, info = padder.pad_batch(inputs)

    assert info is not None
    assert len(padded_inputs.batches[0]) == 2
    assert len(padded_inputs.batches[1]) == 2
    assert padded_inputs.batches[0][0] is ctxs_r0[0]
    assert padded_inputs.batches[0][1] is ctxs_r0[1]
    assert padded_inputs.batches[1][0] is ctxs_r1[0]
    dummy = padded_inputs.batches[1][1]
    assert kv_manager.contains(dummy)

    _release_info(info, kv_manager, pipeline)


def test_dummies_have_generated_length(uneven_tg_batch: UnevenTGBatch) -> None:
    """Dummies allocated for TG batches have generated_length > 0."""
    padder, kv_manager, pipeline, _ctxs_r0, _ctxs_r1, inputs = uneven_tg_batch
    padded_inputs, info = padder.pad_batch(inputs)
    assert info is not None

    dummy = padded_inputs.batches[1][1]
    assert dummy.tokens.generated_length > 0

    _release_info(info, kv_manager, pipeline)


def test_three_replicas_uneven() -> None:
    """3 replicas [3, 1, 2] are all padded to 3."""
    padder, kv_manager, pipeline = _make_padder(dp_size=3)
    ctxs_r0 = _make_contexts(3)
    ctxs_r1 = _make_contexts(1)
    ctxs_r2 = _make_contexts(2)
    _claim_and_alloc(kv_manager, ctxs_r0, replica_idx=0)
    _claim_and_alloc(kv_manager, ctxs_r1, replica_idx=1)
    _claim_and_alloc(kv_manager, ctxs_r2, replica_idx=2)

    inputs = _make_inputs([ctxs_r0, ctxs_r1, ctxs_r2], batch_type=BatchType.TG)
    padded_inputs, info = padder.pad_batch(inputs)

    assert info is not None
    assert len(padded_inputs.batches[0]) == 3
    assert len(padded_inputs.batches[1]) == 3
    assert len(padded_inputs.batches[2]) == 3
    dummy_ids = _get_dummy_ids(
        padded_inputs.batches, [ctxs_r0, ctxs_r1, ctxs_r2]
    )
    # Replica 1 needed 2 dummies, replica 2 needed 1.
    assert len(dummy_ids) == 3

    _release_info(info, kv_manager, pipeline)


def test_dummies_are_appended_to_short_replica(padder_kv: PadderKV) -> None:
    """Dummy contexts are appended after real contexts on the short replica."""
    padder, kv_manager, pipeline = padder_kv
    ctxs_r0 = _make_contexts(3)
    ctxs_r1 = _make_contexts(1)
    _claim_and_alloc(kv_manager, ctxs_r0, replica_idx=0)
    _claim_and_alloc(kv_manager, ctxs_r1, replica_idx=1)

    padded_inputs, info = padder.pad_batch(
        _make_inputs([ctxs_r0, ctxs_r1], batch_type=BatchType.TG)
    )
    assert info is not None

    dummy_ids = _get_dummy_ids(padded_inputs.batches, [ctxs_r0, ctxs_r1])
    assert len(dummy_ids) == 2
    # Dummies should be the last 2 entries on replica 1.
    for ctx in padded_inputs.batches[1][1:]:
        assert ctx.request_id in dummy_ids

    _release_info(info, kv_manager, pipeline)


# ---------------------------------------------------------------------------
# Context type of padding dummies
# ---------------------------------------------------------------------------


def _pad_uneven_batch(
    padder: DPBatchPadder, kv_manager: PagedKVCacheManager
) -> tuple[TextContext, DPPaddingInfo]:
    """Pads a [2, 1] TG batch and returns the dummy context plus its info."""
    ctxs_r0 = _make_contexts(2)
    ctxs_r1 = _make_contexts(1)
    _claim_and_alloc(kv_manager, ctxs_r0, replica_idx=0)
    _claim_and_alloc(kv_manager, ctxs_r1, replica_idx=1)

    padded_inputs, info = padder.pad_batch(
        _make_inputs([ctxs_r0, ctxs_r1], batch_type=BatchType.TG)
    )
    assert info is not None
    return padded_inputs.batches[1][1], info


def test_padding_dummies_use_vision_context_type() -> None:
    """VLM padders build dummies of the arch's TextAndVisionContext subclass.

    The overlap pipeline's vision drive narrows every context in an
    executed batch via `isinstance(ctx, TextAndVisionContext)`, so plain
    TextContext dummies crash VLM decode with DP padding.
    """
    padder, kv_manager, pipeline = _make_padder(
        dp_size=2, context_type=_KimiLikeContext
    )
    dummy, info = _pad_uneven_batch(padder, kv_manager)

    assert isinstance(dummy, TextAndVisionContext)
    assert type(dummy) is _KimiLikeContext
    assert dummy._is_padding_ctx
    # Empty vision fields: the dummy is a valid "no image" context.
    assert not dummy.needs_vision_encoding

    _release_info(info, kv_manager, pipeline)


def test_padding_dummies_default_to_plain_text_context() -> None:
    """Text-only archs (the default context type) keep plain TextContext dummies."""
    padder, kv_manager, pipeline = _make_padder(dp_size=2)
    dummy, info = _pad_uneven_batch(padder, kv_manager)

    assert type(dummy) is TextContext
    assert dummy._is_padding_ctx

    _release_info(info, kv_manager, pipeline)


def test_padding_dummies_subclass_required_fields() -> None:
    """Context subclasses with required fields supply their own padding defaults."""
    padder, kv_manager, pipeline = _make_padder(
        dp_size=2, context_type=_Gemma4LikeContext
    )
    dummy, info = _pad_uneven_batch(padder, kv_manager)

    assert type(dummy) is _Gemma4LikeContext
    assert isinstance(dummy, TextAndVisionContext)
    assert not dummy.needs_vision_encoding

    _release_info(info, kv_manager, pipeline)


# ---------------------------------------------------------------------------
# Release lifecycle
# ---------------------------------------------------------------------------


def test_release_frees_dummy_kv_entries(uneven_tg_batch: UnevenTGBatch) -> None:
    """After releasing dummies, their request IDs are gone from the KV manager."""
    padder, kv_manager, pipeline, ctxs_r0, ctxs_r1, inputs = uneven_tg_batch
    padded_inputs, info = padder.pad_batch(inputs)

    assert info is not None
    dummy = padded_inputs.batches[1][1]
    assert kv_manager.contains(dummy)

    _release_info(info, kv_manager, pipeline)

    assert not kv_manager.contains(dummy)
    assert kv_manager.contains(ctxs_r0[0])
    assert kv_manager.contains(ctxs_r1[0])


def test_release_idempotent(uneven_tg_batch: UnevenTGBatch) -> None:
    """Releasing the same dummies twice does not raise."""
    padder, kv_manager, pipeline, _ctxs_r0, _ctxs_r1, inputs = uneven_tg_batch

    _, info = padder.pad_batch(inputs)
    assert info is not None

    _release_info(info, kv_manager, pipeline)
    _release_info(info, kv_manager, pipeline)


def test_multiple_pad_calls_independent() -> None:
    """Two pad_batch calls produce independent dummy sets."""
    padder, kv_manager, pipeline = _make_padder(dp_size=2)

    ctxs1_r0 = _make_contexts(2)
    ctxs1_r1 = _make_contexts(1)
    _claim_and_alloc(kv_manager, ctxs1_r0, replica_idx=0)
    _claim_and_alloc(kv_manager, ctxs1_r1, replica_idx=1)
    padded1, info1 = padder.pad_batch(
        _make_inputs([ctxs1_r0, ctxs1_r1], batch_type=BatchType.TG)
    )
    assert info1 is not None
    dummy1 = padded1.batches[1][1]

    ctxs2_r0 = _make_contexts(1)
    ctxs2_r1 = _make_contexts(2)
    _claim_and_alloc(kv_manager, ctxs2_r0, replica_idx=0)
    _claim_and_alloc(kv_manager, ctxs2_r1, replica_idx=1)
    padded2, info2 = padder.pad_batch(
        _make_inputs([ctxs2_r0, ctxs2_r1], batch_type=BatchType.TG)
    )
    assert info2 is not None
    dummy2 = padded2.batches[0][1]

    _release_info(info1, kv_manager, pipeline)
    assert not kv_manager.contains(dummy1)
    assert kv_manager.contains(dummy2)

    _release_info(info2, kv_manager, pipeline)
    assert not kv_manager.contains(dummy2)


def test_release_calls_pipeline_release_for_dummies() -> None:
    """Releasing dummies calls pipeline.release() for each dummy request ID."""
    padder, kv_manager, pipeline = _make_padder(dp_size=2)

    ctxs_r0 = _make_contexts(2)
    ctxs_r1 = _make_contexts(1)
    _claim_and_alloc(kv_manager, ctxs_r0, replica_idx=0)
    _claim_and_alloc(kv_manager, ctxs_r1, replica_idx=1)

    padded_inputs, info = padder.pad_batch(
        _make_inputs([ctxs_r0, ctxs_r1], batch_type=BatchType.TG)
    )
    assert info is not None
    dummy_ids = _get_dummy_ids(padded_inputs.batches, [ctxs_r0, ctxs_r1])
    assert len(dummy_ids) == 1

    _release_info(info, kv_manager, pipeline)

    released_ids = {call.args[0] for call in pipeline.release.call_args_list}
    assert released_ids == dummy_ids


# ---------------------------------------------------------------------------
# Null block (dummy KV slots)
# ---------------------------------------------------------------------------


def test_null_block_reserved_per_replica(padder_kv: PadderKV) -> None:
    """Each replica reserves block 0 as the pool null block at init."""
    _padder, kv_manager, _pipeline = padder_kv

    for rank in range(2):
        null_block = kv_manager._replica[
            rank
        ].block_manager.device_block_pool.null_block
        assert null_block.is_null
        assert null_block.bid == 128


def test_padding_does_not_allocate_new_blocks() -> None:
    """Dummies map to the null block; used page count stays constant."""
    padder, kv_manager, pipeline = _make_padder(dp_size=2, total_num_pages=128)

    ctxs_r0 = _make_contexts(3)
    ctxs_r1 = _make_contexts(1)
    _claim_and_alloc(kv_manager, ctxs_r0, replica_idx=0)
    _claim_and_alloc(kv_manager, ctxs_r1, replica_idx=1)

    used_after_real_r0 = kv_manager.block_count(replica_idx=0).used
    used_after_real_r1 = kv_manager.block_count(replica_idx=1).used

    _, info = padder.pad_batch(
        _make_inputs([ctxs_r0, ctxs_r1], batch_type=BatchType.TG)
    )
    assert info is not None

    assert kv_manager.block_count(replica_idx=0).used == used_after_real_r0
    # Dummies use the null block, so used page count should not increase.
    assert kv_manager.block_count(replica_idx=1).used == used_after_real_r1

    _release_info(info, kv_manager, pipeline)


def test_repeated_pad_and_release_cycles() -> None:
    """Padding the same replica multiple times reuses the null block safely."""
    padder, kv_manager, pipeline = _make_padder(dp_size=2, total_num_pages=128)

    for _ in range(5):
        ctxs_r0 = _make_contexts(3)
        ctxs_r1 = _make_contexts(1)
        _claim_and_alloc(kv_manager, ctxs_r0, replica_idx=0)
        _claim_and_alloc(kv_manager, ctxs_r1, replica_idx=1)

        used_before = kv_manager.block_count(replica_idx=1).used
        _, info = padder.pad_batch(
            _make_inputs([ctxs_r0, ctxs_r1], batch_type=BatchType.TG)
        )
        assert info is not None
        assert kv_manager.block_count(replica_idx=1).used == used_before

        _release_info(info, kv_manager, pipeline)

        for ctx in ctxs_r0:
            kv_manager.release(ctx)
        for ctx in ctxs_r1:
            kv_manager.release(ctx)


# ---------------------------------------------------------------------------
# TextBatchConstructor integration
# ---------------------------------------------------------------------------


def test_construct_batch_pads_unequal_replicas() -> None:
    """construct_batch() pads short TG replicas when dp_padder is set."""
    bc, _padder, _kv_manager, _pipeline = _make_batch_constructor(dp_size=2)

    ctxs_r0 = _make_contexts(2)
    ctxs_r1 = _make_contexts(1)
    for ctx in ctxs_r0:
        bc.enqueue_new_request(ctx, replica_idx=0)
    for ctx in ctxs_r1:
        bc.enqueue_new_request(ctx, replica_idx=1)

    # First construct_batch produces a CE batch (no padding).
    ce_inputs = bc.construct_batch()
    assert bc._current_dp_padding is None

    # Simulate pipeline execution so contexts transition CE → TG.
    _simulate_execute(ctxs_r0 + ctxs_r1)
    bc.advance_requests(ce_inputs)

    # Second construct_batch produces a TG batch (triggers padding).
    tg_inputs = bc.construct_batch()

    assert len(tg_inputs.batches[0]) == len(tg_inputs.batches[1])
    assert len(tg_inputs.batches[0]) == 2
    assert bc._current_dp_padding is not None


def test_advance_requests_skips_dummies() -> None:
    """advance_requests() should not add dummy padding contexts to TG queues."""
    bc, _padder, _kv_manager, _pipeline = _make_batch_constructor(dp_size=2)

    ctxs_r0 = _make_contexts(2)
    ctxs_r1 = _make_contexts(1)
    for ctx in ctxs_r0:
        bc.enqueue_new_request(ctx, replica_idx=0)
    for ctx in ctxs_r1:
        bc.enqueue_new_request(ctx, replica_idx=1)
    ce_inputs = bc.construct_batch()
    _simulate_execute(ctxs_r0 + ctxs_r1)
    bc.advance_requests(ce_inputs)

    # Second batch is TG — triggers padding.
    tg_inputs = bc.construct_batch()
    assert bc._current_dp_padding is not None
    dummy_ids = _get_dummy_ids(tg_inputs.batches, [ctxs_r0, ctxs_r1])

    bc.advance_requests(tg_inputs)

    all_tg_ids = set(bc.all_tg_reqs.keys())
    assert all_tg_ids.isdisjoint(dummy_ids)

    real_ids = {ctx.request_id for ctx in ctxs_r0 + ctxs_r1}
    assert all_tg_ids.issubset(real_ids)


def test_advance_requests_deferred_release() -> None:
    """advance_requests() releases previous batch's dummies, not current."""
    bc, _padder, kv_manager, _pipeline = _make_batch_constructor(dp_size=2)

    ctxs1_r0 = _make_contexts(2)
    ctxs1_r1 = _make_contexts(1)
    for ctx in ctxs1_r0:
        bc.enqueue_new_request(ctx, replica_idx=0)
    for ctx in ctxs1_r1:
        bc.enqueue_new_request(ctx, replica_idx=1)
    ce_inputs = bc.construct_batch()
    _simulate_execute(ctxs1_r0 + ctxs1_r1)
    bc.advance_requests(ce_inputs)

    # --- TG batch: unequal, triggers padding. ---
    tg_inputs = bc.construct_batch()
    info1 = bc._current_dp_padding
    assert info1 is not None
    dummies = info1.dummies
    assert dummies

    # First advance_requests on TG: no previous padding, shifts current to prev.
    bc.advance_requests(tg_inputs)
    assert bc._prev_dp_padding is info1
    assert bc._current_dp_padding is None
    # Dummies from TG batch should still be alive.
    for dummy in dummies:
        assert kv_manager.contains(dummy)

    # Construct an empty batch just to have valid inputs for advance.
    inputs2 = _make_inputs([[], []])

    # Second advance_requests: releases the TG batch's dummies.
    bc.advance_requests(inputs2)

    for dummy in dummies:
        assert not kv_manager.contains(dummy)

    assert bc._prev_dp_padding is None
    assert bc._current_dp_padding is None
