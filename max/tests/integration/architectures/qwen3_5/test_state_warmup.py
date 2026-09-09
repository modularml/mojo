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
"""Tests for GatedDeltaNetStateCache and Qwen3_5Model graph-capture warmup.

Validates the ``SupportsSSMStateWarmup`` contract so the overlap pipeline's
``_warmup_model_inputs`` context manager can claim and release state pool
slots during graph-capture probes without exhausting the pool.

These tests run CPU-only (no model load, no GPU) so they complete quickly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.pipelines.architectures.qwen3_5.model import Qwen3_5Model
from max.pipelines.architectures.qwen3_5.state_cache import (
    GatedDeltaNetStateCache,
)
from max.pipelines.lib import SupportsSSMStateWarmup
from max.pipelines.modeling.types import RequestID


def _make_cache(max_slots: int = 4) -> GatedDeltaNetStateCache:
    """Construct a small CPU-resident GatedDeltaNetStateCache for testing."""
    return GatedDeltaNetStateCache(
        num_layers=2,
        conv_dim=8,
        conv_kernel_size=4,
        num_v_heads=4,
        key_head_dim=8,
        value_head_dim=8,
        max_slots=max_slots,
        devices=[CPU()],
        dtype=DType.bfloat16,
    )


def _model_with(cache: GatedDeltaNetStateCache) -> Qwen3_5Model:
    """A stand-in carrying only what ``release_warmup_state`` reads.

    The real method touches nothing but ``_state_cache``, so this exercises
    the shipped implementation without weights or a GPU. Calling it unbound
    is deliberate: a stub that reimplemented the release loop would pass
    while a no-op method body shipped.
    """
    return cast("Qwen3_5Model", SimpleNamespace(_state_cache=cache))


def test_warmup_probe_exhaustion_with_release() -> None:
    """Simulates the warmup loop: max_slots=4, probe batch_sizes 4..1.

    Without release after each probe the pool would be exhausted at the
    second probe. Cleanup goes through ``Qwen3_5Model.release_warmup_state``
    so a no-op body fails here rather than passing on a direct
    ``cache.release`` the production path never makes.
    """
    max_slots = 4
    cache = _make_cache(max_slots=max_slots)
    model = _model_with(cache)
    preallocs = [Buffer.zeros([max_slots], DType.uint32, CPU())]

    for batch_size in range(max_slots, 0, -1):
        warmup_rids = [RequestID() for _ in range(batch_size)]
        for rid in warmup_rids:
            cache.claim(rid)
        slot_idx_views = cache.slot_idx_for(warmup_rids, preallocs)
        assert tuple(slot_idx_views[0].shape) == (batch_size,)
        # The staged values must be the claimed slots, in batch order.
        # `claim` is idempotent, so re-claiming reads back the assignment.
        np.testing.assert_array_equal(
            slot_idx_views[0].to_numpy(),
            np.array(
                [cache.claim(rid) for rid in warmup_rids], dtype=np.uint32
            ),
        )

        Qwen3_5Model.release_warmup_state(model, warmup_rids)

        assert cache.num_free_slots == max_slots, (
            f"Probe batch_size={batch_size}: expected {max_slots} free slots "
            f"after release, got {cache.num_free_slots}"
        )


def test_release_warmup_state_tolerates_no_state_cache() -> None:
    """A model with no linear-attention layers has no pool to release."""
    model = cast("Qwen3_5Model", SimpleNamespace(_state_cache=None))
    Qwen3_5Model.release_warmup_state(model, [RequestID()])


def test_slot_zeroed_on_re_claim() -> None:
    """After warmup writes nonzero data to a slot, a fresh claim zeros it.

    This is what makes graph-capture warmup's writes into unclaimed slots
    harmless: the next real claim wipes the rows before the first forward
    that reads them.
    """
    cache = _make_cache(max_slots=1)
    rid = RequestID()
    slot = cache.claim(rid)

    # Write nonzero bytes into slot 0 of the conv pool via the uint16
    # reinterpret trick (numpy has no native bfloat16).
    nonzero_u16 = np.full((1, 8, 3), 0x3F80, dtype=np.uint16)
    nonzero_bf16_buf = (
        Buffer.from_numpy(nonzero_u16).to(CPU()).view(DType.bfloat16)
    )
    cache.conv_pools[0].inplace_copy_from(nonzero_bf16_buf)

    cache.release(rid)
    rid2 = RequestID()
    slot2 = cache.claim(rid2)
    assert slot2 == slot, "Should reuse the same slot."

    result_u16 = cache.conv_pools[0].view(DType.uint16).to_numpy()
    np.testing.assert_array_equal(
        result_u16, np.zeros((1, 8, 3), dtype=np.uint16)
    )


def test_qwen3_5_model_implements_supports_ssm_state_warmup() -> None:
    """Qwen3_5Model satisfies the SupportsSSMStateWarmup Protocol.

    Checks the @runtime_checkable Protocol and the existence of
    ``release_warmup_state`` without instantiating the full model (which
    requires weights and a GPU).
    """
    assert issubclass(Qwen3_5Model, SupportsSSMStateWarmup), (
        "Qwen3_5Model must implement SupportsSSMStateWarmup so the overlap "
        "pipeline can release warmup state slots after each probe."
    )
    assert callable(getattr(Qwen3_5Model, "release_warmup_state", None)), (
        "Qwen3_5Model.release_warmup_state must be callable"
    )
