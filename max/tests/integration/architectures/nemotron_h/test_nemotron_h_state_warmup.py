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
"""Tests for NemotronHStateCache and NemotronHModel graph-capture warmup.

Validates the ``SupportsSSMStateWarmup`` contract so that the overlap
pipeline's ``_warmup_model_inputs`` context manager can safely claim and
release SSM/conv state pool slots during graph-capture probes without
exhausting the pool.

These tests run CPU-only (no model load, no GPU) so they complete quickly.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.pipelines.architectures.nemotron_h.model import NemotronHModel
from max.pipelines.architectures.nemotron_h.state_cache import (
    NemotronHStateCache,
)
from max.pipelines.lib import SupportsSSMStateWarmup
from max.pipelines.modeling.types import RequestID

# ---------------------------------------------------------------------------
# NemotronHStateCache unit tests
# ---------------------------------------------------------------------------


def _make_cache(max_slots: int = 4) -> NemotronHStateCache:
    """Construct a small CPU-resident NemotronHStateCache for testing."""
    return NemotronHStateCache(
        num_mamba_layers=2,
        conv_dim=8,
        conv_kernel=4,
        nheads=4,
        head_dim=8,
        dstate=8,
        max_slots=max_slots,
        device=CPU(),
        conv_dtype=DType.bfloat16,
    )


def test_claim_and_release_cycle() -> None:
    """claim -> release -> claim succeeds without exhausting the pool."""
    cache = _make_cache(max_slots=2)
    rid_a = RequestID()
    rid_b = RequestID()

    slot_a = cache.claim(rid_a)
    slot_b = cache.claim(rid_b)
    assert slot_a != slot_b
    assert cache.num_free_slots == 0

    cache.release(rid_a)
    assert cache.num_free_slots == 1

    rid_c = RequestID()
    slot_c = cache.claim(rid_c)
    # slot_c reuses the just-freed slot.
    assert slot_c == slot_a


def test_warmup_probe_exhaustion_with_release() -> None:
    """Simulates the warmup loop: max_slots=4, probe batch_sizes 4..1.

    Without release after each probe the pool would be exhausted at
    batch_size=2.  With release, all four probes complete without error.
    """
    max_slots = 4
    cache = _make_cache(max_slots=max_slots)
    prealloc = Buffer.zeros([max_slots], DType.uint32, CPU())

    # Probe batch_sizes 4, 3, 2, 1 (largest-first, matching warmup_pre_ready).
    for batch_size in range(max_slots, 0, -1):
        warmup_rids = [RequestID() for _ in range(batch_size)]
        for rid in warmup_rids:
            cache.claim(rid)
        # Simulate what the kernel would do: access via slot_idx.
        slot_idx = cache.slot_idx_for(warmup_rids, prealloc)
        assert tuple(slot_idx.shape) == (batch_size,)

        # This is what SupportsSSMStateWarmup.release_warmup_state does:
        for rid in warmup_rids:
            cache.release(rid)

        assert cache.num_free_slots == max_slots, (
            f"Probe batch_size={batch_size}: expected {max_slots} free slots "
            f"after release, got {cache.num_free_slots}"
        )


def test_pool_buffer_identity_preserved_across_claim_release() -> None:
    """Pool Buffer objects are the SAME Python objects before and after claim/release.

    This is the key correctness property for graph-capture replay:
    ``inplace_copy_from`` short-circuits when ``self is src`` (same Buffer
    object), so in-place SSM/conv state updates survive replay without being
    overwritten.
    """
    cache = _make_cache(max_slots=2)
    conv_pool_ids_before = [id(b) for b in cache.conv_pools]
    ssm_pool_ids_before = [id(b) for b in cache.ssm_pools]

    rid = RequestID()
    cache.claim(rid)
    cache.release(rid)

    conv_pool_ids_after = [id(b) for b in cache.conv_pools]
    ssm_pool_ids_after = [id(b) for b in cache.ssm_pools]

    assert conv_pool_ids_before == conv_pool_ids_after, (
        "conv_pools Buffer objects must not be recreated across claim/release"
    )
    assert ssm_pool_ids_before == ssm_pool_ids_after, (
        "ssm_pools Buffer objects must not be recreated across claim/release"
    )


def test_slot_zeroed_on_re_claim() -> None:
    """After warmup writes nonzero data to a slot, a fresh claim zeros it.

    Ensures that warmup-corrupted slots are safe for production requests.
    """
    cache = _make_cache(max_slots=1)
    rid = RequestID()
    slot = cache.claim(rid)

    # Manually write nonzero bytes into slot 0 of the conv pool via the uint16
    # reinterpret trick (numpy has no native bfloat16; we write raw uint16
    # patterns instead).
    nonzero_u16 = np.full((1, 8, 3), 0x3F80, dtype=np.uint16)
    nonzero_bf16_buf = (
        Buffer.from_numpy(nonzero_u16).to(CPU()).view(DType.bfloat16)
    )
    cache.conv_pools[0].inplace_copy_from(nonzero_bf16_buf)

    # Release and re-claim the slot.
    cache.release(rid)
    rid2 = RequestID()
    slot2 = cache.claim(rid2)
    assert slot2 == slot, "Should reuse the same slot."

    # The slot should now be zeroed by claim().
    result_u16 = cache.conv_pools[0].view(DType.uint16).to_numpy()
    np.testing.assert_array_equal(
        result_u16, np.zeros((1, 8, 3), dtype=np.uint16)
    )


# ---------------------------------------------------------------------------
# SupportsSSMStateWarmup protocol conformance
# ---------------------------------------------------------------------------


def test_nemotron_h_model_implements_supports_ssm_state_warmup() -> None:
    """NemotronHModel satisfies the SupportsSSMStateWarmup Protocol.

    Checks the @runtime_checkable Protocol and the existence of
    ``release_warmup_state`` without instantiating the full model (which
    requires weights and a GPU).
    """
    assert issubclass(NemotronHModel, SupportsSSMStateWarmup), (
        "NemotronHModel must implement SupportsSSMStateWarmup so the overlap "
        "pipeline can release warmup SSM state slots after each probe."
    )
    assert callable(getattr(NemotronHModel, "release_warmup_state", None)), (
        "NemotronHModel.release_warmup_state must be callable"
    )
