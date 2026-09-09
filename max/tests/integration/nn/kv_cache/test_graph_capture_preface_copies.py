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

"""CPU unit tests for the batched graph-capture replay-preface copy path.

Two layers, no GPU required:

- Real CPU ``Buffer`` objects: verify ``batch_inplace_copy`` produces the same
  values as a per-buffer ``inplace_copy_from`` loop and that identity pairs
  (``dst is src``) are no-ops.
- Fake-device simulation: exercise ``ServeGraphCaptureRunner.replay``'s routing
  without a GPU -- host destinations copy inline, accelerator destinations go
  through the batched call, destinations spanning devices preserve output
  parity, and the identity-stable short-circuit still holds.
  Per-device submit grouping is the driver's job and is covered by the
  multi-GPU driver tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import Buffer, batch_inplace_copy
from max.nn.kv_cache import BatchCharacteristics, MLAAttnKey
from max.pipelines.lib import graph_capture
from max.pipelines.lib.graph_capture import GraphEntry, ServeGraphCaptureRunner

# ---------------------------------------------------------------------------
# Fakes for the routing tests (no GPU required).
# ---------------------------------------------------------------------------


class _FakeDevice:
    """Minimal stand-in for a driver ``Device`` keyed by ``(label, id)``."""

    def __init__(
        self, label: str, device_id: int, *, is_host: bool = False
    ) -> None:
        self.label = label
        self.id = device_id
        self.is_host = is_host

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _FakeDevice)
            and self.label == other.label
            and self.id == other.id
        )

    def __hash__(self) -> int:
        return hash((self.label, self.id))

    def __repr__(self) -> str:
        return f"{self.label}:{self.id}"


class _FakeBuffer:
    """Minimal buffer fake tracking a ``value`` so copies assert without a GPU."""

    def __init__(self, device: _FakeDevice, *, value: object = None) -> None:
        self.device = device
        self.value = value

    def inplace_copy_from(self, src: _FakeBuffer) -> None:
        self.value = src.value


def _fake_batch_inplace_copy(
    dsts: Sequence[_FakeBuffer], srcs: Sequence[_FakeBuffer]
) -> None:
    """Mirrors the real driver call: skip identity pairs, copy the rest."""
    for dst, src in zip(dsts, srcs, strict=True):
        if dst is not src:
            dst.inplace_copy_from(src)


@pytest.fixture
def fake_batch_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Routes ``replay``'s batched copies at the fake buffers.

    ``batch_inplace_copy`` is a module-level driver call rather than a Buffer
    method, so the fakes are installed by patching the name ``graph_capture``
    resolves. The real-Buffer tests below hold their own reference and are
    unaffected.
    """
    monkeypatch.setattr(
        graph_capture, "batch_inplace_copy", _fake_batch_inplace_copy
    )


def _make_replay_runner() -> ServeGraphCaptureRunner:
    """Builds a runner wired only for :meth:`replay` (no warmup/capture)."""
    runner = ServeGraphCaptureRunner.__new__(ServeGraphCaptureRunner)
    runner._records = {}
    runner.graph_entries = {}
    runner._model = MagicMock()
    return runner


def _install_scenario(
    runner: ServeGraphCaptureRunner,
    device_ids: list[int],
    host_device_ids: set[int] | None = None,
) -> tuple[BatchCharacteristics, Any, list[_FakeBuffer]]:
    """Installs one captured graph whose copies target ``device_ids`` in order.

    Returns the characteristics + model_inputs to pass to :meth:`replay` and
    the destination (captured) buffers so callers can assert post-replay values.
    Ids in ``host_device_ids`` are modelled as host-resident (``is_host``).
    """
    host_ids = host_device_ids or set()
    devices: dict[int, _FakeDevice] = {}
    src_buffers: list[_FakeBuffer] = []
    dst_buffers: list[_FakeBuffer] = []
    for i, device_id in enumerate(device_ids):
        device = devices.setdefault(
            device_id,
            _FakeDevice(
                "host" if device_id in host_ids else "gpu",
                device_id,
                is_host=device_id in host_ids,
            ),
        )
        src_buffers.append(_FakeBuffer(device, value=("v", device_id, i)))
        dst_buffers.append(_FakeBuffer(device, value=None))

    bc = BatchCharacteristics(
        batch_size=2, max_prompt_length=1, max_cache_valid_length=128
    )
    key = MLAAttnKey(batch_size=2, max_prompt_length=1, num_partitions=3)
    runner._records[bc] = key
    runner.graph_entries[key] = cast(GraphEntry, (tuple(dst_buffers), None))
    model_inputs = cast(Any, SimpleNamespace(buffers=tuple(src_buffers)))
    return bc, model_inputs, dst_buffers


# -- Real CPU Buffer tests -------------------------------------------------


def test_batch_inplace_copy_cpu_parity() -> None:
    """batch_inplace_copy on CPU buffers matches a per-copy loop.

    Five source buffers each filled with a distinct scalar; batch-copies all
    into zero-initialised destinations and checks every pair bit-exactly.
    """
    n = 5
    srcs = [
        Buffer.from_numpy(np.full((4,), float(i), dtype=np.float32))
        for i in range(n)
    ]
    dsts = [
        Buffer.from_numpy(np.zeros((4,), dtype=np.float32)) for _ in range(n)
    ]
    batch_inplace_copy(dsts, srcs)

    for i, (dst, src) in enumerate(zip(dsts, srcs, strict=True)):
        np.testing.assert_array_equal(
            dst.to_numpy(),
            src.to_numpy(),
            err_msg=f"batch copy mismatch at index {i}",
        )


def test_batch_inplace_copy_skips_identity() -> None:
    """Identity pairs (src is dst) are no-ops; adjacent pairs still copy.

    Passes one self-referential pair plus one real copy in a single batch call.
    The self-referential destination must be unchanged; the other must update.
    """
    stable = Buffer.from_numpy(np.array([42.0], dtype=np.float32))
    other_src = Buffer.from_numpy(np.array([99.0], dtype=np.float32))
    other_dst = Buffer.from_numpy(np.zeros(1, dtype=np.float32))

    batch_inplace_copy([stable, other_dst], [stable, other_src])

    np.testing.assert_array_equal(stable.to_numpy(), [42.0])
    np.testing.assert_array_equal(other_dst.to_numpy(), [99.0])


# -- Routing tests via fake buffers (no GPU required) ---------------------


def test_batched_preface_multi_device_output_parity(
    fake_batch_copy: None,
) -> None:
    """Batched preface delivers each destination its paired source value.

    Three device IDs interleaved; replay() hands them all to one
    batch_inplace_copy and the driver groups by destination device. Every
    destination must end up with its paired source's value.
    """
    runner = _make_replay_runner()
    bc, model_inputs, dst_buffers = _install_scenario(
        runner, [0, 1, 2, 0, 1, 2]
    )
    expected = [src.value for src in model_inputs.buffers]

    runner.replay(model_inputs=model_inputs, batch_characteristics=bc)

    assert [buf.value for buf in dst_buffers] == expected


def test_batched_preface_host_destinations_copy_inline(
    fake_batch_copy: None,
) -> None:
    """Host-resident destinations copy inline via inplace_copy_from.

    Device ID 9 is modelled as host-resident; its copies must go through the
    per-copy inline path (not the batch path) and still deliver the correct
    values.
    """
    runner = _make_replay_runner()
    bc, model_inputs, dst_buffers = _install_scenario(
        runner, [0, 9, 0], host_device_ids={9}
    )
    expected = [src.value for src in model_inputs.buffers]

    runner.replay(model_inputs=model_inputs, batch_characteristics=bc)

    assert [buf.value for buf in dst_buffers] == expected


def test_batched_preface_single_device(fake_batch_copy: None) -> None:
    """Single-accelerator model: all pairs go through one batch_inplace_copy."""
    runner = _make_replay_runner()
    bc, model_inputs, dst_buffers = _install_scenario(runner, [0, 0, 0])
    expected = [src.value for src in model_inputs.buffers]

    runner.replay(model_inputs=model_inputs, batch_characteristics=bc)

    assert [buf.value for buf in dst_buffers] == expected


def test_batched_preface_identity_short_circuit(fake_batch_copy: None) -> None:
    """replay() identity pairs are no-ops; adjacent non-identity pairs copy.

    One slot where src IS dst (identity-stable device metadata, short-circuits)
    and one slot where src is a distinct object. After replay the identity
    slot must be unchanged and the other slot must carry the source's value.
    """
    runner = _make_replay_runner()
    gpu_dev = _FakeDevice("gpu", 0)

    stable = _FakeBuffer(gpu_dev, value="stable")
    fresh_src = _FakeBuffer(gpu_dev, value="fresh_val")
    fresh_dst = _FakeBuffer(gpu_dev, value=None)

    bc = BatchCharacteristics(
        batch_size=2, max_prompt_length=1, max_cache_valid_length=128
    )
    key = MLAAttnKey(batch_size=2, max_prompt_length=1, num_partitions=3)
    runner._records[bc] = key
    # stable is both captured input and live input → src is dst.
    runner.graph_entries[key] = cast(GraphEntry, ((stable, fresh_dst), None))
    model_inputs = cast(Any, SimpleNamespace(buffers=(stable, fresh_src)))

    runner.replay(model_inputs=model_inputs, batch_characteristics=bc)

    assert stable.value == "stable"  # identity slot: no-op
    assert fresh_dst.value == "fresh_val"  # non-identity slot: copied
