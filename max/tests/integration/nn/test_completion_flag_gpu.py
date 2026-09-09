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
"""Smoke tests for the ``CompletionFlag`` nanobind class.

Exercises the host-side API in isolation (allocation, reset, signal,
load). The producer/consumer methods (``DeviceQueue.wait_for_host_value``,
``mo.wait_host_value``) that actually use the flag's device pointer land
in follow-on PRs in the same stack and have their own GPU tests.
"""

import pytest
from max.driver import Accelerator, CompletionFlag


@pytest.fixture
def accelerator() -> Accelerator:
    return Accelerator()


def test_completion_flag_construction_initializes_to_zero(
    accelerator: Accelerator,
) -> None:
    """A freshly allocated flag reads back as zero."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip("CompletionFlag is only supported on CUDA/HIP")

    flag = CompletionFlag(accelerator)
    assert flag.load() == 0


def test_completion_flag_device_ptr_is_nonzero(
    accelerator: Accelerator,
) -> None:
    """``device_ptr`` is the device-visible address of the flag, suitable
    for passing to graph ops or stream APIs that wait on a memory value.
    Must be non-null."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip("CompletionFlag is only supported on CUDA/HIP")

    flag = CompletionFlag(accelerator)
    assert flag.device_ptr != 0


def test_completion_flag_signal_then_load_round_trip(
    accelerator: Accelerator,
) -> None:
    """``signal(value)`` writes a 64-bit value with release semantics;
    ``load()`` reads it back with acquire semantics."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip("CompletionFlag is only supported on CUDA/HIP")

    flag = CompletionFlag(accelerator)
    flag.signal(1)
    assert flag.load() == 1

    flag.signal(0xDEADBEEF)
    assert flag.load() == 0xDEADBEEF


def test_completion_flag_reset_after_signal_clears_to_zero(
    accelerator: Accelerator,
) -> None:
    """``reset()`` after a ``signal()`` returns the flag to zero."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip("CompletionFlag is only supported on CUDA/HIP")

    flag = CompletionFlag(accelerator)
    flag.signal(42)
    assert flag.load() == 42
    flag.reset()
    assert flag.load() == 0


def test_completion_flag_raises_on_cpu_device() -> None:
    """Only CUDA backends override ``allocateCompletionFlag``; others must
    surface the MLRT error as a Python ``RuntimeError`` rather than
    crashing."""
    from max.driver import CPU

    cpu = CPU()
    with pytest.raises(RuntimeError, match="not supported"):
        CompletionFlag(cpu)


def test_completion_flag_unsafe_ptr_is_distinct_nonzero(
    accelerator: Accelerator,
) -> None:
    """``_unsafe_ptr`` returns the raw address of the underlying C++
    ``M::Driver::CompletionFlag`` for packing into a graph-op payload.
    Must be non-null and distinct from ``device_ptr`` (which addresses
    the 8-byte flag slot; this one addresses the wrapper object
    itself)."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip("CompletionFlag is only supported on CUDA/HIP")

    flag = CompletionFlag(accelerator)
    assert flag._unsafe_ptr != 0
    assert flag._unsafe_ptr != flag.device_ptr


def test_completion_flag_multiple_allocations_are_independent(
    accelerator: Accelerator,
) -> None:
    """Two flags allocated against the same device must have distinct
    storage; mutating one through ``reset()`` should not affect the other.
    Catches a regression where someone might accidentally cache a single
    pinned page across allocations.

    We can only observe ``reset() == 0`` from Python, so the strongest
    portable check is that the two ``device_ptr``s differ.
    """
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip("CompletionFlag is only supported on CUDA/HIP")

    a = CompletionFlag(accelerator)
    b = CompletionFlag(accelerator)
    assert a.device_ptr != b.device_ptr
    assert a._unsafe_ptr != b._unsafe_ptr
