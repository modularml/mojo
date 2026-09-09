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
"""Tests for ``max.driver.Device.__unsafe_enqueue_py_host_func``."""

import pytest
from max import driver


@pytest.fixture
def accelerator() -> driver.Accelerator:
    if not driver.accelerator_count():
        pytest.skip("Requires GPU")
    return driver.Accelerator()


def test_unsafe_enqueue_host_func_runs_callback(
    accelerator: driver.Accelerator,
) -> None:
    """Callback runs on an accelerator after preceding work completes."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip(
            "__unsafe_enqueue_py_host_func is only implemented for CUDA/HIP devices"
        )

    calls: list[str] = []

    def cb() -> None:
        calls.append("hello")

    for _ in range(4):
        accelerator.__unsafe_enqueue_py_host_func(cb)

    accelerator.synchronize()
    assert calls == ["hello"] * 4


def test_unsafe_enqueue_host_func_with_raising_callback(
    accelerator: driver.Accelerator,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Callback raises an exception."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip(
            "__unsafe_enqueue_py_host_func is only implemented for CUDA/HIP devices"
        )

    def cb() -> None:
        raise RuntimeError("boom")

    accelerator.__unsafe_enqueue_py_host_func(cb)

    # The callback exception is printed to stderr.
    # Synchronize does not raise an error.
    accelerator.synchronize()

    assert "RuntimeError: boom" in capsys.readouterr().err


def test_unsafe_enqueue_host_func_on_non_cuda_accelerator_raises(
    accelerator: driver.Accelerator,
) -> None:
    """Non-CUDA/HIP accelerators raise until we add support."""
    if accelerator.api in ("cuda", "hip"):
        pytest.skip("Covered by test_unsafe_enqueue_host_func_runs_callback")

    with pytest.raises(RuntimeError):
        accelerator.__unsafe_enqueue_py_host_func(lambda: None)


def test_unsafe_enqueue_host_func_on_cpu_raises() -> None:
    """CPU devices do not support host callbacks."""
    cpu = driver.CPU()
    with pytest.raises(RuntimeError):
        cpu.__unsafe_enqueue_py_host_func(lambda: None)
