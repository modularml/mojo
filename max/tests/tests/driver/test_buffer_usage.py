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
"""Tests for the Buffer Usage descriptor (MXF-635)."""

from enum import Flag

import numpy as np
from max.driver import CPU, Buffer, Usage
from max.dtype import DType


def test_usage_is_a_flag() -> None:
    assert issubclass(Usage, Flag)
    assert Usage.DEFAULT == Usage(0)
    assert Usage.STAGING != Usage.DEFAULT


def test_usage_flag_composition() -> None:
    combined = Usage.STAGING | Usage.DEFAULT
    assert Usage.STAGING in combined
    assert not (Usage.DEFAULT & Usage.STAGING)


def test_default_buffer_reports_default_usage() -> None:
    buf = Buffer(dtype=DType.float32, shape=[4], device=CPU())
    assert buf.usage == Usage.DEFAULT
    assert not buf.pinned


def test_staging_on_host_device_succeeds() -> None:
    # A host device can't page-lock, so STAGING degrades to a plain host
    # allocation: intent is honored (`usage`), mechanism is not (`pinned`).
    buf = Buffer(
        dtype=DType.float32,
        shape=[4],
        device=CPU(),
        usage=Usage.STAGING,
    )
    assert buf.usage == Usage.STAGING
    assert not buf.pinned


def test_staging_on_host_device_is_readable_and_writable() -> None:
    buf = Buffer(
        dtype=DType.float32,
        shape=[4],
        device=CPU(),
        usage=Usage.STAGING,
    )
    arr = buf.to_numpy()
    arr[:] = np.arange(4, dtype=np.float32)
    np.testing.assert_array_equal(
        buf.to_numpy(), np.arange(4, dtype=np.float32)
    )


def test_staging_zeros_on_host_device_zeroes() -> None:
    buf = Buffer.zeros(
        shape=[4], dtype=DType.float32, device=CPU(), usage=Usage.STAGING
    )
    assert buf.usage == Usage.STAGING
    assert not buf.pinned
    np.testing.assert_array_equal(buf.to_numpy(), np.zeros(4, dtype=np.float32))
