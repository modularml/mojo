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
"""Correctness smoke for the FFI bench module.

The actual timing is in `bench.py` (manual target). This pytest test only
verifies the four exposed functions still produce the right values, so that
the bench surface is guaranteed to be wired up and any wrong number `bench.py`
prints reflects perf, not a broken binding.
"""

import mojo_module  # type: ignore[import-not-found]


def test_noop_def_returns_argument() -> None:
    assert mojo_module.noop_def(1) == 1
    assert mojo_module.noop_def("foo") == "foo"


def test_noop_raw_returns_argument() -> None:
    assert mojo_module.noop_raw(1) == 1
    assert mojo_module.noop_raw("foo") == "foo"


def test_add_def_sums_ints() -> None:
    assert mojo_module.add_def(1, 2) == 3
    assert mojo_module.add_def(-5, 10) == 5


def test_add_raw_sums_ints() -> None:
    assert mojo_module.add_raw(1, 2) == 3
    assert mojo_module.add_raw(-5, 10) == 5


def test_noop_raw_fastcall_returns_argument() -> None:
    assert mojo_module.noop_raw_fastcall(1) == 1
    assert mojo_module.noop_raw_fastcall("foo") == "foo"


def test_add_raw_fastcall_sums_ints() -> None:
    assert mojo_module.add_raw_fastcall(1, 2) == 3
    assert mojo_module.add_raw_fastcall(-5, 10) == 5
