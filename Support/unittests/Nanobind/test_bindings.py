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

import bindings  # type: ignore
import pytest


def test_logical_result() -> None:
    assert bindings.return_logical_result_success()
    assert not bindings.return_logical_result_failure()


def test_error_or_success() -> None:
    assert bindings.return_error_or_success_success() is None
    with pytest.raises(RuntimeError):
        bindings.return_error_or_success_failure()


def test_error_or() -> None:
    assert bindings.return_error_or_success() == 42
    with pytest.raises(RuntimeError):
        bindings.return_error_or_failure()
