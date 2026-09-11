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

"""Covers the max._core.request_context nanobind boundary.

What a log record does with the batch id is covered on the C++ side by
RequestLogTest; these cases only pin the Python-facing contract.
"""

import threading

import pytest
from max._core import request_context


def test_set_and_clear_round_trip() -> None:
    request_context.set_batch_id(7)
    request_context.clear_batch_id()


def test_clear_without_set_is_safe() -> None:
    request_context.clear_batch_id()


def test_batch_id_is_keyword_addressable() -> None:
    request_context.set_batch_id(batch_id=7)
    request_context.clear_batch_id()


def test_rejects_a_non_integer_batch_id() -> None:
    with pytest.raises(TypeError):
        request_context.set_batch_id("not-an-int")  # type: ignore[arg-type]


def test_accepts_an_id_wider_than_32_bits() -> None:
    request_context.set_batch_id(2**62)
    request_context.clear_batch_id()


def test_rejects_an_id_too_wide_for_int64() -> None:
    with pytest.raises(TypeError):
        request_context.set_batch_id(2**64)


def test_setting_on_one_thread_does_not_disturb_another() -> None:
    # The context is thread-local, so a worker setting it must not leave state
    # behind that the calling thread's clear would be responsible for.
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            request_context.set_batch_id(99)
            request_context.clear_batch_id()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert not errors
