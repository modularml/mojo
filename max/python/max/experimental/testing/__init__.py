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

"""Testing utilities for the experimental :class:`~max.experimental.tensor.Tensor` API."""

from __future__ import annotations

from max.driver import DLPackArray
from max.experimental.tensor import NestedArray, Number, Tensor

__all__ = ["assert_all_close"]


def assert_all_close(
    t1: DLPackArray | NestedArray | Number,
    t2: Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    """Asserts two tensors are elementwise close within the given tolerances.

    Args:
        t1: The expected value. Converted to a :class:`Tensor` matching
            ``t2``'s dtype and device if it is not already one.
        t2: The actual :class:`Tensor` to compare against.
        atol: The maximum allowed absolute difference.
        rtol: The maximum allowed relative difference.

    Raises:
        AssertionError: If any element exceeds ``atol`` or ``rtol``.
    """
    if not isinstance(t1, Tensor):
        t1 = Tensor(t1, dtype=t2.dtype, device=t2.device)

    absolute_difference = abs(t1 - t2)
    # TODO: div0
    left_relative_difference = abs(absolute_difference / t1)
    right_relative_difference = abs(absolute_difference / t2)

    if (d := absolute_difference.max()) > atol:
        idx = absolute_difference.argmax().item()
        raise AssertionError(
            f"atol: tensors not close at index {idx}, {d.item()} > {atol}: \n"
            f"   left[{idx}] = {t1[idx].item()}\n"
            f"  right[{idx}] = {t2[idx].item()}\n"
        )
    elif (d := left_relative_difference.max()) > rtol:
        idx = left_relative_difference.argmax().item()
        raise AssertionError(
            f"rtol: tensors not close at index {idx}, {d.item()} > {rtol}: \n"
            f"   left[{idx}] = {t1[idx].item()}\n"
            f"  right[{idx}] = {t2[idx].item()}\n"
        )
    elif (d := right_relative_difference.max()) > rtol:
        idx = right_relative_difference.argmax().item()
        raise AssertionError(
            f"rtol: tensors not close at index {idx}, {d.item()} > {rtol}: \n"
            f"   left[{idx}] = {t1[idx].item()}\n"
            f"  right[{idx}] = {t2[idx].item()}\n"
        )
