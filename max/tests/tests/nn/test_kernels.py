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

"""Checks for CPU-testable helpers in ``kernels.py``.

``_fp6_format_code`` is the boundary between the Python ``"e2m3"``/``"e3m2"``
encoding names and the integer ``FP6_FORMAT`` op parameter; a silent
off-by-one or swapped mapping there would route a matmul to the wrong OCP
encoding without any shape or dtype check catching it.
"""

from __future__ import annotations

import pytest
from max.nn.kernels import _fp6_format_code


@pytest.mark.parametrize(
    "fp6_format, expected_code",
    [
        ("e2m3", 0),
        ("e3m2", 1),
    ],
)
def test_fp6_format_code_matches_op_parameter(
    fp6_format: str, expected_code: int
) -> None:
    assert _fp6_format_code(fp6_format) == expected_code


def test_fp6_format_code_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="fp6_format must be one of"):
        _fp6_format_code("e1m4")
