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

"""Custom exceptions for MAX serving infrastructure."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager


class OOMError(RuntimeError):
    """Custom exception for out-of-memory errors with helpful guidance."""

    def __init__(self, _: str = ""):
        super().__init__("""
GPU ran out of memory during model execution.

This typically happens when:
1. The model's runtime memory usage is too large for your GPU's memory
2. The batch size is too large for the available memory
3. The sequence length (max_length) is too large

Suggested solutions:
1. Reduce --device-memory-utilization to a smaller value
2. Reduce batch size with --max-batch-size parameter
3. Reduce sequence length with --max-length parameter
4. Reduce max batch input tokens with --max-batch-input-tokens parameter
""")


# TODO: include other patterns (e.g. AMD) here
_oom_pattern = re.compile(".*OUT_OF_MEMORY.*")


@contextmanager
def detect_and_wrap_ooms() -> Iterator[None]:
    """
    Detect  OOM errors and wrap them in a more helpful exception.

    This function checks if the given exception is a out-of-memory error
    and wraps it in a OOMError with helpful guidance if so.

    Args:
        exception: The exception to check

    Returns:
        OOMError if it's OOM, otherwise the original exception
    """

    try:
        yield
    except Exception as exception:
        msg = str(exception)
        if isinstance(exception, ValueError) and _oom_pattern.match(msg):
            raise OOMError() from exception
        raise
