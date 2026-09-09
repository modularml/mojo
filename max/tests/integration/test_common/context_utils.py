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
"""Context utility functions for use in testing infrastructure."""

import numpy as np
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.modeling.types import RequestID


def create_text_context(
    tokens: np.ndarray, max_length: int = 1000
) -> TextContext:
    # TokenBuffer requires non-empty arrays
    if len(tokens) == 0:
        raise ValueError(
            "create_text_context requires non-empty token array. "
            "TokenBuffer does not support empty arrays."
        )

    # Ensure tokens are int64 as required by TokenBuffer
    if tokens.dtype != np.int64:
        tokens = tokens.astype(np.int64)

    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(tokens),
    )
