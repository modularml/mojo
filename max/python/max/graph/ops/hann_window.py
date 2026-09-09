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
"""Op implementation for hann_window."""

from __future__ import annotations

import numpy as np
from max.dtype import DType

from ..type import DeviceRef
from ..value import TensorValue
from .constant import constant
from .elementwise import cos
from .range import range


def hann_window(
    window_length: int,
    device: DeviceRef,
    periodic: bool = True,
    dtype: DType = DType.float32,
) -> TensorValue:
    """Computes a Hann window of a given length.

    For a symmetric window of ``N`` points where ``N > 1``, the value at index
    ``n`` is:

    .. code-block:: text

        w[n] = 0.5 * (1 - cos(2 * pi * n / (N - 1)))

    When ``N`` is 0, the result is an empty tensor, and when ``N`` is 1, the
    result is ``[1]``. A periodic window instead computes ``N + 1`` points and
    drops the last one, which makes it suitable for spectral analysis.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("hann_window_example") as graph:
            graph.output(ops.hann_window(4, device, periodic=True))

        model = InferenceSession().load(graph)
        result = model.execute()[0]
        # result holds [0.0, 0.5, 1.0, 0.5].

    Args:
        window_length: The number of points in the window.
        device: The device the window lives on.
        periodic: Whether to return a periodic window. When ``True``, computes
            a symmetric window of ``window_length + 1`` points and drops the
            last point, so that
            ``hann_window(L, periodic=True)`` equals
            ``hann_window(L + 1, periodic=False)[:-1]``. Defaults to ``True``.
        dtype: The output tensor's data type. Defaults to ``float32``.

    Returns:
        A 1-D ``TensorValue`` of shape ``(window_length,)`` representing the
        window.

    Raises:
        ValueError: If ``window_length`` is negative.
        TypeError: If ``window_length`` isn't an integer.
    """
    if not isinstance(window_length, int):
        raise TypeError(
            f"window_length must be an integer, got {type(window_length).__name__}"
        )
    if window_length < 0:
        raise ValueError("window_length must be non-negative")
    if window_length == 0:
        return constant([], dtype, device)
    elif window_length == 1:
        return constant([1], dtype, device)

    if periodic:
        window_length += 1

    window = range(0, window_length, 1, dtype=dtype, device=device)
    window = window * (2.0 * np.pi / np.float64(window_length - 1))
    window = cos(window) * (-0.5) + 0.5

    if periodic:
        window = window[:-1]  # Drop the last point for periodic windows

    return window
