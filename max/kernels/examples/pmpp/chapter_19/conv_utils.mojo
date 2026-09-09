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

"""Convolution utilities for Chapter 19.

This module provides common utilities for convolution implementations including:
- Indexing functions for 4D tensors
- CPU reference implementation
- Data initialization and verification
"""

from std.itertools import product
from std.math import abs
from std.random import rand


@always_inline
def idx_x(
    n: Int, c: Int, h: Int, w: Int, N: Int, C: Int, H: Int, W: Int
) -> Int:
    """Index into X tensor: (N, C, H, W) layout."""
    return n * (C * H * W) + c * (H * W) + h * W + w


@always_inline
def idx_f(m: Int, c: Int, p: Int, q: Int, M: Int, C: Int, K: Int) -> Int:
    """Index into F (filter) tensor: (M, C, K, K) layout."""
    return m * (C * K * K) + c * (K * K) + p * K + q


@always_inline
def idx_y(
    n: Int, m: Int, h: Int, w: Int, N: Int, M: Int, H_out: Int, W_out: Int
) -> Int:
    """Index into Y (output) tensor: (N, M, H_out, W_out) layout."""
    return n * (M * H_out * W_out) + m * (H_out * W_out) + h * W_out + w


def conv_cpu(
    N: Int,
    M: Int,
    C: Int,
    H: Int,
    W: Int,
    K: Int,
    X: UnsafePointer[mut=False, Float32, _],
    F: UnsafePointer[mut=False, Float32, _],
    Y: UnsafePointer[mut=True, Float32, _],
):
    """CPU implementation of batched convolution (Figure 19.4 logic).

    Performs a basic convolution with no padding and stride=1.

    Args:
        N: Batch size.
        M: Output channels.
        C: Input channels.
        H: Input height.
        W: Input width.
        K: Kernel size (K x K).
        X: Input tensor pointer.
        F: Filter tensor pointer.
        Y: Output tensor pointer.
    """
    var H_out = H - K + 1
    var W_out = W - K + 1

    for n, m, h, w in product(range(N), range(M), range(H_out), range(W_out)):
        var acc: Float32 = 0.0
        for c, p, q in product(range(C), range(K), range(K)):
            var x_idx = idx_x(n, c, h + p, w + q, N, C, H, W)
            var f_idx = idx_f(m, c, p, q, M, C, K)
            acc += X[x_idx] * F[f_idx]

        var y_idx = idx_y(n, m, h, w, N, M, H_out, W_out)
        Y[y_idx] = acc


def init_data(data: UnsafePointer[mut=True, Float32, _], size: Int):
    """Initialize data with random values in [-0.5, 0.5].

    Args:
        data: Pointer to data.
        size: Number of elements.
    """
    rand(data, size)
    for i in range(size):
        data[i] -= 0.5


def verify_results(
    h_Y_cpu: UnsafePointer[mut=False, Float32, _],
    h_Y_gpu: UnsafePointer[mut=False, Float32, _],
    N: Int,
    M: Int,
    H_out: Int,
    W_out: Int,
) -> Bool:
    """Verify GPU results against CPU reference.

    Args:
        h_Y_cpu: CPU reference output.
        h_Y_gpu: GPU computed output.
        N: Batch size.
        M: Output channels.
        H_out: Output height.
        W_out: Output width.

    Returns:
        True if results match within tolerance, False otherwise.
    """
    var total_elements = N * M * H_out * W_out
    var max_error: Float32 = 0.0
    comptime TOLERANCE: Float32 = 1e-3

    for i in range(total_elements):
        var diff = abs(h_Y_cpu[i] - h_Y_gpu[i])
        if diff > max_error:
            max_error = diff

        if diff > TOLERANCE:
            print(
                "Mismatch at index",
                i,
                ": CPU",
                h_Y_cpu[i],
                ", GPU",
                h_Y_gpu[i],
                ", Diff",
                diff,
            )
            return False

    print("Max Error:", max_error)
    return True
