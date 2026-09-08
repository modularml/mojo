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
"""Bitwise operations: manipulation, counting, rotation, and power-of-two utilities.

The `bit` package provides low-level bitwise operations and utilities for
efficient bit manipulation on integer values and SIMD vectors. It includes
functions for counting bits, reversing bit patterns, byte swapping, rotating
bits, and computing power-of-two boundaries. These operations are fundamental
for systems programming, cryptography, data compression, and performance-critical
algorithms.

Use this package for low-level bit manipulation, implementing custom hash
functions, optimizing memory layouts, working with packed data structures, or
any algorithm requiring direct bit-level control.
"""

from .bit import (
    bit_not,
    bit_reverse,
    bit_width,
    byte_swap,
    count_leading_zeros,
    count_trailing_zeros,
    log2_ceil,
    log2_floor,
    next_power_of_two,
    pop_count,
    prev_power_of_two,
    rotate_bits_left,
    rotate_bits_right,
)
