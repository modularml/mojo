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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# Reject SIMD types whose resolved length is not within the acceptable range.
# FIXME: These should be correct on construction from the Mojo code itself,
# but as per MOCO-2839 we're (often) silently dropping assertions during the
# folding of @always_inline("builtin") functions. This is a clumsy fallback
# but a good backstop to ensure we don't generate invalid code.


def main():
    # CHECK: error: SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<0, f32>'
    var x = SIMD[.float32, 0](0)
    print(x)
    # CHECK: error: SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<-1, f32>'
    var y = SIMD[.float32, -1](0)
    print(y)
    # CHECK: error: SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<3, f32>'
    var z = SIMD[.float32, 3](0)
    print(z)
