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

# RUN: %mojo %s | FileCheck %s

# Smaller reproduction of the split + MCLink `_PromotedConst` miscompile
# on macos.
#
# Two specializations of a String-parameterized struct each hold a distinct
# compile-time SIMD constant. With optimization and parallel (per-function
# split) codegen, the `ab` specialization loads the `aaaa...` specialization's
# promoted constant, so its whole-vector compare no longer matches.

from std.testing import assert_true

comptime _SimdVec[W: Int] = SIMD[.uint8, W]


struct Matcher[pattern: String](Copyable, Movable):
    comptime _W = Self.pattern.byte_length()
    comptime _src = Self.pattern.as_bytes()

    var _lit: _SimdVec[Self._W]

    def __init__(out self):
        var v = _SimdVec[Self._W](0)
        comptime for i in range(Self._W):
            comptime b = Self._src[i]
            v[i] = b
        self._lit = rebind_var[type_of(self._lit)](v)

    def matches_prefix(self, data: Span[Byte, _]) -> Bool:
        var lit = rebind[_SimdVec[Self._W]](self._lit)
        print(lit)
        var chunk = _SimdVec[Self._W](0)
        comptime for k in range(Self._W):
            chunk[k] = data[k]
        return chunk == lit


# @no_inline keeps each specialization in its own per-function codegen split,
# which is required to trigger the cross-split promoted-constant collision.
@no_inline
def check[S: String]() raises:
    var m = Matcher[S]()
    var s = String(S)
    # Two uses of the constant are needed: promotion to a shared global only
    # fires with more than one use, and that global is what collides.
    assert_true(m.matches_prefix(s.as_bytes()))
    assert_true(m.matches_prefix(s.as_bytes()))


def main() raises:
    # CHECK: [97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97]
    # CHECK: [97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97]
    check["aaaaaaaaaaaaaaaa"]()

    # CHECK: [97, 98]
    # CHECK: [97, 98]
    check["ab"]()
