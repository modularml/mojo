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

# Regression test: a closure capturing a pointer whose origin is a field
# projection of `self` must keep that origin mutable.

from std.memory import Pointer


@fieldwise_init
struct Inner(Copyable, Movable):
    var value: Int


struct Outer:
    var inner: Inner

    def __init__(out self):
        self.inner = Inner(1)

    def apply(mut self):
        var p = Pointer(to=self.inner)

        @always_inline
        def closure() {imm}:
            p[].value = 5

        closure()


def main():
    var o = Outer()
    o.apply()
    # CHECK: 5
    print(o.inner.value)
