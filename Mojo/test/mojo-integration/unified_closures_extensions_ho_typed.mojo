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
# RUN: %mojo %s 3 1 | FileCheck %s

# COM: Support transitive where clauses.
# COM: In this case, G.T is bound to R so we have to prove that G.T eq Inner.T

from std.sys import argv


def sinkHO[
    R: Movable & Deinitable,
    //,
    Inner: def() -> R,
    F: def(cb: Inner) -> R,
](*, call: F, cb: Inner) -> R:
    return call(cb)


def forwardHO[
    T: Movable & Deinitable, //, Inner: def() -> T, G: def(cb: Inner) -> T
](*, call: G, cb: Inner) -> T:
    return sinkHO(call=call, cb=cb)


def main() raises:
    var x = atol(argv()[1])
    var y = atol(argv()[2])

    def make() {var} -> Int:
        return x + y

    comptime X = type_of(make)

    def apply(cb: X) {var} -> Int:
        return cb() + 1

    # CHECK: ho_typed: 5
    print("ho_typed:", forwardHO(call=apply, cb=make))
