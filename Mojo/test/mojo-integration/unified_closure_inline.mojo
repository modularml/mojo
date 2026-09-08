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
# RUN: kgen %s -elaborate -S -o - | FileCheck %s

# COM: Closure and wrapper are inlined away
# CHECK-NOT: __call__{{.*}}def(a: Int, b: Int, c: Int) -> Int


@no_inline
def callee_three_arg[
    func: def(a: Int, b: Int, c: Int) -> Int
](impl: func, x: Int) -> Int:
    return impl(x, x, x)


def test_always_inline_closure():
    var y = 42

    @always_inline
    def three_arg(a: Int, b: Int, c: Int) {var y} -> Int:
        return a + b + c + y

    _ = callee_three_arg(three_arg, 1)


# COM: Closure body is not inlined. Wrapper __call__ is still inlined.
# CHECK: kgen.func @"{{.*}}bool_closure{{.*}}"
# CHECK-NOT: __call__{{.*}}def(flag: Bool, count: Int) -> Bool


@no_inline
def callee_bool[
    func: def(flag: Bool, count: Int) -> Bool
](impl: func, x: Bool, n: Int) -> Bool:
    return impl(x, n)


def test_wrapper_inlined_no_annotation():
    var y = 7

    @no_inline
    def bool_closure(flag: Bool, count: Int) {var y} -> Bool:
        return flag and count > y

    _ = callee_bool(bool_closure, True, 10)


def main():
    test_always_inline_closure()
    test_wrapper_inlined_no_annotation()
