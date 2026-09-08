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
#
# Various integration tests for function types.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


def target(foo: Foo) -> Int:
    foo.closure_taking_a_foo(foo)
    return foo.thing


def bar(var s: Foo):
    print(s.thing)


struct Foo(Deinitable, ImplicitlyCopyable):
    var thing: Int
    var closure_taking_a_foo: def(var s: Foo) thin

    def __init__(out self, x: Int, f: def(var s: Foo) thin):
        self.thing = x
        self.closure_taking_a_foo = f


def main():
    # COM: runtime
    # CHECK: 4
    var foo_run = Foo(4, bar)
    _ = target(foo_run)

    # COM: comptime
    # CHECK: 2
    comptime foo = Foo(2, bar)
    comptime z = target(foo)
    var y = materialize[foo]()
    _ = target(y)
