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

# RUN: %parse-mojo-isolated --verify-diagnostics %s

# Basic tests for when multiple parent traits define a function with
# conflicting signature and no override in the child struct.


trait Foo1:
    # expected-note @+1 {{original default implementation from trait 'Foo1' here}}
    def foo(self) -> Int:
        return 1


trait Foo2(Foo1):
    # expected-note @+1 {{conflicting implementation from trait 'Foo2' here}}
    def foo(self) -> Int:
        return 2


trait Foo3(Foo1):
    # expected-note @+1 {{conflicting implementation from trait 'Foo3' here}}
    def foo(self) -> Int:
        return 3


@fieldwise_init
# expected-error @+2 {{trait method requirement 'foo' has conflicting default implementations in 'Foo1' and 'Foo2'; you must implement it manually}}
# expected-error @+1 {{trait method requirement 'foo' has conflicting default implementations in 'Foo1' and 'Foo3'; you must implement it manually}}
struct Foo(Foo2, Foo3, Movable where False):
    pass


trait AA1:
    comptime X: ImplicitlyCopyable

    # expected-note @+1 {{original default implementation from trait 'AA1' here}}
    def zork(self, x: Self.X) -> Self.X:
        return x


trait AA2:
    comptime X: ImplicitlyCopyable

    # expected-note @+1 {{conflicting implementation from trait 'AA2' here}}
    def zork(self, x: Self.X) -> Self.X:
        return x


@fieldwise_init
# expected-error @+1 {{trait method requirement 'zork' has conflicting default implementations in 'AA1' and 'AA2'; you must implement it manually}}
struct Bar(AA1, AA2, Movable where False):
    comptime X: ImplicitlyCopyable = Int


trait WithAsyncMethod:
    # expected-error @+1 {{async defaulted trait methods are not supported; remove the method or remove 'async'}}
    async def async_default_method(self) -> Int:
        return 42


# TODO: Should the 'original' implementation not be FooC here? It's ordering
# alphabetically.
@fieldwise_init
struct RP(TrivialRegisterPassable):
    pass


trait FooC:
    # expected-note @+1 {{conflicting implementation from trait 'FooC' here}}
    def foo(self) -> RP:
        return RP()

    def bar(self) -> RP:
        return RP()


trait FooB(FooC):
    # expected-note @+1 {{original default implementation from trait 'FooB' here}}
    def foo(self) -> RP:
        return RP()

    def bar(self) -> RP:
        return RP()


# expected-error @+1 {{trait method requirement 'foo' has conflicting default implementations in 'FooB' and 'FooC'; you must implement it manually}}
struct FooActual(FooB, Movable where False):
    pass
