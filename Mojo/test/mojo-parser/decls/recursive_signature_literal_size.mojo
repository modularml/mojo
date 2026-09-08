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

# RUN: %parse-mojo-isolated -verify-diagnostics %s 2>&1

# A `where` clause on a SIMD constructor whose predicate transitively requires
# the constructor being constrained creates a declaration cycle that poisons
# the `Int` constructor set. Inferring the `__literal_size__` parameter of a
# list-literal constructor materializes an `Int`, so it hits the poisoned set;
# ensure that failure is reported as a normal overload-resolution error instead
# of crashing.

@fieldwise_init
struct Wrapper(ImplicitlyCopyable, RegisterPassable):
    pass


def pred(arg: Int) -> Bool:
    return True


__extension SIMD:
    @implicit
# expected-error @below {{attempt to resolve a recursive reference to declaration '__init__'}}
# expected-note @below {{referenced from here}}
# expected-note @below {{by declaration '__init__'}}
# expected-error @below {{failed to emit constraint expression}}
    def __init__(out self, arg: Wrapper) where pred(0):
        self = {}


# expected-note @below {{candidate not viable: missing required argument: 'move'}}
# expected-note @below {{def __init__(out self, *, deinit move: Self)    # note - generated function}}
struct MyList[T: AnyType]:
# expected-note @below {{candidate not viable: cannot infer the size of the list literal with a previously diagnosed error}}
    def __init__[*, __literal_size__: Int](
        out self,
        var *elements: Self.T,
        __list_literal__: NoneType,
    ):
        pass


def trigger():
# expected-note @below {{referenced through this use}}
# expected-error @below {{no matching function in initialization}}
    var x: MyList[Wrapper] = [Wrapper(), Wrapper()]
