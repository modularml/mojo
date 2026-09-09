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
# RUN: %parse-mojo-isolated -verify-diagnostics %s

# COM: Negative companion to closure_wrapper_marker_conformances.mojo
# COM: (MOCO-4240). Marker-trait conformance on closure wrappers must stay
# COM: gated on the closure's actual properties: a memory-convention closure
# COM: is not RegisterPassable, and a closure move-capturing a non-copyable
# COM: value is not ImplicitlyCopyable.


# A plain (non-register-passable) struct: capturing it forces the closure to
# the memory convention.
struct MemOnly(Copyable, ImplicitlyCopyable, Movable, Deinitable):
    var x: Int

    def __init__(out self):
        self.x = 0


struct MoveOnly(Movable, Deinitable):
    var x: Int

    def __init__(out self):
        self.x = 0


# expected-note @below {{function declared here}}
def needs_rp[
    FuncType: def () -> Int,
    # expected-note @below {{constraint declared here evaluated to False, expected 'conforms_to(FuncType, RegisterPassable)'}}
](func: FuncType) -> Int where conforms_to(FuncType, RegisterPassable):
    return func()


# expected-note @below {{function declared here}}
def needs_ic[
    FuncType: def () -> Int,
    # expected-note @below {{constraint declared here evaluated to False, expected 'conforms_to(FuncType, ImplicitlyCopyable)'}}
](func: FuncType) -> Int where conforms_to(FuncType, ImplicitlyCopyable):
    return func()


def use_mem() -> Int:
    var m = MemOnly()

    def mem_closure() {var m} -> Int:
        return m.x

    # expected-error @below {{invalid call to 'needs_rp': violated constraint}}
    return needs_rp(mem_closure)


def use_move() -> Int:
    var m = MoveOnly()

    def move_closure() {var m^} -> Int:
        return m.x

    # expected-error @below {{invalid call to 'needs_ic': violated constraint}}
    return needs_ic(move_closure)
