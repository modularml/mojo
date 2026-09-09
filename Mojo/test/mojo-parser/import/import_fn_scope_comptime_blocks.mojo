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

# Imports are allowed inside comptime control flow (unlike runtime control
# flow, see import_errors.mojo), but the binding is scoped to the comptime
# block like any other declaration there: it is usable within the block and
# does not escape to the rest of the function, regardless of whether the
# branch is taken. Imports in dead comptime branches are still resolved
# eagerly, so a nonexistent module is an error even under `comptime if False`.

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%S/inputs %s

# The import is usable within the comptime block.


def use_inside_comptime_if():
    comptime if True:
        from wildcard_shadow_a import shadowed_fn

        _ = shadowed_fn()


# // -----

# The import does not escape the block, even from a taken branch.


def use_after_taken_comptime_if():
    comptime if True:
        from wildcard_shadow_a import shadowed_fn
    # expected-error @below {{use of unknown declaration 'shadowed_fn'}}
    _ = shadowed_fn()


# // -----

# A dead branch's import never binds anything after the block.


def use_after_dead_comptime_if():
    comptime if False:
        from wildcard_shadow_a import shadowed_fn
    # expected-error @below {{use of unknown declaration 'shadowed_fn'}}
    _ = shadowed_fn()


# // -----

# Imports in dead comptime branches are still resolved eagerly: a nonexistent
# module is an error even though the branch is never instantiated.


def dead_branch_bad_module():
    comptime if False:
        # expected-error @below {{unable to locate module 'nonexistent_module_for_test'}}
        import nonexistent_module_for_test


# // -----

# An import inside a comptime for does not escape the loop body, even when the
# loop runs (two trips here).


@fieldwise_init
struct _TwoTrips(TrivialRegisterPassable, Iterator):
    comptime Element = Int
    var n: Int

    def __iter__(self) -> Self:
        return self

    def __next__(mut self) raises StopIteration -> Int:
        if self.n >= 2:
            raise StopIteration()
        self.n += 1
        return self.n

    def __len__(self) -> Int:
        return 2 - self.n


def use_after_comptime_for():
    comptime for _i in _TwoTrips(0):
        from wildcard_shadow_a import shadowed_fn

        _ = shadowed_fn()
    # expected-error @below {{use of unknown declaration 'shadowed_fn'}}
    _ = shadowed_fn()
