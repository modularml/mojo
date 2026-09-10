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

# Test that `@inline(value)` picks the inline level per instantiation when the
# value is only known once parameters are bound. Every function here resolves
# to "never", so each keeps a definition of its own in the one `noinline`
# attribute group; the "always" instantiation is inlined and leaves none.

# RUN: mkdir -p %t
# RUN: %mojo-build %s -o %t/inline.ll --emit llvm
# RUN: FileCheck %s --input-file=%t/inline.ll

comptime NEVER = InlineLevel.never


def pick[fast: Bool]() -> InlineLevel:
    return .always if fast else .never


# COM: The definitions appear in source order, and all four share one group.
# CHECK: define{{.*}}chosen_by_function{{.*}}"() #[[#ATTR:]]
@inline(pick[False]())
def chosen_by_function() -> Int:
    return 5


# CHECK: define{{.*}}parametric{{.*}}"() #[[#ATTR]]
@inline(policy)
def parametric[policy: InlineLevel]() -> Int:
    return 4


# CHECK: define{{.*}}via_alias{{.*}}"() #[[#ATTR]]
@inline(NEVER)
def via_alias() -> Int:
    return 3


# COM: A nested function reaches the generator through a different lowering
# COM: path, which must carry the unfolded expression too.
# CHECK: define{{.*}}inner{{.*}}"() #[[#ATTR]]
def with_nested() -> Int:
    @inline(policy)
    def inner[policy: InlineLevel]() -> Int:
        return 7

    return inner[.never]()


def main():
    print(
        parametric[.always]()
        + parametric[.never]()
        + via_alias()
        + with_nested()
        + chosen_by_function()
    )


# CHECK: attributes #[[#ATTR]] = { noinline
