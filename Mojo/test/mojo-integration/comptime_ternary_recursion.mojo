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

# Regression test for MOCO-4518. `countdown` is a parameter constant and a
# return, so `ApplyInliner` gives it an inlined form, and that form applies
# `countdown` itself. The ternary guarding the base case only folds once `n` is
# concrete, so expanding the form while rewriting the generator's own body -
# where `n` is still symbolic - walks the recursion past its base case:
# `countdown[n - 1]`, `countdown[n - 2]`, ... Every level is a fresh apply, so
# no cycle check catches it, and the parameter expression grows until the walk
# overflows the stack. `adaptive_recursion.mojo` covers the `comptime if`
# spelling of the same shape, which folds through a different path.


def countdown[n: Int]() -> Int:
    comptime m = 0 if n == 0 else 1 + countdown[n - 1]()
    return m


# The same shape with the recursion spanning two generators, which no check
# against the generator being expanded can see on its own.
def ping[n: Int]() -> Int:
    comptime m = 0 if n == 0 else pong[n - 1]()
    return m


def pong[n: Int]() -> Int:
    comptime m = 1 if n == 0 else ping[n - 1]()
    return m


def main():
    # CHECK: 4
    print(countdown[4]())
    # CHECK: 0
    print(ping[4]())
    # CHECK: 1
    print(ping[5]())
