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

# RUN: kgen -elaborate %s --verify-diagnostics

# Every generator in this cycle is trivial enough for `ApplyInliner` to give it
# an inlined form: the body is just a parameter constant and a return. That form
# then applies the generator it came from, so expanding it re-derives the same
# apply forever and used to overflow the stack before the elaborator ever saw
# the cycle. The sibling recursion tests route their cycle through a generator
# holding a runtime call, which never gets an inlined form, so the expansion has
# nothing to loop on there. See MOCO-4518.


# expected-note @below {{function instantiation failed}}
# expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
# expected-note @below {{function recursively calls itself in the parameter domain}}
def g[n: Int]() -> Int:
    comptime m = g[n]()
    return m


# expected-error @below {{function instantiation failed}}
def main():
    # expected-note @below {{call expansion failed with parameter value(s): ("n": 1)}}
    print(g[1]())
