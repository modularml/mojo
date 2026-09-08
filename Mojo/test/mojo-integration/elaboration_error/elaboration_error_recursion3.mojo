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


# This creates recursive cycles: foo[D] -> bar[D] -> foo[D] and foo[D] -> baz[D] -> foo[D]
# expected-note @below {{recursively instantiated through here}}
def bar[D: Int]() -> Int:
    comptime x = foo[D]()
    return x


# expected-note @below {{function instantiation failed}}
# expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
def baz[D: Int]() -> Int:
    comptime x = foo[D]()
    return x


# expected-note @below {{function instantiation failed}}
def foo[D: Int]() -> Int:
    # expected-note @below {{recursively instantiated through here}}
    var x = bar[D]()
    # expected-note @below {{call expansion failed with parameter value(s): ("D": 2)}}
    var y = baz[D]()
    _ = x
    _ = y
    return y


# expected-error @below {{function instantiation failed}}
def main():
    # expected-note @below {{call expansion failed with parameter value(s): ("D": 2)}}
    _ = foo[2]()
