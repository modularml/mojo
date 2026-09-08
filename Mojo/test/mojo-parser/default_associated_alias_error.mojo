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

# Basic tests for when multiple parent traits define a associated alias with
# conflicting signature and no user-provided value in the child struct.


trait B:
    # expected-note@below{{conflicting implementation from trait B here}}
    comptime a: Int = 1


trait A:
    # expected-note@below{{original default implementation from trait A here}}
    comptime a: Int = 2


# expected-error@below{{trait member 'a' has conflicting default implementations in B and A; you must implement it manually}}
struct Foo(A, B, Movable where False):
    pass
