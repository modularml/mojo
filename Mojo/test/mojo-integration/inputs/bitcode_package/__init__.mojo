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
"""Test package with extern functions that are implemented in external bitcode."""


@extern("extern_add")
def extern_add(a: Int32, b: Int32) abi("Mojo") -> Int32:
    ...


@extern("extern_multiply")
def extern_multiply(a: Int32, b: Int32) abi("Mojo") -> Int32:
    ...


# Use the extern add method to compute the result.
def double_add(a: Int32, b: Int32) -> Int32:
    # Another method that directly uses an extern function
    return extern_add(extern_add(a, b), extern_add(a, b))
