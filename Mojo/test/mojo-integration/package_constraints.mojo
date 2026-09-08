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

# Create the inner package with ABSOLUTE file paths.
# Create the outer package with RELATIVE file paths.
# When the main executable imports both packages, there will be a conflict:
# The `constrained_method` function will come from the inner package, and use
# absolute paths. But the usage of `constrained_method` in the outer package
# will use relative paths, and result in a mismatch in def metadata.

# RUN: mojo precompile %S/inputs/where_package -o %S/where_package.mojoc
# RUN: mojo precompile %S/inputs/wrapper_where_package -o %S/wrapper_where_package.mojoc -strip-file-prefix=%S/inputs
# RUN: mojo %s -I %S | FileCheck %s

from where_package import constrained_method
from wrapper_where_package import use_constrained_method


def main():
    # CHECK: result: 42 2
    comptime result = constrained_method[42]()
    comptime result2 = use_constrained_method()
    print("result:", result, result2)
