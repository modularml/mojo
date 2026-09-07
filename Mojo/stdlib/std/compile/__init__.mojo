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
"""Runtime function compilation and introspection: assembly, IR, linkage, metadata.

The `compile` package exposes functionality for compiling individual Mojo
functions and examining their low-level implementation details. It enables
inspecting generated code, obtaining linkage information, and controlling
compilation options at runtime. This package provides tools for
metaprogramming, debugging, and understanding how Mojo code compiles.

This package is particularly useful for:

- Inspecting assembly, LLVM IR, or object code output
- Getting linkage names and module information
- Examining function metadata like captures
- Writing compilation output to files
- Controlling compilation options and targets

Example:
```mojo
from std.compile import compile_info

def my_func():
    print("Hello")

# Get assembly for the function
info = compile_info[my_func]()
print(info.asm)
```
"""

from .compile import CompiledFunctionInfo, compile_info
