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
"""Compile-time reflection utilities for introspecting Mojo types and functions.

This module provides compile-time reflection capabilities including:

- Unified type and struct reflection via `reflect[T]`, a `comptime` alias
  for the `Reflected[T]` handle type. Use `reflect[T].name()`,
  `reflect[T].base_name()`, `reflect[T].field_count()`, etc.
- Function name and linkage introspection (`get_function_name`, `get_linkage_name`).
- Source location introspection (`source_location`, `call_location`).

`reflect` is auto-imported via the prelude. The other names listed above
must be imported explicitly from `std.reflection`.

Example:
```mojo
struct Point:
    var x: Int
    var y: Float64

def print_fields[T: AnyType]():
    comptime names = reflect[T].field_names()
    comptime for i in range(reflect[T].field_count()):
        print(materialize[names[i]]())

def main():
    print_fields[Point]()  # Prints: x, y
```
"""

from .function import ReflectedFn, reflect_fn
from .location import SourceLocation, source_location, call_location
from .reflect import Reflected, reflect
from .type_info import get_linkage_name, get_function_name
