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
"""Provides comprehensive Unicode string functionality.

Core features:

- Unicode support with UTF-8 encoding
- Efficient string slicing and views
- String formatting and interpolation
- Memory-safe string operations
- Unicode case conversion
- Unicode property lookups and validation

Key components:

- [`String`](/docs/std/collections/string/string/String/):
  Mutable and owning string

  - Uses a smart three-mode allocation strategy: static memory
  references for string literals, small string optimization (SSO) for strings
  ≤23 bytes (stored directly within the `String` object with zero allocation
  cost), and reference-counted heap allocation for larger strings. This design
  makes the vast majority of real-world strings extremely fast.

  - Owns its data and manages memory automatically when heap-allocated.

  - Mutable and grows dynamically as needed.

- [`StringSpan`](/docs/std/collections/string/string_span/StringSpan/):
  Non-owning string view

  - Performs zero heap allocations: stores only a pointer and length
  that reference existing string data owned by another object.

  - Does not own the data pointed to, so it can't outlive the data it
  references.

- [`StaticString`](/docs/std/collections/string/string_span/#staticstring):
  Compile-time constant (immutable) string reference

  - Performs zero heap allocations: stores a pointer and length to a
  compile-time constant or static program memory.

  - References data with a static lifetime that exists for the entire program
  duration, unlike `StringSpan` which can reference temporary data.

- [`Codepoint`](/docs/std/collections/string/codepoint/Codepoint/):
  Unicode codepoint representation and operations

  - Represents a single Unicode codepoint as a 32-bit value.

  - Enables iteration over string contents at the Unicode codepoint level
  rather than byte level for proper Unicode text processing.

- [`format`](/docs/std/collections/string/format/): Built-in string
formatting and interpolation utilities.

:::note Note

String stores data using UTF-8, and all operations (unless clearly noted) are intended to
be fully Unicode compliant and maintain correct UTF-8 encoded data.
A handful of operations are known to not be Unicode / UTF-8 compliant yet, but will be
fixed as time permits.

:::

"""

from .codepoint import Codepoint
from .string import String, ascii, atof, atol, chr, ord
from .string_span import (
    BytesIter,
    CodepointsIter,
    GraphemeSliceIter,
    ImmStringSpan,
    ImmStringSlice,
    MutStringSpan,
    MutStringSlice,
    StaticString,
    StringSpan,
    StringSlice,
)
