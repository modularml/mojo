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
"""Binary data encoding: hexadecimal encode/decode functions.

The `hex` package provides functions for encoding and decoding binary data
using hexadecimal (base16) notation. Hexadecimal encoding represents each byte
as two ASCII characters drawn from the digits 0-9 and the letters a-f, making
binary data directly readable and unambiguous at the cost of doubling its size.
Because the mapping is byte-aligned and requires no padding,
easy to inspect by eye, and trivially reversible.

Use this package for rendering hashes, checksums, and cryptographic digests as
text, inspecting or logging raw bytes in a human-readable form, parsing hex
literals from configuration files or wire formats, or converting between binary
and text representations where clarity matters more than compactness.
"""

from .hex import hex_encode, hex_decode
