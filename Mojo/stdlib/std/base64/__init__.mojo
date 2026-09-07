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
"""Binary data encoding: base64 and base16 encode/decode functions.

The `base64` package provides functions for encoding and decoding binary data
using Base64 and Base16 (hexadecimal) encoding schemes. Base64 encoding
converts binary data into ASCII text for transmission over text-based protocols,
while Base16 provides a simpler hexadecimal representation. These encodings are
essential for data interchange, embedding binary data in text formats, and
ensuring data integrity across different systems.

Use this package for encoding binary data in JSON or XML, transmitting binary
data over text protocols (HTTP, email), embedding images or files in text
formats, or converting between binary and text representations while preserving
data integrity.
"""

from .base64 import b16decode, b16encode, b64decode, b64encode
