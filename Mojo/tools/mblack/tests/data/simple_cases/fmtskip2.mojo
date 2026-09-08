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

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/simple_cases/fmtskip2.py
#
# ===----------------------------------------------------------------------=== #

l1 = [
    "This list should be broken up",
    "into multiple lines",
    "because it is way too long",
]
l2 = ["But this list shouldn't", "even though it also has", "way too many characters in it"]  # fmt: skip
l3 = [
    "I have",
    "trailing comma",
    "so I should be braked",
]

# output

l1 = [
    "This list should be broken up",
    "into multiple lines",
    "because it is way too long",
]
l2 = ["But this list shouldn't", "even though it also has", "way too many characters in it"]  # fmt: skip
l3 = [
    "I have",
    "trailing comma",
    "so I should be braked",
]
