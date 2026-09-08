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
#   Path:   tests/data/preview/one_element_subscript.py
#
# ===----------------------------------------------------------------------=== #

# We should not treat the trailing comma
# in a single-element subscript.
a: tuple[
    int,
]
b = tuple[
    int,
]

# The magic comma still applies to multi-element subscripts.
c: tuple[
    int,
    int,
]
d = tuple[
    int,
    int,
]

# Magic commas still work as expected for non-subscripts.
small_list = [
    1,
]
list_of_types = [
    tuple[
        int,
    ],
]

# output
# We should not treat the trailing comma
# in a single-element subscript.
a: tuple[int,]
b = tuple[int,]

# The magic comma still applies to multi-element subscripts.
c: tuple[
    int,
    int,
]
d = tuple[
    int,
    int,
]

# Magic commas still work as expected for non-subscripts.
small_list = [
    1,
]
list_of_types = [
    tuple[int,],
]
