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
#   Path:   tests/data/simple_cases/comments_non_breaking_space.py
#
# ===----------------------------------------------------------------------=== #

from .config import (
    ConfigTypeAttributes,
    Int,
    Path,  # String,
    # DEFAULT_TYPE_ATTRIBUTES,
)

result = 1  # A simple comment
result = (1,)  # Another one

result = 1  #  type: ignore
result = 1  # This comment is talking about type: ignore
square = Square(4)  #  type: Optional[Square]


def function(a: int = 42):
    """This docstring is already formatted
    a
    b
    """
    #    There's a NBSP + 3 spaces before
    #    And 4 spaces on the next line
    pass


# output
from .config import (
    ConfigTypeAttributes,
    Int,
    Path,  # String,
    # DEFAULT_TYPE_ATTRIBUTES,
)

result = 1  # A simple comment
result = (1,)  # Another one

result = 1  #  type: ignore
result = 1  # This comment is talking about type: ignore
square = Square(4)  #  type: Optional[Square]


def function(a: int = 42):
    """This docstring is already formatted
    a
    b
    """
    #    There's a NBSP + 3 spaces before
    #    And 4 spaces on the next line
    pass
