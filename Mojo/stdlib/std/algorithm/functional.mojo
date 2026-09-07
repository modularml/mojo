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
"""Implements higher-order functions.

You can import these APIs from the `algorithm` package. For example:

```mojo
from std.algorithm import map
```
"""

# Re-export backend-independent implementations.
from .backend import (
    BinaryTile1DTileUnitFunc,
    Dynamic1DTileUnitFunc,
    Dynamic1DTileUnswitchUnitFunc,
    Static1DTileUnitFunc,
    Static1DTileUnitFuncWithFlag,
    Static1DTileUnitFuncWithFlags,
    Static1DTileUnswitchUnitFunc,
    Static2DTileUnitFunc,
    SwitchedFunction,
    SwitchedFunction2,
    tile,
    tile_and_unswitch,
    tile_middle_unswitch_boundaries,
    unswitch,
    vectorize,
)

# Re-export CPU implementations.
from .backend.cpu import map
