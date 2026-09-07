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
"""Implements algorithm backend utilities including tiling and unswitching."""

from .tile import (
    BinaryTile1DTileUnitFunc,
    Dynamic1DTileUnitFunc,
    Static1DTileUnitFunc,
    Static2DTileUnitFunc,
    tile,
)
from .unswitch import (
    Dynamic1DTileUnswitchUnitFunc,
    Static1DTileUnitFuncWithFlag,
    Static1DTileUnitFuncWithFlags,
    Static1DTileUnswitchUnitFunc,
    SwitchedFunction,
    SwitchedFunction2,
    tile_and_unswitch,
    tile_middle_unswitch_boundaries,
    unswitch,
)
from .vectorize import vectorize
