# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements the built-in `swap` function.

These are Mojo built-ins, so you don't need to import them.
"""


@always_inline
fn swap[T: Movable](inout lhs: T, inout rhs: T):
    """Swaps the two given arguments.

    Parameters:
       T: Constrained to Movable types.

    Args:
        lhs: Argument value swapped with rhs.
        rhs: Argument value swapped with lhs.
    """
    var tmp = lhs^
    lhs = rhs^
    rhs = tmp^
