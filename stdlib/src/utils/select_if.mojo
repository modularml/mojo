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


@always_inline("nodebug")
fn select[T: AnyTrivialRegType](condition: Bool, lhs: T, rhs: T) -> T:
    """Choose one value based on a condition, without IR-level branching.
        Use this over normal `if` branches to reduce the size of the generated IR.

    Parameters:
        T: The type of the lhs and rhs.

    Args:
        condition: The condition to test.
        lhs: The value to select if the condition is met.
        rhs: The value to select if the condition is not met.

    Returns:
        The value selected based on the condition.
    """
    return __mlir_op.`pop.select`(condition.value, lhs, rhs)
