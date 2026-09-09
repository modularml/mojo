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
"""Implements traits for abstracting MMA operand descriptors in GPU matrix operations."""


trait MMAOperandDescriptor(TrivialRegisterPassable):
    """Trait for abstracting MMA (Matrix Multiply-Accumulate) operand descriptors.

    This trait defines the interface for WGMMA operand descriptors used in GPU matrix operations.
    """

    @always_inline
    def __add__(self, offset: Int) -> Self:
        """Adds an offset to the operand descriptor.

        Args:
            offset: The offset to add to the descriptor.

        Returns:
            A new descriptor with the offset applied.
        """
        ...
