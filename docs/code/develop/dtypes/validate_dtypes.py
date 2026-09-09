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
# DOC: max/develop/dtypes.mdx

from max.dtype import DType


def validate_weights_dtype(dtype: DType) -> None:
    """Ensure weights use a floating-point type."""
    if not dtype.is_float():
        raise TypeError(f"Weights must be float type, got {dtype}")


def validate_indices_dtype(dtype: DType) -> None:
    """Ensure indices use an integer type."""
    if not dtype.is_integral():
        raise TypeError(f"Indices must be integer type, got {dtype}")


# Usage
weights_dtype = DType.float16
indices_dtype = DType.int32

validate_weights_dtype(weights_dtype)  # OK
validate_indices_dtype(indices_dtype)  # OK
