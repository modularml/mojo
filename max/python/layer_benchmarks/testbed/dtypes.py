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

"""Shared dtype mapping for harnesses."""

from collections.abc import Mapping

# torch is a caller-supplied dep, see BUILD.bazel
import torch  # type: ignore[import-not-found]
from max.dtype import DType

DTYPE_MAP: Mapping[str, tuple[DType, torch.dtype]] = {
    "bfloat16": (DType.bfloat16, torch.bfloat16),
    "float32": (DType.float32, torch.float32),
    "float16": (DType.float16, torch.float16),
}
