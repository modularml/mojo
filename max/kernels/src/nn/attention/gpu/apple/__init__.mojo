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
"""Apple (Metal) GPU attention kernels.

Decode-only split-K naive flash-attention for Apple silicon GPUs.
"""

from .naive_fa_decode import (
    naive_fa_decode_apple,
    naive_fa_decode_apple_core,
    naive_fa_decode_apple_stitch,
)
