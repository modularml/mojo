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

"""Re-exports the depth-512 multi-head attention kernel package for NVIDIA SM100 GPUs."""

from .barriers import Depth512MBars
from .config import Depth512SM100Config
from .dispatch import mha_sm100_depth512_dispatch
from .kernel import SM100MHADepth512
from .load_warp import depth512_load
from .mma_warp import depth512_mma
from .smem import Depth512AttentionSMem
from .softmax_warp import depth512_softmax
