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
"""Python wrappers for the ``state_space`` Mojo kernel package.

The module layout mirrors ``max/kernels/src/state_space/``. All ops here are
registered as graph-compiler builtins, so no ``custom_extensions`` plumbing
is needed. The legacy Mamba-1 eager wrappers intentionally stay in the Mamba
architecture (their sole consumer) to keep ``max.nn`` from depending on
``max.experimental.functional``.
"""

from .gated_delta import gated_delta_conv1d_fwd, gated_delta_recurrence_fwd
from .gated_group_rmsnorm import gated_group_rmsnorm
from .kimi_delta import kda_decode
from .mamba2_ssd_scan import (
    mamba2_ssd_chunk_scan_varlen_fwd,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace,
)
from .varlen_causal_conv1d import causal_conv1d_varlen_fwd

__all__ = [
    "causal_conv1d_varlen_fwd",
    "gated_delta_conv1d_fwd",
    "gated_delta_recurrence_fwd",
    "gated_group_rmsnorm",
    "kda_decode",
    "mamba2_ssd_chunk_scan_varlen_fwd",
    "mamba2_ssd_chunk_scan_varlen_fwd_inplace",
]
