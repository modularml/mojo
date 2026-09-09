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
"""Eagle3 draft model for a DeepseekV3 target.

Thin named alias over the shared :class:`Eagle3MLADraft`; the implementation
lives in ``architectures/eagle_common/eagle_mla_draft.py``. Kept as a subclass
so existing imports (``from .eagle3_draft import Eagle3DeepseekV3``) and the
``Eagle3DeepseekV3Unified`` graph wiring stay unchanged.
"""

from __future__ import annotations

from ..eagle_common.eagle_mla_draft import Eagle3MLADraft

__all__ = ["Eagle3DeepseekV3"]


class Eagle3DeepseekV3(Eagle3MLADraft):
    """Eagle3 draft paired with a DeepseekV3 target.

    See :class:`Eagle3MLADraft` for the implementation.
    """
