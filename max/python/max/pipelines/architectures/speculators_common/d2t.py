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
"""Draft-to-target vocabulary mapping for pruned-vocab DSpark drafters."""

from __future__ import annotations

from max.graph import TensorValue, ops


def map_draft_to_target_vocab(
    draft_ids: TensorValue, d2t: TensorValue
) -> TensorValue:
    """Maps draft-vocab token ids to target-vocab ids via the d2t offsets.

    ``d2t`` is an offset table (Eagle3 convention):
    ``target_id = draft_id + d2t[draft_id]``.
    """
    return draft_ids + ops.gather(d2t, draft_ids, axis=0)
