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
"""The target-side contract a spec-decode driver verifies through.

A target is the model being verified, so nothing here depends on how the draft
proposes. The drafting strategy shows up in one place only: which batch type
the driver hands :meth:`SpecDecodeTarget.verify`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from max.graph import BufferType, TensorType, TensorValue

__all__ = ["SpecDecodeTarget", "Verified"]

_BatchT = TypeVar("_BatchT", contravariant=True)
"""The driver's batch type. Contravariant: ``verify`` only consumes it."""


@dataclass(frozen=True)
class Verified:
    """What the target contributes: verification logits and hidden states.

    Naming the two results is the point. The target output tuple is
    ``[last_logits?][logits, offsets?][hidden per device...]``, and whether the
    leading entry exists depends on ``emit_last_token_logits`` -- which is why
    structurally identical code reads ``[1]`` and ``[3 : 3 + n]`` in one arch
    and ``[0]`` and ``[2 : 2 + n]`` in another. The adapter owns that layout.
    """

    logits: TensorValue
    hidden: list[TensorValue]


class SpecDecodeTarget(Protocol[_BatchT]):
    """The target model's entry point and output layout."""

    def verify(self, batch: _BatchT) -> Verified:
        """Runs the target over the merged prompt + draft tokens."""
        ...

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        """Expert-parallel input types to append to the graph signature."""
        ...
