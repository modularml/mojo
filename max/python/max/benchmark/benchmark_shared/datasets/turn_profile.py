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

from __future__ import annotations

from dataclasses import dataclass

from .distribution import BaseDistribution, DistributionParameter


@dataclass(frozen=True)
class TurnTargets:
    """One turn's targets, one sample from each of a profile's distributions."""

    input_len: int
    output_len: int
    delay_ms: float | None
    """``None`` when the profile has no delay distribution."""


@dataclass(frozen=True)
class TurnProfile:
    """The input length, output length and delay one turn draws from.

    A human turn and an agent-loop round draw the same way, from different
    profiles, so both use this.
    """

    input_len: BaseDistribution
    output_len: BaseDistribution
    delay: BaseDistribution | None

    @classmethod
    def from_parameters(
        cls,
        *,
        input_len: DistributionParameter,
        output_len: DistributionParameter,
        delay: DistributionParameter | None,
    ) -> TurnProfile:
        """Build a profile from the distribution specs a user wrote.

        Args:
            input_len: Spec for the input length.
            output_len: Spec for the output length.
            delay: Spec for the delay in milliseconds, or ``None`` for none.

        Returns:
            The parsed profile.
        """
        return cls(
            input_len=BaseDistribution.from_distribution_parameter_or_raise(
                input_len
            ),
            output_len=BaseDistribution.from_distribution_parameter_or_raise(
                output_len
            ),
            delay=BaseDistribution.from_distribution_parameter(delay),
        )

    def draw(self, *, min_input_len: int, min_output_len: int) -> TurnTargets:
        """Draw one turn's targets: one sample from each distribution.

        Args:
            min_input_len: Floor for the input length.
            min_output_len: Floor for the output length.

        Returns:
            The sampled lengths and delay.
        """
        return TurnTargets(
            input_len=max(round(self.input_len.sample_value()), min_input_len),
            output_len=max(
                round(self.output_len.sample_value()), min_output_len
            ),
            delay_ms=(
                max(float(self.delay.sample_value()), 0.0)
                if self.delay is not None
                else None
            ),
        )
