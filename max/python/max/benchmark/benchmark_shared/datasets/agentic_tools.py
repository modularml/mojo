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

"""The agent-loop tool profiles behind ``--agentic-tool-profiles``.

A profile describes the load one tool call produces, not a tool the model may
invoke: nothing here reaches the wire as an OpenAI ``tools=[...]`` definition.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
)

from .distribution import DistributionParameter
from .turn_profile import TurnProfile


class ToolConfig(BaseModel):
    """One tool the agent loop can call, and the load one call produces.

    ``input_len`` is the result it returns, ``output_len`` the assistant
    message that invokes it -- a tool call, so usually small -- and ``delay``
    how long it took, in milliseconds. The three travel together because they
    correlate within a tool: a file read returns a lot and is slow, a shell
    exit code returns almost nothing and is fast.

    The fields hold the distribution strings the user wrote; :attr:`profile`
    holds them parsed, which is the form a human turn arrives in too.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    weight: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    input_len: DistributionParameter = Field(alias="input-len")
    output_len: DistributionParameter = Field(alias="output-len")
    delay: DistributionParameter = 0

    # Parsed once at validation, so an unparseable distribution fails when the
    # flag is read rather than mid-generation, and sampling stays cheap.
    _profile: TurnProfile = PrivateAttr()

    def model_post_init(self, context: object, /) -> None:
        self._profile = TurnProfile.from_parameters(
            input_len=self.input_len,
            output_len=self.output_len,
            delay=self.delay,
        )

    @property
    def profile(self) -> TurnProfile:
        """This tool's distributions, parsed once at validation.

        A round draws through the same :class:`TurnProfile` a human turn does,
        so neither side needs its own sampling path.
        """
        return self._profile

    @field_validator("input_len", "output_len", "delay")
    @classmethod
    def _reject_two_part_spec(
        cls, value: DistributionParameter, info: ValidationInfo
    ) -> DistributionParameter:
        """Reject ``first;rest``: it belongs to the ``random_*`` knobs."""
        if isinstance(value, str) and ";" in value:
            raise ValueError(
                f"{info.field_name}: ';' (first;rest) is not supported per"
                " tool -- it applies to the --random-* human-turn knobs."
                " Give this tool a single distribution."
            )
        return value


def tool_profiles_from_mapping(
    profiles: Mapping[str, object],
) -> list[ToolConfig]:
    """Build the per-tool load profiles from a ``tools`` mapping.

    Args:
        profiles: A mapping holding a ``tools`` list.

    Returns:
        The tools each round draws from, biased by ``weight``.

    Raises:
        ValueError: If the mapping holds keys other than ``tools``, or no tool
            that can ever be selected.
    """
    loaded = dict(profiles)
    raw = loaded.pop("tools", None)
    if extra := sorted(loaded):
        raise ValueError(
            f"--agentic-tool-profiles: unexpected key(s) {extra}; the mapping"
            " holds only a 'tools' list."
        )
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "--agentic-tool-profiles: 'tools' must list at least one tool;"
            f" got {raw!r}."
        )

    tools = [ToolConfig.model_validate(tool) for tool in raw]
    # Rounds pick with random.choices, which cannot draw from all-zero weights.
    if not any(tool.weight > 0 for tool in tools):
        raise ValueError(
            "--agentic-tool-profiles: at least one tool must have weight > 0,"
            " otherwise no tool can ever be selected."
        )
    return tools
