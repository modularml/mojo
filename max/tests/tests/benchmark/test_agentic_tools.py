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

"""Tests for the ``--agentic-tool-profiles`` models and resolution."""

from __future__ import annotations

import pytest
from max.benchmark.benchmark_shared.datasets.agentic_tools import (
    ToolConfig,
    tool_profiles_from_mapping,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# ToolConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["input-len", "output-len", "delay"])
def test_tool_spec_rejects_two_part_spec(field: str) -> None:
    """``first;rest`` is a human-turn concept and must not bind per tool."""
    payload = {"input-len": "100", "output-len": "20"}
    payload[field] = "N(500,100);N(200,50)"
    with pytest.raises(ValidationError, match="first;rest"):
        ToolConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# tool_profiles_from_mapping
# ---------------------------------------------------------------------------


def test_mapping_builds_the_tool_list() -> None:
    """The kebab keys the docs show, with the defaults they omit."""
    tools = tool_profiles_from_mapping(
        {
            "tools": [
                {
                    "weight": 5,
                    "input-len": "N(30,20)",
                    "output-len": "N(60,30)",
                    "delay": "N(50,20)",
                },
                {"input-len": "LN(7.5,1.2)", "output-len": "N(190,60)"},
            ]
        }
    )
    assert [t.weight for t in tools] == [5.0, 1.0]
    assert tools[0].delay == "N(50,20)"
    assert tools[1].delay == 0, "omitting delay means instant"


@pytest.mark.parametrize(
    ("profiles", "message"),
    [
        ({"tools": []}, "at least one tool"),
        (
            {"tools": [{"weight": 0, "input-len": "10", "output-len": "5"}]},
            "weight > 0",
        ),
        (
            {
                "tools": [{"input-len": "10", "output-len": "5"}],
                "round": 3,
            },
            "unexpected key",
        ),
    ],
    ids=["empty-list", "all-zero-weights", "typo'd-key"],
)
def test_rejected_mapping_names_the_problem(
    profiles: dict[str, object], message: str
) -> None:
    """Each rejection names the flag and what to fix, never a bare traceback."""
    with pytest.raises(ValueError, match=message):
        tool_profiles_from_mapping(profiles)
