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
"""Request-handling registration parity between graph and ModuleV3 arches.

The ModuleV3 arch must default the same serving-layer request-handling
features as its graph counterpart: tool-call parsing, reasoning parsing, and
the structured-output (constrained decoding) backend. The parsers themselves
live at the serving layer and are shared; only the ``SupportedArchitecture``
defaults differ per registration.
"""

from __future__ import annotations

import pytest
from max.pipelines.architectures.gemma4.arch import (
    gemma4_arch,
    gemma4_unified_arch,
)
from max.pipelines.architectures.gemma4_modulev3.arch import (
    gemma4_modulev3_arch,
    gemma4_unified_modulev3_arch,
)
from max.pipelines.lib import reasoning, tool_parsing
from max.pipelines.lib.registry import SupportedArchitecture


@pytest.mark.parametrize(
    ("graph_arch", "modulev3_arch"),
    [
        pytest.param(gemma4_arch, gemma4_modulev3_arch, id="gemma4"),
        pytest.param(
            gemma4_unified_arch,
            gemma4_unified_modulev3_arch,
            id="gemma4_unified",
        ),
    ],
)
def test_gemma4_request_handling_parity(
    graph_arch: SupportedArchitecture,
    modulev3_arch: SupportedArchitecture,
) -> None:
    """ModuleV3 registrations mirror the graph arch's request-handling defaults."""
    assert modulev3_arch.tool_parser == graph_arch.tool_parser
    assert modulev3_arch.reasoning_parser == graph_arch.reasoning_parser
    assert (
        modulev3_arch.default_structured_output_backend
        == graph_arch.default_structured_output_backend
    )


def test_gemma4_modulev3_default_parsers_are_registered() -> None:
    """Importing the ModuleV3 arch registers the parsers it names.

    Parser registration is an import side effect (the gemma4 package's
    ``__init__`` imports ``tool_parser``/``reasoning``); a missing import
    would surface as "Unknown ... parser" per request at serve time.
    """
    assert isinstance(gemma4_modulev3_arch.tool_parser, str)
    assert (
        tool_parsing.get_parser_cls(gemma4_modulev3_arch.tool_parser)
        is not None
    )
    assert (
        reasoning.get_parser_cls(gemma4_modulev3_arch.reasoning_parser)
        is not None
    )
