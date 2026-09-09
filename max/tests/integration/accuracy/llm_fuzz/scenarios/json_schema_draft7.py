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
"""Scenario: JSON Schema draft 7 test suite

Every vendored draft 7 schema is sent verbatim (no filtering, no wrapping),
driven four ways (tools auto/required/named, ``response_format``) with an
adversarial break-the-schema prompt. The ``strict`` flag is never sent: the
server is expected to match the schema regardless, so output is validated with
the ``jsonschema`` ``Draft7Validator`` and a non-conforming output or a 4xx is
always a FAIL. Remote ``$ref``s are inlined from the vendored ``remotes/``.

The shared driver and vendored-data attribution live in
``scenarios._json_schema_suite``.
"""

from __future__ import annotations

from scenarios import register_scenario
from scenarios._json_schema_suite import JsonSchemaSuiteScenario


@register_scenario
class JsonSchemaTestSuiteDraft7(JsonSchemaSuiteScenario):
    name = "json_schema_draft7"
    description = (
        "Drive every draft 7 JSON Schema Test Suite schema four ways "
        "(tools auto/required/named, response_format) using an adversarial "
        "break-the-schema prompt; validate with the jsonschema Draft7Validator. "
        "Faithful reproduction of the suite: no keyword/format filtering and no "
        "object wrapping -- schemas are sent verbatim. The strict flag is never "
        "sent; the server must constrain output regardless, so 400s and "
        "non-conforming output are always FAILs."
    )
    tags = ["json", "schema", "structured", "tools", "response_format"]
    scenario_type = "fuzz"
    wrap_tool_modes = False
    reject_400_is_pass = False
