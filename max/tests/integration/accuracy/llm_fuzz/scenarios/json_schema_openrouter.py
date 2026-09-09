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
"""Scenario: OpenRouter-style JSON Schema test suite

Every vendored draft 7 schema is driven four ways (tools auto/required/named,
``response_format``) with an adversarial break-the-schema prompt. Each
tool-calling request is issued twice: once with the schema verbatim and once
wrapped in a root object, so schemas with array or non-object root types are
still exercised through the tool-calling path. The ``strict`` flag is never sent:
the server is expected to match the schema regardless, so output is validated
with the ``jsonschema`` ``Draft7Validator``. A non-conforming output is a FAIL,
while a 400 (the server rejecting the schema up front) is a PASS. Remote
``$ref``s are inlined from the vendored ``remotes/``.

The shared driver and vendored-data attribution live in
``scenarios._json_schema_suite``.
"""

from __future__ import annotations

from scenarios import register_scenario
from scenarios._json_schema_suite import JsonSchemaSuiteScenario


@register_scenario
class JsonSchemaOpenRouter(JsonSchemaSuiteScenario):
    name = "json_schema_openrouter"
    description = (
        "Drive every draft 7 JSON Schema Test Suite schema four ways "
        "(tools auto/required/named, response_format) using an adversarial "
        "break-the-schema prompt; validate with the jsonschema Draft7Validator. "
        "Each tool-calling request is issued twice -- schema verbatim and "
        "wrapped in a root object -- so array/non-object root schemas are "
        "exercised through tool calls. The strict flag is never sent; "
        "non-conforming output is a FAIL, while a 400 (schema rejected up "
        "front) is a PASS."
    )
    tags = ["json", "schema", "structured", "tools", "response_format"]
    scenario_type = "fuzz"
    wrap_tool_modes = True
    reject_400_is_pass = True
