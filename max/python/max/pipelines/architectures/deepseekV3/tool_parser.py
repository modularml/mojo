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
# ruff: noqa: RUF001, RUF002, RUF003
"""Tool call parsers for DeepSeek V3 model family.

DeepSeek V3 ships two tool-call grammars depending on checkpoint revision:

V3 (markdown-wrapped — original ``deepseek-ai/DeepSeek-V3`` release)::

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}
    ```json
    {"key": "value"}
    ```<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

V3.1 (raw JSON — ``deepseek-ai/DeepSeek-V3.1`` and later)::

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{"key": "value"}<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

Both checkpoints register under the same architecture name
(``DeepseekV3ForCausalLM``); :func:`resolve_deepseekv3_tool_parser`
inspects the chat template and returns the matching registered parser
name.

References:

- vllm/tool_parsers/deepseekv3_1_tool_parser.py
- sglang/srt/function_call/deepseekv3_detector.py
- sglang/srt/function_call/deepseekv3_1_detector.py
"""

from __future__ import annotations

import json
import logging
import os
import re

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError
from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    generate_call_id,
    partial_tag_overlap,
    register,
)
from max.pipelines.modeling.types import ParsedToolCall
from max.pipelines.weights.hf_utils import HuggingFaceRepo

logger = logging.getLogger(__name__)

# Structural tags shared across V3 variants. The "｜" character is the
# fullwidth vertical line (U+FF5C) and "▁" is U+2581.
TOOL_CALLS_SECTION_BEGIN = "<｜tool▁calls▁begin｜>"
TOOL_CALLS_SECTION_END = "<｜tool▁calls▁end｜>"
TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
TOOL_CALL_END = "<｜tool▁call▁end｜>"
TOOL_CALL_SEP = "<｜tool▁sep｜>"

# Markdown fences used by the V3 (original) format that were removed in V3.1.
_V3_ARGS_OPEN = "\n```json\n"
_V3_ARGS_CLOSE = "\n```"

# V3: <｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n{args}\n```<｜tool▁call▁end｜>
_V3_TOOL_CALL_PATTERN = re.compile(
    rf"{re.escape(TOOL_CALL_BEGIN)}"
    rf".*?{re.escape(TOOL_CALL_SEP)}"
    rf"(?P<function_name>.*?)"
    rf"{re.escape(_V3_ARGS_OPEN)}"
    rf"(?P<arguments>.*?)"
    rf"{re.escape(_V3_ARGS_CLOSE)}"
    rf"{re.escape(TOOL_CALL_END)}",
    re.DOTALL,
)

# V3.1: <｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{args}<｜tool▁call▁end｜>
_V3_1_TOOL_CALL_PATTERN = re.compile(
    rf"{re.escape(TOOL_CALL_BEGIN)}"
    rf"(?P<function_name>.*?)"
    rf"{re.escape(TOOL_CALL_SEP)}"
    rf"(?P<arguments>.*?)"
    rf"{re.escape(TOOL_CALL_END)}",
    re.DOTALL,
)


def _parse_json_args(arguments_str: str) -> str:
    """Re-serializes ``arguments_str`` to canonical JSON when possible."""
    try:
        return json.dumps(json.loads(arguments_str))
    except json.JSONDecodeError:
        return arguments_str


@register("deepseekv3_1")
class DeepseekV3_1ToolParser(StructuralTagToolParser):
    """Parses DeepSeek V3.1+ tool calls (raw JSON arguments).

    The function name is a plain identifier immediately after
    ``<｜tool▁call▁begin｜>`` and the arguments are raw JSON between
    ``<｜tool▁sep｜>`` and ``<｜tool▁call▁end｜>``.
    """

    SECTION_BEGIN = TOOL_CALLS_SECTION_BEGIN
    SECTION_END = TOOL_CALLS_SECTION_END
    CALL_BEGIN = TOOL_CALL_BEGIN
    CALL_END = TOOL_CALL_END

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        tool_calls: list[ParsedToolCall] = []
        for match in _V3_1_TOOL_CALL_PATTERN.finditer(tool_section):
            func_name = match.group("function_name").strip()
            if not func_name:
                continue
            tool_calls.append(
                ParsedToolCall(
                    id=generate_call_id(),
                    name=func_name,
                    arguments=_parse_json_args(
                        match.group("arguments").strip()
                    ),
                )
            )
        return tool_calls

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits ``name<｜tool▁sep｜>args``."""
        sep_pos = body.find(TOOL_CALL_SEP)
        if sep_pos == -1:
            return None, None
        return body[:sep_pos].strip(), body[sep_pos + len(TOOL_CALL_SEP) :]


@register("deepseekv3")
class DeepseekV3ToolParser(StructuralTagToolParser):
    """Parses DeepSeek V3 (original) tool calls with markdown-wrapped JSON.

    The body format is::

        function<｜tool▁sep｜>{name}
        ```json
        {args}
        ```

    The literal ``function`` type prefix before ``<｜tool▁sep｜>`` is
    ignored. The name follows the separator on its own line, and the
    arguments are raw JSON between the markdown fences.
    """

    SECTION_BEGIN = TOOL_CALLS_SECTION_BEGIN
    SECTION_END = TOOL_CALLS_SECTION_END
    CALL_BEGIN = TOOL_CALL_BEGIN
    CALL_END = TOOL_CALL_END

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        tool_calls: list[ParsedToolCall] = []
        for match in _V3_TOOL_CALL_PATTERN.finditer(tool_section):
            func_name = match.group("function_name").strip()
            if not func_name:
                continue
            tool_calls.append(
                ParsedToolCall(
                    id=generate_call_id(),
                    name=func_name,
                    arguments=_parse_json_args(
                        match.group("arguments").strip()
                    ),
                )
            )
        return tool_calls

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Body format: ``function<｜tool▁sep｜>{name}\\n```json\\n{args}\\n``` ``.

        While streaming, the closing markdown fence may not yet be
        present; we hold back any trailing characters that partially
        match it so they are not emitted as argument content.
        """
        sep_pos = body.find(TOOL_CALL_SEP)
        if sep_pos == -1:
            return None, None

        after_sep = body[sep_pos + len(TOOL_CALL_SEP) :]

        open_pos = after_sep.find(_V3_ARGS_OPEN)
        if open_pos == -1:
            return None, None

        header = after_sep[:open_pos].strip()
        args_text = after_sep[open_pos + len(_V3_ARGS_OPEN) :]

        # Strip the closing fence whenever it has fully landed. While
        # still streaming with only a partial fence, hold back the
        # overlapping suffix to avoid leaking it as argument content.
        close_pos = args_text.rfind(_V3_ARGS_CLOSE)
        if close_pos != -1:
            args_text = args_text[:close_pos]
        elif not is_complete:
            overlap = partial_tag_overlap(args_text, _V3_ARGS_CLOSE)
            if overlap:
                args_text = args_text[:-overlap]

        return header, args_text


# ---------------------------------------------------------------------------
# Per-checkpoint dispatch
# ---------------------------------------------------------------------------

_TOKENIZER_CONFIG = "tokenizer_config.json"
_CHAT_TEMPLATE_FILE = "chat_template.jinja"


def _extract_template(data: object) -> str | None:
    """Pulls the ``chat_template`` string out of a tokenizer_config payload."""
    if not isinstance(data, dict):
        return None
    template = data.get("chat_template")
    if isinstance(template, list) and template:
        entry = template[0]
        if isinstance(entry, dict):
            template = entry.get("template")
    return template if isinstance(template, str) else None


def _read_repo_file(repo: HuggingFaceRepo, filename: str) -> str | None:
    """Reads a file from a HuggingFace repo."""
    subfolder = repo.subfolder
    rel_path = f"{subfolder}/{filename}" if subfolder else filename

    if repo.repo_type == "local":
        local_path = os.path.join(repo.local_path, rel_path)
        if not os.path.isfile(local_path):
            return None
        try:
            with open(local_path, encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            logger.warning("Failed to read %s: %s", local_path, e)
            return None

    try:
        cached = huggingface_hub.hf_hub_download(
            repo_id=repo.repo_id,
            filename=filename,
            revision=repo.revision,
            subfolder=subfolder,
        )
    except EntryNotFoundError:
        # Not every repo ships every template file — fall through quietly.
        return None
    except Exception as e:
        logger.warning(
            "Failed to fetch %s for %s@%s (subfolder=%s): %s",
            filename,
            repo.repo_id,
            repo.revision,
            subfolder,
            e,
        )
        return None

    try:
        with open(cached, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        logger.warning("Failed to read cached %s: %s", cached, e)
        return None


def _load_chat_template(repo: HuggingFaceRepo) -> str | None:
    """Loads the chat template from a model repo's template files.

    Checks two sources, in order:

    1. ``tokenizer_config.json``'s ``chat_template`` field (the
       traditional location, embedded alongside other tokenizer config).
    2. ``chat_template.jinja`` (a standalone template file that newer
       HuggingFace releases ship for long templates).
    """
    raw = _read_repo_file(repo, _TOKENIZER_CONFIG)
    if raw is not None:
        try:
            template = _extract_template(json.loads(raw))
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse %s: %s", _TOKENIZER_CONFIG, e)
            template = None
        if template:
            return template

    return _read_repo_file(repo, _CHAT_TEMPLATE_FILE)


def resolve_deepseekv3_tool_parser(repo: HuggingFaceRepo) -> str:
    """Picks the registered parser name for a DeepSeek V3-family checkpoint.

    DeepSeek V3 (original) and V3.1+ share the same architecture name but
    emit tool calls in different grammars. This resolver inspects the
    model's chat template (respecting ``repo.subfolder``, ``repo.revision``,
    and ``repo.trust_remote_code``) to choose between them:

    - ``"deepseekv3"`` for checkpoints whose chat template wraps tool
      arguments in a ``\\`\\`\\`json`` markdown block (original V3).
    - ``"deepseekv3_1"`` for checkpoints whose chat template emits raw JSON
      between ``<｜tool▁sep｜>`` and ``<｜tool▁call▁end｜>`` (V3.1 and
      later).

    Defaults to ``"deepseekv3_1"`` when the chat template is unavailable.
    """
    chat_template = _load_chat_template(repo)
    if chat_template and "```json" in chat_template:
        return "deepseekv3"
    return "deepseekv3_1"
