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
"""Pluggable grammar backend for constrained decoding.

Abstracts the grammar/matcher engine behind two protocols so MAX can support
more than one structured-output backend:

* :class:`GrammarBackend` owns grammar compilation, matcher construction, and
  bitmask allocation/filling (the engine-level, non-per-token entry points).
* :class:`GrammarMatcher` is the per-request object stepped each decode step.
  Method names mirror llguidance's ``LLMatcher`` so the hot decode path is
  backend-agnostic by duck typing.

``llguidance`` is the original (and default) backend; its native ``LLMatcher``
already satisfies :class:`GrammarMatcher`, so :class:`LlguidanceBackend` is a
thin pass-through with no behavior change.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Collection
from functools import wraps
from typing import Any, Protocol, TypeVar, cast

import llguidance
import llguidance.hf
import llguidance.numpy
import numpy as np
import numpy.typing as npt
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max import _xgrammar as xgrammar
from max._xgrammar.builtin_structural_tag import (
    get_inkling_response_format_branch,
)
from max._xgrammar.structural_tag import (
    Format,
    JSONSchemaFormat,
    OrFormat,
    StructuralTag,
)
from max.pipelines.context import GrammarMatcher
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.json_schema import schema_shape
from max.pipelines.lib.tool_parsing import get_parser_cls
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

logger = logging.getLogger("max.pipelines")

# Log a line whenever a grammar compile exceeds this threshold (milliseconds).
# Compilation runs synchronously on the decode thread, so a cold build stalls
# co-batched requests. Override via ``MAX_GRAMMAR_COMPILE_LOG_MS``; the 10ms
# default surfaces real cold compiles without spamming the warm path.
_GRAMMAR_COMPILE_LOG_MS = float(
    os.environ.get("MAX_GRAMMAR_COMPILE_LOG_MS", "10.0")
)

# Cap on each run of consecutive whitespace a whitespace-tolerant
# response_format grammar accepts between JSON tokens.
STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN = 16

_CompileFn = TypeVar("_CompileFn", bound=Callable[..., Any])


def _structural_tag_schemas(tag: dict[str, Any]) -> list[Any] | None:
    """Returns the schemas a structural tag embeds, or None if not a tag.

    A tag's ``format`` is an object where a JSON Schema's is a string, which
    is what tells the two apart. The scan stops at each ``json_schema``, so
    the tag and its schemas are walked once each.
    """
    if not isinstance(tag.get("format"), dict):
        return None
    schemas: list[Any] = []
    stack: list[Any] = [tag]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "json_schema":
                    schemas.append(value)
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(v for v in node if isinstance(v, (dict, list)))
    return schemas


def _compiled_shape(body: Any) -> tuple[int, int] | None:
    """Returns ``(max_depth, total_subschemas)`` of what a compile was handed.

    A tool grammar is a structural tag wrapping one schema per tool, so its
    shape is the deepest of those schemas and the total across them. Anything
    else is measured as a schema in its own right.
    """
    if isinstance(body, (str, bytes)):
        try:
            body = json.loads(body)
        except (ValueError, TypeError):
            return None
    if not isinstance(body, dict):
        return None

    embedded = _structural_tag_schemas(body)
    if embedded is None:
        return schema_shape(body)
    shapes = [s for s in map(schema_shape, embedded) if s is not None]
    if not shapes:
        return None
    return max(d for d, _ in shapes), sum(n for _, n in shapes)


def _log_if_slow(fn: _CompileFn) -> _CompileFn:
    """Times a compile entry point and logs when it exceeds the threshold.

    Duration, method, backend and schema shape ride along as ``extra`` so they
    are queryable; the schema body never does, being large and possibly
    sensitive. Shape is walked only past the threshold and is linear in
    subschema count, so the warm path is unaffected.

    Typical shapes, for recognizing an outlier: a ``response_format`` schema or
    a single tool's arguments sit at single-digit depth with at most a few dozen
    subschemas, and a tool grammar holds that depth while its node count scales
    with the tool count (20 tools of 5 subschemas each reports depth 4, 100
    nodes). Compile cost grows super-linearly in both depth and breadth, so a
    schema hundreds deep, or a thousand wide, sits orders of magnitude above
    typical.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return fn(self, *args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if elapsed_ms > _GRAMMAR_COMPILE_LOG_MS:
                extra: dict[str, Any] = {
                    "event": "grammar_compile_slow",
                    "grammar_compile_method": fn.__name__,
                    "grammar_compile_time_ms": elapsed_ms,
                    "grammar_backend": self.name,
                }
                # Both wrapped methods take exactly one argument, so a
                # keyword call puts the payload in the sole kwargs entry.
                payload = args[0] if args else next(iter(kwargs.values()), None)
                # Never let diagnostics break the call they are describing.
                try:
                    shape = _compiled_shape(payload)
                except Exception:
                    shape = None
                if shape is not None:
                    extra["grammar_schema_depth"] = shape[0]
                    extra["grammar_schema_nodes"] = shape[1]
                logger.info(
                    "grammar %s took %.1fms (%s backend)",
                    fn.__name__,
                    elapsed_ms,
                    self.name,
                    extra=extra,
                )

    return cast(_CompileFn, wrapper)


class _TikTokenAdapter:
    """Adapter to make TikToken-based tokenizers compatible with llguidance.

    llguidance's TokenizerWrapper expects a tokenizer object with specific
    attributes (eos_token_id, bos_token_id, tokens, special_token_ids) and
    a callable interface for encoding. This adapter wraps TikToken-based
    tokenizers (which don't inherit from PreTrainedTokenizerFast) to provide
    that interface.

    Raises:
        ValueError: If the tokenizer is not a TikToken-based tokenizer.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if "TikToken" not in type(tokenizer).__name__:
            raise ValueError(
                f"Structured output requires PreTrainedTokenizerFast or "
                f"TikToken-based tokenizers, but got {type(tokenizer).__name__}"
            )

        self._tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.special_token_ids = getattr(tokenizer, "all_special_ids", [])

        # convert_ids_to_tokens returns the byte->unicode surface form, not the
        # token's true bytes; reverse it via the tokenizer's byte_decoder, or
        # llguidance masks the wrong bytes and leaks control chars into output.
        byte_decoder = getattr(tokenizer, "byte_decoder", None)
        if byte_decoder is None:
            raise ValueError(
                "TikToken-based structured output requires a tokenizer with a "
                "`byte_decoder` (byte-level BPE inverse map); "
                f"{type(tokenizer).__name__} does not provide one."
            )
        vocab_size = len(tokenizer.get_vocab())
        self._tokens: list[bytes] = []
        for i in range(vocab_size):
            token_str = tokenizer.convert_ids_to_tokens(i)
            if token_str is None:
                self._tokens.append(b"")
            else:
                try:
                    self._tokens.append(
                        bytes(byte_decoder[c] for c in token_str)
                    )
                except KeyError:
                    self._tokens.append(
                        token_str.encode("utf-8", errors="replace")
                    )

    @property
    def tokens(self) -> list[bytes]:
        """Returns byte representation of each token in vocabulary."""
        return self._tokens

    def __call__(self, text: str | bytes) -> list[int]:
        """Encode text to token IDs."""
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")

        return self._tokenizer.encode(text, allow_special_tokens=True)


# Backend-specific compiled-grammar handle, kept consistent across a backend's
# compile/validate/create_matcher methods.
GrammarT = TypeVar("GrammarT")


class GrammarBackend(Protocol[GrammarT]):
    """Engine-level entry points: compile grammars, build matchers, bitmasks.

    Only the model worker holds a backend: it owns the single grammar
    compile, and rejects what it cannot build with an :class:`InputError`
    the API server turns into an HTTP 400.
    """

    name: str

    def compile_json_schema(self, json_schema: str) -> GrammarT:
        """Compile a JSON schema to a grammar handle for this backend."""
        ...

    def create_matcher(self, grammar: GrammarT | str) -> GrammarMatcher:
        """Build a matcher from a compiled grammar handle or grammar string.

        The string form is a raw serialized grammar (e.g. a tool-call grammar).
        """
        ...

    def validate_grammar(self, grammar: GrammarT) -> None:
        """Raise if the compiled grammar is invalid.

        Catches grammars that compile but are semantically invalid (most
        importantly unsatisfiable schemas). Backends that already reject
        these at compile time may do nothing.
        """
        ...

    def allocate_token_bitmask(
        self, batch_size: int, vocab_size: int
    ) -> npt.NDArray[np.int32]:
        """Allocate a packed ``[batch_size, ceil(vocab_size/32)]`` int32 bitmask."""
        ...

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Fill ``bitmask`` row ``index`` with the matcher's allowed tokens."""
        ...


class LlguidanceBackend(GrammarBackend[Any]):
    """llguidance backend. Thin pass-through over the native ``LLMatcher``."""

    name = "llguidance"

    def __init__(
        self, tokenizer_info: Any, any_whitespace: bool = False
    ) -> None:
        self._tokenizer_info = tokenizer_info
        self._any_whitespace = any_whitespace

    @classmethod
    def from_tokenizer_delegate(
        cls,
        tokenizer_delegate: PreTrainedTokenizerBase,
        vocab_size: int,
        any_whitespace: bool = False,
    ) -> LlguidanceBackend:
        """Build the llguidance tokenizer info from a tokenizer delegate."""
        if isinstance(tokenizer_delegate, PreTrainedTokenizerFast):
            tokenizer_info = llguidance.hf.from_tokenizer(
                tokenizer_delegate, n_vocab=vocab_size
            )
        else:
            adapter = _TikTokenAdapter(tokenizer_delegate)
            wrapper = TokenizerWrapper(adapter)
            tokenizer_info = LLTokenizer(wrapper, n_vocab=vocab_size)
        return cls(tokenizer_info, any_whitespace=any_whitespace)

    @_log_if_slow
    def compile_json_schema(self, json_schema: str) -> Any:
        """Compile a JSON schema to a grammar handle for this backend."""
        # The empty whitespace pattern pins compact JSON (no whitespace
        # between tokens); omitting it uses llguidance's whitespace-tolerant
        # default. Unlike xgrammar, llguidance has no whitespace-run cap: a
        # bounded whitespace_pattern regex does not bound consecutive
        # whitespace (the pattern repeats), so whitespace-tolerant mode on
        # this backend is unbounded.
        return LLMatcher.grammar_from_json_schema(
            json_schema,
            overrides=(
                None if self._any_whitespace else {"whitespace_pattern": ""}
            ),
        )

    @_log_if_slow
    def create_matcher(self, grammar: Any) -> GrammarMatcher:
        """Build a matcher from a compiled grammar (backend-specific handle)."""
        return LLMatcher(self._tokenizer_info, grammar)

    def validate_grammar(self, grammar: Any) -> None:
        """Raise if the compiled grammar is invalid.

        llguidance's matcher path fails open on unsatisfiable schemas, so
        this explicit check is the gate that rejects them at admission.
        """
        error = LLMatcher.validate_grammar(grammar)
        if error:
            raise ValueError(error)

    def allocate_token_bitmask(
        self, batch_size: int, vocab_size: int
    ) -> npt.NDArray[np.int32]:
        """Allocate a packed ``[batch_size, ceil(vocab_size/32)]`` int32 bitmask."""
        return llguidance.numpy.allocate_token_bitmask(batch_size, vocab_size)

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Fill ``bitmask`` row ``index`` with the matcher's allowed tokens."""
        assert isinstance(matcher, LLMatcher)
        llguidance.numpy.fill_next_token_bitmask(matcher, bitmask, index=index)


class XgrammarMatcher:
    """Adapter exposing an xgrammar ``GrammarMatcher`` as a :class:`GrammarMatcher`."""

    def __init__(self, matcher: Any) -> None:
        self._matcher = matcher

    def try_consume_tokens(self, tokens: list[int]) -> int:
        """Advance the matcher; returns the number of tokens consumed."""
        consumed = 0
        for token in tokens:
            if not self._matcher.accept_token(token):
                break
            consumed += 1
        return consumed

    def is_accepting(self) -> bool:
        """Whether the matcher is at an accepting (stoppable) state."""
        return bool(self._matcher.is_completed())

    def is_stopped(self) -> bool:
        """Whether the matcher has reached a terminal state."""
        return bool(self._matcher.is_terminated())

    def get_error(self) -> str | None:
        """Error message for the last rejection, if any (diagnostics)."""
        return None

    def get_grammar_warnings(self) -> Any:
        """Grammar compilation warnings, if any (diagnostics)."""
        return None

    def deep_copy(self) -> XgrammarMatcher:
        """Independent copy for speculative walks (never mutates the original)."""
        return XgrammarMatcher(self._matcher.fork())


def content_stripped_special_token_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> set[int]:
    """Return the ids of special/added tokens that carry no string content.

    A control/special token that decodes to the empty string with
    ``skip_special_tokens=True`` is never valid JSON string content: if left in
    the byte-level content vocabulary it can satisfy a character-count minimum
    while decoding to nothing, silently bypassing ``minLength``. Marking such
    ids special excludes them from byte-content matching while keeping them
    emittable at grammar positions that reference them by id. Sourced generally
    from the tokenizer's own special/added-token metadata plus a decode-strip
    check -- no model-specific id list.
    """
    candidates: set[int] = set()
    for tid in getattr(tokenizer, "all_special_ids", None) or []:
        candidates.add(int(tid))
    for tid in getattr(tokenizer, "added_tokens_decoder", None) or {}:
        candidates.add(int(tid))
    return {
        tid
        for tid in candidates
        if tokenizer.decode([tid], skip_special_tokens=True) == ""
    }


def special_token_ids_for_markers(
    markers: Collection[str],
    tokenizer_delegate: PreTrainedTokenizerBase,
) -> set[int]:
    """Resolve structural marker strings to their single vocabulary token ids.

    A marker the tokenizer encodes as one token is registered special so its
    bytes are masked out of byte-level string content; a marker with no
    single-token form is skipped (the grammar keeps its byte-literal form).
    """
    unk_id = getattr(tokenizer_delegate, "unk_token_id", None)
    ids: set[int] = set()
    for marker in markers:
        token_id = tokenizer_delegate.convert_tokens_to_ids(marker)
        if token_id is None or token_id == unk_id:
            continue
        ids.add(int(token_id))
    return ids


# Structural tool-call markers a model's grammar references by token id (rather
# than by byte literal), keyed by the parser's ``XGRAMMAR_FORMAT`` model key.
# They must be masked out of byte-level string content so a bare string value's
# length bound cannot be satisfied by the marker's own bytes, while the marker
# stays emittable where the grammar references it by id.
STRUCTURAL_MARKERS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "glm_4_7": ("<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"),
}


def special_token_ids_for(
    tool_parser_name: str | None,
    tokenizer_delegate: PreTrainedTokenizerBase,
) -> set[int]:
    """Return every special token id the grammar backend must mask."""
    parser_cls = get_parser_cls(tool_parser_name)
    model = (
        getattr(parser_cls, "XGRAMMAR_FORMAT", None)
        if isinstance(parser_cls, type)
        else None
    )
    markers = STRUCTURAL_MARKERS_BY_MODEL.get(model, ()) if model else ()
    return content_stripped_special_token_ids(
        tokenizer_delegate
    ) | special_token_ids_for_markers(markers, tokenizer_delegate)


# xgrammar's compiled-grammar cache is unbounded by default; a long-running
# server accumulating unique schemas would grow it without limit. Bound it like
# vLLM does (its VLLM_XGRAMMAR_CACHE_MB, default 512 MB); override the MB budget
# with MODULAR_XGRAMMAR_CACHE_MB.
_DEFAULT_XGRAMMAR_CACHE_MB = 512


def _xgrammar_cache_limit_bytes() -> int:
    """Return the byte budget for xgrammar's compiled-grammar LRU cache."""
    mb = os.environ.get("MODULAR_XGRAMMAR_CACHE_MB")
    return (int(mb) if mb else _DEFAULT_XGRAMMAR_CACHE_MB) * 1024 * 1024


class XgrammarBackend(GrammarBackend[Any]):
    """xgrammar backend.

    Compiles JSON schemas with full ``$ref``/``$defs``/``anyOf``/type-list
    enforcement (where llguidance fails open). The packed int32 bitmask layout
    matches llguidance's, and ``fill_next_token_bitmask`` writes numpy arrays
    directly, so the decode hot path stays torch-free.
    """

    name = "xgrammar"

    def __init__(
        self,
        compiler: Any,
        any_whitespace: bool = False,
    ) -> None:
        self._compiler = compiler
        self._any_whitespace = any_whitespace

    @classmethod
    def from_tokenizer_delegate(
        cls,
        tokenizer_delegate: PreTrainedTokenizerBase,
        vocab_size: int,
        stop_token_ids: Collection[int] | None = None,
        special_token_ids: Collection[int] = (),
        any_whitespace: bool = False,
    ) -> XgrammarBackend:
        """Build the xgrammar tokenizer info and compiler from a delegate."""
        stop_token_ids = (
            list(stop_token_ids) if stop_token_ids is not None else None
        )
        if isinstance(tokenizer_delegate, PreTrainedTokenizerFast):
            tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
                tokenizer_delegate,
                vocab_size=vocab_size,
                stop_token_ids=stop_token_ids,
                special_token_ids=sorted(set(special_token_ids)),
            )
        else:
            adapter = _TikTokenAdapter(tokenizer_delegate)
            if stop_token_ids is None and adapter.eos_token_id is not None:
                stop_token_ids = [adapter.eos_token_id]
            tokenizer_info = xgrammar.TokenizerInfo(
                adapter.tokens,
                vocab_type=xgrammar.VocabType.RAW,
                vocab_size=vocab_size,
                stop_token_ids=stop_token_ids,
            )
        return cls(
            xgrammar.GrammarCompiler(
                tokenizer_info, max_memory_bytes=_xgrammar_cache_limit_bytes()
            ),
            any_whitespace=any_whitespace,
        )

    @_log_if_slow
    def compile_json_schema(self, json_schema: Any) -> Any:
        """Compile a JSON schema (str or dict) to a grammar handle."""
        schema = (
            json_schema
            if isinstance(json_schema, str)
            else json.dumps(json_schema)
        )
        # Compact mode pins the separators; whitespace-tolerant mode must
        # leave them unset (xgrammar's flexible-whitespace grammar) and caps
        # each whitespace run so a looping model cannot emit whitespace
        # forever.
        return self._compiler.compile_json_schema(
            schema,
            any_whitespace=self._any_whitespace,
            separators=None if self._any_whitespace else (",", ":"),
            max_whitespace_cnt=(
                STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN
                if self._any_whitespace
                else None
            ),
            # MAX fails closed: reject schemas the engine cannot faithfully
            # enforce rather than fall back to unconstrained decoding.
            reject_unsupported=True,
        )

    @_log_if_slow
    def create_matcher(self, grammar: Any) -> GrammarMatcher:
        """Build a matcher from a compiled grammar or structural-tag JSON."""
        if isinstance(grammar, xgrammar.CompiledGrammar):
            compiled = grammar
        elif isinstance(grammar, str):
            tag = xgrammar.StructuralTag.model_validate_json(grammar)
            compiled = self._compiler.compile_structural_tag(tag)
        else:
            raise InputError(
                f"The xgrammar backend received an unsupported grammar of "
                f"type {type(grammar).__name__}."
            )
        return XgrammarMatcher(xgrammar.GrammarMatcher(compiled))

    def validate_grammar(self, grammar: Any) -> None:
        """Raise if the compiled grammar is invalid.

        Xgrammar rejects unsatisfiable schemas at compile time, so the
        compiled grammar reaching here is already valid (nothing to check).
        """
        return

    def allocate_token_bitmask(
        self, batch_size: int, vocab_size: int
    ) -> npt.NDArray[np.int32]:
        """Allocate a packed ``[batch_size, ceil(vocab_size/32)]`` int32 bitmask."""
        # -1 == all bits set == unconstrained (matches llguidance + MAX's
        # bitmask convention); fill_next_token_bitmask overwrites filled rows.
        words = (vocab_size + 31) // 32
        return np.full((batch_size, words), -1, dtype=np.int32)

    def fill_next_token_bitmask(
        self,
        matcher: GrammarMatcher,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Fill ``bitmask`` row ``index`` with the matcher's allowed tokens.

        Once ``matcher`` is stopped, xgrammar's own fill raises a fatal C++
        check rather than returning a mask -- unlike llguidance, which
        computes a real EOS-only mask in that state. ``stop_token_ids`` is a
        plain property, safe to read even after termination, so the row is
        built by hand here instead: only the stop tokens stay allowed,
        giving xgrammar the same forced termination llguidance already
        provides rather than leaving the row unconstrained.
        """
        assert isinstance(matcher, XgrammarMatcher)
        if matcher.is_stopped():
            bitmask[index, :] = 0
            for stop_id in matcher._matcher.stop_token_ids:
                bitmask[index, stop_id // 32] |= np.int32(1) << np.int32(
                    stop_id % 32
                )
            return
        matcher._matcher.fill_next_token_bitmask(bitmask, index)


def build_xgrammar_tool_grammar(
    model_format: str,
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any],
    response_format_schema: dict[str, Any] | None = None,
) -> str:
    """Build a serialized xgrammar tool-call grammar (StructuralTag JSON).

    Uses xgrammar's built-in per-model tool-call format (e.g. ``"kimi"``),
    which frames the model's tool-call envelope and constrains each call's
    arguments to that tool's JSON schema. The returned JSON string is passed
    as a grammar to :meth:`XgrammarBackend.create_matcher`.

    When ``response_format_schema`` is provided, the grammar accepts *either* a
    tool call *or* a JSON response matching that schema, mirroring the
    llguidance path's ``start: tool_calls | json_response`` alternation.

    Args:
        model_format: xgrammar model-format key (e.g. ``"kimi"``).
        tools: OpenAI-style tool dicts (``{"type": "function", "function": ...}``).
        tool_choice: ``"auto"``, ``"required"``, or a named choice.
        response_format_schema: Optional JSON schema for a ``response_format``
            json_schema response. When set, the grammar allows a
            schema-conforming JSON response as an alternative to a tool call.

    Returns:
        The StructuralTag serialized as a JSON string.
    """
    effective_tool_choice = tool_choice
    if response_format_schema is not None and tool_choice == "auto" and tools:
        # ``auto`` makes the tool call optional, so free-form text is a valid
        # output. OR-ing that with the response schema would let arbitrary text
        # through and defeat the schema. Use the mandatory (``required``) tool
        # section so the only accepting outputs are a complete tool call or a
        # schema-conforming JSON.
        effective_tool_choice = "required"
    tag = xgrammar.get_builtin_structural_tag(
        model_format,
        tools=tools,
        tool_choice=effective_tool_choice,
        reasoning=False,
    )
    if response_format_schema is not None:
        json_branch: Format = JSONSchemaFormat(
            json_schema=response_format_schema,
            any_whitespace=False,
            separators=(",", ":"),
            reject_unsupported=True,
        )
        if model_format == "inkling":
            # An Inkling turn is a sequence of framed messages; bare JSON is
            # not something its parsers can read.
            json_branch = get_inkling_response_format_branch(json_branch)
        tag = StructuralTag(format=OrFormat(elements=[tag.format, json_branch]))
    return tag.model_dump_json()


def make_grammar_backend(
    name: str,
    tokenizer_delegate: PreTrainedTokenizerBase,
    vocab_size: int,
    *,
    tool_parser_name: str | None = None,
    stop_token_ids: Collection[int] | None = None,
    any_whitespace: bool = False,
) -> GrammarBackend[Any]:
    """Construct the structured-output backend selected by ``name``.

    Args:
        name: Backend identifier (``"llguidance"`` or ``"xgrammar"``).
        tokenizer_delegate: HuggingFace/TikToken tokenizer to build vocab info.
        vocab_size: Vocabulary size from the tokenizer.
        tool_parser_name: Active tool parser, used to derive the special-token
            mask and the fail-closed compile policy (xgrammar only).
        stop_token_ids: The full set of stop token IDs.
        any_whitespace: Whether ``response_format`` grammars accept whitespace
            between JSON tokens. ``False`` (the default) pins compact JSON.

    Returns:
        A configured :class:`GrammarBackend`.

    Raises:
        ValueError: If ``name`` is not a known backend.
    """
    if name == "llguidance":
        return LlguidanceBackend.from_tokenizer_delegate(
            tokenizer_delegate, vocab_size, any_whitespace=any_whitespace
        )
    if name == "xgrammar":
        return XgrammarBackend.from_tokenizer_delegate(
            tokenizer_delegate,
            vocab_size,
            stop_token_ids=stop_token_ids,
            special_token_ids=special_token_ids_for(
                tool_parser_name, tokenizer_delegate
            ),
            any_whitespace=any_whitespace,
        )
    raise ValueError(
        f"unknown structured output backend: {name!r} "
        f"(supported: 'llguidance', 'xgrammar')"
    )
