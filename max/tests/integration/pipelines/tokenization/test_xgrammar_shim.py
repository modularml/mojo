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

"""Parity/integration test for the ``max._xgrammar`` shim.

Validates that the shim restores the upstream xgrammar Python-layer surface MAX's
structured-output backend depends on -- over the nanobind binding --
including the structural-tag bridge (vendored pydantic models ->
``model_dump_json`` -> the C++ structural-tag compiler) used by the tool-call
path.
"""

import importlib.util
import json
import math
import sys
import time
from typing import Any

import numpy as np
import pytest
from max import _xgrammar as xgr
from max._xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OrFormat,
    TagFormat,
    TriggeredTagsFormat,
)

_VOCAB = [
    "{",
    "}",
    "[",
    "]",
    '"',
    ":",
    ",",
    " ",
    "a",
    "b",
    "1",
    "true",
    "null",
    "<eos>",
]


def _compiler() -> xgr.GrammarCompiler:
    tokenizer_info = xgr.TokenizerInfo(
        _VOCAB,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[len(_VOCAB) - 1],
    )
    return xgr.GrammarCompiler(tokenizer_info)


def _accepts(compiled: xgr.CompiledGrammar, s: str) -> bool:
    # Whether the grammar accepts s as a complete value.
    matcher = xgr.GrammarMatcher(compiled)
    return matcher.accept_string(s) and matcher.is_completed()


def _toolcall(schema: dict[str, Any]) -> xgr.StructuralTag:
    # Build a tool-call structural tag wrapping one JSON-schema arg. Tool-call
    # arguments are always a JSON object, so opt into require_object_root on the
    # JSONSchemaFormat (the flag defaults false on the general path).
    return xgr.StructuralTag(
        format=TriggeredTagsFormat(
            triggers=["<t>"],
            tags=[
                TagFormat(
                    begin="<t>",
                    content=JSONSchemaFormat(
                        json_schema=schema,
                        require_object_root=True,
                    ),
                    end="</t>",
                )
            ],
        )
    )


def _set_require_object_root(fmt: Any) -> None:
    # Recursively opt every JSONSchemaFormat in a tag tree into
    # require_object_root (the flag defaults false); mirrors what a model's
    # tool-call tag would set. Walks the container fields of the format models.
    if isinstance(fmt, JSONSchemaFormat):
        fmt.require_object_root = True
    for child_field in ("content",):
        child = getattr(fmt, child_field, None)
        if child is not None:
            _set_require_object_root(child)
    for list_field in ("elements", "tags"):
        for child in getattr(fmt, list_field, None) or []:
            _set_require_object_root(child)


def test_shim_is_torch_free() -> None:
    assert "torch" not in sys.modules
    assert importlib.util.find_spec("torch") is None


def test_shim_surface_present() -> None:
    assert hasattr(xgr.TokenizerInfo, "from_huggingface")
    assert callable(xgr.get_builtin_structural_tag)
    for name in (
        "TokenizerInfo",
        "VocabType",
        "GrammarCompiler",
        "CompiledGrammar",
        "GrammarMatcher",
        "StructuralTag",
        "StructuralTagItem",
        "allocate_token_bitmask",
    ):
        assert hasattr(xgr, name)


def test_allocate_token_bitmask() -> None:
    bitmask = xgr.allocate_token_bitmask(2, 100)
    assert bitmask.shape == (2, (100 + 31) // 32)
    assert bitmask.dtype == np.int32
    assert (bitmask == -1).all()


def test_json_schema_path_through_shim() -> None:
    compiled = _compiler().compile_json_schema('{"type": "object"}')
    assert isinstance(compiled, xgr.CompiledGrammar)
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = np.full((xgr.get_bitmask_size(len(_VOCAB)),), -1, dtype=np.int32)
    assert matcher.fill_next_token_bitmask(bitmask) is True
    assert int(bitmask[0]) != -1


def _nested_object_schema(depth: int) -> str:
    head = "".join(
        f'{{"type": "object", "properties": {{"p{i}":' for i in range(depth)
    )
    return head + '{"type": "string"}' + "}}" * depth


def _compile_cpu_seconds(compiler: xgr.GrammarCompiler, schema: str) -> float:
    started = time.process_time()
    compiled = compiler.compile_json_schema(schema, reject_unsupported=True)
    elapsed = time.process_time() - started
    assert isinstance(compiled, xgr.CompiledGrammar)
    return elapsed


def test_deep_nesting_compile_cost_scales_subcubically() -> None:
    # CPU-time ratios exclude host scheduling noise from this complexity check.
    compiler = _compiler()
    compiler.compile_json_schema('{"type": "null"}', reject_unsupported=True)
    shallow = _compile_cpu_seconds(compiler, _nested_object_schema(100))
    deep = _compile_cpu_seconds(compiler, _nested_object_schema(400))
    assert shallow > 0
    assert deep / shallow < 30


def test_near_identical_subschemas_do_not_share() -> None:
    # Conflating these two keys would leak one maxLength into the other.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "properties":'
        ' {"a": {"type": "object", "properties":'
        ' {"x": {"type": "string", "maxLength": 5}}},'
        ' "b": {"type": "object", "properties":'
        ' {"x": {"type": "string", "maxLength": 6}}}},'
        ' "required": ["a", "b"]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"a": {"x": "12345"}, "b": {"x": "123456"}}')
    assert not _accepts(compiled, '{"a": {"x": "123456"}, "b": {"x": "1"}}')


def test_property_name_escapes_cannot_forge_a_cache_key() -> None:
    forged_name = 'a":{},"b'
    # Without key escaping, the one-name map serializes identically to the
    # two-name map. Compiling them as siblings exercises the cache lookup.
    compiled = _compiler().compile_json_schema(
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "object",
                        "properties": {forged_name: {}},
                        "additionalProperties": False,
                    },
                    "y": {
                        "type": "object",
                        "properties": {"a": {}, "b": {}},
                        "additionalProperties": False,
                    },
                },
                "required": ["x", "y"],
            }
        ),
        reject_unsupported=True,
    )
    assert _accepts(
        compiled,
        json.dumps({"x": {forged_name: None}, "y": {"a": 1, "b": 2}}),
    )
    assert not _accepts(
        compiled,
        json.dumps({"x": {}, "y": {forged_name: None}}),
    )


def test_property_key_with_embedded_quote_enforces_value_type() -> None:
    key = 'foo"bar'
    compiled = _compiler().compile_json_schema(
        json.dumps(
            {
                "properties": {key: {"$ref": "#/definitions/foo%22bar"}},
                "definitions": {key: {"type": "number"}},
            }
        )
    )
    assert _accepts(compiled, json.dumps({key: 1}))
    assert not _accepts(compiled, json.dumps({key: "1"}))


def test_property_key_with_embedded_backslash_enforces_value_type() -> None:
    key = "a\\b"
    compiled = _compiler().compile_json_schema(
        json.dumps(
            {
                "properties": {key: {"type": "number"}},
            }
        )
    )
    assert _accepts(compiled, json.dumps({key: 1}))
    assert not _accepts(compiled, json.dumps({key: "1"}))


# Rejection of unenforceable keywords is opt-in: it happens only when the caller
# passes reject_unsupported=True. The default (exercised by the guard tests below)
# falls back to best-effort decoding instead.


def test_non_local_ref_rejected() -> None:
    # A non-local $ref (external URI) cannot be resolved into a grammar; it
    # would silently under-constrain, so reject_unsupported rejects it.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"$ref": "http://example.com"}', reject_unsupported=True
        )


def test_plain_name_anchor_rejected() -> None:
    # "#foo" is a plain-name anchor, not a JSON pointer; ResolveRef returns an
    # unconstrained AnySpec, so it is rejected rather than under-constrain.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"$ref": "#foo"}', reject_unsupported=True
        )


def test_local_ref_compiles() -> None:
    # A local JSON-pointer $ref ("#/...") is resolvable, so unlike a non-local or
    # plain-name ref it must NOT be rejected -- it compiles and enforces the
    # referenced schema.
    compiled = _compiler().compile_json_schema(
        '{"$ref": "#/$defs/S", "$defs": {"S": {"type": "string"}}}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)
    assert _accepts(compiled, '"a"')


# An embedded resource: "sub" declares its own "$id" and its own "$defs/x", so
# the "#/$defs/x" inside it names sub's string, not the document's integer.
# Resolving against the document root binds the wrong one of the two.
_EMBEDDED_RESOURCE_SCHEMA = (
    '{"$defs": {"x": {"type": "integer"},'
    ' "sub": {"$id": "https://example.com/sub",'
    ' "$defs": {"x": {"type": "string"}},'
    ' "type": "object", "properties": {"v": {"$ref": "#/$defs/x"}},'
    ' "required": ["v"]}},'
    ' "$ref": "#/$defs/sub"}'
)


def test_ref_under_nested_id_refused() -> None:
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            _EMBEDDED_RESOURCE_SCHEMA, reject_unsupported=True
        )


def test_hash_ref_under_nested_id_refused() -> None:
    # A bare "#" is a fragment too: inside an embedded resource it means that
    # resource's root, not the document's.
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"$defs": {"sub": {"$id": "https://example.com/sub",'
            ' "type": "object", "properties": {"v": {"$ref": "#"}}}},'
            ' "type": "object", "properties": {"w": {"$ref": "#/$defs/sub"}}}',
            reject_unsupported=True,
        )


def test_ref_reached_through_nested_id_refused() -> None:
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"$defs": {"x": {"type": "integer"},'
            ' "sub": {"$id": "https://example.com/sub",'
            ' "$defs": {"x": {"type": "string"}},'
            ' "properties": {"inner": {"type": "object",'
            ' "properties": {"v": {"$ref": "#/$defs/x"}},'
            ' "required": ["v"]}}}},'
            ' "type": "object",'
            ' "properties": {"w": {"$ref": "#/$defs/sub/properties/inner"}},'
            ' "required": ["w"]}',
            reject_unsupported=True,
        )


def test_same_ref_text_in_both_resources_refused() -> None:
    # The root's "#/$defs/x" parses first, so a scope-blind parse cache would
    # hand its answer to sub's identically spelled one.
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"$defs": {"x": {"type": "integer"},'
            ' "sub": {"$id": "https://example.com/sub",'
            ' "$defs": {"x": {"type": "string"}},'
            ' "type": "object", "properties": {"v": {"$ref": "#/$defs/x"}},'
            ' "required": ["v"]}},'
            ' "type": "object",'
            ' "properties": {"a": {"$ref": "#/$defs/x"},'
            ' "b": {"$ref": "#/$defs/sub"}},'
            ' "required": ["a", "b"]}',
            reject_unsupported=True,
        )


# The refusal is a property of the document, not of any one reference: a
# declaration below the root is refused even where every reference in the
# document happens to be a root-resource one. Resolving those correctly needs a
# base URI per subschema, which this converter does not compute.
def test_unused_nested_id_refuses_the_document() -> None:
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"$defs": {"x": {"type": "string"},'
            ' "unused": {"$id": "https://example.com/unused",'
            ' "type": "integer"}},'
            ' "type": "object", "properties": {"v": {"$ref": "#/$defs/x"}},'
            ' "required": ["v"]}',
            reject_unsupported=True,
        )


def test_allof_beside_unused_nested_id_refuses_the_document() -> None:
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"$defs": {"x": {"type": "string"},'
            ' "unused": {"$id": "https://example.com/unused", "type": "integer"}},'
            ' "allOf": [{"$ref": "#/$defs/x"}]}',
            reject_unsupported=True,
        )


# An unknown extension keyword is an annotation whose value is arbitrary JSON,
# so an "$id" member inside it is a name and declares nothing.
def test_annotation_keyword_id_does_not_declare_a_resource() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "x-metadata": {"$id": "not-a-schema"}}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, "1")


def test_empty_nested_id_refuses_the_document() -> None:
    # "" is a valid URI-reference, so this is a declaration with empty text
    # rather than no declaration, and sub's own "x" is what its $ref names.
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"$defs": {"x": {"type": "integer"},'
            ' "sub": {"$id": "", "$defs": {"x": {"type": "string"}},'
            ' "$ref": "#/$defs/x"}},'
            ' "$ref": "#/$defs/sub"}',
            reject_unsupported=True,
        )


def test_definition_named_id_does_not_open_resource() -> None:
    compiled = _compiler().compile_json_schema(
        '{"$defs": {"$id": {"type": "string"}},'
        ' "allOf": [{"$ref": "#/$defs/$id"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, "1")


def test_definition_named_id_can_contain_embedded_resource() -> None:
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"$defs": {"x": {"type": "integer"},'
            ' "$id": {"$id": "https://example.com/sub",'
            ' "$defs": {"x": {"type": "string"}},'
            ' "$ref": "#/$defs/x"}},'
            ' "$ref": "#/$defs/$id"}',
            reject_unsupported=True,
        )


def test_ref_under_root_id_still_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        '{"$id": "https://example.com/root",'
        ' "$defs": {"x": {"type": "string"}},'
        ' "type": "object", "properties": {"v": {"$ref": "#/$defs/x"}},'
        ' "required": ["v"]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"v": "a"}')
    assert not _accepts(compiled, '{"v": 1}')


# The root's "$id" text repeated below the root, on instance data. "examples"
# holds values rather than subschemas, so the scan does not descend into it and
# the only declaration in this document is the root's own.
_ROOT_ID_IN_INSTANCE_DATA_SCHEMA = (
    '{"$id": "https://example/root",'
    ' "examples": [{"$id": "https://example/root"}],'
    ' "$defs": {"x": {"type": "string"}},'
    ' "$ref": "#/$defs/x"}'
)


def test_ref_under_root_id_repeated_below_root_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        _ROOT_ID_IN_INSTANCE_DATA_SCHEMA, reject_unsupported=True
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, "1")


def test_root_id_repeated_below_root_permissive_unchanged() -> None:
    compiled = _compiler().compile_json_schema(_ROOT_ID_IN_INSTANCE_DATA_SCHEMA)
    assert _accepts(compiled, '"a"')


def test_instance_id_does_not_rebase_allof_ref() -> None:
    compiled = _compiler().compile_json_schema(
        '{"examples": [{"$id": "https://example.com/instance"}],'
        ' "$defs": {"x": {"type": "string"}},'
        ' "allOf": [{"$ref": "#/$defs/x"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, "1")


def test_ref_under_duplicate_root_id_refused() -> None:
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"$id": "https://example.com/root",'
            ' "$defs": {"x": {"type": "integer"},'
            ' "sub": {"$id": "https://example.com/root",'
            ' "$defs": {"x": {"type": "string"}},'
            ' "type": "object", "properties": {"v": {"$ref": "#/$defs/x"}},'
            ' "required": ["v"]}},'
            ' "$ref": "#/$defs/sub"}',
            reject_unsupported=True,
        )


def test_ref_under_nested_id_permissive_unchanged() -> None:
    compiled = _compiler().compile_json_schema(_EMBEDDED_RESOURCE_SCHEMA)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_dynamic_ref_rejected() -> None:
    # $dynamicRef is a reference (it resolves to a subschema the instance must
    # satisfy), not a no-op annotation; with no other type keyword it would fall
    # back to unconstrained, so it is rejected.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"$dynamicRef": "#node"}', reject_unsupported=True
        )


def test_dynamic_and_recursive_ref_with_type_rejected() -> None:
    for keyword in ("$dynamicRef", "$recursiveRef"):
        with pytest.raises(
            Exception, match=f"unsupported schema keyword: \\{keyword}"
        ):
            _compiler().compile_json_schema(
                f'{{"type": "integer", "{keyword}": "#"}}',
                reject_unsupported=True,
            )


def test_legacy_id_keyword_rejected() -> None:
    # Draft 4 spells the identifier "id", so a nested one declares a resource
    # on the same terms as "$id", and the refusal quotes the keyword the
    # document actually used rather than one it never wrote.
    with pytest.raises(Exception) as excinfo:
        _compiler().compile_json_schema(
            '{"$schema": "http://json-schema.org/draft-04/schema#",'
            ' "definitions": {"x": {"type": "integer"},'
            ' "sub": {"id": "sub", "definitions": {"x": {"type": "string"}},'
            ' "type": "object", "properties": {"v": {"$ref": "#/definitions/x"}},'
            ' "required": ["v"]}}, "$ref": "#/definitions/sub"}',
            reject_unsupported=True,
        )
    message = str(excinfo.value)
    assert "declares a resource below the document root" in message
    assert '"id" "sub"' in message
    assert "$id" not in message


def test_draft_03_refused_as_an_unmodelled_dialect() -> None:
    # Draft 3 names resources with "id" and puts subschemas under "extends",
    # which the scan does not walk. Reading it as a modern document would walk
    # past the resource this one declares, so the dialect is refused instead.
    with pytest.raises(Exception, match='unsupported "\\$schema" dialect'):
        _compiler().compile_json_schema(
            '{"$schema": "http://json-schema.org/draft-03/schema#",'
            ' "type": "string", "extends": {"id": "sub"}}',
            reject_unsupported=True,
        )


def test_custom_schema_uri_spelling_a_draft_is_not_that_draft() -> None:
    # A vendor URI that merely spells a draft is not that draft, and is not a
    # dialect this converter models either, so the document is refused rather
    # than read under a guessed identifier spelling.
    with pytest.raises(Exception, match='unsupported "\\$schema" dialect'):
        _compiler().compile_json_schema(
            '{"$schema": "https://example.com/schemas/draft-04-migration/v1",'
            ' "$defs": {"x": {"type": "integer"},'
            ' "sub": {"$id": "https://example.com/sub",'
            ' "$defs": {"x": {"type": "string"}},'
            ' "$ref": "#/$defs/x"}},'
            ' "$ref": "#/$defs/sub"}',
            reject_unsupported=True,
        )


@pytest.mark.parametrize(
    "metaschema",
    [
        "http://json-schema.org/draft-06/schema#",
        "http://json-schema.org/draft-07/schema#",
        "https://json-schema.org/draft/2019-09/schema",
        "https://json-schema.org/draft/2020-12/schema",
    ],
)
def test_recognized_modern_metaschemas_compile(metaschema: str) -> None:
    compiled = _compiler().compile_json_schema(
        json.dumps(
            {
                "$schema": metaschema,
                "definitions": {"s": {"type": "string"}},
                "$ref": "#/definitions/s",
            }
        ),
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, "1")


def test_modern_dialect_bare_id_is_not_an_identifier() -> None:
    # Draft 6 renamed the identifier to "$id" and left "id" an ordinary
    # annotation, so a draft 7 subschema carrying one declares no resource.
    compiled = _compiler().compile_json_schema(
        '{"$schema": "http://json-schema.org/draft-07/schema#",'
        ' "definitions": {"s": {"id": "metadata", "type": "string"}},'
        ' "$ref": "#/definitions/s"}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, "1")


def test_malformed_root_legacy_id_rejected() -> None:
    # The root's identifier declares nothing to refuse, but an unreadable one
    # is unreadable there too.
    with pytest.raises(Exception, match='"id" must be a string'):
        _compiler().compile_json_schema(
            '{"$schema": "http://json-schema.org/draft-04/schema#",'
            ' "id": {}, "type": "string"}',
            reject_unsupported=True,
        )


def test_permissive_raw_percent_pointer_unchanged() -> None:
    # Upstream read a fragment raw, so a member whose name carries a literal
    # "%" still resolves when unsupported-schema rejection is off.
    compiled = _compiler().compile_json_schema(
        '{"definitions": {"x%": {"type": "string"}}, "$ref": "#/definitions/x%"}'
    )
    assert _accepts(compiled, '"a"')


def test_root_legacy_id_still_compiles() -> None:
    # A root "id" names the document's sole resource, exactly as a root "$id"
    # does, so every fragment still resolves against the document.
    compiled = _compiler().compile_json_schema(
        '{"$schema": "http://json-schema.org/draft-04/schema#",'
        ' "id": "https://example.com/root",'
        ' "definitions": {"x": {"type": "string"}},'
        ' "$ref": "#/definitions/x"}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, "1")


def test_percent_encoded_pointer_separator_compiles() -> None:
    # RFC 6901 evaluates a fragment once percent-decoded, so encoding the
    # pointer's own separators names the same place as writing them plainly.
    compiled = _compiler().compile_json_schema(
        '{"$defs": {"x": {"type": "string"}}, "$ref": "#%2F$defs%2Fx"}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, "1")


def test_percent_encoded_local_ref_compiles() -> None:
    for schema in (
        '{"$defs": {"a b": {"type": "string"}}, "$ref": "#/$defs/a%20b"}',
        '{"$defs": {"x": {"type": "string"}}, "$ref": "#/%24defs/x"}',
        '{"$defs": {"a": {"b": {"type": "string"}},'
        ' "a/b": {"type": "integer"}}, "$ref": "#/$defs/a%2Fb"}',
    ):
        compiled = _compiler().compile_json_schema(
            schema, reject_unsupported=True
        )
        assert _accepts(compiled, '"a"')
        assert not _accepts(compiled, "1")


@pytest.mark.parametrize("ref", ["#/x%", "#/x%GG"])
def test_malformed_percent_ref_rejected(ref: str) -> None:
    with pytest.raises(
        Exception,
        match=r"malformed percent escape.*must be followed by two hexadecimal digits",
    ):
        _compiler().compile_json_schema(
            json.dumps({"$ref": ref}), reject_unsupported=True
        )


def test_malformed_id_rejected_without_recursive_scan() -> None:
    with pytest.raises(Exception, match='"\\$id" must be a string'):
        _compiler().compile_json_schema(
            '{"$defs": {"s": {"$id": {"$id": {"$id": "leaf"}},'
            ' "type": "string"}}, "$ref": "#/$defs/s"}',
            reject_unsupported=True,
        )


@pytest.mark.parametrize(
    ("schema", "error"),
    [
        (
            '{"$defs": {"x": {"type": "integer"},'
            ' "sub": {"$id": {}, "properties":'
            ' {"v": {"$ref": "#/$defs/x"}}}},'
            ' "$ref": "#/$defs/sub/properties/v"}',
            '"\\$id" must be a string',
        ),
        (
            '{"$schema": "http://json-schema.org/draft-04/schema#",'
            ' "definitions": {"x": {"type": "integer"},'
            ' "sub": {"id": "sub", "properties":'
            ' {"v": {"$ref": "#/definitions/x"}}}},'
            ' "$ref": "#/definitions/sub/properties/v"}',
            "declares a resource below the document root",
        ),
    ],
)
def test_ref_path_validates_ancestor_identifier(
    schema: str, error: str
) -> None:
    with pytest.raises(Exception, match=error):
        _compiler().compile_json_schema(schema, reject_unsupported=True)


@pytest.mark.parametrize(
    ("schema", "error"),
    [
        (
            '{"$defs": {"sub": {"$id": {},'
            ' "properties": {"v": {"type": "integer"}}}},'
            ' "allOf": [{"$ref": "#/$defs/sub/properties/v"}]}',
            '"\\$id" must be a string',
        ),
        (
            '{"$schema": "http://json-schema.org/draft-04/schema#",'
            ' "definitions": {"sub": {"id": "sub",'
            ' "properties": {"v": {"type": "integer"}}}},'
            ' "allOf": [{"$ref": "#/definitions/sub/properties/v"}]}',
            "declares a resource below the document root",
        ),
    ],
)
def test_allof_ref_path_validates_ancestor_identifier(
    schema: str, error: str
) -> None:
    with pytest.raises(Exception, match=error):
        _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_unknown_keyword_without_type_rejected() -> None:
    # A type-less schema whose only keyword is unrecognized would fall back to
    # unconstrained decoding; reject rather than silently emit an any-value
    # grammar.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"x-custom": "value"}', reject_unsupported=True
        )


def test_annotation_only_without_type_compiles() -> None:
    # A type-less schema carrying only annotation keywords is genuinely
    # unconstrained; it compiles to the any-value grammar rather than being
    # rejected.
    compiled = _compiler().compile_json_schema('{"title": "foo"}')
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_empty_schema_enforces_single_json_value() -> None:
    # An explicit empty schema ("{}", the any-value AnySpec) is ENFORCED, not
    # unconstrained: it accepts exactly one well-formed JSON value and REJECTS
    # trailing prose after that value. A boolean "true" schema compiles to
    # the same any-value grammar.
    for schema in ("{}", "true"):
        compiled = _compiler().compile_json_schema(schema)
        assert isinstance(compiled, xgr.CompiledGrammar)
        assert _accepts(compiled, "1")
        assert _accepts(compiled, "true")
        assert _accepts(compiled, "{}")
        # A lone value followed by prose is not a single JSON value: rejected.
        assert not _accepts(compiled, "1 ab")
        assert not _accepts(compiled, "{} a")


def test_multiple_of_rejected() -> None:
    # multipleOf has no faithful CFG encoding; reject rather than emit an
    # unconstrained number.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "number", "multipleOf": 5}', reject_unsupported=True
        )


def test_integer_multiple_of_rejected() -> None:
    # multipleOf is also rejected on the integer path (ParseInteger), distinct
    # from the number path above.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "integer", "multipleOf": 3}', reject_unsupported=True
        )


def test_numeric_bound_out_of_range_rejected() -> None:
    # A numeric bound beyond the int64-representable range cannot be enforced by
    # the generated regex, so it is rejected rather than left to fall open.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "number", "minimum": 1e30}', reject_unsupported=True
        )


def test_numeric_bound_at_int64_ulp_boundary() -> None:
    k_max_enforceable = float(2**63 - 1)
    reject = int(k_max_enforceable)
    accept = int(math.nextafter(k_max_enforceable, 0.0))
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            f'{{"type": "number", "minimum": {reject}}}',
            reject_unsupported=True,
        )
    compiled = _compiler().compile_json_schema(
        f'{{"type": "number", "minimum": {accept}}}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_unenforceable_top_level_keywords_rejected() -> None:
    # Each of these is categorically unenforceable and rejected on presence by
    # the top-level keyword check.
    for schema in (
        '{"not": {"type": "string"}}',
        '{"dependentRequired": {"a": ["b"]}}',
        '{"dependentSchemas": {"a": {"type": "string"}}}',
        '{"dependencies": {"a": ["b"]}}',
    ):
        with pytest.raises(Exception):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_ambiguous_type_keywords_rejected() -> None:
    # minLength implies string, minItems implies array: JSON Schema keywords are
    # type-conditional and do not union, so an ambiguous keyword mix is rejected
    # rather than picking an arbitrary type.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"minLength": 5, "minItems": 2}', reject_unsupported=True
        )


def test_conflicting_type_keywords_rejected() -> None:
    # pattern implies string, minimum implies number: another ambiguous mix.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"pattern": "[a-z]+", "minimum": 0}', reject_unsupported=True
        )


def test_annotation_only_keyword_accepted() -> None:
    # An unknown annotation keyword (x-custom) alongside a concrete type does
    # not constrain the instance and must not block compilation.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "x-custom": "value"}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_array_without_items_compiles() -> None:
    # A single-type-family schema (only array keywords) is unambiguous and
    # compiles even without an items schema.
    compiled = _compiler().compile_json_schema(
        '{"type": "array", "minItems": 2}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_typeless_single_family_keywords_compile() -> None:
    # With no explicit "type", validation keywords that all imply ONE type select
    # that arm of the type-inference dispatch and compile.
    for schema in (
        '{"properties": {"a": {"type": "string"}}}',  # object
        '{"minItems": 1}',  # array
        '{"minLength": 1}',  # string
        '{"minimum": 0}',  # number
    ):
        compiled = _compiler().compile_json_schema(schema)
        assert isinstance(compiled, xgr.CompiledGrammar)


def test_array_unique_items_rejected() -> None:
    # uniqueItems (like contains/minContains/maxContains) cannot be enforced by a
    # CFG; reject on the array path.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "array", "uniqueItems": true}', reject_unsupported=True
        )


def test_tool_call_with_multiple_of_in_arg_schema_rejected() -> None:
    # The tool-call path (StructuralTag -> compile_structural_tag) carries
    # reject_unsupported through the tag JSON via JSONSchemaFormat, so an
    # unenforceable keyword in a tool-argument schema is rejected there too.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"n": {"type": "number", "multipleOf": 5}},
            },
            reject_unsupported=True,
        )
    )
    with pytest.raises(Exception):
        _compiler().compile_structural_tag(tag)


def test_unsupported_keyword_compiles_by_default() -> None:
    # Without reject_unsupported, an unenforceable keyword (multipleOf) falls
    # back to best-effort decoding rather than raising -- the default is
    # permissive on the plain compile_json_schema path.
    compiled = _compiler().compile_json_schema(
        '{"type": "number", "multipleOf": 5}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_oneof_type_disjoint_compiles_and_enforces() -> None:
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"type": "string"}, {"type": "integer"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(compiled, "1")
    assert not _accepts(compiled, "true")
    assert not _accepts(compiled, "null")


def test_oneof_const_disjoint_compiles_and_enforces() -> None:
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"const": "a"}, {"const": "b"}]}', reject_unsupported=True
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(compiled, '"b"')
    assert not _accepts(compiled, '"ab"')


def test_const_enum_control_char_string_is_escaped() -> None:
    # A const/enum string value with a control char must render as its escaped
    # JSON form; the grammar must reject the raw control byte (invalid JSON).
    nl = _compiler().compile_json_schema(
        json.dumps({"const": "a\nb"}), reject_unsupported=True
    )
    assert _accepts(nl, '"a\\nb"')  # escaped newline -> valid JSON, accepted
    assert not _accepts(nl, '"a\nb"')  # raw newline -> invalid JSON, rejected

    en = _compiler().compile_json_schema(
        json.dumps({"enum": ["a\nb"]}), reject_unsupported=True
    )
    assert _accepts(en, '"a\\nb"')
    assert not _accepts(en, '"a\nb"')

    nul = _compiler().compile_json_schema(
        json.dumps({"const": "x\x00y"}), reject_unsupported=True
    )
    assert _accepts(nul, '"x\\u0000y"')  # escaped NUL -> valid, accepted
    assert not _accepts(nul, '"x\x00y"')  # raw NUL -> invalid, rejected


def test_oneof_enum_disjoint_compiles_and_enforces() -> None:
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"enum": ["a", "b"]}, {"enum": [1]}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(compiled, "1")
    assert not _accepts(compiled, "true")


def test_oneof_discriminated_objects_compile_and_enforce() -> None:
    # Distinct required discriminator values make the branches disjoint.
    compiled = _compiler().compile_json_schema(
        '{"oneOf": ['
        '{"type": "object", "required": ["a"], '
        '"properties": {"a": {"const": "a"}}}, '
        '{"type": "object", "required": ["a"], '
        '"properties": {"a": {"const": "b"}}}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"a": "a"}')
    assert _accepts(compiled, '{"a": "b"}')
    assert not _accepts(compiled, '{"a": 1}')


def test_oneof_integer_number_overlap_rejected() -> None:
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"oneOf": [{"type": "integer"}, {"type": "number"}]}',
            reject_unsupported=True,
        )


def test_oneof_untyped_branch_rejected() -> None:
    # Type-specific keywords do not imply that an untyped branch has that type.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"oneOf": [{"type": "number"}, {"minimum": 0}]}',
            reject_unsupported=True,
        )


def test_oneof_overlapping_compiles_when_permissive() -> None:
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"type": "integer"}, {"type": "number"}]}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_oneof_sibling_of_anyof_not_dropped() -> None:
    # Dispatch must not drop a sibling union.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"anyOf": [{"type": "string"}], '
            '"oneOf": [{"type": "integer"}, {"type": "number"}]}',
            reject_unsupported=True,
        )


def test_allof_two_nontrivial_branches_merged() -> None:
    # Two enforceable members conjoin into one schema: the length bound each
    # contributes is enforced, not just the first.
    compiled = _compiler().compile_json_schema(
        '{"allOf": [{"type": "string", "minLength": 2}, '
        '{"type": "string", "maxLength": 3}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"ab"')
    assert not _accepts(compiled, '"a"')
    assert not _accepts(compiled, '"abab"')


def test_allof_with_only_empty_branch_compiles() -> None:
    # A single empty-object member is a no-op (0 enforceable branches):
    # no constraint to apply, so it compiles.
    compiled = _compiler().compile_json_schema('{"allOf": [{}]}')
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_allof_with_true_member_compiles() -> None:
    # A boolean-true member accepts everything -- a no-op, dropped like {}; the
    # 0-enforceable allOf compiles.
    compiled = _compiler().compile_json_schema('{"allOf": [true]}')
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_allof_with_false_member_rejected() -> None:
    # A boolean-false member matches nothing (not a no-op), so the allOf is
    # unsatisfiable and is rejected.
    with pytest.raises(Exception):
        _compiler().compile_json_schema('{"allOf": [false]}')


def test_allof_with_restricting_additional_properties_rejected() -> None:
    # additionalProperties governs the keys its own object leaves unnamed. The
    # base here names none, so it accepts only {}; folding the member's "a" in
    # would make the merged schema accept an object the base forbids.
    for value in ("false", '{"type": "string"}'):
        with pytest.raises(Exception):
            _compiler().compile_json_schema(
                '{"type": "object", "allOf": [{"properties": {"a": {"type": '
                '"string"}}}], "additionalProperties": ' + value + "}",
                reject_unsupported=True,
            )


def test_oneof_type_array_arms_disjoint_compiles() -> None:
    # A "type" array is a union of types; ["string","null"] and integer share none.
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"type": ["string", "null"]}, {"type": "integer"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(compiled, "null")
    assert _accepts(compiled, "7")


def test_oneof_type_array_arms_overlapping_rejected() -> None:
    # integer is a subset of number, so the arms overlap inside the array.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"oneOf": [{"type": ["integer", "null"]}, {"type": "number"}]}',
            reject_unsupported=True,
        )


def test_oneof_const_versus_type_disjoint_compiles() -> None:
    # A finite arm against a typed arm: "a" is not an integer.
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"const": "a"}, {"type": "integer"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(compiled, "7")
    assert not _accepts(compiled, '"b"')


def test_oneof_fractional_const_versus_integer_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"const": 2.5}, {"type": "integer"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, "2.5")
    assert _accepts(compiled, "2")


def test_oneof_const_versus_type_overlapping_rejected() -> None:
    # 1 is a number, so the finite arm falls inside the typed arm.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"oneOf": [{"const": 1}, {"type": "number"}]}',
            reject_unsupported=True,
        )


def test_oneof_huge_enums_rejected_not_hung() -> None:
    # Branch counts and enum sizes come from the request, and the proof is
    # quadratic in both. Over budget must refuse, not run for minutes.
    a = json.dumps(list(range(10000)))
    b = json.dumps(list(range(10000, 20000)))
    started = time.monotonic()
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"oneOf": [{"enum": ' + a + '}, {"enum": ' + b + "}]}",
            reject_unsupported=True,
        )
    assert time.monotonic() - started < 10


def test_oneof_many_discriminator_candidates_rejected_not_hung() -> None:
    keys = [f"k{i}" for i in range(50000)]
    option = {
        "type": "object",
        "required": keys,
        "properties": {key: {"const": 0} for key in keys},
    }
    schema = json.dumps({"oneOf": [option, option]})
    started = time.monotonic()
    with pytest.raises(Exception):
        _compiler().compile_json_schema(schema, reject_unsupported=True)
    assert time.monotonic() - started < 10


def test_allof_conflicting_id_rejected() -> None:
    # $id rebases $ref resolution, so two members declaring different ones
    # disagree about something an instance can observe.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"allOf": [{"$id": "https://example.com/a", "type": "string"}, '
            '{"$id": "https://example.com/b", "minLength": 1}]}',
            reject_unsupported=True,
        )


def test_allof_matching_id_refused() -> None:
    # Agreeing on the identifier does not make the document one resource: both
    # members still declare one below the root, and a fragment written in
    # either would name a place inside it.
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"allOf": [{"$id": "https://example.com/a", "type": "string"}, '
            '{"$id": "https://example.com/a", "minLength": 2}]}',
            reject_unsupported=True,
        )


def test_oneof_mixed_integer_and_float_consts_compiles() -> None:
    # 1 and 2.5 are disjoint. The proof must compare numbers, not picojson's
    # storage tag, which differs between an integer and a float literal.
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"const": 1}, {"const": 2.5}]}', reject_unsupported=True
    )
    assert _accepts(compiled, "1")
    assert _accepts(compiled, "2.5")
    assert not _accepts(compiled, "3")


def test_oneof_equal_integer_and_float_consts_rejected() -> None:
    # 1 and 1.0 are the same value, so the branches are not disjoint.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"oneOf": [{"const": 1}, {"const": 1.0}]}', reject_unsupported=True
        )


def test_allof_with_open_additional_properties_merged() -> None:
    # additionalProperties: true restricts nothing, so nothing is lost by
    # folding the member's declared property in beside it.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "allOf": [{"properties": {"a": {"type": '
        '"string"}}}], "additionalProperties": true}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"a": "b"}')
    assert not _accepts(compiled, '{"a": 1}')


def test_allof_with_sibling_object_keyword_merged() -> None:
    # A required sibling of an allOf binds the member's declared property: the
    # merge is what makes the two visible to one ObjectSpec.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "allOf": [{"properties": {"a": {"type": '
        '"string"}}}], "required": ["a"]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"a": "b"}')
    assert not _accepts(compiled, "{}")


def test_if_then_rejected() -> None:
    # An ACTIVE if/then conditional cannot be enforced; reject it.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "string", "if": {"minLength": 5}, '
            '"then": {"pattern": "^hello"}}',
            reject_unsupported=True,
        )


def test_bare_if_no_then_compiles() -> None:
    # A lone if with no then/else is a draft-7 no-op and compiles.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "if": {"minLength": 5}}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_anyof_compiles() -> None:
    # anyOf IS representable as a grammar union (GenerateAnyOf emits
    # alternatives), so unlike oneOf a multi-branch anyOf compiles. It is the
    # base-vs-branch key *conflict* that is rejected, not anyOf itself.
    compiled = _compiler().compile_json_schema(
        '{"anyOf": [{"type": "string"}, {"type": "number"}]}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_unsupported_keyword_in_tag_compiles_by_default() -> None:
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"n": {"type": "number", "multipleOf": 5}},
            }
        )
    )
    compiled = _compiler().compile_structural_tag(tag)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_anyof_base_branch_conflict_rejected() -> None:
    # A key set by BOTH the base schema and a branch is an unmergeable conflict;
    # reject it rather than silently letting the branch value win.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"minLength": 2, "anyOf": [{"minLength": 5}, {"type": "string"}]}',
            reject_unsupported=True,
        )


def test_anyof_boolean_branch_with_base_rejected() -> None:
    # A boolean anyOf branch is representable (true collapses the anyOf to the
    # base schema, false drops out), but the base-merge only folds base keys
    # into an object branch, so a boolean branch alongside a base schema is
    # rejected conservatively.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"minLength": 2, "anyOf": [{"type": "string"}, true]}',
            reject_unsupported=True,
        )


def test_anyof_annotation_key_on_branch_and_base_compiles() -> None:
    # Annotation-only keywords (description, title, ...) constrain nothing, so
    # a branch and the base schema both carrying one is not a merge conflict
    # even in strict mode.
    compiled = _compiler().compile_json_schema(
        '{"description": "base", "anyOf": ['
        '{"type": "string", "description": "branch"}, {"type": "number"}]}',
        reject_unsupported=True,
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_anyof_base_merge_compiles() -> None:
    # The happy path of the base-merge: base keywords (type, required) are folded
    # into each object branch without conflict, so base AND (B1 OR B2) compiles.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "required": ["x"], "anyOf": ['
        '{"properties": {"x": {"type": "string"}}}, '
        '{"properties": {"x": {"type": "number"}}}]}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_anyof_base_merge_accepts_union() -> None:
    # Semantic check (not just "compiles"): the merged grammar must accept the
    # UNION of both branches -- both "a" and "b" -- and nothing outside it.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "anyOf": [{"const": "a"}, {"const": "b"}]}'
    )
    assert _accepts(compiled, '"a"')  # branch 1
    assert _accepts(
        compiled, '"b"'
    )  # branch 2 -> proves the union, not one arm
    assert not _accepts(compiled, '"c"')  # outside the union is rejected


def test_anyof_base_merge_enforces_base() -> None:
    # The folded base constraint must actually bind: with an empty branch the
    # union is "any value", so only base type:string keeps a number out. If the
    # base were dropped, '1' would be accepted.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "anyOf": [{"const": "a"}, {}]}'
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(
        compiled, '"b"'
    )  # {} branch (as string) accepts other strings
    assert not _accepts(compiled, "1")  # base type:string bars a bare number


def test_unsupported_format_idn_email_rejected() -> None:
    # idn-email is not in the supported-format set (its regex does not compile
    # into a faithful grammar), so it is rejected.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "string", "format": "idn-email"}', reject_unsupported=True
        )


def test_unsupported_format_iri_rejected() -> None:
    # iri is likewise unsupported and rejected.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "string", "format": "iri"}', reject_unsupported=True
        )


def test_json_pointer_format_compiles() -> None:
    # RFC 6901 json-pointer is a regular language; base ships its regex, so it
    # must be accepted (not rejected by the format allow-list).
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "format": "json-pointer"}', reject_unsupported=True
    )
    assert compiled is not None


def test_relative_json_pointer_format_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "format": "relative-json-pointer"}',
        reject_unsupported=True,
    )
    assert compiled is not None


def test_supported_format_date_compiles() -> None:
    # date is a supported format and compiles.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "format": "date"}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_supported_format_ipv4_compiles() -> None:
    # ipv4 is a supported format and compiles.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "format": "ipv4"}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_supported_formats_compile() -> None:
    for fmt in (
        "email",
        "time",
        "date-time",
        "duration",
        "ipv6",
        "hostname",
        "uuid",
        "uri",
        "uri-reference",
        "uri-template",
    ):
        compiled = _compiler().compile_json_schema(
            f'{{"type": "string", "format": "{fmt}"}}'
        )
        assert isinstance(compiled, xgr.CompiledGrammar)


def test_format_ipv4_enforced() -> None:
    # A supported format compiles into a grammar that binds: a well-formed value
    # is accepted and a malformed one rejected (proves the format regex path,
    # not just that it compiled).
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "format": "ipv4"}'
    )
    assert _accepts(compiled, '"1.1.1.1"')
    assert not _accepts(compiled, '"a"')


def test_format_ipv4_octet_range_enforced() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "format": "ipv4"}'
    )
    assert _accepts(compiled, '"1.1.1.1"')
    assert not _accepts(compiled, '"256.1.1.1"')


def test_format_date_accepts_valid() -> None:
    # The month-aware date regex accepts real calendar dates: the 31st of a
    # 31-day month, the 30th of a 30-day month, and the leap day.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "format": "date"}'
    )
    assert _accepts(compiled, '"2023-01-31"')
    assert _accepts(compiled, '"2023-04-30"')
    assert _accepts(compiled, '"2024-02-29"')


def test_format_date_rejects_invalid() -> None:
    # Month-specific day caps reject impossible dates the old any-month 01-31
    # regex let through (Feb 31, Apr 31), plus an out-of-range month.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "format": "date"}'
    )
    assert not _accepts(compiled, '"2023-02-31"')
    assert not _accepts(compiled, '"2023-04-31"')
    assert not _accepts(compiled, '"2023-13-01"')


def test_format_date_time_accepts_valid() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "format": "date-time"}'
    )
    assert _accepts(compiled, '"2023-01-31T23:59:60Z"')
    assert _accepts(compiled, '"2023-04-30T00:00:00+02:00"')
    assert _accepts(compiled, '"2024-02-29T12:00:00Z"')


def test_format_date_time_rejects_invalid() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "format": "date-time"}'
    )
    assert not _accepts(compiled, '"2023-02-31T12:00:00Z"')
    assert not _accepts(compiled, '"2023-04-31T12:00:00Z"')
    assert not _accepts(compiled, '"2023-13-01T12:00:00Z"')


def test_ecma262_backslash_t_compiles() -> None:
    # The JSON ``"\\t"`` decodes to the two-byte regex ``\t``; the regex
    # converter decodes it to a tab, so it compiles (no pre-filter rejection).
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "\\\\t"}', reject_unsupported=True
    )
    assert compiled is not None


def test_ecma262_backslash_n_compiles() -> None:
    # The JSON ``"\\n"`` decodes to the two-byte regex ``\n``; likewise compiles.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "\\\\n"}', reject_unsupported=True
    )
    assert compiled is not None


def test_ecma262_backslash_x_compiles() -> None:
    # The JSON ``"\\x41"`` decodes to the regex ``\x41`` (the letter A); the
    # converter decodes the \xXX numeric escape, so it compiles.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "\\\\x41"}', reject_unsupported=True
    )
    assert compiled is not None


def test_valid_pattern_compiles_and_enforced() -> None:
    # A pattern with no unsupported escapes takes the accept path (regex ->
    # EBNF) and the grammar actually enforces it: it accepts strings matching
    # [ab]+ and rejects others.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "[ab]+"}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)
    assert _accepts(compiled, '"ab"')
    assert _accepts(compiled, '"aab"')
    assert not _accepts(compiled, '"c"')


def test_pattern_with_control_char_escaped() -> None:
    # A control char with no JSON shorthand (U+0001) reaches the pattern as a
    # raw byte via the JSON escape in the schema below -- a raw byte, not a
    # backslash-escape, so ParseString's escape guard does not reject it.
    # FormatCharLiteral's control path (<= 0x1F) then emits it as the \uXXXX
    # escape: the grammar accepts the escaped form and rejects the raw byte.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "a\\u0001b"}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)
    assert _accepts(compiled, '"a\\u0001b"')
    assert not _accepts(compiled, '"a\x01b"')


def test_single_pattern_properties_compiles() -> None:
    # A single patternProperties pattern with no named `properties` is the
    # accept path of the object handler and compiles.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "patternProperties": {"a.*": {"type": "string"}}}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_multiple_pattern_properties_rejected() -> None:
    # More than one patternProperties pattern needs per-key schema intersection
    # (a key may match several patterns); xgrammar applies only one, so reject.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "object", "patternProperties": '
            '{"a.*": {"type": "string"}, "b.*": {"type": "string"}}}',
            reject_unsupported=True,
        )


def test_properties_with_pattern_properties_rejected() -> None:
    # `properties` coexisting with `patternProperties`/`additionalProperties`
    # needs per-key schema intersection that xgrammar's object grammar cannot
    # express: a declared key's value schema is silently dropped in favor of the
    # pattern/additional alternative (a `bar: array` key accepts the
    # `additionalProperties` integer; `foo`'s `maxItems` is lost). Reject the
    # combination under reject_unsupported rather than fail open. Rejected
    # schemas surface as HTTP 400 at the route boundary.
    schema = (
        '{"properties":{"foo":{"type":"array","maxItems":3},'
        '"bar":{"type":"array"}},"patternProperties":{"f.o":{"minItems":2}},'
        '"additionalProperties":{"type":"integer"}}'
    )
    with pytest.raises(Exception):
        _compiler().compile_json_schema(schema, reject_unsupported=True)


def _qwen_tool(params: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": "f", "parameters": params}}
    ]


def _qwen_compiler() -> xgr.GrammarCompiler:
    # A full printable-ASCII vocab so the qwen tag's literal begin/end markers
    # (``<tool_call>\n<function=...``) can be byte-tokenized.
    vocab = [chr(c) for c in range(32, 127)] + ["\n", "<eos>"]
    info = xgr.TokenizerInfo(
        vocab,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[len(vocab) - 1],
    )
    return xgr.GrammarCompiler(info)


def _compile_qwen_tool(params: dict[str, Any]) -> xgr.CompiledGrammar:
    # The qwen_xml tool path routes through QwenXMLToolCallingToEBNF, which
    # reads require_object_root from the JSONSchemaFormat. Tool-call arguments
    # are always a JSON object, so opt into require_object_root on the tag's
    # JSON-schema contents (the flag defaults false). A non-object root is then
    # rejected during schema parse (before vocab tokenization), so the reject
    # tests do not depend on the vocab; the compile test needs the markers to
    # tokenize.
    tag = xgr.get_builtin_structural_tag(
        "qwen_3_5",
        tools=_qwen_tool(params),
        tool_choice="required",
        reasoning=False,
    )
    _set_require_object_root(tag.format)
    return _qwen_compiler().compile_structural_tag(tag)


def test_tool_call_array_root_rejected() -> None:
    # An array-root tool-argument schema violates require_object_root.
    with pytest.raises(Exception):
        _compile_qwen_tool({"type": "array", "items": {"type": "string"}})


def test_tool_call_string_root_rejected() -> None:
    # A string-root tool-argument schema likewise violates require_object_root.
    with pytest.raises(Exception):
        _compile_qwen_tool({"type": "string"})


def test_tool_call_object_root_compiles() -> None:
    # An object-root tool-argument schema is well-formed and compiles.
    compiled = _compile_qwen_tool(
        {"type": "object", "properties": {"x": {"type": "string"}}}
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_response_format_array_root_compiles() -> None:
    # The response_format path does not require an object root, so an array
    # root compiles (it must NOT be rejected the way a tool-call root is).
    compiled = _compiler().compile_json_schema(
        '{"type": "array", "items": {"type": "string"}}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_structural_tag_bridge() -> None:
    tag = xgr.StructuralTag.from_legacy_structural_tag(
        [xgr.StructuralTagItem(begin="a", schema={"type": "object"}, end="b")],
        triggers=["a"],
    )
    roundtripped = xgr.StructuralTag.model_validate_json(tag.model_dump_json())
    assert isinstance(roundtripped, xgr.StructuralTag)

    compiled = _compiler().compile_structural_tag(tag)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_or_format_compiles() -> None:
    """``OrFormat`` must be usable end to end.

    ``OrFormat`` has a forward-referenced ``List["Format"]`` field, so it must
    appear in ``structural_tag``'s ``model_rebuild()`` block; otherwise
    constructing it raises ``PydanticUserError``. Guards the generic shim layer
    (construct, serialize, compile) independently of any model wiring.
    """
    tag = xgr.StructuralTag(
        format=OrFormat(
            elements=[
                JSONSchemaFormat(json_schema={"type": "object"}),
                ConstStringFormat(value="null"),
            ]
        )
    )
    roundtripped = xgr.StructuralTag.model_validate_json(tag.model_dump_json())
    assert roundtripped.format.type == "or"

    compiled = _compiler().compile_structural_tag(tag)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_unique_items_false_compiles() -> None:
    # uniqueItems: false is a no-op (duplicates allowed = the default), so it
    # must compile even under reject_unsupported; only uniqueItems: true is
    # rejected -- xgrammar's grammar model has no uniqueness tracking, and
    # all-distinct isn't context-free for unbounded-domain arrays.
    compiled = _compiler().compile_json_schema(
        '{"type": "array", "items": {"type": "integer"}, "uniqueItems": false}',
        reject_unsupported=True,
    )
    assert compiled is not None


def test_unique_items_true_rejected() -> None:
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "array", "items": {"type": "integer"}, "uniqueItems": true}',
            reject_unsupported=True,
        )


def test_object_without_properties_compiles() -> None:
    # A bounds-only object {"minProperties": 1} (no declared properties) must
    # compile to an OPEN object (>=1 member of any type), mirroring
    # test_array_without_items_compiles -- not collapse to a closed empty object
    # and be rejected as unsatisfiable under strict mode.
    compiled = _compiler().compile_json_schema('{"minProperties": 1}')
    assert compiled is not None


def test_anyof_with_false_branch_compiles() -> None:
    # A false branch accepts nothing; it should be dropped from the union, not
    # abort the whole schema. The remaining branch is satisfiable.
    compiled = _compiler().compile_json_schema(
        '{"anyOf": [{"type": "string"}, false]}'
    )
    assert compiled is not None


def test_anyof_all_false_rejected() -> None:
    # If every branch is false, nothing is satisfiable -> reject.
    with pytest.raises(Exception):
        _compiler().compile_json_schema('{"anyOf": [false, false]}')


def test_object_with_false_property_compiles() -> None:
    # A property mapped to false forbids that key; skip it (never emit) rather
    # than aborting the object. Other declared properties still compile.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "properties": {"foo": {"type": "string"}, "bar": false}}'
    )
    assert compiled is not None


def test_required_false_property_rejected() -> None:
    # A required property forbidden by a false schema is genuinely unsatisfiable.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "object", "properties": {"bar": false}, "required": ["bar"]}'
        )


def test_pattern_literal_quote_escaped() -> None:
    # A literal `"` in a pattern must be emitted as the escape `\"`; a raw
    # quote (which would close the JSON string) must be rejected.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "a\\"b"}'
    )
    assert _accepts(compiled, '"a\\"b"')
    assert not _accepts(compiled, '"a"b"')


def test_pattern_literal_backslash_escaped() -> None:
    # A literal backslash must be emitted as `\\`; a raw backslash rejected.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "a\\\\\\\\b"}'
    )
    assert _accepts(compiled, '"a\\\\b"')
    assert not _accepts(compiled, '"a\x5cb"')


def test_char_class_control_member_hoisted() -> None:
    # A control member inside a positive class is hoisted out as its \uXXXX
    # escape; the raw byte is rejected.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "[a\\u0001]"}'
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(compiled, '"\\u0001"')
    assert not _accepts(compiled, '"\x01"')


def test_char_class_quote_member_hoisted() -> None:
    # A `"` inside a positive class is hoisted as `\"`; raw quote rejected.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "[a\\"]"}'
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(compiled, '"\\""')
    assert not _accepts(compiled, '"""')


def test_char_class_range_straddling_unsafe() -> None:
    # A range that spans an unsafe codepoint (`"` at 0x22) keeps the safe
    # members and hoists the unsafe one; raw quote still rejected.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "[ -$]"}'
    )
    for c in ('" "', '"!"', '"#"', '"$"', '"\\""'):
        assert _accepts(compiled, c)
    assert not _accepts(compiled, '"""')


def test_negated_char_class_excludes_forbidden() -> None:
    # A negated class has the forbidden set appended, so it cannot match the
    # delimiter/escape/control bytes even though it "negates" only `a`.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "[^a]"}'
    )
    assert _accepts(compiled, '"b"')
    assert not _accepts(compiled, '"a"')
    assert not _accepts(compiled, '"\x01"')
    assert not _accepts(compiled, '"""')


def test_dot_excludes_forbidden() -> None:
    # `.` is compiled to a negated class that excludes the forbidden set, so
    # it cannot match a raw control byte or the string delimiter.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "a.b"}'
    )
    assert _accepts(compiled, '"axb"')
    assert not _accepts(compiled, '"a\x01b"')
    assert not _accepts(compiled, '"a"b"')


def test_pattern_allowed_escape_compiles() -> None:
    # `\d` is not in the ECMA-escape reject set, so it compiles and enforces.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "\\\\d"}'
    )
    assert _accepts(compiled, '"1"')
    assert not _accepts(compiled, '"a"')


def test_pattern_remaining_ecma_escapes_compile() -> None:
    # The rest of the ECMA-262 escapes the regex converter decodes
    # (\t \n \x already covered elsewhere); all compile. \u is tested in
    # its complete \uXXXX form (a bare \u is malformed, not merely
    # unenforceable).
    for pat in (
        "\\\\r",
        "\\\\f",
        "\\\\v",
        "\\\\s",
        "\\\\S",
        "\\\\0",
        "\\\\u0041",
    ):
        compiled = _compiler().compile_json_schema(
            f'{{"type": "string", "pattern": "{pat}"}}',
            reject_unsupported=True,
        )
        assert compiled is not None


def test_negated_class_excludes_multibyte_whitespace() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "\\\\S"}'
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, '" "')
    assert not _accepts(compiled, '"\u00a0"')


def test_negated_ascii_range() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "[^a-z]"}'
    )
    assert _accepts(compiled, '"`"')  # right before 'a'
    assert not _accepts(compiled, '"a"')  # 'a', lower boundary
    assert not _accepts(compiled, '"m"')  # 'm', interior
    assert not _accepts(compiled, '"z"')  # 'z', upper boundary
    assert _accepts(compiled, '"{"')  # right after 'z'
    assert _accepts(compiled, '"\u00e9"')  # multi-byte, still accepted


def test_negated_single_multibyte_codepoint() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "[^\\u00e9]"}'
    )
    assert _accepts(compiled, '"\u00e8"')  # right before U+00E9
    assert not _accepts(compiled, '"\u00e9"')  # excluded
    assert _accepts(compiled, '"\u00ea"')  # right after U+00E9
    assert _accepts(compiled, '"a"')  # ASCII 'a' still accepted


def test_negated_multibyte_range() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "[^\\u5000-\\u8fff]"}'
    )
    assert _accepts(compiled, '"\u4fff"')  # right before the range
    assert not _accepts(compiled, '"\u5000"')  # lower boundary
    assert not _accepts(compiled, '"\u7000"')  # interior
    assert not _accepts(compiled, '"\u8fff"')  # upper boundary
    assert _accepts(compiled, '"\u9000"')  # right after the range


def test_negated_mixed_ascii_and_multibyte() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "[^a\\u00e9]"}'
    )
    assert _accepts(compiled, '"`"')  # right before 'a'
    assert not _accepts(compiled, '"a"')  # 'a', excluded ASCII
    assert _accepts(compiled, '"b"')  # right after 'a'
    assert _accepts(compiled, '"\u00e8"')  # right before U+00E9
    assert not _accepts(compiled, '"\u00e9"')  # excluded multi-byte
    assert _accepts(compiled, '"\u00ea"')  # right after U+00E9


def test_char_class_3byte_range_max_lead_block() -> None:
    # A 3-byte range whose max endpoint's trailing continuation bytes are not
    # both 0xBF (e.g. U+4DFF = E4 B7 BF) must accept every codepoint up to its
    # max, including the whole max-lead-byte block (U+4000-U+4DFF, lead byte
    # 0xE4), and reject just past the max.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "[\\u0800-\\u4dff]"}'
    )
    assert not _accepts(compiled, '"\u07ff"')  # just below the min
    assert _accepts(compiled, '"\u0800"')  # min boundary
    assert _accepts(compiled, '"\u3fff"')  # block below the max lead (0xE3)
    assert _accepts(compiled, '"\u4000"')  # start of the max-lead (0xE4) block
    assert _accepts(compiled, '"\u4500"')  # middle of the max-lead block
    assert _accepts(compiled, '"\u4dff"')  # max boundary
    assert not _accepts(compiled, '"\u4e00"')  # just past the max


def test_pattern_unsupportable_escapes_rejected() -> None:
    # Genuinely unenforceable regex constructs are still hard-rejected by the
    # converter (fail-closed): a backreference, a Unicode property class, and a
    # word boundary. These have no context-free grammar encoding.
    for pat in ("(a)\\\\1", "\\\\p{L}", "a\\\\bc"):
        with pytest.raises(Exception):
            _compiler().compile_json_schema(
                f'{{"type": "string", "pattern": "{pat}"}}',
                reject_unsupported=True,
            )


def test_array_contains_rejected() -> None:
    for schema in (
        '{"type": "array", "contains": {"type": "string"}}',
        '{"type": "array", "contains": {"type": "string"}, "minContains": 1}',
        '{"type": "array", "contains": {"type": "string"}, "maxContains": 2}',
    ):
        with pytest.raises(Exception):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_array_bounds_enforced() -> None:
    c_min = _compiler().compile_json_schema('{"type": "array", "minItems": 2}')
    assert _accepts(c_min, "[1,1]")
    assert not _accepts(c_min, "[1]")
    assert not _accepts(c_min, "[]")
    c_max = _compiler().compile_json_schema('{"type": "array", "maxItems": 2}')
    assert _accepts(c_max, "[1,1]")
    assert not _accepts(c_max, "[1,1,1]")


def test_numeric_negative_and_maximum_bounds_rejected() -> None:
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "number", "minimum": -1e30}', reject_unsupported=True
        )
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "number", "maximum": 1e30}', reject_unsupported=True
        )


def test_top_level_multiple_of_rejected() -> None:
    # multipleOf with no explicit type is caught by the top-level check.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"multipleOf": 5}', reject_unsupported=True
        )


def test_additional_properties_value_kinds_compile() -> None:
    for value in ("false", "true", '{"type": "string"}'):
        compiled = _compiler().compile_json_schema(
            '{"type": "object", "additionalProperties": ' + value + "}"
        )
        assert isinstance(compiled, xgr.CompiledGrammar)


def test_required_undeclared_key_additional_properties_false_rejected() -> None:
    # A required key absent from `properties` is an additional property;
    # additionalProperties:false forbids it, so requiring it is unsatisfiable.
    # Reject rather than fall back to an unconstrained (any-value) property that
    # would accept objects the schema forbids.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "object", "required": ["k"], "additionalProperties": false}',
            reject_unsupported=True,
        )


def test_required_undeclared_key_open_object_compiles() -> None:
    # With additionalProperties absent/true (or a schema), a required key not in
    # `properties` is genuinely constrained by additionalProperties (any when
    # absent/true), so it compiles rather than being rejected.
    for schema in (
        '{"type": "object", "required": ["k"]}',
        '{"type": "object", "required": ["k"], "additionalProperties": true}',
        '{"type": "object", "required": ["k"], '
        '"additionalProperties": {"type": "string"}}',
    ):
        compiled = _compiler().compile_json_schema(schema)
        assert isinstance(compiled, xgr.CompiledGrammar)


def test_allof_other_sibling_keywords_rejected() -> None:
    # Key-language siblings survive the merge only to be refused downstream:
    # patternProperties and propertyNames would each need their key language
    # intersected with the member's declared names, and unevaluatedProperties
    # closes the base to keys the member declares.
    base = '{"type":"object","allOf":[{"properties":{"a":{"type":"string"}}}],'
    for sibling in (
        '"patternProperties":{"x.*":{"type":"string"}}',
        '"propertyNames":{"pattern":"a"}',
        '"unevaluatedProperties":false',
    ):
        with pytest.raises(Exception):
            _compiler().compile_json_schema(
                base + sibling + "}", reject_unsupported=True
            )


def test_allof_countable_sibling_keywords_merged() -> None:
    # A sibling that only counts or names properties composes with the member's
    # declared set, so it merges.
    base = '{"type":"object","allOf":[{"properties":{"a":{"type":"string"}}}],'
    for sibling in (
        '"properties":{"b":{"type":"string"}}',
        '"minProperties":1',
        '"maxProperties":3',
    ):
        compiled = _compiler().compile_json_schema(
            base + sibling + "}", reject_unsupported=True
        )
        assert isinstance(compiled, xgr.CompiledGrammar)


def test_allof_single_enforceable_branch_enforced() -> None:
    compiled = _compiler().compile_json_schema(
        '{"allOf": [{"type": "string", "minLength": 2}]}'
    )
    assert _accepts(compiled, '"ab"')
    assert not _accepts(compiled, '"a"')


def test_allof_enforceable_plus_noop_member_enforced() -> None:
    # A no-op member (annotation-only) must not tip the enforceable count past
    # one, so the single real branch still compiles and binds.
    compiled = _compiler().compile_json_schema(
        '{"allOf": [{"type": "string", "minLength": 2}, {"title": "x"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"ab"')
    assert not _accepts(compiled, '"a"')


def test_allof_annotation_only_member_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        '{"allOf": [{"title": "ignored", "description": "x"}]}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_allof_lone_if_member_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        '{"allOf": [{"if": {"minLength": 5}}]}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_allof_active_conditional_member_rejected() -> None:
    # An active if/then member is enforceable; paired with another enforceable
    # branch it must reject as multi-option (proves it is not dropped as no-op).
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"allOf": [{"type": "string", "minLength": 1}, '
            '{"if": {"minLength": 5}, "then": {"pattern": "a"}}]}',
            reject_unsupported=True,
        )


def test_allof_three_members_merged() -> None:
    # Three object members, each contributing one required property: the fold
    # has to union all three, not stop once it has merged a pair.
    compiled = _compiler().compile_json_schema(
        '{"allOf": ['
        '{"type": "object", "properties": {"a": {"type": "integer"}}, '
        '"required": ["a"]}, '
        '{"type": "object", "properties": {"b": {"type": "integer"}}, '
        '"required": ["b"]}, '
        '{"type": "object", "properties": {"c": {"type": "integer"}}, '
        '"required": ["c"]}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"a": 1, "b": 1, "c": 1}')
    assert not _accepts(compiled, '{"a": 1, "b": 1}')


def test_allof_property_declared_by_two_members_intersected() -> None:
    # A property both members declare must satisfy both declarations. The
    # declarations become an allOf on that property, merged one level down.
    compiled = _compiler().compile_json_schema(
        '{"allOf": [{"type": "object", "properties": {"a": {"type": '
        '"string"}}}, {"type": "object", "properties": {"a": {"minLength": '
        '2}}}], "required": ["a"]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"a": "ab"}')
    assert not _accepts(compiled, '{"a": "b"}')
    assert not _accepts(compiled, '{"a": 11}')


def test_allof_conflicting_type_rejected() -> None:
    # No single "type" expresses the intersection, and this IR has no type
    # intersection to fall back on.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"allOf": [{"type": "string"}, {"type": "integer"}]}',
            reject_unsupported=True,
        )


def test_allof_two_additional_properties_schemas_intersected() -> None:
    # An unnamed key must satisfy both members' value schemas. The two become
    # an allOf on that value, merged one level down, so both still bind. The
    # boundary check is what makes this well defined: it has already refused
    # any member whose declared names differ, so both govern the same keys.
    compiled = _compiler().compile_json_schema(
        '{"allOf": [{"type": "object", "additionalProperties": {"type": '
        '"string"}}, {"type": "object", "additionalProperties": '
        '{"minLength": 2}}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"k": "ab"}')
    assert not _accepts(compiled, '{"k": "a"}')
    assert not _accepts(compiled, '{"k": 1}')


def test_allof_additional_properties_false_closes_the_object() -> None:
    # false accepts no value, so however the other member describes the unnamed
    # keys the conjunction admits none: the merged object is closed.
    for schema in (
        '{"allOf": [{"type": "object", "additionalProperties": false}, '
        '{"type": "object", "additionalProperties": {"type": "string"}}]}',
        '{"allOf": [{"type": "object", "additionalProperties": {"type": '
        '"string"}}, {"type": "object", "additionalProperties": false}]}',
    ):
        compiled = _compiler().compile_json_schema(
            schema, reject_unsupported=True
        )
        assert _accepts(compiled, "{}")
        assert not _accepts(compiled, '{"k": "ab"}')


def test_allof_additional_properties_asymmetric_names_still_refused() -> None:
    # The intersection above is only sound while both members govern the same
    # keys, so a member that declares a name the other does not stays refused.
    with pytest.raises(Exception, match="forbids unlisted properties"):
        _compiler().compile_json_schema(
            '{"allOf": [{"type": "object", "properties": {"x": {"type": '
            '"string"}, "y": {"type": "string"}}, "additionalProperties": '
            '{"type": "string"}}, {"type": "object", "properties": {"x": '
            '{"type": "string"}}, "additionalProperties": {"minLength": 2}}]}',
            reject_unsupported=True,
        )


def test_allof_property_forbidden_by_one_member_forbidden_in_merge() -> None:
    # false accepts nothing, so a property one member forbids stays forbidden
    # however the other declares it -- and required makes that unsatisfiable.
    compiled = _compiler().compile_json_schema(
        '{"allOf": [{"type": "object", "properties": {"a": {"type": '
        '"string"}, "b": {"type": "string"}}}, {"type": "object", '
        '"properties": {"a": false}}], "required": ["b"]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"b": "1"}')
    assert not _accepts(compiled, '{"a": "1", "b": "1"}')

    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"allOf": [{"type": "object", "properties": {"a": {"type": '
            '"string"}}}, {"type": "object", "properties": {"a": false}}], '
            '"required": ["a"]}',
            reject_unsupported=True,
        )


def test_allof_under_restricting_unevaluated_properties_rejected() -> None:
    # unevaluatedProperties is defined over what every applicator evaluated; a
    # flat object no longer carries that, so the conjunction is not merged.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "object", "unevaluatedProperties": false, '
            '"allOf": [{"required": ["b"]}]}',
            reject_unsupported=True,
        )


def test_allof_pattern_and_length_from_two_members_rejected() -> None:
    with pytest.raises(Exception, match="each admit lengths the other forbids"):
        _compiler().compile_json_schema(
            '{"allOf": [{"type": "string", "pattern": "a+"}, '
            '{"type": "string", "minLength": 2}]}',
            reject_unsupported=True,
        )


def test_pattern_implying_the_length_bounds_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "^a{3,5}$", "maxLength": 10}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"aaa"')
    assert not _accepts(compiled, '"aa"')
    assert not _accepts(compiled, '"aaaaaa"')


def test_pattern_disjoint_from_the_length_bounds_rejected() -> None:
    with pytest.raises(Exception, match="no length in common"):
        _compiler().compile_json_schema(
            '{"type": "string", "pattern": "^a{5,}$", "maxLength": 3}',
            reject_unsupported=True,
        )


def test_pattern_overlapping_the_length_bounds_rejected() -> None:
    with pytest.raises(Exception, match="each admit lengths the other forbids"):
        _compiler().compile_json_schema(
            '{"type": "string", "pattern": "a+", "minLength": 2}',
            reject_unsupported=True,
        )


def test_unbounded_repeat_with_one_required_match_keeps_min_length() -> None:
    with pytest.raises(Exception, match="each admit lengths the other forbids"):
        _compiler().compile_json_schema(
            '{"type": "string", "pattern": "^a{1,}$", "minLength": 2}',
            reject_unsupported=True,
        )


def test_string_length_bounds_must_fit_the_enforceable_range() -> None:
    for schema in (
        '{"type": "string", "minLength": -1}',
        '{"type": "string", "maxLength": -1}',
        '{"type": "string", "pattern": "^a$", "minLength": 2147483648}',
    ):
        with pytest.raises(
            Exception, match="non-negative integer in the enforceable range"
        ):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_allof_pattern_implying_the_length_bounds_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        '{"allOf": [{"type": "string", "pattern": "^a{3,5}$"}, '
        '{"maxLength": 10}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"aaa"')
    assert not _accepts(compiled, '"aa"')


def test_allof_pattern_disjoint_from_the_length_bounds_rejected() -> None:
    with pytest.raises(Exception, match="no length in common"):
        _compiler().compile_json_schema(
            '{"allOf": [{"type": "string", "pattern": "^a{5,}$"}, '
            '{"maxLength": 3}]}',
            reject_unsupported=True,
        )


def test_union_base_length_bound_not_dropped_by_a_branch_pattern() -> None:
    with pytest.raises(Exception, match="each admit lengths the other forbids"):
        _compiler().compile_json_schema(
            '{"oneOf": [{"type": "string", "pattern": "^a+$"}, '
            '{"type": "null"}], "maxLength": 1}',
            reject_unsupported=True,
        )


def test_format_implying_the_length_bounds_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "format": "date", "maxLength": 32}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"2026-01-31"')


def test_format_disjoint_from_the_length_bounds_rejected() -> None:
    with pytest.raises(Exception, match="no length in common"):
        _compiler().compile_json_schema(
            '{"type": "string", "format": "date", "maxLength": 3}',
            reject_unsupported=True,
        )


def test_format_overlapping_the_length_bounds_rejected() -> None:
    with pytest.raises(Exception, match="each admit lengths the other forbids"):
        _compiler().compile_json_schema(
            '{"type": "string", "format": "email", "maxLength": 20}',
            reject_unsupported=True,
        )


def test_format_beside_pattern_rejected() -> None:
    for schema in (
        '{"type": "string", "format": "email", "pattern": "a+"}',
        '{"allOf": [{"type": "string", "format": "email"}, {"pattern": "a+"}]}',
    ):
        with pytest.raises(Exception, match="only one is enforced"):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_unenforceable_format_beside_pattern_rejected() -> None:
    with pytest.raises(Exception, match="unsupported string format"):
        _compiler().compile_json_schema(
            '{"type": "string", "format": "phone", "pattern": "a+"}',
            reject_unsupported=True,
        )


def test_format_beside_pattern_rejected_on_an_older_draft() -> None:
    # The converter enforces format regardless of the declared draft.
    with pytest.raises(Exception, match="only one is enforced"):
        _compiler().compile_json_schema(
            '{"$schema": "http://json-schema.org/draft-04/schema#",'
            ' "type": "string", "format": "email", "pattern": "a+"}',
            reject_unsupported=True,
        )


def test_format_beside_pattern_permissive_keeps_the_pattern() -> None:
    for string_format in ("email", "phone"):
        compiled = _compiler().compile_json_schema(
            f'{{"type": "string", "format": "{string_format}",'
            ' "pattern": "a+"}'
        )
        assert _accepts(compiled, '"aa"'), string_format
        assert not _accepts(compiled, '"b"'), string_format


def test_non_ascii_pattern_with_length_bounds_rejected() -> None:
    # Length is counted in characters and the automaton counts bytes, so a
    # pattern that can leave ASCII is refused rather than measured in the
    # wrong unit.
    with pytest.raises(Exception, match="cannot be measured"):
        _compiler().compile_json_schema(
            '{"type": "string", "pattern": "^\\u00e9+$", "maxLength": 3}',
            reject_unsupported=True,
        )


def test_multichar_escape_length_analysis_fails_closed() -> None:
    schema = json.dumps(
        {"type": "string", "pattern": r"^\u0041$", "minLength": 2}
    )
    with pytest.raises(Exception, match="cannot be measured"):
        _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_misplaced_anchor_length_analysis_fails_closed() -> None:
    schema = json.dumps({"type": "string", "pattern": "a^b", "minLength": 3})
    with pytest.raises(Exception, match="cannot be measured"):
        _compiler().compile_json_schema(schema, reject_unsupported=True)


@pytest.mark.parametrize("pattern", [r"^\^\$$", r"^[$^]{2}$"])
def test_valid_anchor_placements_still_measured(pattern: str) -> None:
    schema = json.dumps({"type": "string", "pattern": pattern, "minLength": 2})
    compiled = _compiler().compile_json_schema(schema, reject_unsupported=True)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_huge_quantifier_with_length_bounds_refused_quickly() -> None:
    for schema in (
        '{"type": "string", "pattern": "^a{100000000}$",'
        ' "maxLength": 100000000}',
        '{"allOf": [{"type": "string", "pattern": "^a{100000000}$"},'
        ' {"maxLength": 100000000}]}',
        '{"oneOf": [{"type": "string", "pattern": "^a{100000000}$"},'
        ' {"type": "null"}], "maxLength": 100000000}',
    ):
        started = time.process_time()
        with pytest.raises(Exception, match="cannot be measured"):
            _compiler().compile_json_schema(schema, reject_unsupported=True)
        assert time.process_time() - started < 1.0


def test_quantifier_beyond_integer_range_is_refused() -> None:
    for pattern in (
        "^a{999999999999999999999999}$",
        "^a{1,999999999999999999999999}$",
    ):
        with pytest.raises(Exception, match="cannot be measured"):
            _compiler().compile_json_schema(
                f'{{"type": "string", "pattern": "{pattern}", "maxLength": 1}}',
                reject_unsupported=True,
            )


def test_length_measurement_budget_is_shared_across_union_branches() -> None:
    branches = ",".join(
        f'{{"type": "string", "pattern": "^{index:04d}a{{1,500}}$"}}'
        for index in range(32)
    )
    with pytest.raises(Exception, match="cannot be measured"):
        _compiler().compile_json_schema(
            f'{{"anyOf": [{branches}], "maxLength": 10000}}',
            reject_unsupported=True,
        )


def test_moderate_quantifier_with_length_bounds_still_measured() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "^a{1,500}$", "maxLength": 9999}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"aa"')
    assert not _accepts(compiled, '""')


def test_string_tier_alone_still_compiles() -> None:
    for schema in (
        '{"type": "string", "pattern": "^a{3,5}$"}',
        '{"type": "string", "minLength": 2, "maxLength": 4}',
        '{"type": "string", "format": "date"}',
    ):
        compiled = _compiler().compile_json_schema(
            schema, reject_unsupported=True
        )
        assert isinstance(compiled, xgr.CompiledGrammar)


def test_pattern_and_length_bounds_permissive_unchanged() -> None:
    # Without reject_unsupported the converter keeps upstream's bargain: the
    # first tier wins and the rest are dropped.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "pattern": "a+", "minLength": 2}'
    )
    assert _accepts(compiled, '"a"')


def test_allof_lone_ref_member_still_compiles() -> None:
    # The pydantic wrapper shape, bare and with the annotation pydantic adds.
    # $defs and description beside the $ref are bookkeeping, not competing
    # constraints, so the merge leaves the ref alone and its target still binds.
    defs = '{"$defs": {"s": {"type": "string", "minLength": 2}}, '
    for tail in (
        '"allOf": [{"$ref": "#/$defs/s"}]}',
        '"allOf": [{"$ref": "#/$defs/s"}], "description": "wrapped"}',
    ):
        compiled = _compiler().compile_json_schema(
            defs + tail, reject_unsupported=True
        )
        assert _accepts(compiled, '"ab"')
        assert not _accepts(compiled, '"a"')


def test_allof_ref_member_beside_other_constraint_enforced() -> None:
    # Inlining the target removes the $ref before dispatch sees it, so the other
    # member's constraint survives instead of being dropped.
    compiled = _compiler().compile_json_schema(
        '{"$defs": {"s": {"type": "string"}}, '
        '"allOf": [{"$ref": "#/$defs/s"}, {"minLength": 2}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"ab"')
    assert not _accepts(compiled, '"a"')
    assert not _accepts(compiled, "1")


def test_allof_zero_enforceable_with_object_sibling_compiles() -> None:
    # allOf is all no-op (0 enforceable) so the sibling-reject guard is skipped;
    # the object siblings are the effective schema and it compiles.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "allOf": [{}], "required": ["a"], '
        '"properties": {"a": {"type": "string"}}}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_anyof_boolean_branch_without_base_compiles() -> None:
    # With no base siblings the boolean-branch reject is skipped; a true branch
    # is the accept-any arm.
    compiled = _compiler().compile_json_schema(
        '{"anyOf": [{"type": "string"}, true]}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_typeless_inference_enforced() -> None:
    c_num = _compiler().compile_json_schema('{"minimum": 0}')
    assert _accepts(c_num, "1")
    assert not _accepts(c_num, '"a"')
    c_str = _compiler().compile_json_schema('{"minLength": 1}')
    assert _accepts(c_str, '"a"')
    assert not _accepts(c_str, "1")


def test_ambiguous_object_number_keywords_rejected() -> None:
    # required implies object, minimum implies number -> ambiguous.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"required": ["a"], "minimum": 0}', reject_unsupported=True
        )


def test_tool_call_carries_reject_unsupported() -> None:
    # The tool-call path (triggered tags -> JSONSchemaFormat content) carries
    # reject_unsupported through the tag JSON, so an unenforceable arg schema
    # (a oneOf whose branches are not provably disjoint) is rejected there too.
    tag = xgr.StructuralTag(
        format=TriggeredTagsFormat(
            triggers=["<t>"],
            tags=[
                TagFormat(
                    begin="<t>",
                    content=JSONSchemaFormat(
                        json_schema={
                            "oneOf": [{"type": "number"}, {"minimum": 0}]
                        },
                        reject_unsupported=True,
                    ),
                    end="</t>",
                )
            ],
        )
    )
    with pytest.raises(Exception):
        _compiler().compile_structural_tag(tag)


def test_or_format_json_schema_rejects_unenforceable() -> None:
    tag = xgr.StructuralTag(
        format=OrFormat(
            elements=[
                JSONSchemaFormat(
                    json_schema={"type": "number", "multipleOf": 5},
                    reject_unsupported=True,
                ),
                ConstStringFormat(value="null"),
            ]
        )
    )
    with pytest.raises(Exception):
        _compiler().compile_structural_tag(tag)


def test_string_value_forbidden_tokens_without_delimiter_rejected() -> None:
    # Token-level string-content exclusion only applies to token-delimited string
    # values, so string_value_forbidden_tokens without string_value_delimiter_token
    # is a misconfiguration and is rejected rather than silently ignored.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={"type": "object"},
            string_value_forbidden_tokens=["<x>"],
        )
    )
    with pytest.raises(Exception):
        _compiler().compile_structural_tag(tag)


def test_tool_call_requires_object_root() -> None:
    # Every tool-call argument schema must be a JSON object; non-object roots
    # (string / array / number) are rejected.
    obj = _compiler().compile_structural_tag(
        _toolcall({"type": "object", "properties": {"a": {"type": "string"}}})
    )
    assert isinstance(obj, xgr.CompiledGrammar)
    for non_object in (
        {"type": "string"},
        {"type": "array", "items": {"type": "string"}},
        {"type": "number"},
    ):
        with pytest.raises(Exception):
            _compiler().compile_structural_tag(_toolcall(non_object))


def test_tool_call_empty_schema_coerced_to_object() -> None:
    # An empty (any) tool-call schema is coerced to an open object, not rejected.
    compiled = _compiler().compile_structural_tag(_toolcall({}))
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_tool_call_anyof_non_object_branch_rejected() -> None:
    # Object-root verification recurses through anyOf: if any branch could be a
    # non-object, the root is not guaranteed to be an object, so reject.
    with pytest.raises(Exception):
        _compiler().compile_structural_tag(
            _toolcall({"anyOf": [{"type": "object"}, {"type": "string"}]})
        )


def test_tool_call_oneof_object_branches_accepted() -> None:
    # Object-root verification recurses through oneOf the same way it does
    # through anyOf: all-object branches ARE an object root.
    compiled = _compiler().compile_structural_tag(
        _toolcall(
            {
                "oneOf": [
                    {
                        "type": "object",
                        "required": ["a"],
                        "properties": {"a": {"const": "a"}},
                    },
                    {
                        "type": "object",
                        "required": ["a"],
                        "properties": {"a": {"const": "b"}},
                    },
                ]
            }
        )
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_tool_call_oneof_non_object_branch_rejected() -> None:
    # The counterpart: a provably-disjoint oneOf still fails the object root
    # when a branch is a scalar.
    with pytest.raises(Exception):
        _compiler().compile_structural_tag(
            _toolcall({"oneOf": [{"type": "string"}, {"type": "integer"}]})
        )


def test_general_path_allows_non_object_root() -> None:
    # The general compile_json_schema path does NOT require an object root
    # (require_object_root=false), unlike the tool-call path.
    for schema in ('{"type": "string"}', '{"type": "number"}'):
        compiled = _compiler().compile_json_schema(schema)
        assert isinstance(compiled, xgr.CompiledGrammar)


_GEMMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                    "opts": {
                        "type": "object",
                        "properties": {"verbose": {"type": "boolean"}},
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["location"],
            },
        },
    }
]


def test_gemma_4_builtin_tag_registered() -> None:
    # Previously raised "Unknown format type: gemma_4".
    tag = xgr.get_builtin_structural_tag(
        "gemma_4", tools=_GEMMA_TOOLS, tool_choice="auto", reasoning=False
    )
    assert isinstance(tag, xgr.StructuralTag)
    # The arguments are constrained with the Gemma string style, now carried by
    # the JSON style plus the delimiter/bare-key config on the format.
    dumped = tag.model_dump_json().replace(" ", "")
    assert '"style":"json"' in dumped
    assert '"string_value_delimiter_token":"<|\\"|>"' in dumped
    assert '"additional_bare_key_terminal":"[a-zA-Z_][-a-zA-Z0-9_.]*"' in dumped


# The real Gemma tokenizer encodes <|"|>, <|tool_call>, and <tool_call|> as
# SINGLE tokens, and the "gemma" style now references <|"|> by token ID
# (Token()/ExcludeToken()) — there is no byte-literal fallback. So the test
# vocab must contain those delimiters as single entries; spelling them out as
# separate byte tokens is a hard error at compile time (delimiter not found).
_GEMMA_VOCAB = (
    [chr(c) for c in range(32, 127)]
    + ["<|tool_call>", "<tool_call|>", '<|"|>']
    + ["<eos>"]
)
_GEMMA_SPECIALS = ("<|tool_call>", "<tool_call|>", '<|"|>')
_GEMMA_ID = {tok: i for i, tok in enumerate(_GEMMA_VOCAB)}


def _gemma_encode(s: str) -> list[int]:
    """Tokenize s, emitting the multi-char Gemma specials as single tokens."""
    out: list[int] = []
    i = 0
    while i < len(s):
        for sp in _GEMMA_SPECIALS:
            if s.startswith(sp, i):
                out.append(_GEMMA_ID[sp])
                i += len(sp)
                break
        else:
            out.append(_GEMMA_ID[s[i]])
            i += 1
    return out


def _gemma_compiler() -> xgr.GrammarCompiler:
    info = xgr.TokenizerInfo(
        _GEMMA_VOCAB,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[_GEMMA_ID["<eos>"]],
        special_token_ids=[_GEMMA_ID[tok] for tok in _GEMMA_SPECIALS],
    )
    return xgr.GrammarCompiler(info)


def _gemma_matcher(
    tools: list[dict[str, Any]], tool_choice: str = "required"
) -> xgr.GrammarMatcher:
    tag = xgr.get_builtin_structural_tag(
        "gemma_4", tools=tools, tool_choice=tool_choice, reasoning=False
    )
    return xgr.GrammarMatcher(_gemma_compiler().compile_structural_tag(tag))


def _accept_ids(matcher: xgr.GrammarMatcher, ids: list[int]) -> bool:
    return all(matcher.accept_token(tid) for tid in ids)


_SIMPLE_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        },
    }
]


def test_gemma_4_style_compiles() -> None:
    # The rich schema (string, integer, enum, nested object, array) must lower
    # to valid EBNF and compile against a vocab that contains the <|"|>
    # delimiter token.
    tag = xgr.get_builtin_structural_tag(
        "gemma_4", tools=_GEMMA_TOOLS, tool_choice="required", reasoning=False
    )
    compiled = _gemma_compiler().compile_structural_tag(tag)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_gemma_4_accepts_delimiters_as_single_tokens() -> None:
    # Feed <|"|>, <|tool_call>, <tool_call|> as single token IDs (as the real
    # tokenizer emits them); the Token()-referenced grammar must accept them.
    matcher = _gemma_matcher(_SIMPLE_TOOL)
    good = '<|tool_call>call:f{x:<|"|>v<|"|>}<tool_call|>'
    assert _accept_ids(matcher, _gemma_encode(good))
    assert matcher.is_completed()


def _gemma_bit_set(bitmask: np.ndarray, token_id: int) -> bool:
    return bool((int(bitmask[token_id // 32]) >> (token_id % 32)) & 1)


def test_gemma_4_bitmask_allows_special_tokens_only_where_referenced() -> None:
    # The filled bitmask (production guidance path) must stay consistent with
    # accept_token: a special token is allowed exactly where the grammar
    # references it by id, and masked as string content everywhere else.
    matcher = _gemma_matcher(_SIMPLE_TOOL)
    size = xgr.get_bitmask_size(len(_GEMMA_VOCAB))

    # At the start the begin marker is the only structural opener.
    bitmask = np.full((size,), -1, dtype=np.int32)
    assert matcher.fill_next_token_bitmask(bitmask)
    assert _gemma_bit_set(bitmask, _GEMMA_ID["<|tool_call>"])
    assert not _gemma_bit_set(bitmask, _GEMMA_ID["a"])

    # After opening a string, the delimiter (id-referenced) is allowed and plain
    # content is allowed, but the tool-call marker is masked out of the body.
    assert _accept_ids(matcher, _gemma_encode('<|tool_call>call:f{x:<|"|>'))
    bitmask = np.full((size,), -1, dtype=np.int32)
    assert matcher.fill_next_token_bitmask(bitmask)
    assert _gemma_bit_set(bitmask, _GEMMA_ID['<|"|>'])
    assert _gemma_bit_set(bitmask, _GEMMA_ID["a"])
    assert not _gemma_bit_set(bitmask, _GEMMA_ID["<|tool_call>"])


def test_gemma_4_string_content_excludes_marker_token_not_text() -> None:
    # Inside a string body the tool-call marker is excluded as a single TOKEN,
    # but the same characters emitted as ordinary text tokens are allowed -- a
    # split marker is handled by the structural-tag envelope, not ExcludeToken.
    into_string = _gemma_encode('<|tool_call>call:f{x:<|"|>')
    marker = "<|tool_call>"

    rejected = _gemma_matcher(_SIMPLE_TOOL)
    assert _accept_ids(rejected, into_string)
    assert not rejected.accept_token(_GEMMA_ID[marker])

    as_text = _gemma_matcher(_SIMPLE_TOOL)
    assert _accept_ids(as_text, into_string)
    assert _accept_ids(as_text, [_GEMMA_ID[ch] for ch in marker])


def test_gemma_4_max_length_string_excludes_marker_token() -> None:
    # A maxLength-bounded string body must still exclude the tool-call marker
    # as a single token.
    tool = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string", "maxLength": 20}},
                    "required": ["x"],
                },
            },
        }
    ]
    matcher = _gemma_matcher(tool)
    assert _accept_ids(matcher, _gemma_encode('<|tool_call>call:f{x:<|"|>'))
    assert not matcher.accept_token(_GEMMA_ID["<|tool_call>"])


def test_gemma_4_max_length_rejects_overlong_string() -> None:
    # maxLength counts characters, not tokens.
    tool = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string", "maxLength": 5}},
                    "required": ["x"],
                },
            },
        }
    ]
    m = _gemma_matcher(tool)
    assert _accept_ids(m, _gemma_encode('<|tool_call>call:f{x:<|"|>'))
    accepted = all(m.accept_token(t) for t in _gemma_encode("abcdef"))
    assert not accepted  # 6 chars > maxLength 5


def test_gemma_4_max_length_accepts_string_within_bound() -> None:
    # The complement of the reject case: a string at the maxLength bound (built
    # from a single multi-char token) is accepted and the value can close.
    tool = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string", "maxLength": 5}},
                    "required": ["x"],
                },
            },
        }
    ]
    m = _gemma_matcher(tool)
    assert _accept_ids(m, _gemma_encode('<|tool_call>call:f{x:<|"|>'))
    assert _accept_ids(m, _gemma_encode('abcde<|"|>'))  # 5 chars == maxLength


def test_gemma_4_min_length_rejects_too_short_string() -> None:
    # minLength is likewise a CHARACTER count: closing the string delimiter before
    # the minimum number of characters is reached must be rejected, while a string
    # meeting the minimum is accepted.
    tool = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string", "minLength": 3}},
                    "required": ["x"],
                },
            },
        }
    ]
    short = _gemma_matcher(tool)
    assert _accept_ids(short, _gemma_encode('<|tool_call>call:f{x:<|"|>ab'))
    assert not short.accept_token(
        _GEMMA_ID['<|"|>']
    )  # only 2 chars < minLength

    ok = _gemma_matcher(tool)
    assert _accept_ids(ok, _gemma_encode('<|tool_call>call:f{x:<|"|>abc'))
    assert ok.accept_token(_GEMMA_ID['<|"|>'])  # 3 chars == minLength


def test_gemma_4_min_length_bitmask_forbids_early_close_delimiter() -> None:
    # Fill/accept consistency guard for the FILL/BITMASK path (the path the
    # server constrains generation through -- fill_next_token_bitmask, not
    # accept_string). The close delimiter <|"|> is a special token re-enabled by
    # SetSpecialTokenBitmaskBits; that re-enabling scans the current parser
    # state set, so it honors the {min,} repeat gate exactly like accept_token:
    # the delimiter bit stays FORBIDDEN before minLength is met (else the model
    # could sample it early and close the string below minLength). This asserts
    # fill and accept agree at the min boundary so a future change to the
    # special-token re-enabling cannot silently diverge from acceptance.
    tool = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "properties": {
                        "bar": {
                            "type": "string",
                            "minLength": 4,
                            "default": "bad",
                        }
                    }
                },
            },
        }
    ]
    size = xgr.get_bitmask_size(len(_GEMMA_VOCAB))
    delim = _GEMMA_ID['<|"|>']
    base = '<|tool_call>call:f{bar:<|"|>'

    # Sweep every body length 0..5 across the minLength=4 boundary. At each
    # position the filled bitmask's close-delimiter bit must equal what
    # accept_token would allow: FORBIDDEN below 4 chars, ALLOWED at/above 4.
    for ncontent in range(6):
        prefix = base + "a" * ncontent
        ids = _gemma_encode(prefix)

        m = _gemma_matcher(tool)
        assert _accept_ids(m, ids)
        bitmask = np.full((size,), -1, dtype=np.int32)
        assert m.fill_next_token_bitmask(bitmask)
        fill_allows_close = _gemma_bit_set(bitmask, delim)

        accept = _gemma_matcher(tool)
        assert _accept_ids(accept, ids)
        accept_allows_close = accept.accept_token(delim)

        assert fill_allows_close == accept_allows_close, (
            f"fill/accept disagree on close delimiter at {ncontent} chars"
        )
        assert accept_allows_close == (ncontent >= 4), (
            f"close delimiter gate wrong at {ncontent} chars"
        )
        # Content is always still allowed below the (unbounded) max.
        assert _gemma_bit_set(bitmask, _GEMMA_ID["a"])


def test_gemma_4_declared_key_excluded_from_additional_properties() -> None:
    # With strict_mode=False, objects that don't set additionalProperties:false
    # get an additional-properties branch whose value is unconstrained. The
    # additional-key pattern must still exclude the DECLARED property names --
    # otherwise an optional declared key (`foo`, typed integer) can be emitted
    # through that branch with an arbitrary value, producing grammar-legal but
    # schema-violating output.
    params = {"properties": {"foo": {"type": "integer"}}}

    # Declared key with its declared type is accepted.
    ok = _gemma_matcher_params(params)
    assert _accept_ids(
        ok, _gemma_encode("<|tool_call>call:f{foo:42}<tool_call|>")
    )
    assert ok.is_completed()

    # Undeclared keys still flow through the additional branch (the point of
    # strict_mode=False), including keys that merely extend a declared name.
    extra = _gemma_matcher_params(params)
    assert _accept_ids(
        extra,
        _gemma_encode('<|tool_call>call:f{bar:<|"|>x<|"|>}<tool_call|>'),
    )
    assert extra.is_completed()

    prefixed = _gemma_matcher_params(params)
    assert _accept_ids(
        prefixed,
        _gemma_encode('<|tool_call>call:f{foobar:<|"|>x<|"|>}<tool_call|>'),
    )
    assert prefixed.is_completed()

    # The declared key must NOT be matchable as an additional property: a
    # string value for integer-typed `foo` is rejected.
    bad = _gemma_matcher_params(params)
    assert not _accept_ids(bad, _gemma_encode('<|tool_call>call:f{foo:<|"|>'))


def test_gemma_4_false_schema_property_key_never_emittable() -> None:
    # A `false` property schema forbids the key outright: the key must never
    # be emittable with any value -- not even through the strict_mode=False
    # additional-properties branch that admits undeclared keys.
    params = {"properties": {"foo": True, "bar": False}}

    # `foo: true` accepts any value.
    ok = _gemma_matcher_params(params)
    assert _accept_ids(
        ok, _gemma_encode("<|tool_call>call:f{foo:42}<tool_call|>")
    )
    assert ok.is_completed()

    # Keys merely extending the forbidden name stay emittable.
    prefixed = _gemma_matcher_params(params)
    assert _accept_ids(
        prefixed,
        _gemma_encode('<|tool_call>call:f{barx:<|"|>x<|"|>}<tool_call|>'),
    )
    assert prefixed.is_completed()

    # The forbidden key itself must never be emittable, with any value.
    bad = _gemma_matcher_params(params)
    assert not _accept_ids(bad, _gemma_encode("<|tool_call>call:f{bar:1"))


def test_gemma_4_only_false_schema_properties_still_excluded() -> None:
    # An object whose declared properties are ALL false-schema behaves like
    # an object with no declared properties -- undeclared keys stay
    # emittable -- but the forbidden names must still never be emitted.
    params = {"properties": {"bar": False}}

    ok = _gemma_matcher_params(params)
    assert _accept_ids(
        ok, _gemma_encode('<|tool_call>call:f{barx:<|"|>x<|"|>}<tool_call|>')
    )
    assert ok.is_completed()

    bad = _gemma_matcher_params(params)
    assert not _accept_ids(bad, _gemma_encode("<|tool_call>call:f{bar:1"))


def test_gemma_4_forbidden_keys_scoped_to_their_object() -> None:
    # Forbidden (false-schema) keys apply per object level: a name forbidden
    # at the outer level stays emittable inside a nested object and vice
    # versa.
    params = {
        "properties": {
            "bar": False,
            "inner": {"type": "object", "properties": {"qux": False}},
        }
    }

    ok = _gemma_matcher_params(params)
    assert _accept_ids(
        ok, _gemma_encode("<|tool_call>call:f{inner:{bar:1}}<tool_call|>")
    )
    assert ok.is_completed()

    bad_inner = _gemma_matcher_params(params)
    assert not _accept_ids(
        bad_inner, _gemma_encode("<|tool_call>call:f{inner:{qux:1")
    )

    bad_outer = _gemma_matcher_params(params)
    assert not _accept_ids(bad_outer, _gemma_encode("<|tool_call>call:f{bar:1"))


def test_gemma_4_property_names_with_properties_rejected() -> None:
    # Combining propertyNames with declared properties (including keys
    # forbidden by a false schema) can't be enforced correctly: the two key
    # constraints would have to be intersected. Rather than accept a schema it
    # would only partially enforce, the combination is rejected under
    # reject_unsupported.
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "properties": {"foo": {"type": "integer"}},
                "propertyNames": {"pattern": "^[a-z]+$"},
            }
        )

    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "properties": {"bar": False},
                "propertyNames": {"pattern": "^[a-z]+$"},
            }
        )


def test_gemma_4_cache_key_annotation_named_property_not_collided() -> None:
    # A property NAME that happens to spell an annotation keyword (title,
    # description, ...) is a name, not an annotation. Sibling subschemas
    # differing only in such a property are different schemas and must not
    # share behavior: in `a` the key is integer-typed, in `b` it is
    # forbidden.
    params = {
        "type": "object",
        "properties": {
            "a": {
                "type": "object",
                "properties": {"description": {"type": "integer"}},
            },
            "b": {"type": "object", "properties": {"description": False}},
        },
    }

    ok = _gemma_matcher_params(params)
    assert _accept_ids(
        ok, _gemma_encode("<|tool_call>call:f{a:{description:1}}<tool_call|>")
    )
    assert ok.is_completed()

    # In b, description is forbidden outright -- it must not be accepted
    # just because a declares an integer-typed property of the same name.
    bad = _gemma_matcher_params(params)
    assert not _accept_ids(
        bad, _gemma_encode("<|tool_call>call:f{b:{description:1")
    )


@pytest.mark.parametrize(
    ("terminal", "literal_forbidden", "pattern_forbidden"),
    [
        pytest.param(
            r"[a-zA-Z_][-a-zA-Z0-9_.]*",
            "",
            "",
            id="terminal_only",
        ),
        pytest.param(
            "",
            r":{},\x00-\x20\x7f",
            r" \t\n\r\f:{},\"\\\x00-\x1f",
            id="all_but_terminal",
        ),
        pytest.param(
            "",
            r":{},\x00-\x20\x7f",
            "",
            id="literal_forbidden_only",
        ),
    ],
)
def test_partial_bare_key_config_rejected(
    terminal: str, literal_forbidden: str, pattern_forbidden: str
) -> None:
    # The bare-key options are a group: setting the terminal without the
    # forbidden-character sets (or vice versa) would leave the parser and
    # converter disagreeing about whether keys are bare, so a partial set is
    # rejected rather than silently producing a mixed grammar.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"foo": {"type": "integer"}},
            },
            string_value_delimiter_token='<|"|>',
            additional_bare_key_terminal=terminal,
            bare_key_literal_forbidden=literal_forbidden,
            bare_key_pattern_forbidden=pattern_forbidden,
            require_object_root=True,
        )
    )
    with pytest.raises(Exception):
        _gemma_compiler().compile_structural_tag(tag)


def test_additional_bare_key_terminal_undecomposable_rejected() -> None:
    # A bare-key terminal in an unsupported shape (here: grouped
    # subexpressions rather than a plain class-with-repetition form) cannot
    # keep declared property names out of the additional keys; the schema
    # must be rejected under reject_unsupported rather than accepted with
    # the key constraint only partially enforced.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"foo": {"type": "integer"}},
            },
            string_value_delimiter_token='<|"|>',
            additional_bare_key_terminal="([a-zA-Z_])([-a-zA-Z0-9_.])*",
            max_whitespace_cnt=1,
            bare_key_literal_forbidden=r":{},\x00-\x20\x7f",
            bare_key_pattern_forbidden=r" \t\n\r\f:{},\"\\\x00-\x1f",
            strict_mode=False,
            require_object_root=True,
            reject_unsupported=True,
        )
    )
    with pytest.raises(Exception):
        _gemma_compiler().compile_structural_tag(tag)


def test_gemma_4_required_forced_property_with_pattern_properties_rejected() -> (
    None
):
    # A key listed in `required` but absent from `properties` is still a
    # declared property of the object, so `required` + `patternProperties`
    # needs the same key intersection as declared properties +
    # patternProperties. It must be rejected even though the schema carries
    # no `properties` keyword at all -- otherwise the pattern constraint is
    # silently dropped for the required key.
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "required": ["x"],
                "patternProperties": {"^[a-z]+$": {"type": "string"}},
            }
        )


def test_bare_key_class_escape_unambiguous_before_hex_digit_member() -> None:
    # A key alphabet mixing a control character and a hex-digit character
    # must keep the two distinct: keys using the hex-digit character stay
    # accepted.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
            },
            string_value_delimiter_token='<|"|>',
            additional_bare_key_terminal=r"[a0\x01][a0\x01]*",
            bare_key_literal_forbidden=r":{},\x00-\x20\x7f",
            bare_key_pattern_forbidden=r" \t\n\r\f:{},\"\\\x00-\x1f",
            max_whitespace_cnt=1,
            strict_mode=False,
            require_object_root=True,
            reject_unsupported=True,
        )
    )
    m = xgr.GrammarMatcher(_gemma_compiler().compile_structural_tag(tag))
    assert _accept_ids(m, _gemma_encode("{0:1}"))
    assert m.is_completed()


def test_cache_key_const_instance_values_not_annotation_filtered() -> None:
    # const/enum values are JSON INSTANCES, not schemas: an object member
    # named "title" inside a const value is data, not an ignorable
    # annotation. Two consts differing only in such a member are different
    # schemas and must not share behavior -- each accepts exactly its own
    # literal value.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "properties": {'
        '"a": {"const": {"title": "a"}}, "b": {"const": {"title": "b"}}}, '
        '"required": ["a", "b"]}'
    )
    # A const value is accepted only in its canonical JSON spelling (no
    # whitespace inside it); the enclosing object allows whitespace.
    assert _accepts(compiled, '{"a": {"title":"a"}, "b": {"title":"b"}}')


def test_cache_key_distinguishes_property_names_default_type() -> None:
    # An empty schema means "any string" when it constrains property names but
    # "any value" when it constrains a property value. Those two readings must
    # stay distinct even when the schema text is identical -- otherwise object
    # keys could be emitted as arbitrary JSON values (e.g. bare numbers)
    # instead of strings.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "properties": {'
        '"b": {}, "a": {"type": "object", "propertyNames": {}}}, '
        '"required": ["a", "b"]}'
    )
    assert _accepts(compiled, '{"b": 1, "a": {"a": 1}}')
    assert not _accepts(compiled, '{"b": 1, "a": {1: 1}}')


def test_property_names_non_string_shape_rejected() -> None:
    for property_names in (
        {"type": ["integer", "string"]},
        {"type": ["string"]},
        {"anyOf": [{"type": "string"}, {"type": ["integer"]}]},
    ):
        schema = {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "propertyNames": property_names,
        }
        with pytest.raises(Exception, match="non-string"):
            _gemma_compile(schema)
        with pytest.raises(Exception, match="non-string"):
            _compiler().compile_json_schema(json.dumps(schema))


def test_property_names_non_string_const_enum_rejected() -> None:
    for property_names in (
        {"const": 42},
        {"const": True},
        {"const": None},
        {"enum": [1, 2]},
        {"enum": ["a", 2]},
        {"enum": [None]},
    ):
        schema = {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "propertyNames": property_names,
        }
        with pytest.raises(Exception, match="non-string"):
            _gemma_compile(schema)
        with pytest.raises(Exception, match="non-string"):
            _compiler().compile_json_schema(json.dumps(schema))


def test_property_names_string_const_enum_accepted() -> None:
    for property_names in ({"const": "foo"}, {"enum": ["a", "b"]}):
        schema = {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "propertyNames": property_names,
        }
        _gemma_compile(schema)
        _compiler().compile_json_schema(json.dumps(schema))


def test_property_names_allof_folds_to_string_shape() -> None:
    for accepted in (
        {"allOf": [{"type": "string"}]},
        {"allOf": [{"type": "string"}, {"minLength": 1}]},
        {"allOf": [{"const": "foo"}]},
    ):
        schema = {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "propertyNames": accepted,
        }
        _compiler().compile_json_schema(json.dumps(schema))
    for rejected in (
        {"allOf": [{"type": "number"}]},
        {"allOf": [{"type": "string"}, {"type": "number"}]},
    ):
        schema = {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "propertyNames": rejected,
        }
        with pytest.raises(
            Exception, match="must be an object that validates string"
        ):
            _compiler().compile_json_schema(json.dumps(schema))


def test_cache_key_not_forgeable_via_property_name() -> None:
    # A property name may contain quotes and colons that make one
    # subschema's text resemble a structurally different sibling's. Such
    # similar-looking schemas must not share behavior: `y` must still
    # accept its own declared shape.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "properties": {'
        '"x": {"type": "object", "properties": {"a\\":true,\\"b": {"type": "integer"}}}, '
        '"y": {"type": "object", "properties": {"a": true, "b": {"type": "integer"}}}}, '
        '"required": ["y"]}'
    )
    assert _accepts(compiled, '{"y": {"a": true, "b": 1}}')


def test_gemma_4_property_names_max_length_zero_rejected_as_schema_error() -> (
    None
):
    # maxLength 0 permits only the empty key, which a bare key cannot
    # represent. This must surface as a schema-level rejection naming
    # propertyNames, not fail later with an opaque internal error.
    with pytest.raises(Exception, match="propertyNames"):
        _gemma_compile({"type": "object", "propertyNames": {"maxLength": 0}})


def test_gemma_4_property_names_explicit_min_length_zero_rejected() -> None:
    # An explicit minLength of 0 affirmatively permits the empty key, which
    # a bare key cannot represent. Silently tightening an author's explicit
    # bound would misenforce the schema, so it is rejected. An ABSENT
    # minimum makes no such promise and compiles: keys simply need at least
    # one character.
    with pytest.raises(Exception, match="minLength"):
        _gemma_compile(
            {
                "type": "object",
                "propertyNames": {"type": "string", "minLength": 0},
            }
        )

    # An absent minimum still compiles (keys need at least one character).
    m = _gemma_matcher_params(
        {"type": "object", "propertyNames": {"type": "string", "maxLength": 3}}
    )
    assert _accept_ids(m, _gemma_encode("<|tool_call>call:f{abc"))


def test_gemma_4_property_names_empty_matching_pattern_rejected() -> None:
    # A propertyNames pattern whose language contains the empty string
    # would let the grammar emit a zero-length bare key (`{: 1}`), which
    # the format cannot represent. Reject under reject_unsupported.
    with pytest.raises(Exception):
        _gemma_compile({"type": "object", "propertyNames": {"pattern": "^a*$"}})


def test_gemma_4_empty_matching_pattern_properties_key_rejected() -> None:
    # Same hole through patternProperties: a key pattern matching the empty
    # string could emit `{: 1}`.
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "patternProperties": {"^a*$": {"type": "integer"}},
            }
        )


def test_gemma_4_property_names_with_pattern_properties_rejected() -> None:
    # patternProperties and propertyNames both constrain the key language,
    # and enforcing both needs an unsupported key intersection -- fail closed
    # rather than enforce only one.
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "propertyNames": {"maxLength": 1},
                "patternProperties": {"^a+$": {"type": "integer"}},
            }
        )


def test_pattern_properties_with_additional_properties_rejected() -> None:
    # patternProperties + additionalProperties (non-false) needs per-key
    # priority: a matching key takes the pattern's subschema, a non-matching
    # key falls back to additionalProperties. The fallback arm needs the
    # complement of a regex as a key pattern, which xgrammar cannot express.
    # Reject the combination under reject_unsupported rather than emit an
    # unsound un-excluded fallback (which would accept a matching key with
    # the additionalProperties value) or silently drop the fallback.
    #
    # All open forms are rejected: a schema value, an empty schema (which
    # accepts any value, same as true), and boolean true.
    schema_base = (
        '{"type":"object",'
        '"patternProperties":{"^[a-z]+$":{"type":"string"}},'
        '"additionalProperties":'
    )
    for addl in ('{"type":"integer"}', "{}", "true"):
        with pytest.raises(Exception, match="patternProperties"):
            _compiler().compile_json_schema(
                schema_base + addl + "}",
                reject_unsupported=True,
            )


def test_pattern_properties_with_additional_properties_false_compiles() -> None:
    # additionalProperties:false closes the object: no fallback arm exists,
    # so the reject guard does not fire and the schema compiles under
    # reject_unsupported. The pattern is still enforced -- a matching key
    # must conform to the pattern's subschema, and a non-matching key is
    # not admitted at all.
    compiled = _compiler().compile_json_schema(
        '{"type":"object",'
        '"patternProperties":{"^[a-z]+$":{"type":"string"}},'
        '"additionalProperties":false}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"a":"a"}')
    assert not _accepts(compiled, '{"a":1}')
    assert not _accepts(compiled, '{"1":1}')


def test_gemma_4_property_names_ref_max_length_zero_rejected() -> None:
    # The propertyNames bare-key guards must inspect the COMPLETE schema, not
    # only the literal propertyNames object: a $ref to a maxLength:0 subschema
    # admits only the empty key and must be rejected the same as a literal
    # maxLength:0.
    with pytest.raises(Exception, match="propertyNames"):
        _gemma_compile(
            {
                "type": "object",
                "propertyNames": {"$ref": "#/$defs/k"},
                "$defs": {"k": {"maxLength": 0}},
            }
        )


def test_gemma_4_property_names_ref_explicit_min_length_zero_rejected() -> None:
    # min-length guard through a $ref: an explicit minLength:0 behind a ref
    # affirmatively permits the empty key and must be rejected.
    with pytest.raises(Exception, match="minLength"):
        _gemma_compile(
            {
                "type": "object",
                "propertyNames": {"$ref": "#/$defs/k"},
                "$defs": {"k": {"type": "string", "minLength": 0}},
            }
        )


def test_gemma_4_property_names_anyof_empty_pattern_rejected() -> None:
    # empty-match guard through a combinator: an anyOf branch whose pattern
    # matches the empty string can emit a zero-length bare key through the
    # union, so the guard must see into the combinator and reject.
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "propertyNames": {"anyOf": [{"pattern": "^a*$"}]},
            }
        )


def test_gemma_4_property_names_allof_forbidden_char_rejected() -> None:
    # forbidden-char guard through a combinator: an allOf member pattern that
    # needs a structural byte (`:`) not allowed in a bare key must be rejected.
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "propertyNames": {"allOf": [{"pattern": "^a:b$"}]},
            }
        )


def test_gemma_4_property_names_ref_bare_safe_compiles() -> None:
    # The complete-schema resolution must not over-reject: a $ref to a
    # bare-safe key schema still compiles.
    compiled = _gemma_compile(
        {
            "type": "object",
            "propertyNames": {"$ref": "#/$defs/k"},
            "$defs": {"k": {"type": "string", "pattern": "^[a-z_]+$"}},
        }
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_xml_root_forbidden_key_excluded_from_additional_branch() -> None:
    # The XML formats must honor forbidden (false-schema) keys like every
    # other format: `bar` must never be emittable, even though
    # additionalProperties:true admits arbitrary other keys.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"bar": False},
                "additionalProperties": True,
            },
            style="qwen_xml",
        )
    )
    compiled = _compiler().compile_structural_tag(tag)
    assert _accepts(compiled, "<parameter=baz>1</parameter>")
    assert not _accepts(compiled, "<parameter=bar>1</parameter>")


def test_xml_root_declared_key_excluded_from_additional_branch() -> None:
    # Same for declared keys: `foo` is integer-typed, so it must not be
    # emittable through the any-valued additional branch with a non-integer
    # value.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"foo": {"type": "integer"}},
                "additionalProperties": True,
            },
            style="qwen_xml",
        )
    )
    compiled = _compiler().compile_structural_tag(tag)
    assert _accepts(compiled, "<parameter=foo>1</parameter>")
    assert not _accepts(compiled, "<parameter=foo>x</parameter>")


def test_bare_key_alphabet_overlapping_key_terminator_rejected() -> None:
    # A bare-key alphabet that includes the characters used to terminate a key
    # (whitespace or the colon) can't have declared names excluded reliably: an
    # excluded name could be respelled as a distinct key that lenient parsers
    # treat as the same one (e.g. `foo` + space). Such a terminal must fail
    # closed under reject_unsupported.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
            },
            string_value_delimiter_token='<|"|>',
            additional_bare_key_terminal=r"[a][a ]*",
            bare_key_literal_forbidden=r":{},\x00-\x1f\x7f",
            bare_key_pattern_forbidden=r"\t\n\r\f:{},\"\\\x00-\x1f",
            max_whitespace_cnt=1,
            strict_mode=False,
            require_object_root=True,
            reject_unsupported=True,
        )
    )
    with pytest.raises(Exception):
        _gemma_compiler().compile_structural_tag(tag)

    # The full ASCII whitespace set is covered, not just the grammar's own
    # separator characters: a form feed is strippable by lenient downstream
    # parsers just like a space.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
            },
            string_value_delimiter_token='<|"|>',
            additional_bare_key_terminal=r"[a][a\x0c]*",
            bare_key_literal_forbidden=r":{},\x00-\x1f\x7f",
            bare_key_pattern_forbidden=r"\t\n\r\f:{},\"\\\x00-\x1f",
            max_whitespace_cnt=1,
            strict_mode=False,
            require_object_root=True,
            reject_unsupported=True,
        )
    )
    with pytest.raises(Exception):
        _gemma_compiler().compile_structural_tag(tag)


def test_additional_bare_key_terminal_non_ascii_rejected() -> None:
    # Excluding declared names from a bare-key alphabet is only supported
    # for ASCII alphabets; a terminal with a non-ASCII member must fail
    # closed under reject_unsupported rather than be partially enforced.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
            },
            string_value_delimiter_token='<|"|>',
            additional_bare_key_terminal="[aé][aé]*",
            bare_key_literal_forbidden=r":{},\x00-\x20\x7f",
            bare_key_pattern_forbidden=r" \t\n\r\f:{},\"\\\x00-\x1f",
            max_whitespace_cnt=1,
            strict_mode=False,
            require_object_root=True,
            reject_unsupported=True,
        )
    )
    with pytest.raises(Exception):
        _gemma_compiler().compile_structural_tag(tag)


def test_gemma_4_declared_key_outside_alphabet_still_typed() -> None:
    # A declared name containing a character outside the format's key
    # alphabet (`$`) is still usable, and its value keeps its declared type.
    params = {"properties": {"a$b": {"type": "integer"}}}

    ok = _gemma_matcher_params(params)
    assert _accept_ids(
        ok, _gemma_encode("<|tool_call>call:f{a$b:1}<tool_call|>")
    )
    assert ok.is_completed()

    bad = _gemma_matcher_params(params)
    assert not _accept_ids(bad, _gemma_encode('<|tool_call>call:f{a$b:<|"|>'))


def test_gemma_4_non_ascii_declared_key_supported() -> None:
    # Non-ASCII declared property names are supported, and their values keep
    # the declared type.
    vocab = _GEMMA_VOCAB + ["é"]
    ids = {tok: i for i, tok in enumerate(vocab)}
    info = xgr.TokenizerInfo(
        vocab,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[ids["<eos>"]],
        special_token_ids=[ids[tok] for tok in _GEMMA_SPECIALS],
    )
    tag = xgr.get_builtin_structural_tag(
        "gemma_4",
        tools=_gemma_tool_with_params(
            {"properties": {"clé": {"type": "integer"}}}
        ),
        tool_choice="required",
        reasoning=False,
    )
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(tag)

    def encode(text: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            for sp in _GEMMA_SPECIALS:
                if text.startswith(sp, i):
                    out.append(ids[sp])
                    i += len(sp)
                    break
            else:
                out.append(ids[text[i]])
                i += 1
        return out

    ok = xgr.GrammarMatcher(compiled)
    assert _accept_ids(ok, encode("<|tool_call>call:f{clé:1}<tool_call|>"))
    assert ok.is_completed()

    bad = xgr.GrammarMatcher(compiled)
    assert not _accept_ids(bad, encode('<|tool_call>call:f{clé:<|"|>'))


def test_gemma_4_property_names_non_ascii_pattern_keys() -> None:
    # A propertyNames pattern may constrain keys to non-ASCII characters.
    vocab = _GEMMA_VOCAB + ["é"]
    ids = {tok: i for i, tok in enumerate(vocab)}
    info = xgr.TokenizerInfo(
        vocab,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[ids["<eos>"]],
        special_token_ids=[ids[tok] for tok in _GEMMA_SPECIALS],
    )
    tag = xgr.get_builtin_structural_tag(
        "gemma_4",
        tools=_gemma_tool_with_params(
            {"type": "object", "propertyNames": {"pattern": "^é+$"}}
        ),
        tool_choice="required",
        reasoning=False,
    )
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(tag)

    def encode(text: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            for sp in _GEMMA_SPECIALS:
                if text.startswith(sp, i):
                    out.append(ids[sp])
                    i += len(sp)
                    break
            else:
                out.append(ids[text[i]])
                i += 1
        return out

    ok = xgr.GrammarMatcher(compiled)
    assert _accept_ids(ok, encode("<|tool_call>call:f{é:1}<tool_call|>"))
    assert ok.is_completed()

    bad = xgr.GrammarMatcher(compiled)
    assert not _accept_ids(bad, encode("<|tool_call>call:f{a"))


def test_property_names_max_length_zero_permissive_forces_empty_object() -> (
    None
):
    # Without reject_unsupported the schema is not rejected, but the grammar
    # must still never emit a zero-length bare key: since only the empty key
    # would satisfy maxLength 0, no property is emittable at all and the
    # object can only be `{}`.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={"type": "object", "propertyNames": {"maxLength": 0}},
            string_value_delimiter_token='<|"|>',
            additional_bare_key_terminal=r"[a-zA-Z_][-a-zA-Z0-9_.]*",
            bare_key_literal_forbidden=r":{},\x00-\x20\x7f",
            bare_key_pattern_forbidden=r" \t\n\r\f:{},\"\\\x00-\x1f",
            max_whitespace_cnt=1,
            strict_mode=False,
            require_object_root=True,
            reject_unsupported=False,
        )
    )
    compiled = _gemma_compiler().compile_structural_tag(tag)

    ok = xgr.GrammarMatcher(compiled)
    assert _accept_ids(ok, _gemma_encode("{}"))
    assert ok.is_completed()

    no_empty_key = xgr.GrammarMatcher(compiled)
    assert _accept_ids(no_empty_key, _gemma_encode("{"))
    assert not no_empty_key.accept_token(_GEMMA_ID[":"])

    no_key = xgr.GrammarMatcher(compiled)
    assert _accept_ids(no_key, _gemma_encode("{"))
    assert not no_key.accept_token(_GEMMA_ID["a"])


def test_property_names_max_length_zero_with_declared_properties_forces_empty() -> (
    None
):
    # propertyNames maxLength 0 constrains every key, including declared
    # ones, to the empty key -- which a bare key cannot represent. So the
    # only valid instance is `{}` and no key, declared or otherwise, is
    # emittable.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"foo": {"type": "integer"}},
                "propertyNames": {"maxLength": 0},
            },
            string_value_delimiter_token='<|"|>',
            additional_bare_key_terminal=r"[a-zA-Z_][-a-zA-Z0-9_.]*",
            bare_key_literal_forbidden=r":{},\x00-\x20\x7f",
            bare_key_pattern_forbidden=r" \t\n\r\f:{},\"\\\x00-\x1f",
            max_whitespace_cnt=1,
            strict_mode=False,
            require_object_root=True,
            reject_unsupported=False,
        )
    )
    compiled = _gemma_compiler().compile_structural_tag(tag)

    ok = xgr.GrammarMatcher(compiled)
    assert _accept_ids(ok, _gemma_encode("{}"))
    assert ok.is_completed()

    no_key = xgr.GrammarMatcher(compiled)
    assert _accept_ids(no_key, _gemma_encode("{"))
    assert not no_key.accept_token(_GEMMA_ID["f"])


def test_property_names_empty_only_with_min_properties_rejected() -> None:
    # propertyNames admits only the empty key (unrepresentable as a bare key),
    # so the object must be empty; minProperties>=1 makes that unsatisfiable.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "minProperties": 1,
                "propertyNames": {"maxLength": 0},
            },
            string_value_delimiter_token='<|"|>',
            additional_bare_key_terminal=r"[a-zA-Z_][-a-zA-Z0-9_.]*",
            bare_key_literal_forbidden=r":{},\x00-\x20\x7f",
            bare_key_pattern_forbidden=r" \t\n\r\f:{},\"\\\x00-\x1f",
            max_whitespace_cnt=1,
            strict_mode=False,
            require_object_root=True,
            reject_unsupported=False,
        )
    )
    with pytest.raises(Exception, match="propertyNames"):
        _gemma_compiler().compile_structural_tag(tag)


def test_property_names_empty_only_with_required_rejected() -> None:
    # Same conflict via required: a required key cannot be present when only
    # the empty (unrepresentable) key is allowed.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "required": ["x"],
                "propertyNames": {"maxLength": 0},
            },
            string_value_delimiter_token='<|"|>',
            additional_bare_key_terminal=r"[a-zA-Z_][-a-zA-Z0-9_.]*",
            bare_key_literal_forbidden=r":{},\x00-\x20\x7f",
            bare_key_pattern_forbidden=r" \t\n\r\f:{},\"\\\x00-\x1f",
            max_whitespace_cnt=1,
            strict_mode=False,
            require_object_root=True,
            reject_unsupported=False,
        )
    )
    with pytest.raises(Exception, match="propertyNames"):
        _gemma_compiler().compile_structural_tag(tag)


def test_property_names_empty_only_with_additional_properties_forces_empty() -> (
    None
):
    # additionalProperties present but no lower bound: the object still
    # collapses to `{}` (no key is representable) rather than admitting an
    # additional key.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "propertyNames": {"maxLength": 0},
                "additionalProperties": {"type": "string"},
            },
            string_value_delimiter_token='<|"|>',
            additional_bare_key_terminal=r"[a-zA-Z_][-a-zA-Z0-9_.]*",
            bare_key_literal_forbidden=r":{},\x00-\x20\x7f",
            bare_key_pattern_forbidden=r" \t\n\r\f:{},\"\\\x00-\x1f",
            max_whitespace_cnt=1,
            strict_mode=False,
            require_object_root=True,
            reject_unsupported=False,
        )
    )
    compiled = _gemma_compiler().compile_structural_tag(tag)

    ok = xgr.GrammarMatcher(compiled)
    assert _accept_ids(ok, _gemma_encode("{}"))
    assert ok.is_completed()

    no_key = xgr.GrammarMatcher(compiled)
    assert _accept_ids(no_key, _gemma_encode("{"))
    assert not no_key.accept_token(_GEMMA_ID["a"])


def test_gemma_4_property_names_bounded_empty_pattern_rejected() -> None:
    # A bounded-repetition pattern with a zero lower bound (`[a-z]{0,5}`) can
    # match the empty string, so it could emit a zero-length bare key just
    # like `[a-z]*`; it must be rejected under reject_unsupported.
    with pytest.raises(Exception):
        _gemma_compile(
            {"type": "object", "propertyNames": {"pattern": "[a-z]{0,5}"}}
        )
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "patternProperties": {"[a-z]{0,5}": {"type": "integer"}},
            }
        )


def test_gemma_4_property_names_bounded_nonempty_pattern_compiles() -> None:
    # The complement: a bounded pattern that cannot match empty (`[a-z]{1,5}`)
    # is a valid key constraint and still compiles.
    m = _gemma_matcher_params(
        {"type": "object", "propertyNames": {"pattern": "[a-z]{1,5}"}}
    )
    assert _accept_ids(m, _gemma_encode("<|tool_call>call:f{ab"))


def test_gemma_4_property_names_key_cannot_be_empty() -> None:
    # A bare key has no delimiters, so a zero-length key is unrepresentable:
    # an unconstrained-string propertyNames must still require at least one
    # key character (otherwise the grammar emits `{: 1}`).
    m = _gemma_matcher_params(
        {"type": "object", "propertyNames": {"type": "string"}}
    )
    assert _accept_ids(m, _gemma_encode("<|tool_call>call:f{a"))
    ok = _gemma_matcher_params(
        {"type": "object", "propertyNames": {"type": "string"}}
    )
    assert _accept_ids(ok, _gemma_encode("<|tool_call>call:f{"))
    assert not ok.accept_token(_GEMMA_ID[":"])


# Control/special tokens that the real Gemma tokenizer decodes to the empty
# string with skip_special_tokens=True (channel/turn/tool/think markers, etc.).
# The production backend derives exactly these from the tokenizer and force-marks
# them special so they are masked out of byte-level string content -- otherwise
# each one satisfies a character-minimum while decoding to nothing, silently
# bypassing minLength.
# <|channel>/<channel|> are also the reasoning-prefix markers, referenced by
# token id, so marking them special must not break tool_choice=auto/reasoning.
_GEMMA_CONTROL_TOKENS = (
    "<|channel>",
    "<channel|>",
    "<|turn>",
    "<turn|>",
    "<|think|>",
    "<|tool>",
    "<tool|>",
)
_GEMMA_CTRL_VOCAB = (
    [chr(c) for c in range(32, 127)]
    + ["\n"]  # the reasoning-prefix "thought\n" literal needs a newline token
    + ["<|tool_call>", "<tool_call|>", '<|"|>']
    + list(_GEMMA_CONTROL_TOKENS)
    + ["<eos>"]
)
# All the multi-char markers/control tokens are single vocab entries here, mirror
# the tokenizer; every one of them is force-marked special.
_GEMMA_CTRL_SPECIALS = (
    "<|tool_call>",
    "<tool_call|>",
    '<|"|>',
    *_GEMMA_CONTROL_TOKENS,
)
_GEMMA_CTRL_ID = {tok: i for i, tok in enumerate(_GEMMA_CTRL_VOCAB)}


def _gemma_ctrl_compiler() -> xgr.GrammarCompiler:
    info = xgr.TokenizerInfo(
        _GEMMA_CTRL_VOCAB,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[_GEMMA_CTRL_ID["<eos>"]],
        special_token_ids=[_GEMMA_CTRL_ID[tok] for tok in _GEMMA_CTRL_SPECIALS],
    )
    return xgr.GrammarCompiler(info)


def _gemma_ctrl_encode(s: str) -> list[int]:
    multi = sorted(
        {*_GEMMA_CTRL_SPECIALS, *_GEMMA_CONTROL_TOKENS}, key=len, reverse=True
    )
    out: list[int] = []
    i = 0
    while i < len(s):
        for tok in multi:
            if s.startswith(tok, i):
                out.append(_GEMMA_CTRL_ID[tok])
                i += len(tok)
                break
        else:
            out.append(_GEMMA_CTRL_ID[s[i]])
            i += 1
    return out


def _gemma_ctrl_matcher(
    tools: list[dict[str, Any]],
    tool_choice: Any = "required",
    reasoning: bool = False,
) -> xgr.GrammarMatcher:
    tag = xgr.get_builtin_structural_tag(
        "gemma_4", tools=tools, tool_choice=tool_choice, reasoning=reasoning
    )
    return xgr.GrammarMatcher(
        _gemma_ctrl_compiler().compile_structural_tag(tag)
    )


def test_gemma_4_control_tokens_forbidden_as_string_content() -> None:
    # The realistic-tokenizer regression the toy 3-marker vocab missed: with a
    # full control-token set force-marked special, every decode-to-empty control
    # token must be FORBIDDEN inside a minLength-gated string body, so "hi"+<ctrl>
    # cannot satisfy the character minimum by decoding to nothing.
    tool = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "properties": {
                        "bar": {
                            "type": "string",
                            "minLength": 4,
                            "default": "bad",
                        }
                    }
                },
            },
        }
    ]
    for ctrl in _GEMMA_CONTROL_TOKENS:
        m = _gemma_ctrl_matcher(tool)
        assert _accept_ids(
            m, _gemma_ctrl_encode('<|tool_call>call:f{bar:<|"|>hi')
        )
        # Two chars in; a control token decodes to nothing, so admitting it as
        # content would let the string close below minLength: it must be rejected.
        assert not m.accept_token(_GEMMA_CTRL_ID[ctrl]), (
            f"control token {ctrl!r} leaked into string content"
        )


def test_gemma_4_control_special_still_emits_tool_calls() -> None:
    # Force-marking the control tokens special (including <|channel>/<channel|>)
    # must not break tool-call emission for auto / required / named tool choice.
    call = '<|tool_call>call:f{x:<|"|>hi<|"|>}<tool_call|>'
    named: dict[str, Any] = {
        "type": "function",
        "function": {"name": "f"},
    }
    for tool_choice in ("auto", "required", named):
        m = _gemma_ctrl_matcher(_SIMPLE_TOOL, tool_choice=tool_choice)
        assert _accept_ids(m, _gemma_ctrl_encode(call)), tool_choice
        assert m.is_completed(), tool_choice


def test_gemma_4_control_special_reasoning_prefix_opens_and_closes() -> None:
    # The reasoning prefix references <|channel>/<channel|> by token id, so with
    # those markers force-marked special the prefix must still open, carry free
    # thinking text, close, and let a following tool call complete.
    m = _gemma_ctrl_matcher(
        _SIMPLE_TOOL, tool_choice="required", reasoning=True
    )
    prefix = "<|channel>thought\nlet me think<channel|>"
    assert _accept_ids(m, _gemma_ctrl_encode(prefix))
    call = '<|tool_call>call:f{x:<|"|>hi<|"|>}<tool_call|>'
    assert _accept_ids(m, _gemma_ctrl_encode(call))
    assert m.is_completed()


# A vocab carrying a multi-CHARACTER content token ("word", 4 chars, one token)
# so the character-vs-token distinction is observable: single-char content tokens
# count identically as chars and as tokens, so they cannot tell a per-CHARACTER
# body apart from a per-TOKEN one. A genuine multi-char content token can.
_GEMMA_MULTICHAR_CONTENT = "word"
_GEMMA_MC_VOCAB = _GEMMA_VOCAB[:-1] + [_GEMMA_MULTICHAR_CONTENT] + ["<eos>"]
_GEMMA_MC_ID = {tok: i for i, tok in enumerate(_GEMMA_MC_VOCAB)}


def _gemma_mc_compiler() -> xgr.GrammarCompiler:
    info = xgr.TokenizerInfo(
        _GEMMA_MC_VOCAB,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[_GEMMA_MC_ID["<eos>"]],
        special_token_ids=[_GEMMA_MC_ID[tok] for tok in _GEMMA_SPECIALS],
    )
    return xgr.GrammarCompiler(info)


def _gemma_mc_encode(s: str) -> list[int]:
    # Greedy: prefer the multi-char content token and the multi-char specials.
    multi = sorted(
        [*_GEMMA_SPECIALS, _GEMMA_MULTICHAR_CONTENT], key=len, reverse=True
    )
    out: list[int] = []
    i = 0
    while i < len(s):
        for tok in multi:
            if s.startswith(tok, i):
                out.append(_GEMMA_MC_ID[tok])
                i += len(tok)
                break
        else:
            out.append(_GEMMA_MC_ID[s[i]])
            i += 1
    return out


def test_gemma_4_min_length_counts_characters_not_tokens() -> None:
    # minLength is a character count: a string meeting it via a single
    # multi-character token must be accepted.
    tool = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string", "minLength": 4}},
                    "required": ["x"],
                },
            },
        }
    ]
    tag = xgr.get_builtin_structural_tag(
        "gemma_4", tools=tool, tool_choice="required", reasoning=False
    )
    matcher = xgr.GrammarMatcher(
        _gemma_mc_compiler().compile_structural_tag(tag)
    )
    # "word" is 4 characters but a single token: char-counting accepts and can
    # close (4 chars == minLength); token-counting would reject (1 token < 4).
    assert _accept_ids(
        matcher, _gemma_mc_encode('<|tool_call>call:f{x:<|"|>word<|"|>}')
    )
    assert _accept_ids(matcher, _gemma_mc_encode("<tool_call|>"))
    assert matcher.is_completed()


def test_gemma_4_matcher_rejects_json_quoting() -> None:
    # Standard-JSON quoted key `"x"` must be rejected: Gemma keys are bare.
    matcher = _gemma_matcher(_SIMPLE_TOOL)
    assert not _accept_ids(matcher, _gemma_encode('<|tool_call>call:f{"x"'))


def test_gemma_4_requires_delimiter_token_in_vocab() -> None:
    # No byte-literal fallback: a vocab without the <|"|> token is a hard error.
    # The tool-call markers are present (they are referenced by token id too), so
    # resolution reaches the missing string delimiter and fails on it.
    info = xgr.TokenizerInfo(
        [chr(c) for c in range(32, 127)]
        + ["<|tool_call>", "<tool_call|>", "<eos>"],
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[97],
    )
    tag = xgr.get_builtin_structural_tag(
        "gemma_4", tools=_SIMPLE_TOOL, tool_choice="required", reasoning=False
    )
    with pytest.raises(Exception, match=r'<\|"\|>'):
        xgr.GrammarCompiler(info).compile_structural_tag(tag)


def _gemma_tool_with_params(params: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": "f", "parameters": params}}
    ]


def _gemma_compile(params: dict[str, Any]) -> xgr.CompiledGrammar:
    tag = xgr.get_builtin_structural_tag(
        "gemma_4",
        tools=_gemma_tool_with_params(params),
        tool_choice="required",
        reasoning=False,
    )
    return _gemma_compiler().compile_structural_tag(tag)


def test_gemma_4_tool_call_rejects_unsupported_keyword_without_walker() -> None:
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "properties": {"n": {"type": "number", "multipleOf": 5}},
            }
        )


def test_gemma_4_tool_call_non_object_root_rejected_without_walker() -> None:
    # The JSON_CONFIG enable makes require_object_root real on the gemma path:
    # a non-object (array) arg-schema root is rejected, with NO test-level
    # flag walker.
    with pytest.raises(Exception):
        _gemma_compile({"type": "array", "items": {"type": "string"}})


def test_qwen_tool_call_permits_unsupported_keyword_by_default() -> None:
    tag = xgr.get_builtin_structural_tag(
        "qwen_3_5",
        tools=_qwen_tool(
            {
                "type": "object",
                "properties": {"n": {"type": "number", "multipleOf": 5}},
            }
        ),
        tool_choice="required",
        reasoning=False,
    )
    compiled = _qwen_compiler().compile_structural_tag(tag)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_gemma_4_pattern_properties_bare_safe_compiles() -> None:
    # A bare-key-safe patternProperties pattern (only lowercase/underscore, all
    # in the bare-key alphabet) compiles on the gemma path.
    compiled = _gemma_compile(
        {
            "type": "object",
            "patternProperties": {"^[a-z_]+$": {"type": "string"}},
        }
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_gemma_4_property_names_bare_safe_compiles() -> None:
    # A bare-key-safe propertyNames pattern compiles on the gemma path.
    compiled = _gemma_compile(
        {
            "type": "object",
            "propertyNames": {"type": "string", "pattern": "^[a-z_]+$"},
        }
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_gemma_4_structural_char_pattern_rejected() -> None:
    # A pattern that can positively match a structural byte (`:`) would let a
    # bare key over-run its boundary; the sound guard must reject it fail-closed.
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "patternProperties": {"^[a:b]+$": {"type": "string"}},
            }
        )
    # A bare literal `:` in the pattern is likewise rejected.
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "propertyNames": {"type": "string", "pattern": "^a:b$"},
                "additionalProperties": {"type": "string"},
            }
        )


def test_gemma_4_pattern_properties_format_agnostic_rejects_preserved() -> None:
    # The format-agnostic intersection limits still reject on the gemma path:
    # multiple simultaneous patternProperties patterns, and `properties`
    # combined with `patternProperties`.
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "patternProperties": {
                    "^[a-z]+$": {"type": "string"},
                    "^[a-z_]+$": {"type": "string"},
                },
            }
        )
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "properties": {"foo": {"type": "string"}},
                "patternProperties": {"^[a-z_]+$": {"type": "string"}},
            }
        )


def _gemma_matcher_params(params: dict[str, Any]) -> xgr.GrammarMatcher:
    return _gemma_matcher(
        _gemma_tool_with_params(params), tool_choice="required"
    )


def test_gemma_4_property_names_format_enforced() -> None:
    # propertyNames with a bare-safe `format` (date) constrains the bare key on
    # the gemma path: a date-shaped key is accepted, a non-date key is rejected.
    params = {
        "type": "object",
        "propertyNames": {"format": "date"},
        "additionalProperties": {"type": "string"},
    }
    # Compiles (bare-safe format).
    assert isinstance(_gemma_compile(params), xgr.CompiledGrammar)
    # A date-shaped key is accepted.
    good = '<|tool_call>call:f{2020-01-01:<|"|>v<|"|>}<tool_call|>'
    m = _gemma_matcher_params(params)
    assert _accept_ids(m, _gemma_encode(good))
    assert m.is_completed()
    # A non-date key is rejected (the key format is enforced).
    m2 = _gemma_matcher_params(params)
    assert not _accept_ids(m2, _gemma_encode("<|tool_call>call:f{abc:"))


def test_gemma_4_property_names_format_with_colon_rejected() -> None:
    # A propertyNames `format` whose regex needs a byte forbidden in a bare key
    # (date-time / time contain `:`) fails CLOSED at compile.
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "propertyNames": {"format": "date-time"},
                "additionalProperties": {"type": "string"},
            }
        )


def test_gemma_4_property_names_const_bare_representability() -> None:
    # A propertyNames const/enum key carrying a bare-key-forbidden byte (`:`) is
    # rejected; a bare-safe const key compiles.
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "propertyNames": {"const": "a:b"},
                "additionalProperties": {"type": "string"},
            }
        )
    assert isinstance(
        _gemma_compile(
            {
                "type": "object",
                "propertyNames": {"const": "abc"},
                "additionalProperties": {"type": "string"},
            }
        ),
        xgr.CompiledGrammar,
    )


def test_gemma_4_property_names_true_compiles() -> None:
    # propertyNames: true is a no-op key constraint (accept, no bare-key
    # narrowing) rather than a hard reject; it compiles on the gemma path.
    compiled = _gemma_compile(
        {
            "type": "object",
            "propertyNames": True,
            "additionalProperties": {"type": "string"},
        }
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_gemma_4_property_names_false_rejected() -> None:
    # propertyNames: false remains rejected (out of scope for the no-op path).
    with pytest.raises(Exception):
        _gemma_compile(
            {
                "type": "object",
                "propertyNames": False,
                "additionalProperties": {"type": "string"},
            }
        )


def test_gemma_4_pattern_properties_full_match() -> None:
    # gemma patternProperties anchors to a FULL match (no unanchored bare*
    # padding), so a key with a leading/trailing byte outside the pattern is
    # rejected; an exact match is accepted.
    params = {
        "type": "object",
        "patternProperties": {"[a-z]+": {"type": "string"}},
    }
    assert isinstance(_gemma_compile(params), xgr.CompiledGrammar)
    good = '<|tool_call>call:f{abc:<|"|>v<|"|>}<tool_call|>'
    m = _gemma_matcher_params(params)
    assert _accept_ids(m, _gemma_encode(good))
    assert m.is_completed()
    # Leading digit: full match now rejects `9abc`.
    m2 = _gemma_matcher_params(params)
    assert not _accept_ids(m2, _gemma_encode("<|tool_call>call:f{9abc:"))
    # Trailing digit: full match now rejects `a9`.
    m3 = _gemma_matcher_params(params)
    assert not _accept_ids(m3, _gemma_encode("<|tool_call>call:f{a9:"))


# A single-char printable vocab so base-path JSON strings can be fed
# byte-by-byte through the matcher.
_JSON_VOCAB = [chr(c) for c in range(32, 127)] + ["<eos>"]
_JSON_ID = {tok: i for i, tok in enumerate(_JSON_VOCAB)}


def _json_compiler() -> xgr.GrammarCompiler:
    info = xgr.TokenizerInfo(
        _JSON_VOCAB,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[_JSON_ID["<eos>"]],
    )
    return xgr.GrammarCompiler(info)


def _json_encode(s: str) -> list[int]:
    return [_JSON_ID[ch] for ch in s]


def _json_matcher(schema: str) -> xgr.GrammarMatcher:
    return xgr.GrammarMatcher(_json_compiler().compile_json_schema(schema))


def test_refactor_base_string_pattern_accept_reject() -> None:
    # Base GenerateString `pattern` branch (shared ladder). The string is JSON
    # quoted; the body is constrained to the pattern alphabet.
    schema = '{"type": "string", "pattern": "[ab]+"}'
    assert _accept_ids(_json_matcher(schema), _json_encode('"abba"'))
    # A byte outside the pattern alphabet is rejected inside the body.
    assert not _accept_ids(_json_matcher(schema), _json_encode('"abc'))


def test_refactor_base_string_length_accept_reject() -> None:
    # Base GenerateString length branch (shared GenerateStringLengthBody).
    schema = '{"type": "string", "minLength": 2, "maxLength": 3}'
    assert _accept_ids(_json_matcher(schema), _json_encode('"ab"'))
    # Below minLength: a 1-char string cannot close.
    assert not _accept_ids(_json_matcher(schema), _json_encode('"a"'))


def test_refactor_base_const_enum_object_accept_reject() -> None:
    # Base GenerateConst/GenerateEnum object literal (unchanged serialize()-based
    # path). Exact serialized object is accepted; a wrong value is rejected.
    const_schema = '{"const": {"a": "b"}}'
    assert _accept_ids(_json_matcher(const_schema), _json_encode('{"a":"b"}'))
    assert not _accept_ids(
        _json_matcher(const_schema), _json_encode('{"a":"c"')
    )

    enum_schema = '{"enum": [{"a": "b"}, {"c": "d"}]}'
    assert _accept_ids(_json_matcher(enum_schema), _json_encode('{"c":"d"}'))
    assert not _accept_ids(_json_matcher(enum_schema), _json_encode('{"a":"z"'))


def test_refactor_gemma_string_pattern_accept_reject() -> None:
    # Gemma GenerateString `pattern` branch: the <|"|> token delimiters wrap a
    # pattern-constrained byte body.
    params = {
        "type": "object",
        "properties": {"x": {"type": "string", "pattern": "[ab]+"}},
        "required": ["x"],
    }
    good = '<|tool_call>call:f{x:<|"|>abba<|"|>}<tool_call|>'
    assert _accept_ids(_gemma_matcher_params(params), _gemma_encode(good))
    # A byte outside the pattern alphabet is rejected inside the body.
    bad = '<|tool_call>call:f{x:<|"|>abc'
    assert not _accept_ids(_gemma_matcher_params(params), _gemma_encode(bad))


def test_refactor_gemma_string_length_accept_reject() -> None:
    # Gemma GenerateString length branch via the shared GenerateStringLengthBody
    # override ({min}-prefix workaround) wrapped in <|"|> tokens.
    params = {
        "type": "object",
        "properties": {"x": {"type": "string", "minLength": 2, "maxLength": 3}},
        "required": ["x"],
    }
    good = '<|tool_call>call:f{x:<|"|>ab<|"|>}<tool_call|>'
    assert _accept_ids(_gemma_matcher_params(params), _gemma_encode(good))
    # Below minLength: a 1-char value cannot close.
    bad = '<|tool_call>call:f{x:<|"|>a<|"|>'
    assert not _accept_ids(_gemma_matcher_params(params), _gemma_encode(bad))


def test_refactor_gemma_const_enum_object_accept_reject() -> None:
    # Gemma GenerateConst/GenerateEnum object literal now routes through the
    # shared RenderJSONValueLiteral (bare keys, <|"|>-wrapped string values).
    const_params = {
        "type": "object",
        "properties": {"x": {"const": {"a": "b"}}},
        "required": ["x"],
    }
    good = '<|tool_call>call:f{x:{a:<|"|>b<|"|>}}<tool_call|>'
    assert _accept_ids(_gemma_matcher_params(const_params), _gemma_encode(good))
    # Wrong nested string value is rejected.
    bad = '<|tool_call>call:f{x:{a:<|"|>c<|"|>'
    assert not _accept_ids(
        _gemma_matcher_params(const_params), _gemma_encode(bad)
    )

    enum_params = {
        "type": "object",
        "properties": {"x": {"enum": [{"a": "b"}, {"c": "d"}]}},
        "required": ["x"],
    }
    good_enum = '<|tool_call>call:f{x:{c:<|"|>d<|"|>}}<tool_call|>'
    assert _accept_ids(
        _gemma_matcher_params(enum_params), _gemma_encode(good_enum)
    )


def test_gemma_4_nested_property_names_additional_properties_value_enforced() -> (
    None
):
    # propertyNames constrains the KEY; the schema-valued additionalProperties
    # constrains the VALUE. A key that matches propertyNames but carries a
    # wrong-typed value must be rejected -- the value schema must be enforced.
    params = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "queryParams": {
                "type": "object",
                "propertyNames": {"pattern": "^[a-z]+$"},
                "additionalProperties": {"type": "integer"},
            },
        },
        "required": ["queryParams"],
    }
    assert isinstance(_gemma_compile(params), xgr.CompiledGrammar)

    ok = _gemma_matcher_params(params)
    assert _accept_ids(
        ok,
        _gemma_encode("<|tool_call>call:f{queryParams:{page:1}}<tool_call|>"),
    )
    assert ok.is_completed()

    # Integer-typed value: a string value must be rejected.
    bad = _gemma_matcher_params(params)
    assert not _accept_ids(
        bad, _gemma_encode('<|tool_call>call:f{queryParams:{page:<|"|>')
    )


def test_gemma_4_property_names_with_additional_properties_false_rejects_key() -> (
    None
):
    # propertyNames + additionalProperties:false with no declared properties:
    # every key is "additional" and forbidden, so only the empty object is
    # valid. A key that matches propertyNames is still forbidden by
    # additionalProperties:false and must be rejected.
    params = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "queryParams": {
                "type": "object",
                "propertyNames": {"pattern": "^[a-z]+$"},
                "additionalProperties": False,
            },
        },
        "required": ["queryParams"],
    }
    assert isinstance(_gemma_compile(params), xgr.CompiledGrammar)

    bad = _gemma_matcher_params(params)
    assert not _accept_ids(
        bad, _gemma_encode("<|tool_call>call:f{queryParams:{abc:")
    )


def _excludes_vocab_compiler() -> xgr.GrammarCompiler:
    # The shared _VOCAB has no c/d/e, and the exclude tries below need them.
    # Insert before "<eos>" so it stays the last entry, as stop_token_ids below
    # assumes.
    vocab = [*_VOCAB[:-1], "c", "d", "e", "x", _VOCAB[-1]]
    return xgr.GrammarCompiler(
        xgr.TokenizerInfo(
            vocab,
            vocab_type=xgr.VocabType.RAW,
            stop_token_ids=[len(vocab) - 1],
        )
    )


def test_excludes_overlapping_prefixes() -> None:
    # Back edges used to fall back only to the start state and its depth-one
    # children, losing any suffix of length >= 2. Following "abc" down the "abce"
    # branch forgot it had just read "bc", so "abcd" slipped past an exclude of
    # "bcd" -- letting the model emit exactly what the grammar forbids.
    tag = xgr.StructuralTag(format=AnyTextFormat(excludes=["bcd", "abce"]))
    compiled = _excludes_vocab_compiler().compile_structural_tag(tag)

    # "abcd" contains "bcd", so it must be rejected. This is the regression.
    assert not _accepts(compiled, "abcd")
    # The pattern that owns the branch is still excluded.
    assert not _accepts(compiled, "abce")
    # And a string containing neither is still accepted.
    assert _accepts(compiled, "abxe")


def test_anyof_base_additional_properties_not_folded_into_branch() -> None:
    # additionalProperties is defined against the names its OWN object declares,
    # so folding a copy into a branch closes over the wrong set. The base here
    # declares nothing and closes the object, so base AND branch admits only {};
    # the fold used to accept {"a": "a"}.
    with pytest.raises(Exception, match="forbids unlisted properties"):
        _compiler().compile_json_schema(
            '{"anyOf": [{"type": "object", "properties":'
            ' {"a": {"type": "string"}}}], "additionalProperties": false}',
            reject_unsupported=True,
        )


def test_anyof_base_additional_properties_over_own_names_compiles() -> None:
    # The rule is name-aware, not a ban on additionalProperties beside a union:
    # when the closing side declares the names itself, the fold is exact.
    compiled = _compiler().compile_json_schema(
        '{"anyOf": [{"required": ["a"]}], "type": "object", "properties":'
        ' {"a": {"type": "string"}}, "additionalProperties": false}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"a": "x"}')
    assert not _accepts(compiled, '{"a": "x", "b": 1}')


def test_anyof_open_base_still_folds() -> None:
    # additionalProperties: true names nothing and closes nothing, so there is
    # no boundary to misplace and the fold stays available.
    compiled = _compiler().compile_json_schema(
        '{"anyOf": [{"type": "object", "properties":'
        ' {"a": {"type": "string"}}}], "additionalProperties": true}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"a": "x"}')


def test_anyof_base_items_boundary_not_folded_into_branch() -> None:
    # Same defect on the array side: base "items": false closes over the base's
    # own prefixItems (none), so folding it into a branch that sets prefixItems
    # would let the branch's positional entries through.
    with pytest.raises(Exception, match="forbids extra items"):
        _compiler().compile_json_schema(
            '{"anyOf": [{"type": "array", "prefixItems":'
            ' [{"type": "string"}]}], "items": false}',
            reject_unsupported=True,
        )


def test_anyof_base_unevaluated_properties_not_folded() -> None:
    # unevaluatedProperties depends on which properties each applicator
    # evaluated, and a flattened branch no longer records that, so the fold
    # cannot represent it at all.
    with pytest.raises(Exception, match="unevaluatedProperties"):
        _compiler().compile_json_schema(
            '{"anyOf": [{"type": "object", "properties":'
            ' {"a": {"type": "string"}}}], "unevaluatedProperties": false}',
            reject_unsupported=True,
        )


def test_exclusive_keyword_not_folded_across_sides() -> None:
    # A base keyword folded beside a branch's const is dropped by the
    # dispatch, so the fold refuses it for either union spelling.
    for schema in (
        '{"anyOf": [{"const": "ab"}], "minLength": 5}',
        '{"oneOf": [{"const": "ab"}], "minLength": 5}',
    ):
        with pytest.raises(Exception, match='combines "const"'):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_exclusive_keyword_beside_sibling_union_rejected() -> None:
    # $ref, const and enum win the dispatch outright and the sibling union is
    # never looked at again, so its branches are dropped rather than enforced.
    # No fold-time guard can see this: ParseAnyOf never runs on that path.
    for schema in (
        '{"const": "ab", "anyOf": [{"const": "zz"}]}',
        '{"enum": ["ab"], "anyOf": [{"const": "zz"}]}',
        '{"$defs": {"n": {"type": "integer"}}, "$ref": "#/$defs/n",'
        ' "oneOf": [{"type": "string"}]}',
    ):
        with pytest.raises(Exception, match="takes dispatch priority"):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_ref_beside_same_object_sibling_rejected() -> None:
    # A sibling in the SAME object shares the $ref's source, so the cross-side
    # fold check cannot see it, yet the dispatch drops it just the same. Draft 7
    # ignores such a sibling and 2019-09 conjoins it, so neither reading is safe
    # to guess. The last shape arrives through a single-member allOf, where the
    # fold has nothing to compare against.
    for schema in (
        '{"$defs": {"x": {"type": "string"}}, "$ref": "#/$defs/x",'
        ' "maxLength": 3}',
        '{"$defs": {"x": {"type": "string"}}, "$ref": "#/$defs/x",'
        ' "pattern": "^zz$"}',
        '{"$defs": {"x": {"type": "string"}}, "$ref": "#/$defs/x",'
        ' "type": "integer"}',
        '{"$defs": {"i": {"type": "object", "properties":'
        ' {"a": {"type": "string"}}}},'
        ' "allOf": [{"$ref": "#/$defs/i", "minProperties": 1}]}',
    ):
        with pytest.raises(Exception, match="takes dispatch priority"):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_finite_value_beside_dropped_sibling_rejected() -> None:
    # ParseConst and ParseEnum read no sibling, so a sibling that rules out one
    # of the listed values is dropped on every draft, not just one. const also
    # outranks enum, so a schema setting both keeps only const.
    for schema in (
        '{"type": "string", "const": "abcd", "maxLength": 1}',
        '{"enum": ["a", "b"], "minLength": 5}',
        '{"type": "integer", "enum": ["a"]}',
        '{"const": "a", "enum": ["a", "b"]}',
    ):
        with pytest.raises(Exception, match="takes dispatch priority"):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_finite_value_beside_satisfied_sibling_compiles() -> None:
    cases = (
        ('{"type": "string", "enum": ["a", "ab"], "maxLength": 2}', '"ab"'),
        ('{"enum": ["ab"], "minLength": 2}', '"ab"'),
        (
            '{"type": "object", "enum": [{"a": 1}], "required": ["a"]}',
            '{"a":1}',
        ),
        # Two bytes, one character.
        ('{"enum": ["\u00e9"], "maxLength": 1}', '"\u00e9"'),
    )
    for schema, accepted in cases:
        compiled = _compiler().compile_json_schema(
            schema, reject_unsupported=True
        )
        assert _accepts(compiled, accepted), schema
        assert not _accepts(compiled, '"zzz"'), schema


def test_finite_value_beside_violated_sibling_rejected() -> None:
    for schema in (
        # One character, two bytes.
        '{"enum": ["\u00e9"], "minLength": 2}',
        '{"type": "object", "enum": [{}], "required": ["a"]}',
    ):
        with pytest.raises(Exception, match="takes dispatch priority"):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_finite_value_beside_malformed_sibling_rejected() -> None:
    for schema in (
        '{"enum": [1], "minLength": -1}',
        '{"enum": [1], "maxLength": -1}',
        '{"enum": ["a"], "minLength": -1}',
        '{"enum": [1], "required": [1]}',
        '{"enum": [{"a": 1}], "required": ["a", "a"]}',
        '{"enum": ["a"], "type": ["string", "string"]}',
    ):
        with pytest.raises(Exception, match="takes dispatch priority"):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_finite_value_sibling_walk_is_budgeted() -> None:
    repeats = 70000
    schema = (
        '{"enum": [1], "required": ['
        + ",".join(f'"a{i}"' for i in range(repeats))
        + "]}"
    )
    with pytest.raises(Exception, match="takes dispatch priority"):
        _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_finite_value_beside_undecidable_sibling_rejected() -> None:
    for schema in (
        '{"enum": ["a"], "pattern": "^a$"}',
        '{"enum": ["a"], "nullable": true}',
        '{"enum": [{"a": "xy"}],'
        ' "properties": {"a": {"type": "string", "maxLength": 2}}}',
        '{"type": "integer", "enum": [5, 7], "minimum": 5}',
    ):
        with pytest.raises(Exception, match="takes dispatch priority"):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_enum_with_annotation_only_sibling_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "enum": ["a"], "enumDescriptions": ["first"]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, '"b"')


def test_enum_with_sibling_type_still_compiles() -> None:
    # An enum beside a plain type keyword is the single most common schema shape
    # there is, and dropping a type that every enumerated value already
    # satisfies widens nothing.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "enum": ["a", "b"]}', reject_unsupported=True
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, '"c"')


def test_exclusive_keyword_with_annotation_siblings_compiles() -> None:
    # An annotation constrains no instance, and $defs is a container for what
    # the $ref names rather than a constraint on the instance, so neither
    # sibling can be dropped by the dispatch.
    compiled = _compiler().compile_json_schema(
        '{"$ref": "#/$defs/x", "description": "d",'
        ' "$defs": {"x": {"type": "string"}}}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert not _accepts(compiled, "1")

    compiled = _compiler().compile_json_schema(
        '{"const": 5, "title": "n"}', reject_unsupported=True
    )
    assert _accepts(compiled, "5")
    assert not _accepts(compiled, "6")


def test_allof_boundary_check_sees_through_union() -> None:
    # A member names a property through an applicator just as plainly as
    # through a sibling "properties". The allOf boundary check used to look
    # only at literal "properties" and merged straight past this.
    with pytest.raises(Exception, match="forbids unlisted properties"):
        _compiler().compile_json_schema(
            '{"allOf": [{"type": "object", "additionalProperties": false},'
            ' {"anyOf": [{"properties": {"a": {"type": "string"}}}]}]}',
            reject_unsupported=True,
        )


def test_excludes_match_beginning_inside_another_branch() -> None:
    # The output-link half, a different defect from the failure-link one above:
    # failure links decide where matching RESUMES, output links whether a pattern
    # ENDED. With excludes {"ab", "xaby"}, "xab" walks the "xaby" branch and never
    # lands on "ab"'s end state.
    tag = xgr.StructuralTag(format=AnyTextFormat(excludes=["ab", "xaby"]))
    compiled = _excludes_vocab_compiler().compile_structural_tag(tag)

    assert not _accepts(compiled, "xab")
    assert not _accepts(compiled, "ab")
    # A string containing neither excluded pattern is still accepted.
    assert _accepts(compiled, "xy")


def test_excludes_output_links_are_transitive() -> None:
    # The failure chain can pass through several states before reaching the
    # excluded end, so the propagation has to close over itself: "zxab" fails
    # to "xab", which is only a match because it fails to "ab".
    tag = xgr.StructuralTag(
        format=AnyTextFormat(excludes=["ab", "xaby", "zxabyy"])
    )
    compiled = _excludes_vocab_compiler().compile_structural_tag(tag)

    assert not _accepts(compiled, "zxab")
    assert not _accepts(compiled, "xab")
    assert _accepts(compiled, "zxy")


def test_allof_bare_ref_with_base_keyword_compiles() -> None:
    # Unlike the lone-ref case above, the base here carries a real keyword, so
    # the target and the base both become contributors. That is the conjunction
    # the merge used to refuse rather than inline.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "definitions": {"addr": {"type": "object",'
        ' "properties": {"street": {"type": "string"}},'
        ' "required": ["street"]}},'
        ' "allOf": [{"$ref": "#/definitions/addr"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"street": "a"}')
    assert not _accepts(compiled, "{}")


def test_allof_ref_plus_extension_unions_required() -> None:
    # The OpenAPI inheritance idiom. Both members name a required key, so the
    # merge has to union them rather than let one win.
    compiled = _compiler().compile_json_schema(
        '{"definitions": {"addr": {"type": "object", "properties":'
        ' {"street": {"type": "string"}}, "required": ["street"]}},'
        ' "allOf": [{"$ref": "#/definitions/addr"},'
        ' {"properties": {"zip": {"type": "string"}}, "required": ["zip"]}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"street": "a", "zip": "1"}')
    assert not _accepts(compiled, '{"street": "a"}')
    assert not _accepts(compiled, '{"zip": "1"}')


def test_allof_nested_and_flat_spellings_agree() -> None:
    # Wrapping a member in its own allOf says nothing new, so the grammar must not
    # change. The flatten is one pass for this reason: a second pass merges the
    # nested member into an already-merged object, and since the object grammar
    # fixes one key order, the two spellings would demand different orders.
    flat = (
        '{"type": "object", "allOf": ['
        '{"properties": {"a": {"type": "string"}}, "required": ["a"]},'
        '{"properties": {"b": {"type": "string"}}, "required": ["b"]}]}'
    )
    nested = (
        '{"type": "object", "allOf": ['
        '{"allOf": [{"properties": {"a": {"type": "string"}},'
        ' "required": ["a"]}]},'
        '{"properties": {"b": {"type": "string"}}, "required": ["b"]}]}'
    )
    doc = '{"a": "x", "b": "y"}'
    for schema in (flat, nested):
        compiled = _compiler().compile_json_schema(
            schema, reject_unsupported=True
        )
        assert _accepts(compiled, doc)


def test_allof_recursive_ref_refused() -> None:
    # Inlining a definition that reaches itself does not terminate.
    with pytest.raises(Exception, match="reaches itself"):
        _compiler().compile_json_schema(
            '{"definitions": {"node": {"type": "object",'
            ' "allOf": [{"$ref": "#/definitions/node"}]}},'
            ' "allOf": [{"$ref": "#/definitions/node"}]}',
            reject_unsupported=True,
        )


def test_allof_ref_with_sibling_constraint_still_refused() -> None:
    # Only a BARE $ref inlines. A $ref carrying a sibling constraint is the shape
    # where draft 7 ignores the sibling and 2019-09 conjoins it.
    with pytest.raises(Exception, match='combines "\\$ref"'):
        _compiler().compile_json_schema(
            '{"definitions": {"addr": {"type": "object", "properties":'
            ' {"street": {"type": "string"}}}},'
            ' "allOf": [{"$ref": "#/definitions/addr", "minProperties": 2},'
            ' {"properties": {"zip": {"type": "string"}}}]}',
            reject_unsupported=True,
        )


# Two definitions whose names collide unless RFC 6901 escapes are decoded: the
# pointer "#/$defs/a~1b" names "a/b", and "#/$defs/a~01b" names "a~1b".
_ESCAPED_KEY_DEFS = (
    '"$defs": {"a/b": {"type": "object",'
    ' "properties": {"x": {"type": "string"}}, "required": ["x"]},'
    ' "a~1b": {"type": "object",'
    ' "properties": {"y": {"type": "string"}}, "required": ["y"]}}'
)


def test_allof_ref_pointer_decodes_escaped_slash() -> None:
    # "~1" is "/", so the inlined target is the "a/b" definition. Comparing the
    # token literally would inline "a~1b" and enforce the wrong schema.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", '
        + _ESCAPED_KEY_DEFS
        + ', "allOf": [{"$ref": "#/$defs/a~1b"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"x": "a"}')
    assert not _accepts(compiled, '{"y": "a"}')


def test_allof_ref_pointer_decodes_escaped_tilde() -> None:
    # The decoy of the test above: "~0" is "~", so "a~01b" names the literal
    # "a~1b" definition, the one the undecoded comparison would have picked.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", '
        + _ESCAPED_KEY_DEFS
        + ', "allOf": [{"$ref": "#/$defs/a~01b"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"y": "a"}')
    assert not _accepts(compiled, '{"x": "a"}')


def test_allof_ref_pointer_empty_token_names_empty_key() -> None:
    # An empty reference token names the "" member, so "#/$defs/" is a pointer
    # two levels deep, not a one-level pointer with a stray separator.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "$defs": {"": {"type": "object",'
        ' "properties": {"x": {"type": "string"}}, "required": ["x"]}},'
        ' "allOf": [{"$ref": "#/$defs/"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"x": "a"}')


def test_allof_ref_pointer_malformed_escape_refused() -> None:
    # A "~" carries "0" or "1" and nothing else, so "a~2b" resolves to no
    # target and the $ref survives the merge for the dispatch check to refuse.
    with pytest.raises(Exception, match='combines "\\$ref"'):
        _compiler().compile_json_schema(
            '{"type": "object", '
            + _ESCAPED_KEY_DEFS
            + ', "allOf": [{"$ref": "#/$defs/a~2b"}]}',
            reject_unsupported=True,
        )


@pytest.mark.parametrize("ref", ["#/x%", "#/x%GG"])
def test_allof_malformed_percent_ref_rejected(ref: str) -> None:
    with pytest.raises(
        Exception,
        match=r"malformed percent escape.*must be followed by two hexadecimal digits",
    ):
        _compiler().compile_json_schema(
            json.dumps({"type": "object", "allOf": [{"$ref": ref}]}),
            reject_unsupported=True,
        )


def test_allof_ref_pointer_through_array_refused() -> None:
    # The walk traverses object members only, so a pointer that indexes an array
    # resolves to no target rather than to a guessed element.
    with pytest.raises(Exception, match='combines "\\$ref"'):
        _compiler().compile_json_schema(
            '{"type": "object", "$defs": {"list": [{"type": "object",'
            ' "properties": {"x": {"type": "string"}}, "required": ["x"]}]},'
            ' "allOf": [{"$ref": "#/$defs/list/0"}]}',
            reject_unsupported=True,
        )


def test_allof_ref_under_nested_id_refused() -> None:
    # An "$id" below the root starts a new base URI, and a "#/..." $ref then
    # points into that resource rather than into the document. The merge does
    # not track base URIs, so it refuses instead of resolving against the root.
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"type": "object", "$defs": {"s": {"$id": "https://example.com/s",'
            ' "type": "object", "properties": {"x": {"type": "string"}},'
            ' "required": ["x"]}},'
            ' "allOf": [{"$ref": "#/$defs/s"}]}',
            reject_unsupported=True,
        )


def test_allof_ref_declared_under_nested_id_refused() -> None:
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"$defs": {"x": {"type": "integer"},'
            ' "sub": {"$id": "https://example.com/sub",'
            ' "$defs": {"x": {"type": "string"}},'
            ' "allOf": [{"$ref": "#/$defs/x"}]}},'
            ' "$ref": "#/$defs/sub"}',
            reject_unsupported=True,
        )


def test_allof_sibling_id_does_not_rebase_a_contributor() -> None:
    # The flatten merges every contributor key into one object, "$id" included,
    # so a sibling's identifier would otherwise scope references the second
    # contributor inlined from the root resource. Refusing the document keeps
    # the merge from having to carry a base URI it never computed.
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"$defs": {"x": {"type": "string"},'
            ' "obj": {"type": "object",'
            ' "properties": {"v": {"$ref": "#/$defs/x"}},'
            ' "required": ["v"]}},'
            ' "allOf": [{"$id": "https://example.com/sub", "type": "object"},'
            ' {"$ref": "#/$defs/obj"}]}',
            reject_unsupported=True,
        )


def test_allof_ref_beside_id_refused() -> None:
    # "$id" on the member holding the $ref rebases that $ref itself, which the
    # bare-ref test would otherwise wave through as a mere annotation.
    with pytest.raises(
        Exception, match="declares a resource below the document root"
    ):
        _compiler().compile_json_schema(
            '{"type": "object", "$defs": {"s": {"type": "object",'
            ' "properties": {"x": {"type": "string"}}, "required": ["x"]}},'
            ' "allOf": [{"$ref": "#/$defs/s",'
            ' "$id": "https://example.com/other"}]}',
            reject_unsupported=True,
        )


def test_allof_ref_beside_malformed_id_rejected() -> None:
    with pytest.raises(Exception, match='"\\$id" must be a string'):
        _compiler().compile_json_schema(
            '{"$defs": {"s": {"type": "string"}},'
            ' "allOf": [{"$id": {}, "$ref": "#/$defs/s"}]}',
            reject_unsupported=True,
        )


def test_allof_annotation_only_malformed_id_rejected() -> None:
    with pytest.raises(Exception, match='"\\$id" must be a string'):
        _compiler().compile_json_schema(
            '{"type": "string", "allOf": [{"$id": {}}]}',
            reject_unsupported=True,
        )


def test_allof_ref_under_root_id_still_inlines() -> None:
    # The root's own "$id" names the document, so a "#/..." $ref still points
    # into it. Refusing this shape would reject the common annotated schema.
    compiled = _compiler().compile_json_schema(
        '{"$id": "https://example.com/root", "type": "object",'
        ' "$defs": {"s": {"type": "object",'
        ' "properties": {"x": {"type": "string"}}, "required": ["x"]}},'
        ' "allOf": [{"$ref": "#/$defs/s"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"x": "a"}')
    assert not _accepts(compiled, "{}")


def test_allof_empty_schema_does_not_close_the_object() -> None:
    # `additionalProperties: {}` imposes nothing, so it does not close the object
    # and the merge may proceed. Only `false` closes it.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "additionalProperties": {},'
        ' "allOf": [{"properties": {"a": {"type": "string"}}}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"a": "x"}')
    with pytest.raises(Exception, match="forbids unlisted properties"):
        _compiler().compile_json_schema(
            '{"type": "object", "additionalProperties": false,'
            ' "allOf": [{"properties": {"a": {"type": "string"}}}]}',
            reject_unsupported=True,
        )


def test_oneof_nullable_string_with_length_bound() -> None:
    # A declared type bounds the branch while its other constraints still apply.
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"type": "string", "maxLength": 4}, {"type": "null"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"abcd"')
    assert _accepts(compiled, "null")
    assert not _accepts(compiled, '"abcde"')
    assert not _accepts(compiled, "1")


def test_oneof_nullable_object() -> None:
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"type": "object", "properties": {"a": {"type": "string"}},'
        ' "required": ["a"]}, {"type": "null"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '{"a": "x"}')
    assert _accepts(compiled, "null")
    assert not _accepts(compiled, "{}")
    assert not _accepts(compiled, "1")


def test_oneof_same_type_arms_refused() -> None:
    # Bounds are not part of the disjointness proof.
    with pytest.raises(Exception, match="non-provably-disjoint"):
        _compiler().compile_json_schema(
            '{"oneOf": [{"type": "string", "maxLength": 4},'
            ' {"type": "string", "minLength": 6}]}',
            reject_unsupported=True,
        )


def test_oneof_branch_without_declared_type_refused() -> None:
    # `properties` also admits every non-object, so it does not imply `object`.
    with pytest.raises(Exception, match="non-provably-disjoint"):
        _compiler().compile_json_schema(
            '{"oneOf": [{"properties": {"a": {"type": "string"}}},'
            ' {"type": "null"}]}',
            reject_unsupported=True,
        )


def test_union_base_string_tier_judged_across_sides() -> None:
    for schema in (
        '{"anyOf": [{"type": "string", "pattern": "^a+$"},'
        ' {"type": "null"}], "maxLength": 1}',
        '{"oneOf": [{"type": "string", "maxLength": 1},'
        ' {"type": "null"}], "pattern": "^a+$"}',
    ):
        with pytest.raises(
            Exception, match="each admit lengths the other forbids"
        ):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_union_base_pattern_within_length_bound_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"type": "string", "pattern": "^a$"},'
        ' {"type": "null"}], "maxLength": 10}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(compiled, "null")
    assert not _accepts(compiled, '"aa"')


def test_union_base_length_bound_folds_without_a_pattern() -> None:
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"type": "string"}, {"type": "null"}], "maxLength": 1}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(compiled, "null")
    assert not _accepts(compiled, '"aa"')


def test_oneof_finite_arm_with_dropped_sibling_refused() -> None:
    for schema in (
        '{"oneOf": [{"const": "a", "type": "number"}, {"const": "b"}]}',
        '{"oneOf": [{"enum": ["a"], "minLength": 5}, {"const": "b"}]}',
        '{"oneOf": [{"const": "a", "enum": ["b"]}, {"const": "b"}]}',
    ):
        with pytest.raises(Exception, match="non-provably-disjoint"):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


def test_oneof_discriminator_with_dropped_sibling_refused() -> None:
    with pytest.raises(Exception, match="non-provably-disjoint"):
        _compiler().compile_json_schema(
            '{"oneOf": ['
            '{"type": "object", "required": ["k"],'
            ' "properties": {"k": {"const": "a", "type": "number"}}}, '
            '{"type": "object", "required": ["k"],'
            ' "properties": {"k": {"const": "b"}}}]}',
            reject_unsupported=True,
        )


def test_oneof_finite_arm_with_redundant_siblings_compiles() -> None:
    compiled = _compiler().compile_json_schema(
        '{"oneOf": [{"const": "a", "type": "string", "title": "A"},'
        ' {"const": 1, "type": "integer"}]}',
        reject_unsupported=True,
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(compiled, "1")
    assert not _accepts(compiled, '"b"')
