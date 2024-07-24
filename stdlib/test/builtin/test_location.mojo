# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s
# RUN: %mojo -debug-level full %s

from builtin._location import (
    __call_location,
    __source_location,
    _SourceLocation,
)
from testing import assert_equal, assert_true


fn check_source_loc(line: Int, col: Int, source_loc: _SourceLocation) raises:
    """Utility function to help writing source location tests."""
    assert_equal(source_loc.line, line)
    assert_equal(source_loc.col, col)
    assert_true(String(source_loc.file_name).endswith("test_location.mojo"))


fn get_locs() -> (_SourceLocation, _SourceLocation):
    return (
        __source_location(),
        source_loc_with_debug(),
    )


@always_inline
fn get_locs_inlined() -> (_SourceLocation, _SourceLocation):
    return (
        __source_location(),
        source_loc_with_debug(),
    )


fn get_four_locs() -> (
    _SourceLocation,
    _SourceLocation,
    _SourceLocation,
    _SourceLocation,
):
    var p1 = get_locs()
    var p2 = get_locs_inlined()
    return (p1[0], p1[1], p2[0], p2[1])


@always_inline
fn get_four_locs_inlined() -> (
    _SourceLocation,
    _SourceLocation,
    _SourceLocation,
    _SourceLocation,
):
    var p1 = get_locs()
    var p2 = get_locs_inlined()
    return (p1[0], p1[1], p2[0], p2[1])


fn test_builtin_source_loc() raises:
    var source_loc = __source_location()
    check_source_loc(70, 39, source_loc)
    check_source_loc(72, 47, __source_location())

    var l = (33, 34, 41, 42)
    var c = (26, 30, 26, 30)
    var loc_pair = get_locs()
    check_source_loc(l[0], c[0], loc_pair[0])
    check_source_loc(l[1], c[1], loc_pair[1])

    loc_pair = get_locs_inlined()
    check_source_loc(l[2], c[2], loc_pair[0])
    check_source_loc(l[3], c[3], loc_pair[1])

    var loc_quad = get_four_locs()
    check_source_loc(l[0], c[0], loc_quad[0])
    check_source_loc(l[1], c[1], loc_quad[1])
    check_source_loc(l[2], c[2], loc_quad[2])
    check_source_loc(l[3], c[3], loc_quad[3])

    loc_quad = get_four_locs_inlined()
    check_source_loc(l[0], c[0], loc_quad[0])
    check_source_loc(l[1], c[1], loc_quad[1])
    check_source_loc(l[2], c[2], loc_quad[2])
    check_source_loc(l[3], c[3], loc_quad[3])


fn get_inner_location_statically() -> _SourceLocation:
    return __source_location()


fn get_inner_location_statically_with_debug() -> _SourceLocation:
    return source_loc_with_debug()


@always_inline("nodebug")
fn get_callsite_statically() -> _SourceLocation:
    return __call_location()


fn test_parameter_context() raises:
    # TODO: enable these in parameter contexts
    alias sloc = __source_location()
    assert_equal(sloc.line, 0)
    assert_equal(sloc.col, 0)
    assert_equal(sloc.file_name, "<unknown location in parameter context>")

    alias cloc = get_callsite_statically()
    assert_equal(cloc.line, 0)
    assert_equal(cloc.col, 0)
    assert_equal(cloc.file_name, "<unknown location in parameter context>")

    alias iloc = get_inner_location_statically()
    check_source_loc(98, 29, iloc)
    alias iloc2 = get_inner_location_statically_with_debug()
    check_source_loc(102, 33, iloc2)


@always_inline
fn capture_call_loc(cond: Bool = False) -> _SourceLocation:
    if (
        not cond
    ):  # NOTE: we test that __call_location works even in a nested scope.
        return __call_location()
    return _SourceLocation(-1, -1, "")


@always_inline("nodebug")
fn capture_call_loc_nodebug(cond: Bool = False) -> _SourceLocation:
    if (
        not cond
    ):  # NOTE: we test that __call_location works even in a nested scope.
        return __call_location()
    return _SourceLocation(-1, -1, "")


fn get_call_locs() -> (_SourceLocation, _SourceLocation):
    return (
        capture_call_loc(),
        capture_call_loc_nodebug(),
    )


@always_inline("nodebug")
fn get_call_locs_inlined() -> (_SourceLocation, _SourceLocation):
    return (
        capture_call_loc(),
        capture_call_loc_nodebug(),
    )


fn get_four_call_locs() -> (
    _SourceLocation,
    _SourceLocation,
    _SourceLocation,
    _SourceLocation,
):
    var p1 = get_call_locs()
    var p2 = get_call_locs_inlined()
    return (p1[0], p1[1], p2[0], p2[1])


@always_inline
fn get_four_call_locs_inlined() -> (
    _SourceLocation,
    _SourceLocation,
    _SourceLocation,
    _SourceLocation,
):
    var p1 = get_call_locs()
    var p2 = get_call_locs_inlined()
    return (p1[0], p1[1], p2[0], p2[1])


fn test_builtin_call_loc() raises:
    var l = (148, 149, 156, 157)
    var c = (25, 33, 25, 33)
    var loc_pair = get_call_locs()
    check_source_loc(l[0], c[0], loc_pair[0])
    check_source_loc(l[1], c[1], loc_pair[1])

    loc_pair = get_call_locs_inlined()
    check_source_loc(l[2], c[2], loc_pair[0])
    check_source_loc(l[3], c[3], loc_pair[1])

    var loc_quad = get_four_call_locs()
    check_source_loc(l[0], c[0], loc_quad[0])
    check_source_loc(l[1], c[1], loc_quad[1])
    check_source_loc(l[2], c[2], loc_quad[2])
    check_source_loc(l[3], c[3], loc_quad[3])

    loc_quad = get_four_call_locs_inlined()
    check_source_loc(l[0], c[0], loc_quad[0])
    check_source_loc(l[1], c[1], loc_quad[1])
    check_source_loc(l[2], c[2], loc_quad[2])
    check_source_loc(l[3], c[3], loc_quad[3])


@always_inline
fn source_loc_with_debug() -> _SourceLocation:
    var line: __mlir_type.index
    var col: __mlir_type.index
    var file_name: __mlir_type.`!kgen.string`
    line, col, file_name = __mlir_op.`kgen.source_loc`[
        _properties = __mlir_attr.`{inlineCount = 0 : i64}`,
        _type = (
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ),
    ]()

    return _SourceLocation(line, col, file_name)


fn test_source_location_struct() raises:
    var source_loc = _SourceLocation(50, 60, "/path/to/some_file.mojo")
    assert_equal(str(source_loc), "/path/to/some_file.mojo:50:60")


fn main() raises:
    test_source_location_struct()
    test_builtin_source_loc()
    test_parameter_context()
    test_builtin_call_loc()
