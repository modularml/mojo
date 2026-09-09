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
#


from std.reflection import (
    call_location,
    source_location,
    SourceLocation,
)
from std.memory import MaybeUninit
from std.sys import size_of
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def check_source_loc(line: Int, col: Int, source_loc: SourceLocation) raises:
    """Utility function to help writing source location tests."""
    assert_equal(source_loc.line(), line)
    assert_equal(source_loc.column(), col)
    assert_true(String(source_loc.file_name()).endswith("test_location.mojo"))


def get_locs() -> Tuple[SourceLocation, SourceLocation]:
    return (
        source_location(),
        source_loc_with_debug(),
    )


@always_inline
def get_locs_inlined() -> Tuple[SourceLocation, SourceLocation]:
    return (
        source_location(),
        source_loc_with_debug(),
    )


def get_four_locs() -> (
    Tuple[
        SourceLocation,
        SourceLocation,
        SourceLocation,
        SourceLocation,
    ]
):
    var p1 = get_locs()
    var p2 = get_locs_inlined()
    return (p1[0], p1[1], p2[0], p2[1])


@always_inline
def get_four_locs_inlined() -> (
    Tuple[
        SourceLocation,
        SourceLocation,
        SourceLocation,
        SourceLocation,
    ]
):
    var p1 = get_locs()
    var p2 = get_locs_inlined()
    return (p1[0], p1[1], p2[0], p2[1])


def test_builtin_source_loc() raises:
    var source_loc = source_location()
    check_source_loc(76, 37, source_loc)
    check_source_loc(78, 45, source_location())

    var l = (35, 36, 43, 44)
    var c = (24, 30, 24, 30)
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


def get_inner_location_statically() -> SourceLocation:
    return source_location()


def get_inner_location_statically_with_debug() -> SourceLocation:
    return source_loc_with_debug()


@always_inline("nodebug")
def get_callsite_statically() -> SourceLocation:
    return call_location()


def test_parameter_context() raises:
    # TODO: enable these in parameter contexts
    comptime sloc = source_location()
    assert_equal(sloc.line(), 0)
    assert_equal(sloc.column(), 0)
    assert_equal(sloc.file_name(), "<unknown location in parameter context>")

    comptime cloc = get_callsite_statically()
    assert_equal(cloc.line(), 0)
    assert_equal(cloc.column(), 0)
    assert_equal(cloc.file_name(), "<unknown location in parameter context>")

    comptime iloc = get_inner_location_statically()
    check_source_loc(104, 27, iloc)
    comptime iloc2 = get_inner_location_statically_with_debug()
    check_source_loc(108, 33, iloc2)


@always_inline
def capture_call_loc[depth: Int = 1](cond: Bool = False) -> SourceLocation:
    if (
        not cond
    ):  # NOTE: we test that call_location works even in a nested scope.
        return call_location[inline_count=depth]()
    return SourceLocation(-1, -1, "")


@always_inline("nodebug")
def capture_call_loc_nodebug[
    depth: Int = 1
](cond: Bool = False) -> SourceLocation:
    if (
        not cond
    ):  # NOTE: we test that call_location works even in a nested scope.
        return call_location[inline_count=depth]()
    return SourceLocation(-1, -1, "")


def get_call_locs() -> Tuple[SourceLocation, SourceLocation]:
    return (
        capture_call_loc(),
        capture_call_loc_nodebug(),
    )


@always_inline("nodebug")
def get_call_locs_inlined[
    depth: Int = 1
]() -> Tuple[SourceLocation, SourceLocation]:
    return (
        capture_call_loc[depth](),
        capture_call_loc_nodebug[depth](),
    )


@always_inline
def get_call_locs_inlined_twice[
    depth: Int = 1
]() -> Tuple[SourceLocation, SourceLocation]:
    return get_call_locs_inlined[depth]()


def get_four_call_locs() -> (
    Tuple[
        SourceLocation,
        SourceLocation,
        SourceLocation,
        SourceLocation,
    ]
):
    var p1 = get_call_locs()
    var p2 = get_call_locs_inlined()
    return (p1[0], p1[1], p2[0], p2[1])


@always_inline
def get_four_call_locs_inlined() -> (
    Tuple[
        SourceLocation,
        SourceLocation,
        SourceLocation,
        SourceLocation,
    ]
):
    var p1 = get_call_locs()
    var p2 = get_call_locs_inlined()
    return (p1[0], p1[1], p2[0], p2[1])


def test_builtin_call_loc() raises:
    var l = (156, 157, 166, 167)
    var c = (25, 33, 32, 40)
    var loc_pair = get_call_locs()
    check_source_loc(l[0], c[0], loc_pair[0])
    check_source_loc(l[1], c[1], loc_pair[1])

    loc_pair = get_call_locs_inlined()
    check_source_loc(l[2], c[2], loc_pair[0])
    check_source_loc(l[3], c[3], loc_pair[1])

    loc_pair = get_call_locs_inlined_twice[2]()
    check_source_loc(175, 40, loc_pair[0])
    check_source_loc(175, 40, loc_pair[1])

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
def source_loc_with_debug() -> SourceLocation:
    var line: __mlir_type.index
    var col: __mlir_type.index
    var file_name: __mlir_type.`!kgen.string`
    line, col, file_name = __mlir_op.`kgen.source_loc`[
        inlineCount=Int(0).__mlir_index__(),
        _type=Tuple[
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ],
    ]()

    return SourceLocation(
        SIMDLength(mlir_value=line),
        SIMDLength(mlir_value=col),
        StaticString(file_name),
    )


def test_source_location_struct() raises:
    var source_loc = SourceLocation(50, 60, "/path/to/some_file.mojo")
    assert_equal(String(source_loc), "/path/to/some_file.mojo:50:60")


def test_source_location_niche() raises:
    assert_equal(SourceLocation.niche_count(), 1)
    assert_equal(size_of[SourceLocation](), size_of[Optional[SourceLocation]]())

    var storage = MaybeUninit[SourceLocation]()

    SourceLocation.write_niche(Pointer(to=storage))
    assert_true(SourceLocation.isa_niche(Pointer(to=storage)))

    storage.unsafe_write(SourceLocation(50, 60, "/path/to/some_file.mojo"))
    assert_false(SourceLocation.isa_niche(Pointer(to=storage)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
