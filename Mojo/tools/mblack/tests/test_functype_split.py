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

# Tests for splitting long function-type expressions (e.g. FFI signatures like
# `def() thin abi("C") -> ...`). A `named_effect`'s parens stay inline; a long
# signature wraps at the parameter or return-type bracket instead.

from tests.util import assert_mojo_format


def test_ffi_functype_thin_abi_is_stable():
    """A thin C-ABI FFI signature formats unchanged, with `abi("C")` inline."""
    source = (
        "from std.ffi import c_char\n"
        "\n"
        "\n"
        "comptime version_query_callback = Optional[\n"
        '    def() thin abi("C") -> UnsafePointer[\n'
        "        UnsafePointer[c_char, ImmutAnyOrigin], ImmutAnyOrigin\n"
        "    ]\n"
        "]\n"
    )
    assert_mojo_format(source, source)


def test_long_functype_splits_return_type_not_named_effect():
    """A too-long functype splits at the return-type bracket, `abi("C")` intact.

    The named effect's parenthesized string must stay on one line rather than
    being chosen as the split point.
    """
    source = (
        "from std.ffi import c_char\n"
        "\n"
        "\n"
        "comptime version_query_callback = Optional[\n"
        '    def() thin abi("C") -> UnsafePointer['
        "UnsafePointer[c_char, ImmutAnyOrigin], ImmutAnyOrigin]\n"
        "]\n"
    )
    expected = (
        "from std.ffi import c_char\n"
        "\n"
        "\n"
        "comptime version_query_callback = Optional[\n"
        '    def() thin abi("C") -> UnsafePointer[\n'
        "        UnsafePointer[c_char, ImmutAnyOrigin], ImmutAnyOrigin\n"
        "    ]\n"
        "]\n"
    )
    assert_mojo_format(source, expected)


def test_async_ffi_functype_is_stable():
    """An async function type also keeps its named effect inline."""
    source = (
        "from std.ffi import c_char\n"
        "\n"
        "\n"
        "comptime version_query_callback = Optional[\n"
        '    async def() thin abi("C") -> UnsafePointer[\n'
        "        UnsafePointer[c_char, ImmutAnyOrigin], ImmutAnyOrigin\n"
        "    ]\n"
        "]\n"
    )
    assert_mojo_format(source, source)


def test_unsplittable_functype_left_intact():
    """A too-long functype whose only bracket is the effect is left intact.

    The empty parameter list and non-bracketed return type leave only the named
    effect's parens, which must stay inline, so the over-long line is emitted
    unchanged.
    """
    source = (
        "comptime AVeryLongReturnTypeAliasNameToForceThisFunctionSignatureWellOverTheColumnLimit = Float64\n"
        "\n"
        "\n"
        "comptime f = Optional[\n"
        '    def() thin abi("C") ->'
        " AVeryLongReturnTypeAliasNameToForceThisFunctionSignatureWellOverTheColumnLimit\n"
        "]\n"
    )
    assert_mojo_format(source, source)


def test_functype_with_params_splits_at_parameter_list():
    """A functype with a plain (non-bracketed) return type splits at its params.

    The parameter list, not the `abi("C")` effect, is the split point, so the
    C-ABI signature stays readable (an FFI-heavy file like `_cpython.mojo`
    relies on this).
    """
    source = (
        "comptime some_math_function = Optional[\n"
        "    def(Float64, Float64, Float64, Float64, Float64, Float64)"
        ' thin abi("C") -> Float64\n'
        "]\n"
    )
    expected = (
        "comptime some_math_function = Optional[\n"
        "    def(\n"
        "        Float64, Float64, Float64, Float64, Float64, Float64\n"
        '    ) thin abi("C") -> Float64\n'
        "]\n"
    )
    assert_mojo_format(source, expected)


def test_real_def_still_splits_at_parameters():
    """Regression guard: a genuine `def` with no effect splits at its params."""
    source = (
        "def a_function_with_a_fairly_long_name("
        "argument_one: Int, argument_two: Int, arg3: Int) -> Int:\n"
        "    return argument_one\n"
    )
    expected = (
        "def a_function_with_a_fairly_long_name(\n"
        "    argument_one: Int, argument_two: Int, arg3: Int\n"
        ") -> Int:\n"
        "    return argument_one\n"
    )
    assert_mojo_format(source, expected)


def test_real_def_with_named_effect_and_empty_params():
    """A statement-level `def` with a named effect keeps the effect inline too.

    `named_effect` appears on both function types and real definitions, so a
    genuine `def foo() abi("C") -> ...:` header wraps at the return type.
    """
    source = (
        "def some_ffi_wrapper_function_with_a_long_name()"
        ' abi("C") -> SIMD[DType.float64, 4]:\n'
        "    return SIMD[DType.float64, 4](0)\n"
    )
    expected = (
        "def some_ffi_wrapper_function_with_a_long_name() abi(\"C\") -> (\n"
        "    SIMD[DType.float64, 4]\n"
        "):\n"
        "    return SIMD[DType.float64, 4](0)\n"
    )
    assert_mojo_format(source, expected)
