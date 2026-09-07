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

from tests.util import assert_mojo_format, mojo_format_str


def test_simple_parametric_alias():
    """Test basic parametric alias formatting."""
    source = "comptime addOne[x: Int] : Int = x + 1"
    expected = "comptime addOne[x: Int]: Int = x + 1\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_with_default_values():
    """Test parametric alias with default parameter values."""
    source = "comptime Float64[size: Int = 1] = SIMD[DType.float64, size]"
    expected = "comptime Float64[size: Int = 1] = SIMD[DType.float64, size]\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_with_multiple_parameters():
    """Test parametric alias with multiple parameters."""
    source = "comptime TwoParams[a: Int, b: Int] = SIMD[DType.float32, a + b]"
    expected = "comptime TwoParams[a: Int, b: Int] = SIMD[DType.float32, a + b]\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_with_complex_expression():
    """Test parametric alias with complex expression in body."""
    source = "comptime ComplexExpr[x: Int, y: Int] : Int = (x * y) + (x + y)"
    expected = "comptime ComplexExpr[x: Int, y: Int]: Int = (x * y) + (x + y)\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_with_conditional_expression():
    """Test parametric alias with conditional expression."""
    source = "comptime ConditionalAlias[dt: DType, size: Int] = SIMD[dt, size if size > 0 else 1]"
    expected = "comptime ConditionalAlias[dt: DType, size: Int] = SIMD[\n    dt, size if size > 0 else 1\n]\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_with_nested_expressions():
    """Test parametric alias with nested expressions."""
    source = "comptime NestedAlias[dt: DType] = Scalar[dt]"
    expected = "comptime NestedAlias[dt: DType] = Scalar[dt]\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_with_mixed_defaults():
    """Test parametric alias with mixed default and non-default parameters."""
    source = "comptime MixedDefaults[a: Int, b: Int = 0, c: Int = 1] : Int = a + b + c"
    expected = "comptime MixedDefaults[a: Int, b: Int = 0, c: Int = 1]: Int = a + b + c\n"
    assert_mojo_format(source, expected)


def test_long_parametric_alias_line_breaking():
    """Test that long parametric aliases are properly line-broken."""
    source = (
        "comptime ComplexType[a: Int, b: String, c: Float64] = Int\n"
        "comptime ComplexExpression[a: Int, b: String, c: Float64, d: Bool,"
        " e: DType, f: Int, g: String, h: Float64] = 0\n"
        "\n"
        "comptime VeryLongParametricAliasWithManyParameters["
        "first_param: Int, second_param: String, third_param: Float64, "
        "fourth_param: Bool, fifth_param: DType, sixth_param: Int = 42, "
        'seventh_param: String = "default", eighth_param: Float64 = 3.14]: '
        "ComplexType[first_param, second_param, third_param] = ComplexExpression["
        "first_param, second_param, third_param, fourth_param, "
        "fifth_param, sixth_param, seventh_param, eighth_param]"
    )
    expected = (
        "comptime ComplexType[a: Int, b: String, c: Float64] = Int\n"
        "comptime ComplexExpression[\n"
        "    a: Int,\n"
        "    b: String,\n"
        "    c: Float64,\n"
        "    d: Bool,\n"
        "    e: DType,\n"
        "    f: Int,\n"
        "    g: String,\n"
        "    h: Float64,\n"
        "] = 0\n"
        "\n"
        "comptime VeryLongParametricAliasWithManyParameters[\n"
        "    first_param: Int,\n"
        "    second_param: String,\n"
        "    third_param: Float64,\n"
        "    fourth_param: Bool,\n"
        "    fifth_param: DType,\n"
        "    sixth_param: Int = 42,\n"
        '    seventh_param: String = "default",\n'
        "    eighth_param: Float64 = 3.14,\n"
        "]: ComplexType[first_param, second_param, third_param] = ComplexExpression[\n"
        "    first_param,\n"
        "    second_param,\n"
        "    third_param,\n"
        "    fourth_param,\n"
        "    fifth_param,\n"
        "    sixth_param,\n"
        "    seventh_param,\n"
        "    eighth_param,\n"
        "]\n"
    )
    assert_mojo_format(source, expected)


def test_parametric_alias_with_function_calls_line_breaking():
    """Test parametric alias with function calls that need line breaking."""
    source = (
        "def some_function(a: Int, b: Int, c: Int) -> Int:\n"
        "    return a\n"
        "\n"
        "comptime FunctionCallAlias[param: Int]: Int = "
        "some_function(param, param * 2, param + 1)"
    )
    expected = (
        "def some_function(a: Int, b: Int, c: Int) -> Int:\n"
        "    return a\n"
        "\n"
        "\n"
        "comptime FunctionCallAlias[param: Int]: Int = some_function(\n"
        "    param, param * 2, param + 1\n"
        ")\n"
    )
    assert_mojo_format(source, expected)


def test_parametric_alias_with_complex_arithmetic_line_breaking():
    """Test parametric alias with complex arithmetic that needs line breaking."""
    source = (
        "comptime ComplexArithmeticAlias[a: Int, b: Int, c: Int]: Int = "
        "(a * b) + (b * c) + (c * a)"
    )
    expected = (
        "comptime ComplexArithmeticAlias[a: Int, b: Int, c: Int]: Int = (a * b) + (\n"
        "    b * c\n"
        ") + (c * a)\n"
    )
    assert_mojo_format(source, expected)


def test_parametric_alias_with_conditional_line_breaking():
    """Test parametric alias with conditional expressions that need line breaking."""
    source = (
        "comptime ConditionalParametricAlias[condition: Bool, value: Int]: Int = "
        "value if condition else 0"
    )
    expected = (
        "comptime ConditionalParametricAlias[\n"
        "    condition: Bool, value: Int\n"
        "]: Int = value if condition else 0\n"
    )
    assert_mojo_format(source, expected)


def test_parametric_alias_with_complex_nested_expressions():
    """Test parametric alias with complex nested expressions that need line breaking."""
    source = (
        "comptime ComplexAlias[dt: DType, size: Int, offset: Int = 0] = "
        "SIMD[dt, size + offset]"
    )

    expected = (
        "comptime ComplexAlias[dt: DType, size: Int, offset: Int = 0] = SIMD[\n"
        "    dt, size + offset\n"
        "]\n"
    )
    assert_mojo_format(source, expected)


def test_multiple_parametric_aliases_in_file():
    """Test multiple parametric aliases in a single file."""
    source = (
        "comptime x: Int = 42\n"
        "comptime addOne[x: Int] : Int = x + 1\n"
        "comptime Scalar[dt: DType] = SIMD[dt, 1]\n"
        "comptime Float64[size: Int = 1] = SIMD[DType.float64, size]\n"
    )

    expected = (
        "comptime x: Int = 42\n"
        "comptime addOne[x: Int]: Int = x + 1\n"
        "comptime Scalar[dt: DType] = SIMD[dt, 1]\n"
        "comptime Float64[size: Int = 1] = SIMD[DType.float64, size]\n"
    )
    assert_mojo_format(source, expected)


def test_parametric_alias_with_comments():
    """Test parametric alias with comments."""
    source = (
        "# This is a parametric alias\n"
        "comptime addOne[x: Int] : Int = x + 1  # Add one to x"
    )

    expected = (
        "# This is a parametric alias\n"
        "comptime addOne[x: Int]: Int = x + 1  # Add one to x\n"
    )
    assert_mojo_format(source, expected)


def test_parametric_alias_idempotent():
    """Test that formatting is idempotent."""
    source = "comptime addOne[x: Int]: Int = x + 1"

    # Format once, then check formatting is stable.
    formatted_once = mojo_format_str(source)
    assert_mojo_format(formatted_once, formatted_once)


def test_parametric_alias_excessive_whitespace():
    """Test parametric alias with excessive whitespace"""
    source = "comptime    addOne[   x:    Int   ]    :    Int    =    x    +    1"
    expected = "comptime addOne[x: Int]: Int = x + 1\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_no_whitespace():
    """Test parametric alias with minimal whitespace."""
    source = "comptime addOne[x:Int]:Int=x+1"
    expected = "comptime addOne[x: Int]: Int = x + 1\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_mixed_whitespace():
    """Test parametric alias with inconsistent whitespace."""
    source = "comptime addOne[x:Int] :Int= x+1"
    expected = "comptime addOne[x: Int]: Int = x + 1\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_brackets():
    """Test parametric alias with spaces inside brackets."""
    source = "comptime SpacedBrackets[ x : Int , y : Int ] = SIMD[ DType.float32 , x + y ]"
    expected = "comptime SpacedBrackets[x: Int, y: Int] = SIMD[DType.float32, x + y]\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_around_colon():
    """Test parametric alias with various spacing around colons."""
    source = "comptime ColonSpacing[x:Int]:Int = x + 1"
    expected = "comptime ColonSpacing[x: Int]: Int = x + 1\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_around_equals():
    """Test parametric alias with various spacing around equals."""
    source = "comptime EqualsSpacing[x: Int] =x + 1"
    expected = "comptime EqualsSpacing[x: Int] = x + 1\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_type_annotations():
    """Test parametric alias with spaces in type annotations."""
    source = "comptime TypeSpacing[x : Int, y : String, z : Float64] : Bool = x > 0"
    expected = "comptime TypeSpacing[x: Int, y: String, z: Float64]: Bool = x > 0\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_default_values():
    """Test parametric alias with spaces in default value expressions."""
    source = "comptime DefaultSpacing[x: Int = 0, y: Int = 1, z: Int = 2] : Int = x + y + z"
    expected = "comptime DefaultSpacing[x: Int = 0, y: Int = 1, z: Int = 2]: Int = x + y + z\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_function_calls():
    """Test parametric alias with spaces in function calls."""
    source = (
        "def some_function(a: Int, b: Int, c: Int) -> Int:\n"
        "    return a\n"
        "\n"
        "comptime FunctionSpacing[x: Int] : Int = some_function( x , x * 2 , x + 1 )"
    )
    expected = (
        "def some_function(a: Int, b: Int, c: Int) -> Int:\n"
        "    return a\n"
        "\n"
        "\n"
        "comptime FunctionSpacing[x: Int]: Int = some_function(x, x * 2, x + 1)\n"
    )
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_arithmetic_expressions():
    """Test parametric alias with spaces in arithmetic expressions."""
    source = "comptime ArithmeticSpacing[x: Int, y: Int] : Int = ( x * y ) + ( x + y )"
    expected = "comptime ArithmeticSpacing[x: Int, y: Int]: Int = (x * y) + (x + y)\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_comparison_expressions():
    """Test parametric alias with spaces in comparison expressions."""
    source = "comptime ComparisonSpacing[x: Int, y: Int] : Bool = x == y and x != 0"
    expected = "comptime ComparisonSpacing[x: Int, y: Int]: Bool = x == y and x != 0\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_logical_expressions():
    """Test parametric alias with spaces in logical expressions."""
    source = "comptime LogicalSpacing[x: Bool, y: Bool] : Bool = x and y or not x"
    expected = "comptime LogicalSpacing[x: Bool, y: Bool]: Bool = x and y or not x\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_list_literals():
    """Test parametric alias with spaces in list literals."""
    source = "comptime ListSpacing[x: Int] = [ x , x * 2 , x * 3 ]"
    expected = "comptime ListSpacing[x: Int] = [x, x * 2, x * 3]\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_tuple_literals():
    """Test parametric alias with spaces in tuple literals."""
    source = "comptime TupleSpacing[x: Int, y: Int] = ( x , y , x + y )"
    expected = "comptime TupleSpacing[x: Int, y: Int] = (x, y, x + y)\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_dict_literals():
    """Test parametric alias with spaces in dict literals."""
    source = "comptime DictSpacing[x: String, y: Int] = { x : y , 'default' : 0 }"
    expected = 'comptime DictSpacing[x: String, y: Int] = {x: y, "default": 0}\n'
    assert_mojo_format(source, expected)


def test_spaces_in_comprehensions():
    """Test spaces in comprehensions."""
    source = (
        "def comprehension_spacing(x: Int):\n"
        "    var result = [ i * 2 for i in range( x ) if i > 0 ]\n"
        "    print(len(result))"
    )
    expected = (
        "def comprehension_spacing(x: Int):\n"
        "    var result = [i * 2 for i in range(x) if i > 0]\n"
        "    print(len(result))\n"
    )
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_multiline_expression():
    """Test parametric alias with spaces in multiline expressions."""
    source = (
        "comptime MultilineSpacing[x: Int, y: Int] : Int = ("
        "    x * y\n"
        "    ) + (\n"
        "    x + y\n"
        ")\n"
    )
    expected = "comptime MultilineSpacing[x: Int, y: Int]: Int = (x * y) + (x + y)\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_comments():
    """Test parametric alias with spaces around comments."""
    source = (
        "comptime CommentSpacing[x: Int] : Int = x + 1  # Add one\n"
        "comptime AnotherSpacing[y: Int] : Int = y * 2  # Multiply by two"
    )
    expected = (
        "comptime CommentSpacing[x: Int]: Int = x + 1  # Add one\n"
        "comptime AnotherSpacing[y: Int]: Int = y * 2  # Multiply by two\n"
    )
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_string_literals():
    """Test parametric alias with spaces in string literals (should be preserved)."""
    source = 'comptime StringSpacing[x: String] : String = "  hello  world  "'
    expected = 'comptime StringSpacing[x: String]: String = "  hello  world  "\n'
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_numeric_literals():
    """Test parametric alias with spaces in numeric literals (should be normalized)."""
    source = "comptime NumericSpacing[x: Int = 1_000_000, y: Float64 = 3.141_59] : Float64 = y"
    expected = "comptime NumericSpacing[x: Int = 1_000_000, y: Float64 = 3.141_59]: Float64 = y\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_boolean_literals():
    """Test parametric alias with spaces around boolean literals."""
    source = "comptime BooleanSpacing[x: Bool = True, y: Bool = False] : Bool = x and y"
    expected = "comptime BooleanSpacing[x: Bool = True, y: Bool = False]: Bool = x and y\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_none_literal():
    """Test parametric alias with spaces around None literal."""
    source = "comptime NoneSpacing[x: Optional[Int] = None] : Bool = x is None"
    expected = "comptime NoneSpacing[x: Optional[Int] = None]: Bool = x is None\n"
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_complex_nested_whitespace():
    """Test parametric alias with complex nested whitespace scenarios."""
    source = (
        'comptime ComplexWhitespace[ x : Int = 1 + 2 , y : String = "  hello  " ] = ('
        "    x    *    2    ,\n"
        "    y    .    strip( )    ,\n"
        "    x    +    len( y . bytes( ) )\n"
        ")"
    )
    expected = (
        'comptime ComplexWhitespace[x: Int = 1 + 2, y: String = "  hello  "] = (\n'
        "    x * 2,\n"
        "    y.strip(),\n"
        "    x + len(y.bytes()),\n"
        ")\n"
    )
    assert_mojo_format(source, expected)


def test_parametric_alias_spaces_in_complex_whitespace_edge_cases():
    """Test parametric alias with complex whitespace edge cases."""
    source = (
        'comptime ComplexEdgeCase[ x : Int = 1 + 2 * 3 // 4 % 5 ** 6 , y : String = "  hello  world  " ] = ('
        "    x    +    len( y    .    strip( )    .    split( )    [ 0 ] . bytes( ) )    ,\n"
        "    len( y    .    strip( )    . bytes( ) )    ,\n"
        "    x    **    2    +    y    .    count( ' ' )    *    10\n"
        ")\n"
    )
    expected = (
        "comptime ComplexEdgeCase[\n"
        '    x: Int = 1 + 2 * 3 // 4 % 5**6, y: String = "  hello  world  "\n'
        "] = (\n"
        "    x + len(y.strip().split()[0].bytes()),\n"
        "    len(y.strip().bytes()),\n"
        '    x**2 + y.count(" ") * 10,\n'
        ")\n"
    )
    assert_mojo_format(source, expected)


def test_parametric_alias_long_header_keeps_tuple_body_inline():
    """Test that a long header explodes while a short tuple body stays inline."""
    source = (
        "comptime CollapseBodyLongHeader[ first : Int = 1 + 2 * 3 // 4 % 5 ** 6 , second : Int = 9 - 8 * 7 // 6 ] = ("
        "    first    +    second    ,\n"
        "    first    *    second\n"
        ")\n"
    )
    expected = (
        "comptime CollapseBodyLongHeader[\n"
        "    first: Int = 1 + 2 * 3 // 4 % 5**6, second: Int = 9 - 8 * 7 // 6\n"
        "] = (first + second, first * second)\n"
    )
    assert_mojo_format(source, expected)


def test_bug_report_representative_case():
    """Test the representative case from the bug report (MOTO-1135)."""
    source = (
        "comptime _dtype_to_llvm_type_f8[dtype: DType] = "
        "__mlir_type.`i8` if dtype == DType.float8_e3m4 or "
        "dtype == DType.float8_e4m3fn or dtype == DType.float8_e4m3fnuz or "
        "dtype == DType.float8_e5m2 or dtype == DType.float8_e5m2fnuz else "
        "__mlir_type.`!kgen.none`"
    )
    expected = (
        "comptime _dtype_to_llvm_type_f8[\n"
        "    dtype: DType\n"
        "] = __mlir_type.`i8` if dtype == DType.float8_e3m4 or dtype == DType.float8_e4m3fn or dtype == DType.float8_e4m3fnuz or dtype == DType.float8_e5m2 or dtype == DType.float8_e5m2fnuz else __mlir_type.`!kgen.none`\n"
    )
    assert_mojo_format(source, expected)
