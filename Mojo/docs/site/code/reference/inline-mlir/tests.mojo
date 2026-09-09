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
# test_inline_mlir.mojo
# Tests for inline-mlir.mdx code examples.
from std.testing import assert_equal


# --- Your first inline MLIR ---


def test_hello_mlir() raises:
    """Three builtins working together: type, attr, op."""
    var a: __mlir_type.index = __mlir_attr.`42 : index`
    var b: __mlir_type.index = __mlir_attr.`8 : index`
    var c = __mlir_op.`index.add`(a, b)
    assert_equal(Int(mlir_value=c), 50)


# --- __mlir_type ---


def test_mlir_type_dot_syntax() raises:
    """Dot syntax for simple MLIR type names."""
    var x: __mlir_type.i1 = __mlir_attr.true
    var y: __mlir_type.index = __mlir_attr.`0 : index`
    assert_equal(Bool(x), True)
    assert_equal(Int(mlir_value=y), 0)


def test_mlir_type_backtick_syntax() raises:
    """Backtick syntax for dialect types with special characters."""
    var s: __mlir_type.`!kgen.scalar<index>` = (
        __mlir_op.`pop.cast_from_builtin`[
            _type=__mlir_type.`!kgen.scalar<index>`
        ](__mlir_attr.`7 : index`)
    )
    var result = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.index](s)
    assert_equal(Int(mlir_value=result), 7)


# --- __mlir_attr ---


def test_mlir_attr_bool() raises:
    """Boolean attributes: true and false."""
    var t: __mlir_type.i1 = __mlir_attr.true
    var f: __mlir_type.i1 = __mlir_attr.false
    assert_equal(Bool(t), True)
    assert_equal(Bool(f), False)


def test_mlir_attr_typed_constants() raises:
    """Backtick syntax for typed integer constants."""
    var zero: __mlir_type.index = __mlir_attr.`0 : index`
    var forty_two: __mlir_type.index = __mlir_attr.`42 : index`
    var negative: __mlir_type.index = __mlir_attr.`-1 : index`
    assert_equal(Int(mlir_value=zero), 0)
    assert_equal(Int(mlir_value=forty_two), 42)
    assert_equal(Int(mlir_value=negative), -1)


# --- __mlir_op: index dialect ---


def test_index_arithmetic() raises:
    """Index dialect: add, sub, mul, divs."""
    var a: __mlir_type.index = __mlir_attr.`10 : index`
    var b: __mlir_type.index = __mlir_attr.`3 : index`

    var sum = __mlir_op.`index.add`(a, b)
    assert_equal(Int(mlir_value=sum), 13)

    var diff = __mlir_op.`index.sub`(a, b)
    assert_equal(Int(mlir_value=diff), 7)

    var prod = __mlir_op.`index.mul`(a, b)
    assert_equal(Int(mlir_value=prod), 30)

    var quot = __mlir_op.`index.divs`(a, b)
    assert_equal(Int(mlir_value=quot), 3)


def test_index_bitwise() raises:
    """Index dialect: and, or, xor, shl, shrs."""
    var a: __mlir_type.index = __mlir_attr.`10 : index`  # 0b1010
    var b: __mlir_type.index = __mlir_attr.`12 : index`  # 0b1100

    var and_result = __mlir_op.`index.and`(a, b)
    assert_equal(Int(mlir_value=and_result), 0b1000)  # 8

    var or_result = __mlir_op.`index.or`(a, b)
    assert_equal(Int(mlir_value=or_result), 0b1110)  # 14

    var xor_result = __mlir_op.`index.xor`(a, b)
    assert_equal(Int(mlir_value=xor_result), 0b0110)  # 6

    var one: __mlir_type.index = __mlir_attr.`1 : index`
    var shifted = __mlir_op.`index.shl`(a, one)
    assert_equal(Int(mlir_value=shifted), 0b10100)  # 20

    var rshifted = __mlir_op.`index.shrs`(a, one)
    assert_equal(Int(mlir_value=rshifted), 0b0101)  # 5


def test_index_comparisons() raises:
    """Index dialect: cmp with various predicates."""
    var five: __mlir_type.index = __mlir_attr.`5 : index`
    var ten: __mlir_type.index = __mlir_attr.`10 : index`

    var lt = __mlir_op.`index.cmp`[
        pred=__mlir_attr.`#index.cmp_predicate<slt>`
    ](five, ten)
    assert_equal(Bool(lt), True)

    var eq = __mlir_op.`index.cmp`[pred=__mlir_attr.`#index.cmp_predicate<eq>`](
        five, five
    )
    assert_equal(Bool(eq), True)

    var gt = __mlir_op.`index.cmp`[
        pred=__mlir_attr.`#index.cmp_predicate<sgt>`
    ](five, ten)
    assert_equal(Bool(gt), False)

    var ne = __mlir_op.`index.cmp`[pred=__mlir_attr.`#index.cmp_predicate<ne>`](
        five, ten
    )
    assert_equal(Bool(ne), True)


def test_index_constant() raises:
    """Index dialect: index.constant operation."""
    var c = __mlir_op.`index.constant`[value=__mlir_attr.`99 : index`]()
    assert_equal(Int(mlir_value=c), 99)


# --- __mlir_op: pop dialect ---


def test_pop_bool_ops() raises:
    """Pop dialect: xor, and, or on i1."""
    var t: __mlir_type.i1 = __mlir_attr.true
    var f: __mlir_type.i1 = __mlir_attr.false

    # NOT via XOR with true
    var not_t = __mlir_op.`pop.xor`(t, __mlir_attr.true)
    assert_equal(Bool(not_t), False)

    var not_f = __mlir_op.`pop.xor`(f, __mlir_attr.true)
    assert_equal(Bool(not_f), True)

    # AND
    var t_and_f = __mlir_op.`pop.and`(t, f)
    assert_equal(Bool(t_and_f), False)

    var t_and_t = __mlir_op.`pop.and`(t, t)
    assert_equal(Bool(t_and_t), True)

    # OR
    var t_or_f = __mlir_op.`pop.or`(t, f)
    assert_equal(Bool(t_or_f), True)

    var f_or_f = __mlir_op.`pop.or`(f, f)
    assert_equal(Bool(f_or_f), False)


def test_pop_cast_roundtrip() raises:
    """Pop dialect: cast_from_builtin and cast_to_builtin roundtrip."""
    var idx: __mlir_type.index = __mlir_attr.`42 : index`

    # index → !kgen.scalar<index>
    var pop_val = __mlir_op.`pop.cast_from_builtin`[
        _type=__mlir_type.`!kgen.scalar<index>`
    ](idx)

    # !kgen.scalar<index> → index
    var back = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.index](pop_val)

    assert_equal(Int(mlir_value=back), 42)


def test_pop_simd_splat() raises:
    """Pop dialect: splat a scalar to a SIMD vector, extract element."""
    # Create a scalar Float32
    var idx: __mlir_type.index = __mlir_attr.`3 : index`
    var pop_idx = __mlir_op.`pop.cast_from_builtin`[
        _type=__mlir_type.`!kgen.scalar<index>`
    ](idx)
    var scalar_f32 = __mlir_op.`pop.cast`[
        _type=__mlir_type.`!kgen.scalar<f32>`
    ](pop_idx)

    # Splat to 4-wide SIMD
    var vec = __mlir_op.`pop.simd.splat`[
        _type=__mlir_type.`!kgen.simd<4, f32>`
    ](scalar_f32)

    # Extract element 0 back out
    var elem = __mlir_op.`pop.simd.extractelement`(
        vec,
        __mlir_attr.`0 : index`,
    )

    # Wrap in SIMD scalar to compare
    var result = Float32(mlir_value=elem)
    assert_equal(result, 3.0)


# --- __mlir_op: explicit _type ---


def test_explicit_type_attr() raises:
    """The _type attribute specifies result type when inference fails."""
    var x: __mlir_type.index = __mlir_attr.`255 : index`
    var casted = __mlir_op.`index.castu`[_type=__mlir_type.i1](x)
    # 255 truncated to i1 = true (lowest bit is 1)
    assert_equal(Bool(casted), True)


# --- Wrapper struct pattern ---


struct MiniInt(Writable):
    """Minimal wrapper struct demonstrating the inline MLIR pattern."""

    var _mlir_value: __mlir_type.index

    def __init__(out self):
        self._mlir_value = __mlir_attr.`0 : index`

    def __init__(out self, *, mlir_value: __mlir_type.index):
        self._mlir_value = mlir_value

    @implicit
    def __init__(out self, value: Int):
        self._mlir_value = value.__mlir_index__()

    def __add__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`index.add`(self._mlir_value, rhs._mlir_value)
        )

    def __sub__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`index.sub`(self._mlir_value, rhs._mlir_value)
        )

    def __mul__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`index.mul`(self._mlir_value, rhs._mlir_value)
        )

    def __eq__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<eq>`
        ](self._mlir_value, rhs._mlir_value)

    def __lt__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<slt>`
        ](self._mlir_value, rhs._mlir_value)

    def to_int(self) -> Int:
        return Int(mlir_value=self._mlir_value)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.to_int())


def test_wrapper_struct_arithmetic() raises:
    """Wrapper struct: add, sub, mul through MLIR ops."""
    var a = MiniInt(10)
    var b = MiniInt(3)
    assert_equal((a + b).to_int(), 13)
    assert_equal((a - b).to_int(), 7)
    assert_equal((a * b).to_int(), 30)


def test_wrapper_struct_comparison() raises:
    """Wrapper struct: eq and lt through MLIR cmp."""
    var a = MiniInt(5)
    var b = MiniInt(10)
    assert_equal(a == a, True)
    assert_equal(a == b, False)
    assert_equal(a < b, True)
    assert_equal(b < a, False)


def test_wrapper_struct_write() raises:
    """Wrapper struct: Writable conformance through MLIR value."""
    var x = MiniInt(42)
    assert_equal(String(x), "42")


# --- Multiple operations chained ---


def test_chained_ops() raises:
    """Multiple MLIR ops chained: (a + b) * c."""
    var a: __mlir_type.index = __mlir_attr.`2 : index`
    var b: __mlir_type.index = __mlir_attr.`3 : index`
    var c: __mlir_type.index = __mlir_attr.`7 : index`

    var sum = __mlir_op.`index.add`(a, b)
    var product = __mlir_op.`index.mul`(sum, c)
    assert_equal(Int(mlir_value=product), 35)  # (2+3)*7


def test_conditional_on_mlir_cmp() raises:
    """MLIR comparison result used in Mojo control flow."""
    var x: __mlir_type.index = __mlir_attr.`42 : index`
    var threshold: __mlir_type.index = __mlir_attr.`10 : index`

    var above = __mlir_op.`index.cmp`[
        pred=__mlir_attr.`#index.cmp_predicate<sgt>`
    ](x, threshold)

    var label = "big" if Bool(above) else "small"
    assert_equal(label, "big")


def sum_to(end: Int) -> Int:
    """Mojo while loop with MLIR operations for arithmetic and comparison."""
    var acc: __mlir_type.index = __mlir_attr.`0 : index`
    var i: __mlir_type.index = __mlir_attr.`0 : index`
    var one: __mlir_type.index = __mlir_attr.`1 : index`

    # end.__mlir_index__() converts Int to raw __mlir_type.index
    while Bool(
        __mlir_op.`index.cmp`[pred=__mlir_attr.`#index.cmp_predicate<slt>`](
            i, end.__mlir_index__()
        )
    ):
        acc = __mlir_op.`index.add`(acc, i)
        i = __mlir_op.`index.add`(i, one)

    return Int(mlir_value=acc)


def test_mlir_in_mojo_loop() raises:
    """MLIR operations used in a Mojo while loop to sum integers."""
    assert_equal(sum_to(10), 45)  # 0+1+2+...+9 = 45
    assert_equal(sum_to(0), 0)
    assert_equal(sum_to(1), 0)
    assert_equal(sum_to(5), 10)  # 0+1+2+3+4 = 10


# --- Additional examples added for practical use ---


def test_mlir_types_in_action() raises:
    """Types in action: declare, assign, convert back."""
    var flag: __mlir_type.i1 = __mlir_attr.true
    var count: __mlir_type.index = __mlir_attr.`0 : index`
    assert_equal(Bool(flag), True)
    assert_equal(Int(mlir_value=count), 0)


def test_circle_area_approx() raises:
    """MLIR attrs as constants in a computation."""

    def circle_area_approx(radius: Int) -> Int:
        var r = radius.__mlir_index__()
        var r_squared = __mlir_op.`index.mul`(r, r)
        var pi: __mlir_type.index = __mlir_attr.`3 : index`
        var area = __mlir_op.`index.mul`(pi, r_squared)
        return Int(mlir_value=area)

    assert_equal(circle_area_approx(5), 75)
    assert_equal(circle_area_approx(10), 300)
    assert_equal(circle_area_approx(0), 0)


def test_clamp() raises:
    """Comparisons with pop.select to clamp a value."""

    def clamp(val: Int, low: Int, high: Int) -> Int:
        var v = val.__mlir_index__()
        var lo = low.__mlir_index__()
        var hi = high.__mlir_index__()
        # index.cmp returns i1, but pop.select needs a !kgen.scalar<bool>.
        var too_low = __mlir_op.`pop.cast_from_builtin`[
            _type=__mlir_type.`!kgen.scalar<bool>`
        ](
            __mlir_op.`index.cmp`[pred=__mlir_attr.`#index.cmp_predicate<slt>`](
                v, lo
            )
        )
        var result = __mlir_op.`pop.select`(too_low, lo, v)
        var too_high = __mlir_op.`pop.cast_from_builtin`[
            _type=__mlir_type.`!kgen.scalar<bool>`
        ](
            __mlir_op.`index.cmp`[pred=__mlir_attr.`#index.cmp_predicate<sgt>`](
                result, hi
            )
        )
        result = __mlir_op.`pop.select`(too_high, hi, result)
        return Int(mlir_value=result)

    assert_equal(clamp(15, 0, 10), 10)
    assert_equal(clamp(-5, 0, 10), 0)
    assert_equal(clamp(7, 0, 10), 7)
    assert_equal(clamp(0, 0, 10), 0)
    assert_equal(clamp(10, 0, 10), 10)


struct Counter:
    """Wrapper struct from doc page: counter backed by raw MLIR index."""

    var _mlir_value: __mlir_type.index

    def __init__(out self):
        self._mlir_value = __mlir_attr.`0 : index`

    def __init__(out self, *, mlir_value: __mlir_type.index):
        self._mlir_value = mlir_value

    def increment(mut self):
        var one: __mlir_type.index = __mlir_attr.`1 : index`
        self._mlir_value = __mlir_op.`index.add`(self._mlir_value, one)

    def __add__(self, rhs: Counter) -> Counter:
        return Counter(
            mlir_value=__mlir_op.`index.add`(self._mlir_value, rhs._mlir_value)
        )

    def value(self) -> Int:
        return Int(mlir_value=self._mlir_value)


def test_counter_increment() raises:
    """Counter wrapper struct: increment and read back."""
    var c = Counter()
    assert_equal(c.value(), 0)
    c.increment()
    c.increment()
    c.increment()
    assert_equal(c.value(), 3)


def test_counter_add() raises:
    """Counter with __add__: operations as methods pattern."""
    var a = Counter()
    a.increment()  # 1
    a.increment()  # 2
    var b = Counter()
    b.increment()  # 1
    var c = a + b  # 2 + 1 = 3
    assert_equal(c.value(), 3)


def main() raises:
    test_hello_mlir()
    test_mlir_type_dot_syntax()
    test_mlir_type_backtick_syntax()
    test_mlir_attr_bool()
    test_mlir_attr_typed_constants()
    test_index_arithmetic()
    test_index_bitwise()
    test_index_comparisons()
    test_index_constant()
    test_pop_bool_ops()
    test_pop_cast_roundtrip()
    test_pop_simd_splat()
    test_explicit_type_attr()
    test_wrapper_struct_arithmetic()
    test_wrapper_struct_comparison()
    test_wrapper_struct_write()
    test_chained_ops()
    test_conditional_on_mlir_cmp()
    test_mlir_in_mojo_loop()
    test_mlir_types_in_action()
    test_circle_area_approx()
    test_clamp()
    test_counter_increment()
    test_counter_add()
