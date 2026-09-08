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

from std.collections.dict import (
    Dict,
    DictEntry,
    DictKeyError,
    EmptyDictError,
    StringDict,
)
from std.collections._swisstable import GROUP_WIDTH

from std.hashlib import Hasher, default_comp_time_hasher, default_hasher

from test_utils import (
    CopyCounter,
    DelCounter,
    ExplicitDestroy,
    ExplicitDestroyKey,
    MoveOnly,
    check_write_to,
)
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_dict_construction() raises:
    _ = Dict[Int, Int]()
    _ = Dict[String, Int]()


def test_dict_lazy_allocation() raises:
    var d = Dict[Int, Int]()
    assert_equal(d._reserved(), 0)
    assert_equal(len(d), 0)
    assert_false(d)

    var d_zero = Dict[Int, Int](capacity=0)
    assert_equal(d_zero._reserved(), 0)

    # Empty dict literal `{}` constructs with capacity=0 and stays lazy.
    var d_literal: Dict[Int, Int] = {}
    assert_equal(d_literal._reserved(), 0)

    # Lookups on a lazy dict must not deref the dangling buffer.
    assert_false(1 in d)
    assert_false(d.find(1))
    with assert_raises(contains="DictKeyError"):
        _ = d[1]

    # `pop(key)` raises `DictKeyError` on a lazy dict.
    with assert_raises(contains="DictKeyError"):
        _ = d.pop(1)

    # `pop(key, default)` returns the default without allocating.
    assert_equal(d.pop(1, 42), 42)
    assert_equal(d._reserved(), 0)

    # `popitem` raises `EmptyDictError` on a lazy dict.
    with assert_raises(contains="EmptyDictError"):
        _ = d.popitem()

    # Iteration over a lazy dict yields nothing — no buffer dereference.
    var iter_count = 0
    for _ in d:
        iter_count += 1
    assert_equal(iter_count, 0)

    var keys_count = 0
    for _ in d.keys():
        keys_count += 1
    assert_equal(keys_count, 0)

    var values_count = 0
    for _ in d.values():
        values_count += 1
    assert_equal(values_count, 0)

    var items_count = 0
    for _ in d.items():
        items_count += 1
    assert_equal(items_count, 0)

    # Clearing a never-allocated dict is a no-op.
    d.clear()
    assert_equal(d._reserved(), 0)

    # Copying a lazy dict yields another lazy dict.
    var d_copy = d.copy()
    assert_equal(d_copy._reserved(), 0)

    # `setdefault` on a lazy dict allocates and inserts.
    var d_sd = Dict[Int, Int]()
    assert_equal(d_sd._reserved(), 0)
    assert_equal(d_sd.setdefault(1, 99), 99)
    assert_equal(d_sd._reserved(), 16)
    assert_equal(d_sd[1], 99)

    # First insertion triggers allocation at INITIAL_CAPACITY (16).
    d[1] = 10
    assert_equal(d._reserved(), 16)
    assert_equal(d[1], 10)

    # Removing the last entry must not regress to the lazy state — capacity
    # is preserved so the next insert doesn't re-trigger the lazy path.
    _ = d.pop(1)
    assert_equal(len(d), 0)
    assert_equal(d._reserved(), 16)


def test_dict_literals() raises:
    var a = {"foo": 1, "bar": 2}
    assert_equal(a["foo"], 1)

    var b = {1: 4, 2: 7, 3: 18}
    assert_equal(b[1], 4)
    assert_equal(b[2], 7)
    assert_equal(b[3], 18)
    assert_false(4 in b)


def test_dict_fromkeys() raises:
    comptime keys = [String("a"), "b"]
    var expected_dict = Dict[String, Int]()
    expected_dict["a"] = 1
    expected_dict["b"] = 1
    var dict = Dict.fromkeys(materialize[keys](), 1)

    assert_equal(len(dict), len(expected_dict))

    for k_v in expected_dict.items():
        var k = k_v.key
        var v = k_v.value
        assert_true(k in dict)
        assert_equal(dict[k], v)


def test_dict_fromkeys_optional() raises:
    comptime keys = [String("a"), "b", "c"]
    var expected_dict: Dict[String, Optional[Int]] = {
        "a": None,
        "b": None,
        "c": None,
    }
    var dict = Dict[String, Optional[Int]].fromkeys(materialize[keys](), None)

    assert_equal(len(dict), len(expected_dict))

    for k_v in expected_dict.items():
        var k = k_v.key
        var v = k_v.value
        assert_true(k in dict)
        assert_false(v)


def test_dict_fromkeys_duplicate_keys() raises:
    # Duplicate keys in the input list collapse to a single entry (matching
    # Python's `dict.fromkeys`), all sharing the given value.
    comptime keys = [String("a"), "b", "a", "c", "b"]
    var dict = Dict.fromkeys(materialize[keys](), 7)

    assert_equal(len(dict), 3)
    assert_equal(dict["a"], 7)
    assert_equal(dict["b"], 7)
    assert_equal(dict["c"], 7)


def test_dict_fromkeys_iterable() raises:
    # `fromkeys` accepts any borrowed `Iterable`, not just `List`. Duplicate
    # keys collapse to a single entry.
    var keys: Array[String, 4] = ["a", "b", "a", "c"]
    var dict = Dict.fromkeys(keys, 7)

    assert_equal(len(dict), 3)
    assert_equal(dict["a"], 7)
    assert_equal(dict["b"], 7)
    assert_equal(dict["c"], 7)


def test_dict_fromkeys_moving() raises:
    # Transferring the iterable with `^` selects the move overload of
    # `fromkeys`, consuming it and moving keys into the new dictionary.
    # Duplicate keys collapse to a single entry.
    var keys = [String("a"), "b", "a", "c"]
    var dict = Dict.fromkeys(keys^, 7)

    assert_equal(len(dict), 3)
    assert_equal(dict["a"], 7)
    assert_equal(dict["b"], 7)
    assert_equal(dict["c"], 7)


def test_basic() raises:
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    assert_equal(1, dict["a"])
    assert_equal(2, dict["b"])

    var ptr = Pointer(to=dict["a"])
    assert_equal(1, ptr[])
    ptr[] = 17
    assert_equal(17, dict["a"])


def test_basic_no_copies() raises:
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2

    assert_equal(1, dict["a"])
    assert_equal(2, dict["b"])


def test_multiple_resizes() raises:
    var dict: Dict[String, Int] = {}
    for i in range(20):
        dict[String("key", i)] = i + 1
    assert_equal(11, dict["key10"])
    assert_equal(20, dict["key19"])


def test_bool_conversion() raises:
    var dict: Dict[String, Int] = {}
    assert_false(dict)
    dict["a"] = 1
    assert_true(dict)
    dict["b"] = 2
    assert_true(dict)
    _ = dict.pop("a")
    assert_true(dict)
    _ = dict.pop("b")
    assert_false(dict)


def test_big_dict() raises:
    var dict: Dict[String, Int] = {}
    for i in range(2000):
        dict[String("key", i)] = i + 1
    assert_equal(2000, len(dict))


def test_dict_string_representation_string_int() raises:
    var d: Dict[String, Int] = {"a": 1, "b": 2}
    check_write_to(d, expected="{a: 1, b: 2}", is_repr=False)
    check_write_to(
        d,
        expected="Dict[String, SIMD[DType.int, 1]]({'a': Int(1), 'b': Int(2)})",
        is_repr=True,
    )


def test_dict_string_representation_int_int() raises:
    var d: Dict[Int, Int] = {1: 2, 3: 4}
    check_write_to(d, expected="{1: 2, 3: 4}", is_repr=False)
    check_write_to(
        d,
        expected=(
            "Dict[SIMD[DType.int, 1], SIMD[DType.int, 1]]({Int(1): Int(2),"
            " Int(3): Int(4)})"
        ),
        is_repr=True,
    )


def test_compact() raises:
    var dict: Dict[String, Int] = {}
    for i in range(20):
        var key = String("key", i)
        dict[key] = i + 1
        _ = dict.pop(key)
    assert_equal(0, len(dict))


def test_compact_with_elements() raises:
    var dict: Dict[String, Int] = {}
    for i in range(5):
        var key = String("key", i)
        dict[key] = i + 1
    for i in range(5, 20):
        var key = String("key", i)
        dict[key] = i + 1
        _ = dict.pop(key)
    assert_equal(5, len(dict))


def test_pop_default() raises:
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    assert_equal(1, dict.pop("a", -1))
    assert_equal(2, dict.pop("b", -1))
    assert_equal(-1, dict.pop("c", -1))


def test_key_error() raises:
    var dict: Dict[String, Int] = {}

    with assert_raises(contains="KeyError"):
        _ = dict["a"]
    with assert_raises(contains="KeyError"):
        _ = dict.pop("a")


def _test_iter_bounds[
    I: Iterator, //
](
    var dict_iter: I,
    dict_len: Int,
) raises where conforms_to(
    I.Element, Deinitable
):
    var iter = dict_iter^
    for i in range(dict_len):
        var lower, upper = iter.bounds()
        assert_equal(dict_len - i, lower)
        assert_equal(dict_len - i, upper.value())
        _ = iter.__next__()

    var lower, upper = iter.bounds()
    assert_equal(0, lower)
    assert_equal(0, upper.value())


def test_iter() raises:
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    var keys = String()
    for key in dict:
        keys += key

    assert_equal(keys, "ab")
    _test_iter_bounds(dict.__iter__(), len(dict))

    var empty_dict: Dict[String, Int] = {}
    with assert_raises():
        var it = iter(empty_dict)
        _ = it.__next__()  # raises StopIteration


def test_iter_keys() raises:
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    var keys = String()
    for key in dict.keys():
        keys += key

    assert_equal(keys, "ab")
    _test_iter_bounds(dict.keys(), len(dict))


def test_iter_values() raises:
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    var sum = 0
    for value in dict.values():
        sum += value

    assert_equal(sum, 3)
    _test_iter_bounds(dict.values(), len(dict))


def test_iter_values_mut() raises:
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    for ref value in dict.values():
        value += 1

    assert_equal(2, dict["a"])
    assert_equal(3, dict["b"])
    assert_equal(2, len(dict))
    _test_iter_bounds(dict.values(), len(dict))


def test_iter_items() raises:
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    var keys = String()
    var sum = 0
    for entry in dict.items():
        keys += entry.key
        sum += entry.value

    assert_equal(keys, "ab")
    assert_equal(sum, 3)

    # TODO: _DictItemIter does not conform to `Iterator` yet
    # _test_iter_bounds(dict.values(), len(dict))


def test_iter_take_items() raises:
    var dict: Dict[Int, String] = {0: "a", 1: "b", 2: "c"}

    var values = String()
    var keys = 0

    for entry in dict.take_items():
        keys += entry.key
        values += entry.value

    assert_equal(values, "abc")
    assert_equal(keys, 3)
    assert_equal(len(dict), 0)
    with assert_raises():
        var it = dict.take_items()
        _ = it.__next__()  # raises StopIteration

    for i in range(3):
        with assert_raises(contains="KeyError"):
            _ = dict[i]


def test_iter_take_items_owned() raises:
    # Test that dict `take_items()` works with non-Copyable values
    var dict = Dict[MoveOnly[Int], String]()
    dict[MoveOnly(0)] = "a"
    dict[MoveOnly(1)] = "b"
    dict[MoveOnly(2)] = "c"

    var values = String()
    var keys = 0

    for entry in dict.take_items():
        keys += entry.key.data
        values += entry.value

    assert_equal(values, "abc")
    assert_equal(keys, 3)
    assert_equal(len(dict), 0)
    with assert_raises():
        var it = dict.take_items()
        _ = it.__next__()  # raises StopIteration

    for i in range(3):
        with assert_raises(contains="KeyError"):
            _ = dict[MoveOnly(i)]


def test_iter_take_items_empty() raises:
    var dict: Dict[Int, String] = {}

    var count = 0
    for _ in dict.take_items():
        count += 1
    assert_equal(len(dict), 0)
    assert_equal(count, 0)


def test_dict_contains() raises:
    var dict: Dict[String, Int] = {}
    dict["abc"] = 1
    dict["def"] = 2
    assert_true("abc" in dict)
    assert_true("def" in dict)
    assert_false("c" in dict)


def test_dict_copy() raises:
    var orig: Dict[String, Int] = {}
    orig["a"] = 1

    # test values copied to new Dict
    var copy = orig.copy()
    assert_equal(1, copy["a"])

    # test there are two copies of dict and
    # they don't share underlying memory
    copy["a"] = 2
    assert_equal(2, copy["a"])
    assert_equal(1, orig["a"])


def test_dict_copy_delete_original() raises:
    var orig: Dict[String, Int] = {}
    orig["a"] = 1

    # test values copied to new Dict
    var copy = orig.copy()
    # don't access the original dict, anymore, confirm that
    # deleting the original doesn't violate the integrity of the copy
    assert_equal(1, copy["a"])


def test_dict_copy_add_new_item() raises:
    var orig: Dict[String, Int] = {}
    orig["a"] = 1

    # test values copied to new Dict
    var copy = orig.copy()
    assert_equal(1, copy["a"])

    # test there are two copies of dict and
    # they don't share underlying memory
    copy["b"] = 2
    assert_false(String(2) in orig)


def test_dict_copy_calls_copy_constructor() raises:
    var orig: Dict[String, CopyCounter[]] = {}
    orig["a"] = CopyCounter()

    # test values copied to new Dict
    var copy = orig.copy()
    assert_equal(0, orig["a"].copy_count)
    assert_equal(1, copy["a"].copy_count)
    assert_equal(0, orig._find_ref("a").copy_count)
    assert_equal(1, copy._find_ref("a").copy_count)


def test_dict_update_nominal() raises:
    var orig: Dict[String, Int] = {}
    orig["a"] = 1
    orig["b"] = 2

    var new: Dict[String, Int] = {}
    new["b"] = 3
    new["c"] = 4

    orig.update(new)

    assert_equal(orig["a"], 1)
    assert_equal(orig["b"], 3)
    assert_equal(orig["c"], 4)


def test_dict_update_empty_origin() raises:
    var orig: Dict[String, Int] = {}
    var new: Dict[String, Int] = {}
    new["b"] = 3
    new["c"] = 4

    orig.update(new)

    assert_equal(orig["b"], 3)
    assert_equal(orig["c"], 4)


def test_dict_or() raises:
    var orig: Dict[String, Int] = {}
    var new: Dict[String, Int] = {}

    new["b"] = 3
    new["c"] = 4
    orig["d"] = 5
    orig["b"] = 8

    var out = orig | new

    assert_equal(out["b"], 3)
    assert_equal(out["c"], 4)
    assert_equal(out["d"], 5)

    orig |= new

    assert_equal(orig["b"], 3)
    assert_equal(orig["c"], 4)
    assert_equal(orig["d"], 5)

    orig = Dict[String, Int]()
    new = Dict[String, Int]()
    new["b"] = 3
    new["c"] = 4

    orig |= new

    assert_equal(orig["b"], 3)
    assert_equal(orig["c"], 4)

    orig = Dict[String, Int]()
    orig["a"] = 1
    orig["b"] = 2

    new = Dict[String, Int]()

    orig = orig | new

    assert_equal(orig["a"], 1)
    assert_equal(orig["b"], 2)
    assert_equal(len(orig), 2)

    orig = Dict[String, Int]()
    new = Dict[String, Int]()
    orig["a"] = 1
    orig["b"] = 2
    new["c"] = 3
    new["d"] = 4
    orig |= new
    assert_equal(orig["a"], 1)
    assert_equal(orig["b"], 2)
    assert_equal(orig["c"], 3)
    assert_equal(orig["d"], 4)

    orig = Dict[String, Int]()
    new = Dict[String, Int]()
    assert_equal(len(orig | new), 0)


def test_dict_update_empty_new() raises:
    var orig: Dict[String, Int] = {}
    orig["a"] = 1
    orig["b"] = 2

    var new: Dict[String, Int] = {}

    orig.update(new)

    assert_equal(orig["a"], 1)
    assert_equal(orig["b"], 2)
    assert_equal(len(orig), 2)


@fieldwise_init("implicit")
struct DummyKey(KeyElement):
    var value: Int

    def __hash__[H: Hasher](self, mut hasher: H):
        return self.value.__hash__(hasher)

    def __eq__(self, other: DummyKey) -> Bool:
        return self.value == other.value

    def __ne__(self, other: DummyKey) -> Bool:
        return self.value != other.value


def test_mojo_issue_1729() raises:
    var keys = [
        7005684093727295727,
        2833576045803927472,
        -446534169874157203,
        -5597438459201014662,
        -7007119737006385570,
        7237741981002255125,
        -649171104678427962,
        -6981562940350531355,
    ]
    var d: Dict[DummyKey, Int] = {}
    for i, key in enumerate(keys):
        d[DummyKey(key)] = i
    assert_equal(len(d), len(keys))
    for i, key in enumerate(keys):
        assert_equal(i, d[DummyKey(key)])


def _test_taking_owned_kwargs_dict(var **kwargs: Int) raises:
    assert_equal(len(kwargs), 2)

    assert_true("fruit" in kwargs)
    assert_equal(kwargs["fruit"], 8)
    assert_equal(kwargs["fruit"], 8)

    assert_true("dessert" in kwargs)
    assert_equal(kwargs["dessert"], 9)
    assert_equal(kwargs["dessert"], 9)

    var keys = String()
    for key in kwargs.keys():
        keys += key
    assert_equal(keys, "fruitdessert")

    var sum = 0
    for val in kwargs.values():
        sum += val
    assert_equal(sum, 17)

    assert_false(kwargs.find("salad").__bool__())
    with assert_raises(contains="KeyError"):
        _ = kwargs["salad"]

    kwargs["salad"] = 10
    assert_equal(kwargs["salad"], 10)

    assert_equal(kwargs.pop("fruit"), 8)
    assert_equal(kwargs.pop("fruit", 2), 2)
    with assert_raises(contains="KeyError"):
        _ = kwargs.pop("fruit")

    keys = String()
    sum = 0
    for entry in kwargs.items():
        keys += entry.key
        sum += entry.value
    assert_equal(keys, "dessertsalad")
    assert_equal(sum, 19)


def test_owned_kwargs_dict() raises:
    var owned_kwargs = StringDict[Int]()
    owned_kwargs._insert("fruit", 8)
    owned_kwargs._insert("dessert", 9)
    _test_taking_owned_kwargs_dict(**owned_kwargs^)


def test_string_dict_string_span_getitem() raises:
    var kwargs = StringDict[Int]()
    kwargs._insert("fruit", 8)
    kwargs._insert("dessert", 9)

    # Index with a plain `StringSpan`, no `String` allocation required.
    assert_equal(kwargs[StringSpan("fruit")], 8)
    assert_equal(kwargs[StringSpan("dessert")], 9)

    # A view into a larger buffer resolves to the same entry as the `String`.
    var buffer = StringSpan("fruitcake")
    assert_equal(kwargs[buffer[byte=0:5]], kwargs[String("fruit")])

    # A missing key still raises `DictKeyError`.
    with assert_raises(contains="KeyError"):
        _ = kwargs[StringSpan("salad")]


def test_string_dict_write_to() raises:
    var kwargs = StringDict[Int]()
    kwargs._insert("a", 1)
    kwargs._insert("b", 2)
    check_write_to(kwargs, expected="{a: 1, b: 2}", is_repr=False)
    check_write_to(
        kwargs,
        expected="StringDict[SIMD[DType.int, 1]]({'a': Int(1), 'b': Int(2)})",
        is_repr=True,
    )


def test_string_dict_write_to_empty() raises:
    var kwargs = StringDict[Int]()
    check_write_to(kwargs, expected="{}", is_repr=False)
    check_write_to(
        kwargs,
        expected="StringDict[SIMD[DType.int, 1]]({})",
        is_repr=True,
    )


def test_find_get() raises:
    var some_dict: Dict[String, Int] = {}
    some_dict["key"] = 1
    assert_equal(some_dict.find("key").value(), 1)
    assert_equal(some_dict.get("key").value(), 1)
    assert_equal(some_dict.find("not_key").or_else(0), 0)
    assert_equal(some_dict.get("not_key", 0), 0)


def test_dict_popitem() raises:
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    assert_equal(len(dict), 2)
    var item = dict.popitem()
    assert_equal(item.key, "b")
    assert_equal(item.value, 2)
    assert_equal(len(dict), 1)
    item = dict.popitem()
    assert_equal(item.key, "a")
    assert_equal(item.value, 1)
    assert_equal(len(dict), 0)
    with assert_raises(contains="EmptyDictError"):
        _ = dict.popitem()


def test_dict_popitem_full_drain() raises:
    # `popitem` truncates the insertion-order array as it goes, so drain a dict
    # big enough to have resized and confirm LIFO order still holds throughout,
    # both with and without interleaved removals leaving tombstones behind.
    var dict: Dict[Int, Int] = {}
    for i in range(64):
        dict[i] = i * 10

    for i in range(63, -1, -1):
        var item = dict.popitem()
        assert_equal(item.key, i)
        assert_equal(item.value, i * 10)
        assert_equal(len(dict), i)

    with assert_raises(contains="EmptyDictError"):
        _ = dict.popitem()

    for i in range(32):
        dict[i] = i
    # Remove the odd keys, so the order array is half stale.
    for i in range(1, 32, 2):
        _ = dict.pop(i)

    var seen = List[Int]()
    while len(dict) > 0:
        seen.append(dict.popitem().key)
    assert_equal(len(seen), 16)
    for i in range(16):
        assert_equal(seen[i], 30 - 2 * i)

    # A dict emptied by `pop` rather than `popitem` still reports empty.
    dict[1] = 1
    _ = dict.pop(1)
    with assert_raises(contains="EmptyDictError"):
        _ = dict.popitem()


def test_pop_string_values() raises:
    var dict: Dict[String, String] = {}
    dict["mojo"] = "lang"
    dict["max"] = "engine"
    dict["a"] = ""
    dict[""] = "a"

    assert_equal(dict.pop("mojo"), "lang")
    assert_equal(dict.pop("max"), "engine")
    assert_equal(dict.pop("a"), "")
    assert_equal(dict.pop(""), "a")
    with assert_raises(contains="KeyError"):
        _ = dict.pop("absent")


def test_clear() raises:
    var some_dict: Dict[String, Int] = {}
    some_dict["key"] = 1
    some_dict.clear()
    assert_equal(len(some_dict), 0)
    assert_false(some_dict.get("key"))

    some_dict = Dict[String, Int]()
    some_dict.clear()
    assert_equal(len(some_dict), 0)


def test_init_initial_capacity() raises:
    var initial_capacity = 14
    var x = Dict[Int, Int](capacity=initial_capacity)
    assert_equal(x._reserved(), 16)
    for i in range(initial_capacity):
        x[i] = i
    for i in range(initial_capacity):
        assert_equal(i, x[i])

    var y = Dict[Int, Int](capacity=64)
    assert_equal(y._reserved(), 128)

    # Non-power-of-two capacity is rounded up
    var z = Dict[Int, Int](capacity=50)
    assert_equal(z._reserved(), 64)

    # Small capacity is clamped to minimum (16)
    var w = Dict[Int, Int](capacity=3)
    assert_equal(w._reserved(), 16)


def test_dict_setdefault() raises:
    var some_dict: Dict[String, Int] = {}
    some_dict["key1"] = 1
    some_dict["key2"] = 2
    assert_equal(some_dict.setdefault("key1", 0), 1)
    assert_equal(some_dict.setdefault("key2", 0), 2)
    assert_equal(some_dict.setdefault("not_key", 0), 0)
    assert_equal(some_dict["not_key"], 0)

    # Check that there is no copy of the default value, so it's performant
    var other_dict: Dict[String, CopyCounter[]] = {}
    var a = CopyCounter()
    var a_def = CopyCounter()
    var b_def = CopyCounter()
    other_dict["a"] = a^
    assert_equal(0, other_dict["a"].copy_count)
    _ = other_dict.setdefault("a", a_def^)
    _ = other_dict.setdefault("b", b_def^)
    assert_equal(0, other_dict["a"].copy_count)
    assert_equal(0, other_dict["b"].copy_count)


def test_compile_time_dict() raises:
    comptime N = 10

    def _get_dict() -> Dict[String, Int32, default_comp_time_hasher]:
        var res = Dict[String, Int32, default_comp_time_hasher]()
        for i in range(N):
            res[String(i)] = Int32(i)
        return res^

    comptime my_dict = _get_dict()

    comptime for i in range(N):
        comptime val = my_dict.get(String(i)).value()
        assert_equal(val, Int32(i))


def test_dict_comprehension() raises:
    var d1 = {x: x * x for x in range(10) if x & 1}
    assert_equal(d1, {1: 1, 3: 9, 5: 25, 7: 49, 9: 81})

    var s2 = {a * b: b for a in ["foo", "bar"] for b in [1, 2]}
    var d1reference = {
        "foo": 1,
        "bar": 1,
        "foofoo": 2,
        "barbar": 2,
    }
    assert_equal(s2, d1reference)


def test_dict_repr_wrap() raises:
    var tmp_dict = {"one": 1.0, "two": 2.0}
    assert_equal(
        repr(tmp_dict),
        (
            "Dict[String, SIMD[DType.float64, 1]]"
            "({'one': Float64(1.0), "
            "'two': Float64(2.0)})"
        ),
    )


def test_popitem_no_copies() raises:
    var dict: Dict[String, CopyCounter[]] = {}
    dict["a"] = CopyCounter()
    dict["b"] = CopyCounter()

    assert_equal(len(dict), 2)
    var item = dict.popitem()
    assert_equal(item.key, "b")
    assert_equal(item.value.copy_count, 0)
    assert_equal(len(dict), 1)
    item = dict.popitem()
    assert_equal(item.key, "a")
    assert_equal(item.value.copy_count, 0)
    assert_equal(len(dict), 0)
    with assert_raises(contains="EmptyDictError"):
        _ = dict.popitem()


def test_dict_key_error_repr() raises:
    var e = DictKeyError[Int]()
    check_write_to(
        e, expected="DictKeyError[SIMD[DType.int, 1]]()", is_repr=False
    )
    check_write_to(
        e, expected="DictKeyError[SIMD[DType.int, 1]]()", is_repr=True
    )


def test_empty_dict_error_repr() raises:
    var e = EmptyDictError()
    check_write_to(e, expected="EmptyDictError()", is_repr=False)
    check_write_to(e, expected="EmptyDictError()", is_repr=True)


def test_high_fill() raises:
    """Fill a dict near its 7/8 load factor to exercise resize triggers."""
    var d = Dict[Int, Int]()
    # Insert enough to trigger multiple resizes (initial capacity is 16,
    # 7/8 load factor means resize at 14, then at 28, 56, 112, etc.)
    for i in range(200):
        d[i] = i * 2
    assert_equal(len(d), 200)
    # Verify all entries survived resizes
    for i in range(200):
        assert_equal(d[i], i * 2)


def test_tombstone_accumulation() raises:
    """Repeatedly insert and delete to accumulate tombstones without resize."""
    var d = Dict[Int, Int]()
    # Pre-fill with 10 entries
    for i in range(10):
        d[i] = i
    # Insert and delete many transient entries to create tombstones
    for i in range(100, 500):
        d[i] = i
        _ = d.pop(i)
    # Original entries must still be found correctly despite tombstones
    assert_equal(len(d), 10)
    for i in range(10):
        assert_equal(d[i], i)


def test_ctrl_mirroring_boundary() raises:
    """Keys landing near the end of the ctrl array exercise mirror bytes."""
    var d = Dict[Int, Int](capacity=16)
    # Insert keys that, when hashed, are likely to probe at positions
    # near the end of the 16-slot table (positions 14, 15) where SIMD
    # loads read into the mirror region.
    for i in range(16):
        d[i] = i
    # All entries must be findable
    for i in range(16):
        assert_equal(d[i], i)
    # Delete some near-boundary entries and re-verify
    _ = d.pop(14)
    _ = d.pop(15)
    assert_false(14 in d)
    assert_false(15 in d)
    for i in range(14):
        assert_equal(d[i], i)


def test_delete_and_relookup() raises:
    """Delete entries then look them up to ensure correct miss detection."""
    var d = Dict[Int, Int]()
    for i in range(50):
        d[i] = i
    # Delete every other entry
    for i in range(0, 50, 2):
        _ = d.pop(i)
    assert_equal(len(d), 25)
    # Deleted keys must not be found
    for i in range(0, 50, 2):
        assert_false(i in d)
    # Remaining keys must still be found
    for i in range(1, 50, 2):
        assert_equal(d[i], i)


def test_order_preserved_after_heavy_deletion() raises:
    """Insertion order is preserved even after many deletions."""
    var d = Dict[Int, Int]()
    for i in range(20):
        d[i] = i
    # Delete first 10
    for i in range(10):
        _ = d.pop(i)
    # Iteration should yield 10..19 in insertion order
    var idx = 10
    for k in d:
        assert_equal(k, idx)
        idx += 1
    assert_equal(idx, 20)


def test_order_compaction() raises:
    """The _order array is compacted when it has too many stale entries."""
    # Use a large initial capacity so inserts+deletes don't trigger resize,
    # allowing stale entries to accumulate in _order until compaction fires.
    var d = Dict[Int, Int](capacity=1024)
    # Insert 100 entries (well under 7/8 * 1024 = 896)
    for i in range(100):
        d[i] = i
    # Delete 90, leaving 10 live entries but 100 stale _order entries
    for i in range(90):
        _ = d.pop(i)
    assert_equal(len(d), 10)
    # Now insert new entries. Each insert calls _ensure_capacity which
    # checks compaction (len(_order) > 2 * _len). With 100 order entries
    # and 10 live, compaction should trigger on the next insert.
    d[1000] = 1000
    assert_equal(len(d), 11)
    # Verify all live entries are intact and iteration order is correct
    for i in range(90, 100):
        assert_equal(d[i], i)
    assert_equal(d[1000], 1000)
    # Verify iteration yields entries in insertion order
    var keys = List[Int]()
    for k in d:
        keys.append(k)
    # Should be 90..99 (surviving originals) then 1000
    for i in range(10):
        assert_equal(keys[i], 90 + i)
    assert_equal(keys[10], 1000)


def test_reversed_items() raises:
    """Reversed item iteration must use _order, not _capacity."""
    # Fresh dict: reversed items should be reverse insertion order
    var d = Dict[String, Int]()
    d["a"] = 1
    d["b"] = 2
    d["c"] = 3
    var keys = List[String]()
    var vals = List[Int]()
    for item in reversed(d.items()):
        keys.append(item.key)
        vals.append(item.value)
    assert_equal(len(keys), 3)
    assert_equal(keys[0], "c")
    assert_equal(keys[1], "b")
    assert_equal(keys[2], "a")
    assert_equal(vals[0], 3)
    assert_equal(vals[1], 2)
    assert_equal(vals[2], 1)

    # After deletions: stale _order entries skipped in reverse
    _ = d.pop("b")
    keys = List[String]()
    for item in reversed(d.items()):
        keys.append(item.key)
    assert_equal(len(keys), 2)
    assert_equal(keys[0], "c")
    assert_equal(keys[1], "a")

    # After delete + re-insert: new entries appear at end of _order
    d["x"] = 10
    keys = List[String]()
    vals = List[Int]()
    for item in reversed(d.items()):
        keys.append(item.key)
        vals.append(item.value)
    assert_equal(len(keys), 3)
    assert_equal(keys[0], "x")
    assert_equal(keys[1], "c")
    assert_equal(keys[2], "a")
    assert_equal(vals[0], 10)
    assert_equal(vals[1], 3)
    assert_equal(vals[2], 1)

    # Empty dict: reversed items yields nothing
    var empty = Dict[String, Int]()
    var count = 0
    for _ in reversed(empty.items()):
        count += 1
    assert_equal(count, 0)


def test_minimum_capacity() raises:
    """Once allocated, the minimum capacity is GROUP_WIDTH (16) for SIMD correctness.
    """
    var d = Dict[Int, Int](capacity=16)
    assert_true(d._reserved() >= GROUP_WIDTH)
    # Default constructor is lazy and allocates GROUP_WIDTH on first insertion.
    var d2 = Dict[Int, Int]()
    d2[0] = 0
    assert_true(d2._reserved() >= GROUP_WIDTH)


def test_inplace_rehash() raises:
    """In-place rehash reclaims tombstones without growing capacity."""
    var capacity = 16
    var max_load = capacity * 7 // 8  # 14
    var d = Dict[Int, Int](capacity=capacity)
    var initial_cap = d._reserved()

    # Fill to max_load so _growth_left = 0 after this
    for i in range(max_load):
        d[i] = i

    # Delete most entries -> _len well below capacity*7/16
    var keep = 4
    for i in range(max_load - keep):
        _ = d.pop(i)

    assert_equal(len(d), keep)

    # Next insert triggers _ensure_capacity. Since _len <= capacity*7/16,
    # should rehash in-place, NOT double capacity.
    d[100] = 100
    assert_equal(d._reserved(), initial_cap)
    assert_equal(len(d), keep + 1)

    # Verify all entries are findable
    for i in range(max_load - keep, max_load):
        assert_equal(d[i], i)
    assert_equal(d[100], 100)


def test_inplace_rehash_preserves_order() raises:
    """In-place rehash preserves iteration (insertion) order."""
    var capacity = 16
    var max_load = capacity * 7 // 8  # 14
    var d = Dict[Int, Int](capacity=capacity)
    for i in range(max_load):
        d[i] = i
    # Delete even keys
    for i in range(0, max_load, 2):
        _ = d.pop(i)
    # _len = 7, capacity*7/16 = 7, so this is on the boundary.
    # Insert one more to trigger rehash.
    d[99] = 99
    # Verify iteration order: odd keys 1,3,5,7,9,11,13 then 99
    var keys = List(d.keys())
    assert_equal(keys[0], 1)
    assert_equal(keys[1], 3)
    assert_equal(keys[2], 5)
    assert_equal(keys[3], 7)
    assert_equal(keys[4], 9)
    assert_equal(keys[5], 11)
    assert_equal(keys[6], 13)
    assert_equal(keys[7], 99)


def test_tombstone_heavy_no_capacity_growth() raises:
    """Repeated insert/delete cycles should not grow capacity unboundedly."""
    var capacity = 16
    var d = Dict[Int, Int](capacity=capacity)
    var initial_cap = d._reserved()

    # Repeated insert+delete of same key range - generates tombstones
    # but _len stays low, so in-place rehash should keep capacity stable.
    for cycle in range(20):
        for i in range(10):
            d[1000 + i] = cycle
        for i in range(10):
            _ = d.pop(1000 + i)

    # Capacity should stay at initial_cap thanks to in-place rehash
    assert_equal(d._reserved(), initial_cap)


def test_high_load_still_doubles() raises:
    """When most slots are genuinely occupied, resize should still double."""
    var max_load = 14
    var d = Dict[Int, Int](capacity=max_load)
    assert_equal(d._reserved(), 16)

    # Fill to capacity without deleting
    for i in range(max_load):
        d[i] = i

    # _len(14) > capacity*7/16(7), so next insert should double
    d[max_load] = max_load
    assert_equal(d._reserved(), 32)


def test_inplace_rehash_string_keys() raises:
    """In-place rehash works with non-trivial key types."""
    var capacity = 16
    var max_load = capacity * 7 // 8  # 14
    var d = Dict[String, String](capacity=capacity)

    # Fill to capacity
    for i in range(max_load):
        d[String("key", i)] = String("val", i)

    # Delete most entries
    for i in range(max_load - 4):
        _ = d.pop(String("key", i))

    assert_equal(len(d), 4)
    var cap_before = d._reserved()

    # Trigger in-place rehash
    d["new_key"] = "new_val"

    assert_equal(d._reserved(), cap_before)
    assert_equal(len(d), 5)

    # Verify all remaining entries
    for i in range(max_load - 4, max_load):
        assert_equal(d[String("key", i)], String("val", i))
    assert_equal(d["new_key"], "new_val")


def test_inplace_rehash_via_setdefault() raises:
    """`setdefault` triggers in-place rehash correctly."""
    var capacity = 16
    var max_load = capacity * 7 // 8  # 14
    var d = Dict[Int, Int](capacity=capacity)

    # Fill to capacity, then delete most
    for i in range(max_load):
        d[i] = i
    for i in range(max_load - 4):
        _ = d.pop(i)

    assert_equal(len(d), 4)
    var cap_before = d._reserved()

    # setdefault calls _ensure_capacity, should trigger in-place rehash
    var val = d.setdefault(200, 200)
    assert_equal(val, 200)
    assert_equal(d._reserved(), cap_before)
    assert_equal(len(d), 5)

    # Verify existing entries survived
    for i in range(max_load - 4, max_load):
        assert_equal(d[i], i)
    assert_equal(d[200], 200)


def test_inplace_rehash_all_deleted() raises:
    """In-place rehash when _len == 0 (all entries deleted)."""
    var capacity = 16
    var max_load = capacity * 7 // 8  # 14
    var d = Dict[Int, Int](capacity=capacity)
    var initial_cap = d._reserved()

    # Fill to capacity then delete everything
    for i in range(max_load):
        d[i] = i
    for i in range(max_load):
        _ = d.pop(i)

    assert_equal(len(d), 0)

    # _growth_left is 0 from the 14 inserts, _len(0) <= 7 -> in-place rehash
    d[999] = 999
    assert_equal(d._reserved(), initial_cap)
    assert_equal(len(d), 1)
    assert_equal(d[999], 999)


def test_compile_time_dict_with_rehash() raises:
    """Compile-time dict that triggers in-place rehash."""

    def _build_dict() -> Dict[String, Int32, default_comp_time_hasher]:
        var capacity = 16
        var max_load = capacity * 7 // 8  # 14
        var keep = 4
        var d = Dict[String, Int32, default_comp_time_hasher](capacity=capacity)
        # Fill to max_load
        for i in range(max_load):
            d[String(i)] = Int32(i)
        # Delete most entries to create tombstones
        for i in range(max_load - keep):
            _ = d.pop(String(i), -1)
        # This insert triggers _ensure_capacity -> in-place rehash at compile time
        d["ct"] = 42
        return d^

    comptime ct_dict = _build_dict()

    # Verify values survive compile-time in-place rehash
    comptime for i in range(10, 14):
        comptime val = ct_dict.get(String(i)).value()
        assert_equal(val, Int32(i))

    comptime ct_val = ct_dict.get("ct").value()
    assert_equal(ct_val, 42)


def test_inplace_rehash_via_update() raises:
    """`update()` on a tombstone-heavy dict triggers in-place rehash."""
    var capacity = 16
    var max_load = capacity * 7 // 8  # 14
    var d = Dict[Int, Int](capacity=capacity)

    # Fill to capacity, then delete most
    for i in range(max_load):
        d[i] = i
    for i in range(max_load - 4):
        _ = d.pop(i)

    assert_equal(len(d), 4)
    var cap_before = d._reserved()

    # update() inserts multiple entries; the first triggers in-place rehash
    var other = Dict[Int, Int]()
    for i in range(5):
        other[200 + i] = 200 + i
    d.update(other)

    assert_equal(d._reserved(), cap_before)
    assert_equal(len(d), 9)

    # Verify all entries
    for i in range(max_load - 4, max_load):
        assert_equal(d[i], i)
    for i in range(5):
        assert_equal(d[200 + i], 200 + i)


def test_dict_eq() raises:
    # Equal dicts
    assert_equal(Dict[String, Int](), Dict[String, Int]())
    assert_equal({"a": 1, "b": 2}, {"a": 1, "b": 2})

    # Different insertion order, same content
    var d1 = Dict[String, Int]()
    d1["b"] = 2
    d1["a"] = 1
    var d2 = Dict[String, Int]()
    d2["a"] = 1
    d2["b"] = 2
    assert_equal(d1, d2)

    # Different lengths
    assert_true({"a": 1} != {"a": 1, "b": 2})
    assert_true({"a": 1, "b": 2} != {"a": 1})

    # Different values
    assert_true({"a": 1} != {"a": 2})

    # Different keys
    assert_true({"a": 1} != {"b": 1})


def test_dict_hash() raises:
    # Same content, same hash
    assert_equal(hash({"a": 1, "b": 2}), hash({"a": 1, "b": 2}))

    # Different insertion order, same hash
    var d1 = Dict[String, Int]()
    d1["b"] = 2
    d1["a"] = 1
    var d2 = Dict[String, Int]()
    d2["a"] = 1
    d2["b"] = 2
    assert_equal(hash(d1), hash(d2))

    # Different content, different hash (probabilistic but extremely unlikely
    # to collide for these simple values)
    assert_true(hash({"a": 1}) != hash({"a": 2}))
    assert_true(hash({"a": 1}) != hash({"b": 1}))

    # Empty dicts
    assert_equal(hash(Dict[String, Int]()), hash(Dict[String, Int]()))


def test_dict_float_signed_zero_key() raises:
    var d = Dict[Float64, String]()
    d[Float64(0.0)] = "zero"
    d[Float64(-0.0)] = "negzero"

    # `-0.0 == 0.0`, so the second insert overwrites rather than adding a key.
    assert_equal(len(d), 1)
    assert_equal(d[Float64(0.0)], "negzero")
    assert_equal(d[Float64(-0.0)], "negzero")


struct NonWritable(Copyable, Deinitable):
    pass


def test_dict_conditional_conformances() raises:
    assert_true(conforms_to(Dict[Int, Int], Writable))
    assert_true(conforms_to(Dict[Int, Int], Equatable))
    assert_true(conforms_to(Dict[Int, Int], Hashable))
    assert_false(conforms_to(Dict[Int, NonWritable], Writable))

    # Owned iteration should work for any combination of non-Copyable K/V types
    assert_true(conforms_to(Dict[MoveOnly[Int], Int], IterableOwned))
    assert_true(conforms_to(Dict[Int, MoveOnly[Int]], IterableOwned))
    assert_true(conforms_to(Dict[MoveOnly[Int], MoveOnly[Int]], IterableOwned))

    # Move-only key drops every copy-requiring conformance: each conditional
    # clause on `Dict` includes `conforms_to(K, Copyable)`.
    assert_false(conforms_to(Dict[MoveOnly[Int], Int], Copyable))
    assert_false(conforms_to(Dict[MoveOnly[Int], Int], Equatable))
    assert_false(conforms_to(Dict[MoveOnly[Int], Int], Hashable))
    assert_false(conforms_to(Dict[MoveOnly[Int], Int], Writable))

    # Move-only value: only `Copyable` is dropped. `MoveOnly[Int]` is itself
    # conditionally `Equatable`/`Hashable`/`Writable` when its payload is, so
    # `Dict[Int, MoveOnly[Int]]` keeps those conformances (K is `Copyable`).
    assert_false(conforms_to(Dict[Int, MoveOnly[Int]], Copyable))
    assert_true(conforms_to(Dict[Int, MoveOnly[Int]], Equatable))
    assert_true(conforms_to(Dict[Int, MoveOnly[Int]], Hashable))
    assert_true(conforms_to(Dict[Int, MoveOnly[Int]], Writable))

    # Both axes move-only: K-side `Copyable` failure drops everything.
    assert_false(conforms_to(Dict[MoveOnly[Int], MoveOnly[Int]], Copyable))
    assert_false(conforms_to(Dict[MoveOnly[Int], MoveOnly[Int]], Equatable))
    assert_false(conforms_to(Dict[MoveOnly[Int], MoveOnly[Int]], Hashable))
    assert_false(conforms_to(Dict[MoveOnly[Int], MoveOnly[Int]], Writable))


def test_dict_iter_owned() raises:
    # Test that owned iteration works, for non-Copyable types
    var d = Dict[MoveOnly[String], Int]()
    d[MoveOnly("a")] = 1
    d[MoveOnly("b")] = 2
    d[MoveOnly("c")] = 3

    var keys = List[MoveOnly[String]]()
    for var key in d^:
        keys.append(key^)

    assert_equal(len(keys), 3)
    assert_equal(keys[0], "a")
    assert_equal(keys[1], "b")
    assert_equal(keys[2], "c")


def test_dict_iter_owned_destroys_elements_if_not_consumed() raises:
    var del_count = 0
    var ptr = Pointer(to=del_count).as_imm().as_unsafe_any_origin()
    var d = Dict[Int, DelCounter[ptr.origin]]()
    d[1] = DelCounter(ptr)
    d[2] = DelCounter(ptr)
    d[3] = DelCounter(ptr)
    assert_equal(del_count, 0)

    # Create the owned iterator but never consume it; all values should
    # still be destroyed when the iterator is dropped.
    var _ = d^.__iter__()
    assert_equal(del_count, 3)


def test_dict_iter_owned_destroys_elements_if_partially_consumed() raises:
    var del_count = 0
    var ptr = Pointer(to=del_count).as_imm().as_unsafe_any_origin()
    var d = Dict[Int, DelCounter[ptr.origin]]()
    d[1] = DelCounter(ptr)
    d[2] = DelCounter(ptr)
    d[3] = DelCounter(ptr)
    assert_equal(del_count, 0)

    var it = d^.__iter__()
    _ = it.__next__()  # consume one key; entry (including value) is dropped
    assert_equal(del_count, 1)

    # Drop the iterator with two unconsumed entries remaining.
    _ = it^
    assert_equal(del_count, 3)


def test_dict_iter_owned_bounds() raises:
    var d = Dict[String, Int]()
    d["a"] = 1
    d["b"] = 2
    d["c"] = 3

    var it = d^.__iter__()
    assert_equal(it.bounds()[0], 3)
    _ = it.__next__()
    assert_equal(it.bounds()[0], 2)
    _ = it.__next__()
    assert_equal(it.bounds()[0], 1)
    _ = it.__next__()
    assert_equal(it.bounds()[0], 0)


def test_dict_move_only_value() raises:
    # `MoveOnly[Int]` is not `Copyable`; this exercises the conditional
    # conformance path of `Dict[K, V: Movable & Deinitable, H]`.
    assert_false(conforms_to(Dict[String, MoveOnly[Int]], Copyable))

    var d = Dict[String, MoveOnly[Int]]()
    d["a"] = MoveOnly[Int](1)
    d["b"] = MoveOnly[Int](2)
    d["c"] = MoveOnly[Int](3)
    assert_equal(d["a"], MoveOnly[Int](1))
    assert_equal(d["b"], MoveOnly[Int](2))
    assert_equal(d["c"], MoveOnly[Int](3))
    assert_equal(len(d), 3)
    assert_true("a" in d)
    assert_false("missing" in d)

    # `pop` moves the value out.
    var v = d.pop("a")
    assert_equal(v, MoveOnly[Int](1))
    assert_equal(len(d), 2)
    assert_false("a" in d)

    # `popitem` returns an owned entry, draining the dict.
    var seen: Int = 0
    while len(d) > 0:
        var entry = d.popitem()
        seen += 1
        _ = entry^
    assert_equal(seen, 2)
    assert_equal(len(d), 0)


def test_dict_move_only_key() raises:
    # `MoveOnly[Int]` is not `Copyable`; this exercises the conditional
    # conformance path of `Dict[K: Movable & Hashable & Equatable, V, H]`
    # where the key type is move-only.
    assert_false(conforms_to(Dict[MoveOnly[Int], Int], Copyable))

    var d = Dict[MoveOnly[Int], Int]()
    d[MoveOnly[Int](1)] = 10
    d[MoveOnly[Int](2)] = 20
    d[MoveOnly[Int](3)] = 30
    assert_equal(d[MoveOnly[Int](1)], 10)
    assert_equal(d[MoveOnly[Int](2)], 20)
    assert_equal(d[MoveOnly[Int](3)], 30)
    assert_equal(len(d), 3)
    assert_true(MoveOnly[Int](1) in d)
    assert_false(MoveOnly[Int](99) in d)

    # Updating an existing key by `__setitem__` moves the new key in.
    d[MoveOnly[Int](1)] = 100
    assert_equal(d[MoveOnly[Int](1)], 100)
    assert_equal(len(d), 3)

    # `pop(key)` removes by key without copying the key.
    var v = d.pop(MoveOnly[Int](2))
    assert_equal(v, 20)
    assert_equal(len(d), 2)
    assert_false(MoveOnly[Int](2) in d)

    # `popitem` returns an owned entry by moving the key out.
    var seen: Int = 0
    while len(d) > 0:
        var entry = d.popitem()
        seen += 1
        _ = entry^
    assert_equal(seen, 2)
    assert_equal(len(d), 0)

    # `setdefault` takes the key by `var`, so it moves a move-only key in.
    d = Dict[MoveOnly[Int], Int]()
    ref existing = d.setdefault(MoveOnly[Int](1), 10)
    assert_equal(existing, 10)
    assert_equal(len(d), 1)
    ref already = d.setdefault(MoveOnly[Int](1), 999)
    assert_equal(already, 10)
    assert_equal(len(d), 1)

    # `pop(key, default)` falls back to the default for missing keys.
    assert_equal(d.pop(MoveOnly[Int](42), 7), 7)
    assert_equal(len(d), 1)
    assert_equal(d.pop(MoveOnly[Int](1), 7), 10)
    assert_equal(len(d), 0)

    # `__bool__` and `clear` on a move-only-keyed dict.
    d[MoveOnly[Int](1)] = 1
    assert_true(d.__bool__())
    d.clear()
    assert_false(d.__bool__())
    assert_equal(len(d), 0)


def test_dict_move_only_key_and_value() raises:
    # Both K and V are move-only: confirms the orthogonal conditional
    # conformance clauses on `Dict[K, V, H]` compose correctly.
    assert_false(conforms_to(Dict[MoveOnly[Int], MoveOnly[Int]], Copyable))

    var d = Dict[MoveOnly[Int], MoveOnly[Int]]()
    d[MoveOnly[Int](1)] = MoveOnly[Int](10)
    d[MoveOnly[Int](2)] = MoveOnly[Int](20)
    assert_equal(len(d), 2)
    assert_equal(d[MoveOnly[Int](1)], MoveOnly[Int](10))
    assert_true(MoveOnly[Int](2) in d)

    var v = d.pop(MoveOnly[Int](1))
    assert_equal(v, MoveOnly[Int](10))
    assert_equal(len(d), 1)

    var entry = d.popitem()
    _ = entry^
    assert_equal(len(d), 0)

    # `setdefault` moves both the key and the default value in.
    ref inserted = d.setdefault(MoveOnly[Int](1), MoveOnly[Int](10))
    assert_equal(inserted, MoveOnly[Int](10))
    assert_equal(len(d), 1)
    ref already = d.setdefault(MoveOnly[Int](1), MoveOnly[Int](999))
    assert_equal(already, MoveOnly[Int](10))
    assert_equal(len(d), 1)


def test_dict_conditional_implicitly_deletable() raises:
    assert_true(conforms_to(Dict[Int, Int], Deinitable))

    assert_false(conforms_to(Dict[Int, ExplicitDestroy], Deinitable))


def test_dict_deinit_with() raises:
    # `deinit_with` must hand every entry's key/value to the closure exactly
    # once. Uses a deletable value type because populating a linear-valued dict
    # isn't supported yet (tracked in the linear-usability follow-up).
    var d = Dict[Int, Int]()
    d[1] = 10
    d[2] = 20
    d[3] = 30

    var destroyed = List[Int]()

    def dispose(var key: Int, var value: Int) {mut}:
        destroyed.append(value)

    d^.deinit_with(dispose)

    # Order follows slot layout, not insertion, so check membership.
    assert_equal(len(destroyed), 3)
    assert_true(10 in destroyed)
    assert_true(20 in destroyed)
    assert_true(30 in destroyed)


def test_dict_deinit_with_empty() raises:
    # `deinit_with` on an empty (linear-valued) dict must run and free the
    # backing without invoking the closure — there are no entries.
    var d = Dict[Int, ExplicitDestroy]()
    var calls = 0

    def dispose(var key: Int, var value: ExplicitDestroy) {mut}:
        calls += 1
        value^.destroy()

    d^.deinit_with(dispose)
    assert_equal(calls, 0)


def test_dict_clear_with() raises:
    # `clear_with` must hand every entry's key/value to the closure exactly
    # once, empty the dict, and leave it reusable (capacity retained). Uses a
    # deletable value type since the disposal path is what's under test.
    var d = Dict[Int, Int]()
    d[1] = 10
    d[2] = 20
    d[3] = 30

    var cleared = List[Int]()

    def dispose(var key: Int, var value: Int) {mut}:
        cleared.append(value)

    d.clear_with(dispose)

    # Every entry disposed exactly once, and the dict is now empty.
    assert_equal(len(cleared), 3)
    assert_true(10 in cleared)
    assert_true(20 in cleared)
    assert_true(30 in cleared)
    assert_equal(len(d), 0)

    # Capacity is retained, so the dict is immediately reusable.
    d[4] = 40
    assert_equal(len(d), 1)
    assert_equal(d[4], 40)


def test_dict_clear_with_empty() raises:
    var d = Dict[Int, ExplicitDestroy]()
    var calls = 0

    def dispose(var key: Int, var value: ExplicitDestroy) {mut}:
        calls += 1
        value^.destroy()

    d.clear_with(dispose)
    d^.deinit_with(dispose)
    assert_equal(calls, 0)


def test_dict_clear_with_linear() raises:
    # `clear_with` on a populated linear `Dict` (neither key nor value is
    # `Deinitable`). Populate via `insert`, then verify every entry
    # reaches the closure once, the dict empties, and its capacity is reused.
    var d = Dict[ExplicitDestroyKey, ExplicitDestroy]()
    var disposed = List[Int]()
    var disposed_keys = List[Int]()

    def dispose_kv(
        var key: ExplicitDestroyKey, var value: ExplicitDestroy
    ) {mut}:
        disposed.append(value.value)
        disposed_keys.append(key.value)
        key^.destroy()
        value^.destroy()

    def dispose_entry(
        var entry: DictEntry[
            ExplicitDestroyKey, ExplicitDestroy, default_hasher
        ]
    ) {mut}:
        entry^.deinit_with(dispose_kv)

    d.insert(ExplicitDestroyKey(1), ExplicitDestroy(10)).deinit_with(
        dispose_entry
    )
    d.insert(ExplicitDestroyKey(2), ExplicitDestroy(20)).deinit_with(
        dispose_entry
    )
    d.insert(ExplicitDestroyKey(3), ExplicitDestroy(30)).deinit_with(
        dispose_entry
    )
    var len_before_clear = len(d)

    d.clear_with(dispose_kv)

    var cleared_values = disposed.copy()
    var cleared_keys = disposed_keys.copy()
    var len_after_clear = len(d)

    # Capacity is retained, so the emptied dict is reusable.
    d.insert(ExplicitDestroyKey(4), ExplicitDestroy(40)).deinit_with(
        dispose_entry
    )
    var len_after_reuse = len(d)

    d^.deinit_with(dispose_kv)

    assert_equal(len_before_clear, 3)
    assert_equal(len_after_clear, 0)
    assert_equal(len_after_reuse, 1)

    assert_equal(len(cleared_values), 3)
    assert_true(10 in cleared_values)
    assert_true(20 in cleared_values)
    assert_true(30 in cleared_values)

    assert_equal(len(cleared_keys), 3)
    assert_true(1 in cleared_keys)
    assert_true(2 in cleared_keys)
    assert_true(3 in cleared_keys)

    assert_equal(len(disposed), 4)
    assert_true(40 in disposed)


def test_dict_insert_linear_key_and_value() raises:
    # A linear `Dict` (and a linear `Optional[DictEntry]`) has no implicit
    # destructor, so every linear value is consumed via `deinit_with` before
    # any (raising) assert runs, per the linear-in-`raises` idiom.
    var d = Dict[ExplicitDestroyKey, ExplicitDestroy]()
    var disposed = List[Int]()
    var disposed_keys = List[Int]()

    def dispose_kv(
        var key: ExplicitDestroyKey, var value: ExplicitDestroy
    ) {mut}:
        disposed.append(value.value)
        disposed_keys.append(key.value)
        key^.destroy()
        value^.destroy()

    def dispose_entry(
        var entry: DictEntry[
            ExplicitDestroyKey, ExplicitDestroy, default_hasher
        ]
    ) {mut}:
        entry^.deinit_with(dispose_kv)

    # New keys: no displaced entry, so each returned `Optional` is empty.
    var r1 = d.insert(ExplicitDestroyKey(1), ExplicitDestroy(10))
    var r1_present = Bool(r1)
    r1^.deinit_with(dispose_entry)

    var r2 = d.insert(ExplicitDestroyKey(2), ExplicitDestroy(20))
    var r2_present = Bool(r2)
    r2^.deinit_with(dispose_entry)

    var len_after_new = len(d)

    # Overwrite an existing key: the displaced entry (old key 1, value 10)
    # comes back — not the just-inserted value 99.
    var r3 = d.insert(ExplicitDestroyKey(1), ExplicitDestroy(99))
    var r3_present = Bool(r3)
    r3^.deinit_with(dispose_entry)

    # Snapshot what the overwrite disposed, before teardown adds survivors.
    # A store/return swap (keep old, return new) would still leave the same
    # {10, 20, 99} disposed overall, so the final membership checks alone
    # can't catch it — pinning the displaced pair here can.
    var overwrite_disposed_values = disposed.copy()
    var overwrite_disposed_keys = disposed_keys.copy()

    var len_after_overwrite = len(d)

    # Tear down the linear dict explicitly, disposing the surviving entries.
    d^.deinit_with(dispose_kv)

    # All linear values consumed; asserts may raise freely now.
    assert_false(r1_present)
    assert_false(r2_present)
    assert_true(r3_present)
    assert_equal(len_after_new, 2)
    assert_equal(len_after_overwrite, 2)

    # 1. The overwrite displaced exactly the old pair: value 10, key 1.
    assert_equal(len(overwrite_disposed_values), 1)
    assert_equal(overwrite_disposed_values[0], 10)
    assert_equal(len(overwrite_disposed_keys), 1)
    assert_equal(overwrite_disposed_keys[0], 1)

    # Displaced 10 (overwrite) + surviving 20 and 99 (teardown).
    assert_equal(len(disposed), 3)
    assert_true(10 in disposed)
    assert_true(20 in disposed)
    assert_true(99 in disposed)

    # 2. Keys are explicitly destroyed too: old key 1 (displaced) plus the
    # surviving keys 1 and 2 at teardown.
    assert_equal(len(disposed_keys), 3)


def _test_taking_owned_kwargs_dict_linear_popitem(
    var **kwargs: ExplicitDestroy,
) raises:
    var disposed_keys = List[String]()
    var disposed_vals = List[Int]()

    def dispose_kwarg(var key: String, var value: ExplicitDestroy) {mut}:
        disposed_keys.append(key)
        disposed_vals.append(value.value)
        value^.destroy()

    while True:
        try:
            var entry = kwargs.popitem()
            entry^.deinit_with(dispose_kwarg)
        except e:
            break  # EmptyDictError: kwargs is now empty

    kwargs^.deinit_with(dispose_kwarg)  # empty container still needs teardown

    assert_equal(len(disposed_vals), 2)
    assert_true("linear_fruit" in disposed_keys)
    assert_true("linear_dessert" in disposed_keys)
    assert_true(8 in disposed_vals)
    assert_true(9 in disposed_vals)


def _test_taking_owned_kwargs_dict_linear_insert_pop(
    var **kwargs: ExplicitDestroy,
) raises:
    var disposed = List[Int]()

    def dispose_kwarg(var key: String, var value: ExplicitDestroy) {mut}:
        disposed.append(value.value)
        value^.destroy()

    def dispose_entry(
        var entry: DictEntry[String, ExplicitDestroy, default_comp_time_hasher]
    ) {mut}:
        entry^.deinit_with(dispose_kwarg)

    var popped_val: Int
    var len_after_pop: Int
    try:
        var displaced = kwargs.insert("linear_dessert", ExplicitDestroy(12))
        displaced^.deinit_with(dispose_entry)

        var popped = kwargs.pop("linear_dessert")
        popped_val = popped.value
        popped^.destroy()

        len_after_pop = len(kwargs)
        kwargs^.deinit_with(dispose_kwarg)  # success: consume once
    except e:
        kwargs^.deinit_with(dispose_kwarg)  # error: consume once
        raise e  # then bail

    assert_equal(disposed[0], 9)  # displaced old dessert
    assert_equal(popped_val, 12)  # popped the replacement value
    assert_equal(len_after_pop, 1)  # started 2 (fruit, dessert), popped dessert


def test_owned_kwargs_dict_linear() raises:
    def dispose_kwarg(var key: String, var value: ExplicitDestroy) {mut}:
        value^.destroy()

    def dispose_val(
        var entry: DictEntry[String, ExplicitDestroy, default_comp_time_hasher]
    ) {mut}:
        entry^.deinit_with(dispose_kwarg)

    var kw1 = StringDict[ExplicitDestroy]()
    var a1 = kw1.insert("linear_fruit", ExplicitDestroy(8))
    a1^.deinit_with(dispose_val)
    var a2 = kw1.insert("linear_dessert", ExplicitDestroy(9))
    a2^.deinit_with(dispose_val)
    _test_taking_owned_kwargs_dict_linear_popitem(**kw1^)

    var kw2 = StringDict[ExplicitDestroy]()
    var b1 = kw2.insert("linear_fruit", ExplicitDestroy(8))
    b1^.deinit_with(dispose_val)
    var b2 = kw2.insert("linear_dessert", ExplicitDestroy(9))
    b2^.deinit_with(dispose_val)
    _test_taking_owned_kwargs_dict_linear_insert_pop(**kw2^)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
