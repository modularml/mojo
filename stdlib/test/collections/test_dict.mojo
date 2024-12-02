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

from collections import Dict, KeyElement, Optional
from collections.dict import OwnedKwargsDict

from test_utils import CopyCounter, ValueDestructorRecorder
from testing import assert_equal, assert_false, assert_raises, assert_true
from memory import UnsafePointer


def test_dict_construction():
    _ = Dict[Int, Int]()
    _ = Dict[String, Int]()


def test_dict_fromkeys():
    alias keys = List[String]("a", "b")
    var expected_dict = Dict[String, Int]()
    expected_dict["a"] = 1
    expected_dict["b"] = 1
    var dict = Dict.fromkeys(keys, 1)

    assert_equal(len(dict), len(expected_dict))

    for k_v in expected_dict.items():
        var k = k_v[].key
        var v = k_v[].value
        assert_true(k in dict)
        assert_equal(dict[k], v)


def test_dict_fromkeys_optional():
    alias keys = List[String]("a", "b", "c")
    var expected_dict = Dict[String, Optional[Int]]()
    expected_dict["a"] = None
    expected_dict["b"] = None
    expected_dict["c"] = None
    var dict = Dict[_, Int].fromkeys(keys)

    assert_equal(len(dict), len(expected_dict))

    for k_v in expected_dict.items():
        var k = k_v[].key
        var v = k_v[].value
        assert_true(k in dict)
        assert_false(v)


def test_basic():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2

    assert_equal(1, dict["a"])
    assert_equal(2, dict["b"])


def test_basic_no_copies():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2

    assert_equal(1, dict["a"])
    assert_equal(2, dict["b"])


def test_multiple_resizes():
    var dict = Dict[String, Int]()
    for i in range(20):
        dict["key" + str(i)] = i + 1
    assert_equal(11, dict["key10"])
    assert_equal(20, dict["key19"])


def test_bool_conversion():
    var dict = Dict[String, Int]()
    assert_false(dict)
    dict["a"] = 1
    assert_true(dict)
    dict["b"] = 2
    assert_true(dict)
    _ = dict.pop("a")
    assert_true(dict)
    _ = dict.pop("b")
    assert_false(dict)


def test_big_dict():
    var dict = Dict[String, Int]()
    for i in range(2000):
        dict["key" + str(i)] = i + 1
    assert_equal(2000, len(dict))


def test_dict_string_representation_string_int():
    var some_dict = Dict[String, Int]()
    some_dict["a"] = 1
    some_dict["b"] = 2
    dict_as_string = some_dict.__str__()
    assert_true(
        some_dict._minimum_size_of_string_representation()
        <= len(dict_as_string)
    )
    assert_equal(dict_as_string, "{'a': 1, 'b': 2}")


def test_dict_string_representation_int_int():
    var some_dict = Dict[Int, Int]()
    some_dict[3] = 1
    some_dict[4] = 2
    some_dict[5] = 3
    some_dict[6] = 4
    dict_as_string = some_dict.__str__()
    # one char per key and value, we should have the minimum size of string possible
    assert_equal(
        some_dict._minimum_size_of_string_representation(), len(dict_as_string)
    )
    assert_equal(dict_as_string, "{3: 1, 4: 2, 5: 3, 6: 4}")


def test_compact():
    var dict = Dict[String, Int]()
    for i in range(20):
        var key = "key" + str(i)
        dict[key] = i + 1
        _ = dict.pop(key)
    assert_equal(0, len(dict))


def test_compact_with_elements():
    var dict = Dict[String, Int]()
    for i in range(5):
        var key = "key" + str(i)
        dict[key] = i + 1
    for i in range(5, 20):
        var key = "key" + str(i)
        dict[key] = i + 1
        _ = dict.pop(key)
    assert_equal(5, len(dict))


def test_pop_default():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2

    assert_equal(1, dict.pop("a", -1))
    assert_equal(2, dict.pop("b", -1))
    assert_equal(-1, dict.pop("c", -1))


def test_key_error():
    var dict = Dict[String, Int]()

    with assert_raises(contains="KeyError"):
        _ = dict["a"]
    with assert_raises(contains="KeyError"):
        _ = dict.pop("a")


def test_iter():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2

    var keys = String("")
    for key in dict:
        keys += key[]

    assert_equal(keys, "ab")


def test_iter_keys():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2

    var keys = String("")
    for key in dict.keys():
        keys += key[]

    assert_equal(keys, "ab")


def test_iter_values():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2

    var sum = 0
    for value in dict.values():
        sum += value[]

    assert_equal(sum, 3)


def test_iter_values_mut():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2

    for value in dict.values():
        value[] += 1

    assert_equal(2, dict["a"])
    assert_equal(3, dict["b"])
    assert_equal(2, len(dict))


def test_iter_items():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2

    var keys = String("")
    var sum = 0
    for entry in dict.items():
        keys += entry[].key
        sum += entry[].value

    assert_equal(keys, "ab")
    assert_equal(sum, 3)


def test_dict_copy():
    var orig = Dict[String, Int]()
    orig["a"] = 1

    # test values copied to new Dict
    var copy = Dict(other=orig)
    assert_equal(1, copy["a"])

    # test there are two copies of dict and
    # they don't share underlying memory
    copy["a"] = 2
    assert_equal(2, copy["a"])
    assert_equal(1, orig["a"])


def test_dict_copy_delete_original():
    var orig = Dict[String, Int]()
    orig["a"] = 1

    # test values copied to new Dict
    var copy = Dict(other=orig)
    # don't access the original dict, anymore, confirm that
    # deleting the original doesn't violate the integrity of the copy
    assert_equal(1, copy["a"])


def test_dict_copy_add_new_item():
    var orig = Dict[String, Int]()
    orig["a"] = 1

    # test values copied to new Dict
    var copy = Dict(other=orig)
    assert_equal(1, copy["a"])

    # test there are two copies of dict and
    # they don't share underlying memory
    copy["b"] = 2
    assert_false(str(2) in orig)


def test_dict_copy_calls_copy_constructor():
    var orig = Dict[String, CopyCounter]()
    orig["a"] = CopyCounter()

    # test values copied to new Dict
    var copy = Dict(other=orig)
    assert_equal(0, orig["a"].copy_count)
    assert_equal(1, copy["a"].copy_count)
    assert_equal(0, orig._find_ref("a").copy_count)
    assert_equal(1, copy._find_ref("a").copy_count)


def test_dict_update_nominal():
    var orig = Dict[String, Int]()
    orig["a"] = 1
    orig["b"] = 2

    var new = Dict[String, Int]()
    new["b"] = 3
    new["c"] = 4

    orig.update(new)

    assert_equal(orig["a"], 1)
    assert_equal(orig["b"], 3)
    assert_equal(orig["c"], 4)


def test_dict_update_empty_origin():
    var orig = Dict[String, Int]()
    var new = Dict[String, Int]()
    new["b"] = 3
    new["c"] = 4

    orig.update(new)

    assert_equal(orig["b"], 3)
    assert_equal(orig["c"], 4)


def test_dict_or():
    var orig = Dict[String, Int]()
    var new = Dict[String, Int]()

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


def test_dict_update_empty_new():
    var orig = Dict[String, Int]()
    orig["a"] = 1
    orig["b"] = 2

    var new = Dict[String, Int]()

    orig.update(new)

    assert_equal(orig["a"], 1)
    assert_equal(orig["b"], 2)
    assert_equal(len(orig), 2)


@value
struct DummyKey(KeyElement):
    var value: Int

    @implicit
    fn __init__(out self, value: Int):
        self.value = value

    fn __init__(out self, *, other: Self):
        self = other

    fn __hash__(self) -> UInt:
        return self.value

    fn __eq__(self, other: DummyKey) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: DummyKey) -> Bool:
        return self.value != other.value


def test_mojo_issue_1729():
    var keys = List(
        7005684093727295727,
        2833576045803927472,
        -446534169874157203,
        -5597438459201014662,
        -7007119737006385570,
        7237741981002255125,
        -649171104678427962,
        -6981562940350531355,
    )
    var d = Dict[DummyKey, Int]()
    for i in range(len(keys)):
        d[DummyKey(keys[i])] = i
    assert_equal(len(d), len(keys))
    for i in range(len(d)):
        var k = keys[i]
        assert_equal(i, d[k])


fn test[name: String, test_fn: fn () raises -> object]() raises:
    print("Test", name, "...", end="")
    try:
        _ = test_fn()
    except e:
        print("FAIL")
        raise e
    print("PASS")


def test_dict():
    test_dict_construction()
    test["test_basic", test_basic]()
    test["test_multiple_resizes", test_multiple_resizes]()
    test["test_big_dict", test_big_dict]()
    test["test_compact", test_compact]()
    test["test_compact_with_elements", test_compact_with_elements]()
    test["test_pop_default", test_pop_default]()
    test["test_key_error", test_key_error]()
    test["test_iter", test_iter]()
    test["test_iter_keys", test_iter_keys]()
    test["test_iter_values", test_iter_values]()
    test["test_iter_values_mut", test_iter_values_mut]()
    test["test_iter_items", test_iter_items]()
    test["test_dict_copy", test_dict_copy]()
    test["test_dict_copy_add_new_item", test_dict_copy_add_new_item]()
    test["test_dict_copy_delete_original", test_dict_copy_delete_original]()
    test[
        "test_dict_copy_calls_copy_constructor",
        test_dict_copy_calls_copy_constructor,
    ]()
    test["test_dict_update_nominal", test_dict_update_nominal]()
    test["test_dict_update_empty_origin", test_dict_update_empty_origin]()
    test["test_dict_update_empty_new", test_dict_update_empty_new]()
    test["test_mojo_issue_1729", test_mojo_issue_1729]()
    test["test dict or", test_dict_or]()
    test["test dict popitem", test_dict_popitem]()


def test_taking_owned_kwargs_dict(owned kwargs: OwnedKwargsDict[Int]):
    assert_equal(len(kwargs), 2)

    assert_true("fruit" in kwargs)
    assert_equal(kwargs["fruit"], 8)
    assert_equal(kwargs["fruit"], 8)

    assert_true("dessert" in kwargs)
    assert_equal(kwargs["dessert"], 9)
    assert_equal(kwargs["dessert"], 9)

    var keys = String("")
    for key in kwargs.keys():
        keys += key[]
    assert_equal(keys, "fruitdessert")

    var sum = 0
    for val in kwargs.values():
        sum += val[]
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

    keys = String("")
    sum = 0
    for entry in kwargs.items():
        keys += entry[].key
        sum += entry[].value
    assert_equal(keys, "dessertsalad")
    assert_equal(sum, 19)


def test_owned_kwargs_dict():
    var owned_kwargs = OwnedKwargsDict[Int]()
    owned_kwargs._insert("fruit", 8)
    owned_kwargs._insert("dessert", 9)
    test_taking_owned_kwargs_dict(owned_kwargs^)


def test_find_get():
    var some_dict = Dict[String, Int]()
    some_dict["key"] = 1
    assert_equal(some_dict.find("key").value(), 1)
    assert_equal(some_dict.get("key").value(), 1)
    assert_equal(some_dict.find("not_key").or_else(0), 0)
    assert_equal(some_dict.get("not_key", 0), 0)


def test_dict_popitem():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2

    var item = dict.popitem()
    assert_equal(item.key, "b")
    assert_equal(item.value, 2)
    item = dict.popitem()
    assert_equal(item.key, "a")
    assert_equal(item.value, 1)
    with assert_raises(contains="KeyError"):
        _ = dict.popitem()


def test_pop_string_values():
    var dict = Dict[String, String]()
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


fn test_clear() raises:
    var some_dict = Dict[String, Int]()
    some_dict["key"] = 1
    some_dict.clear()
    assert_equal(len(some_dict), 0)
    assert_false(some_dict.get("key"))

    some_dict = Dict[String, Int]()
    some_dict.clear()
    assert_equal(len(some_dict), 0)


def test_init_initial_capacity():
    var initial_capacity = 16
    var x = Dict[Int, Int](power_of_two_initial_capacity=initial_capacity)
    assert_equal(x._reserved(), initial_capacity)
    for i in range(initial_capacity):
        x[i] = i
    for i in range(initial_capacity):
        assert_equal(i, x[i])

    var y = Dict[Int, Int](power_of_two_initial_capacity=64)
    assert_equal(y._reserved(), 64)


fn test_dict_setdefault() raises:
    var some_dict = Dict[String, Int]()
    some_dict["key1"] = 1
    some_dict["key2"] = 2
    assert_equal(some_dict.setdefault("key1", 0), 1)
    assert_equal(some_dict.setdefault("key2", 0), 2)
    assert_equal(some_dict.setdefault("not_key", 0), 0)
    assert_equal(some_dict["not_key"], 0)

    # Check that there is no copy of the default value, so it's performant
    var other_dict = Dict[String, CopyCounter]()
    var a = CopyCounter()
    var a_def = CopyCounter()
    var b_def = CopyCounter()
    other_dict["a"] = a^
    assert_equal(0, other_dict["a"].copy_count)
    _ = other_dict.setdefault("a", a_def^)
    _ = other_dict.setdefault("b", b_def^)
    assert_equal(0, other_dict["a"].copy_count)
    assert_equal(0, other_dict["b"].copy_count)


def test_dict_adaptive_capacity():
    var result: Int = 0
    var dict_size = 256

    x = Dict[Int, Int]()
    adaptive_capacity = List[Int]()
    for y in range(2):
        # grow the dictionary up to its full size
        for i in range(dict_size):
            var current_size = len(x)
            var current_reserved = x._reserved()
            var over_load = ((3 * current_size) + 1) > (2 * current_reserved)
            x[i] = i
            if over_load:
                assert_equal(len(x), current_size + 1)
                assert_true(x._reserved() != current_reserved)
                assert_equal(x._reserved(), current_reserved * 2)
                adaptive_capacity.append(x._reserved())
        assert_equal(len(x), dict_size)
        # always have more reserved:
        assert_equal(x._reserved(), dict_size * 2)

        # ungrow the dictionary
        for i in range(dict_size):
            var current_reserved = x._reserved()
            var under_load = (
                current_reserved > __type_of(x)._initial_reservation
                and ((len(x) - 1) * 3 < current_reserved)
            )
            result += x.pop(i)
            if under_load:
                assert_true(x._reserved() != current_reserved)
                assert_equal(x._reserved(), current_reserved / 2)
                adaptive_capacity.append(x._reserved())
        assert_equal(len(x), 0)
        assert_equal(x._reserved(), __type_of(x)._initial_reservation)

    assert_equal(
        adaptive_capacity,
        List[Int](
            16,
            32,
            64,
            128,
            256,
            512,
            256,
            128,
            64,
            32,
            16,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            256,
            128,
            64,
            32,
            16,
            8,
        ),
    )

    value_to_compare = 0
    for y in range(2):
        for i in range(256):
            value_to_compare += i
    assert_equal(result, value_to_compare)


def test_dict_adaptive_capacity2():
    x = Dict[Int, ValueDestructorRecorder]()
    y = List[Int]()
    y_ptr = UnsafePointer.address_of(y)
    alias size = 64
    for i in range(size):
        x[i] = ValueDestructorRecorder(i, y_ptr)^
    for i in range(size):
        _ = x.pop(i)
    assert_equal(len(y), 64)
    for i in range(size):
        assert_equal(y[i], i)


def main():
    test_dict()
    test_dict_fromkeys()
    test_dict_fromkeys_optional()
    test_dict_string_representation_string_int()
    test_dict_string_representation_int_int()
    test_owned_kwargs_dict()
    test_bool_conversion()
    test_find_get()
    test_pop_string_values()
    test_clear()
    test_init_initial_capacity()
    test_dict_setdefault()
    test_dict_adaptive_capacity()
    test_dict_adaptive_capacity2()
