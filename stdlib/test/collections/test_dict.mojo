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

from test_utils import CopyCounter
from testing import assert_equal, assert_false, assert_raises, assert_true


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
    var copy = Dict(orig)
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
    var copy = Dict(orig)
    # don't access the original dict, anymore, confirm that
    # deleting the original doesn't violate the integrity of the copy
    assert_equal(1, copy["a"])


def test_dict_copy_add_new_item():
    var orig = Dict[String, Int]()
    orig["a"] = 1

    # test values copied to new Dict
    var copy = Dict(orig)
    assert_equal(1, copy["a"])

    # test there are two copies of dict and
    # they don't share underlying memory
    copy["b"] = 2
    assert_false(str(2) in orig)


def test_dict_copy_calls_copy_constructor():
    var orig = Dict[String, CopyCounter]()
    orig["a"] = CopyCounter()

    # test values copied to new Dict
    var copy = Dict(orig)
    # I _may_ have thoughts about where our performance issues
    # are coming from :)
    assert_equal(1, orig["a"].copy_count)
    assert_equal(2, copy["a"].copy_count)
    assert_equal(0, orig.__get_ref("a")[].copy_count)
    assert_equal(1, copy.__get_ref("a")[].copy_count)


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

    fn __hash__(self) -> Int:
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
    var name_val = name  # FIXME(#26974): Can't pass 'name' directly.
    print("Test", name_val, "...", end="")
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


fn test_clear() raises:
    var some_dict = Dict[String, Int]()
    some_dict["key"] = 1
    some_dict.clear()
    assert_equal(len(some_dict), 0)
    assert_false(some_dict.get("key"))

    some_dict = Dict[String, Int]()
    some_dict.clear()
    assert_equal(len(some_dict), 0)


def main():
    test_dict()
    test_dict_fromkeys()
    test_dict_fromkeys_optional()
    test_dict_string_representation_string_int()
    test_dict_string_representation_int_int()
    test_owned_kwargs_dict()
    test_bool_conversion()
    test_find_get()
    test_clear()
