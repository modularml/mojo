# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from collections import Optional
from collections.dict import Dict, KeyElement

from test_utils import CopyCounter
from testing import *


def test_dict_construction():
    _ = Dict[Int, Int]()
    _ = Dict[String, Int]()


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


def test_big_dict():
    var dict = Dict[String, Int]()
    for i in range(2000):
        dict["key" + str(i)] = i + 1
    assert_equal(2000, len(dict))


def test_compact():
    var dict = Dict[String, Int]()
    for i in range(20):
        var key = "key" + str(i)
        dict[key] = i + 1
        dict.pop(key)
    assert_equal(0, len(dict))


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
    assert_false(2 in orig)


def test_dict_copy_calls_copy_constructor():
    var orig = Dict[String, CopyCounter]()
    orig["a"] = CopyCounter() ^

    # test values copied to new Dict
    var copy = Dict(orig)
    # I _may_ have thoughts about where our performance issues
    # are coming from :)
    assert_equal(5, orig["a"].copy_count)
    assert_equal(6, copy["a"].copy_count)


fn test[name: String, test_fn: fn () raises -> object]() raises:
    var name_val = name  # FIXME(#26974): Can't pass 'name' directly.
    print("Test", name_val, "...", end="")
    try:
        _ = test_fn()
    except e:
        print("FAIL")
        raise e
    print("PASS")


def main():
    test_dict_construction()
    test["test_basic", test_basic]()
    test["test_multiple_resizes", test_multiple_resizes]()
    test["test_big_dict", test_big_dict]()
    test["test_compact", test_compact]()
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
