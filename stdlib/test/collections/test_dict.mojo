# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from collections.dict import Dict, KeyElement
from collections import Optional

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
        let key = "key" + str(i)
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


fn test[name: String, test_fn: fn () raises -> object]() raises:
    var name_val = name  # FIXME(#26974): Can't pass 'name' directly.
    print_no_newline("Test", name_val, "...")
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
