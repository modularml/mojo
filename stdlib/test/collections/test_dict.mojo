# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from collections.dict import Dict, KeyElement
from collections import Optional

from testing import *


@value
struct assert_raises:
    var message: Optional[StringLiteral]

    fn __enter__(self) -> Self:
        return self

    fn __exit__(self) raises:
        var message = String("Test didn't raise!")
        if self.message:
            message += " Expected: " + str(self.message.value())
        assert_true(False, message)

    fn __exit__(self, error: Error) raises -> Bool:
        let message = str(error)
        if self.message:
            let expected = String(self.message.value())
            return expected == message
        else:
            return True


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

    with assert_raises("KeyError"):
        _ = dict["a"]
    with assert_raises("KeyError"):
        _ = dict.pop("a")


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
