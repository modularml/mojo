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

from std.collections.counter import Counter

from test_utils import check_write_to
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.testing import TestSuite


# def test_and() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 1
#     c1["b"] = 2

#     var c2 = Counter[String]()
#     c2["b"] = 3
#     c2["c"] = 4

#     var c3 = c1 & c2

#     assert_equal(c3["a"], 0)
#     assert_equal(c3["b"], 2)
#     assert_equal(c3["c"], 0)

#     c1 &= c2

#     assert_equal(c1["a"], 0)
#     assert_equal(c1["b"], 2)
#     assert_equal(c1["c"], 0)


# def test_bool() raises:
#     var c = Counter[String]()
#     assert_false(c)
#     c["a"] = 1
#     assert_true(c)
#     _ = c.pop("a")
#     assert_false(c)


# def test_clear() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     c.clear()

#     assert_equal(len(c), 0)
#     assert_false(c)


# def test_contains() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     assert_true("a" in c)
#     assert_true("b" in c)
#     assert_false("c" in c)


# def test_copy() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     var copy = c.copy()

#     assert_equal(copy["a"], 1)
#     assert_equal(copy["b"], 2)
#     assert_equal(len(copy), 2)

#     c["c"] = 3

#     assert_equal(copy["c"], 0)
#     assert_equal(len(copy), 2)


# def test_counter_construction() raises:
#     _ = Counter[Int]()
#     _ = Counter[Int](List[Int]())
#     _ = Counter[String](List[String]())
#     _ = Counter[Int](1, 2)


# def test_counter_getitem() raises:
#     c = Counter[Int](1, 2, 2, 3, 3, 3, 4)
#     assert_equal(c[1], 1)
#     assert_equal(c[2], 2)
#     assert_equal(c[3], 3)
#     assert_equal(c[4], 1)
#     assert_equal(c[5], 0)


# def test_fromkeys() raises:
#     var keys = [String("a"), "b", "c"]
#     var c = Counter[String].fromkeys(keys, 3)

#     assert_equal(c["a"], 3)
#     assert_equal(c["b"], 3)
#     assert_equal(c["c"], 3)
#     assert_equal(len(c), 3)


# def test_get() raises:
#     var counter = Counter[String]()
#     counter["a"] = 1
#     counter["b"] = 2

#     var a: Int = counter.get("a").value()
#     var b: Int = counter.get("b").value()
#     var c: Int = counter.get("c", 3)

#     var d: Optional[Int] = counter.get("d")

#     assert_equal(a, 1)
#     assert_equal(b, 2)
#     assert_equal(c, 3)
#     assert_false(d)


# def test_iter() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     var keys = String()
#     for key in c:
#         keys += key

#     assert_equal(keys, "ab")

#     var keys_list = ["a", "b"]
#     var keys_count = 0
#     for i, e in enumerate(c):
#         assert_equal(i, keys_count)
#         assert_equal(e, keys_list[i])
#         keys_count += 1


# def test_iter_keys() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     var keys = String()
#     for key in c.keys():
#         keys += key

#     assert_equal(keys, "ab")


# def test_iter_values() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     var sum = 0
#     for value in c.values():
#         sum += value

#     assert_equal(sum, 3)


# def test_iter_values_mut() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     for ref value in c.values():
#         value += 1

#     assert_equal(2, c["a"])
#     assert_equal(3, c["b"])
#     assert_equal(2, len(c))


# def test_iter_items() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     var keys = String()
#     var sum = 0
#     for entry in c.items():
#         keys += entry.key
#         sum += entry.value

#     assert_equal(keys, "ab")
#     assert_equal(sum, 3)


# def test_len() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     assert_equal(len(c), 2)
#     _ = c.pop("a")
#     assert_equal(len(c), 1)
#     c.clear()
#     assert_equal(len(c), 0)


# def test_total() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     assert_equal(c.total(), 3)


# def test_most_common() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2
#     c["c"] = 3

#    var most_common = c.most_common(2)
#    assert_equal(len(most_common), 2)
#    assert_equal(most_common[0][0][String], "c")
#    # TODO: these asserts crash the compiler due to MOCO-3763
#    # assert_equal(most_common[0][1][Int], 3)
#    assert_equal(most_common[1][0][String], "b")
#    # assert_equal(most_common[1][1][Int], 2)


def test_most_common_clamps() raises:
    var c = Counter[Int](1, 1, 2)

    # Asking for more elements than the counter holds returns all of them,
    # like Python's `Counter.most_common`.
    var top = c.most_common(5)
    assert_equal(len(top), 2)
    assert_equal(top[0]._value, 1)
    assert_equal(top[0]._count, 2)
    assert_equal(top[1]._value, 2)
    assert_equal(top[1]._count, 1)

    assert_equal(len(c.most_common(0)), 0)
    assert_equal(len(c.most_common(-1)), 0)


# def test_eq_and_ne() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 1
#     c1["b"] = 2
#     c1["d"] = 0

#     var c2 = Counter[String]()
#     c2["a"] = 1
#     c2["b"] = 2
#     c2["c"] = 0

#     assert_true(c1.__eq__(c2))
#     assert_false(c1.__ne__(c2))

#     c2["b"] = 3
#     assert_false(c1.__eq__(c2))
#     assert_true(c1.__ne__(c2))


# def test_lt_le_gt_and_ge() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 1
#     c1["b"] = 2
#     c1["d"] = 0

#     var c2 = Counter[String]()
#     c2["a"] = 1
#     c2["b"] = 2
#     c2["c"] = 0

#     assert_false(c1.lt(c2))
#     assert_false(c2.lt(c1))
#     assert_true(c1.le(c2))
#     assert_true(c2.le(c1))
#     assert_false(c1.gt(c2))
#     assert_false(c2.gt(c1))
#     assert_true(c1.ge(c2))
#     assert_true(c2.ge(c1))

#     c2["a"] = 2
#     assert_false(c1.lt(c2))
#     assert_false(c2.lt(c1))
#     assert_true(c1.le(c2))
#     assert_false(c2.le(c1))
#     assert_false(c1.gt(c2))
#     assert_false(c2.gt(c1))
#     assert_false(c1.ge(c2))
#     assert_true(c2.ge(c1))

#     c2["b"] = 3
#     c2["c"] = 1
#     c2["d"] = 1
#     assert_true(c1.lt(c2))
#     assert_false(c2.lt(c1))
#     assert_true(c1.le(c2))
#     assert_false(c2.le(c1))
#     assert_false(c1.gt(c2))
#     assert_true(c2.gt(c1))
#     assert_false(c1.ge(c2))
#     assert_true(c2.ge(c1))


# def test_elements() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2
#     c["c"] = 3

#     var elements = c.elements()

#     assert_equal(len(elements), 6)
#     assert_equal(elements[0], "a")
#     assert_equal(elements[1], "b")
#     assert_equal(elements[2], "b")
#     assert_equal(elements[3], "c")
#     assert_equal(elements[4], "c")
#     assert_equal(elements[5], "c")


# def test_update() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 1
#     c1["b"] = 2

#     var c2 = Counter[String]()
#     c2["b"] = 3
#     c2["c"] = 4

#     c1.update(c2)

#     assert_equal(c1["a"], 1)
#     assert_equal(c1["b"], 5)
#     assert_equal(c1["c"], 4)


# def test_add() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 3
#     c1["b"] = 2
#     c1["d"] = -1  # should be ignored

#     var c2 = Counter[String]()
#     c2["a"] = -2
#     c2["b"] = 3
#     c2["c"] = 4
#     c2["e"] = 0  # should be ignored

#     var c3 = c1 + c2

#     assert_equal(c3["a"], 1)
#     assert_equal(c3["b"], 5)
#     assert_equal(c3["c"], 4)
#     # Check that the original counters are not modified
#     assert_equal(c1["a"], 3)
#     assert_equal(c1["b"], 2)
#     assert_equal(c1["d"], -1)

#     c2 += c1

#     assert_equal(c2["a"], 1)
#     assert_equal(c2["b"], 5)
#     assert_equal(c2["c"], 4)


# def test_subtract() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 4
#     c1["b"] = 2
#     c1["c"] = 0

#     var c2 = Counter[String]()
#     c2["a"] = 1
#     c2["b"] = -2
#     c2["c"] = 3

#     c1.subtract(c2)

#     assert_equal(c1["a"], 3)
#     assert_equal(c1["b"], 4)
#     assert_equal(c1["c"], -3)


# def test_sub() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 4
#     c1["b"] = 2
#     c1["c"] = 0

#     var c2 = Counter[String]()
#     c2["a"] = 1
#     c2["b"] = -2
#     c2["c"] = 3

#     var c3 = c1 - c2

#     assert_equal(c3["a"], 3)
#     assert_equal(c3["b"], 4)
#     assert_equal(c3["c"], 0)

#     # Check that the original counters are not modified
#     assert_equal(c1["a"], 4)
#     assert_equal(c1["b"], 2)
#     assert_equal(c1["c"], 0)

#     c2 -= c1

#     assert_equal(c2["a"], 0)
#     assert_equal(c2["b"], 0)
#     assert_equal(c2["c"], 3)


# def test_counter_setitem() raises:
#     c = Counter[Int]()
#     c[1] = 1
#     c[2] = 2
#     assert_equal(c[1], 1)
#     assert_equal(c[2], 2)
#     assert_equal(c[3], 0)


# def test_neg() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = -2
#     c["c"] = 3

#     var neg = -c

#     assert_equal(neg["a"], 0)
#     assert_equal(neg["b"], 2)
#     assert_equal(neg["c"], 0)


# def test_or() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 1
#     c1["b"] = 2

#     var c2 = Counter[String]()
#     c2["b"] = 3
#     c2["c"] = 4
#     c2["d"] = -1

#     var c3 = c1 | c2

#     assert_equal(c3["a"], 1)
#     assert_equal(c3["b"], 3)
#     assert_equal(c3["c"], 4)
#     assert_equal(c3["d"], 0)

#     c1 |= c2

#     assert_equal(c1["a"], 1)
#     assert_equal(c1["b"], 3)
#     assert_equal(c1["c"], 4)
#     assert_equal(c1["d"], 0)


# def test_pop() raises:
#     var counter = Counter[String]()
#     counter["a"] = 1
#     counter["b"] = 2

#     var a = counter.pop("a")
#     var b = counter.pop("b")
#     var c = counter.pop("c", 3)

#     assert_equal(a, 1)
#     assert_equal(b, 2)
#     assert_equal(c, 3)

#     with assert_raises(contains="KeyError"):
#         _ = counter.pop("not_a_key")


# def test_popitem() raises:
#     var counter = Counter[String]()
#     counter["a"] = 1
#     counter["b"] = 2

#     var item = counter.popitem()
#     assert_equal(item[0][String], "b")
#     assert_equal(item[1][Int], 2)

#     item = counter.popitem()
#     assert_equal(item[0][String], "a")
#     assert_equal(item[1][Int], 1)

#     with assert_raises():
#         _ = counter.popitem()


# def test_write_to() raises:
#     var c = Counter[String]()
#     c["a"] = 3
#     c["b"] = 2
#     check_write_to(c, expected="{a: 3, b: 2}", is_repr=False)

#     var empty = Counter[String]()
#     check_write_to(empty, expected="{}", is_repr=False)

#     var single = Counter[String]()
#     single["x"] = 1
#     check_write_to(single, expected="{x: 1}", is_repr=False)


# def test_write_repr_to() raises:
#     var c = Counter[String]()
#     c["a"] = 3
#     c["b"] = 2
#     check_write_to(
#         c,
#         expected="Counter[String]({'a': Int(3), 'b': Int(2)})",
#         is_repr=True,
#     )

#     var empty = Counter[String]()
#     check_write_to(empty, expected="Counter[String]({})", is_repr=True)

#     var single = Counter[String]()
#     single["x"] = 1
#     check_write_to(
#         single, expected="Counter[String]({'x': Int(1)})", is_repr=True
#     )


# def test_counter_iter_owned() raises:
#     var c = Counter[String]("a", "a", "b", "c")
#     var keys = List[String]()
#     for key in c^:
#         keys.append(key)

#     assert_equal(len(keys), 3)
#     assert_true(String("a") in keys)
#     assert_true(String("b") in keys)
#     assert_true(String("c") in keys)


# def test_counter_iter_owned_bounds() raises:
#     var c = Counter[String]("a", "b", "c")
#     var it = c^.__iter__()
#     assert_equal(it.bounds()[0], 3)
#     _ = it.__next__()
#     assert_equal(it.bounds()[0], 2)
#     _ = it.__next__()
#     assert_equal(it.bounds()[0], 1)
#     _ = it.__next__()
#     assert_equal(it.bounds()[0], 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
