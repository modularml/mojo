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

from collections import Optional
from collections.counter import Counter

from testing import assert_equal, assert_false, assert_raises, assert_true


def test_and():
    var c1 = Counter[String]()
    c1["a"] = 1
    c1["b"] = 2

    var c2 = Counter[String]()
    c2["b"] = 3
    c2["c"] = 4

    var c3 = c1 & c2

    assert_equal(c3["a"], 0)
    assert_equal(c3["b"], 2)
    assert_equal(c3["c"], 0)

    c1 &= c2

    assert_equal(c1["a"], 0)
    assert_equal(c1["b"], 2)
    assert_equal(c1["c"], 0)


def test_bool():
    var c = Counter[String]()
    assert_false(c)
    c["a"] = 1
    assert_true(c)
    c.pop("a")
    assert_false(c)


def test_clear():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    c.clear()

    assert_equal(len(c), 0)
    assert_false(c)


def test_contains():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    assert_true("a" in c)
    assert_true("b" in c)
    assert_false("c" in c)


def test_copy():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var copy = Counter[String](other=c)

    assert_equal(copy["a"], 1)
    assert_equal(copy["b"], 2)
    assert_equal(len(copy), 2)

    c["c"] = 3

    assert_equal(copy["c"], 0)
    assert_equal(len(copy), 2)


def test_counter_construction():
    _ = Counter[Int]()
    _ = Counter[Int](List[Int]())
    _ = Counter[String](List[String]())


def test_counter_getitem():
    c = Counter[Int](List[Int](1, 2, 2, 3, 3, 3, 4))
    assert_equal(c[1], 1)
    assert_equal(c[2], 2)
    assert_equal(c[3], 3)
    assert_equal(c[4], 1)
    assert_equal(c[5], 0)


def test_fromkeys():
    var keys = List[String]("a", "b", "c")
    var c = Counter[String].fromkeys(keys, 3)

    assert_equal(c["a"], 3)
    assert_equal(c["b"], 3)
    assert_equal(c["c"], 3)
    assert_equal(len(c), 3)


def test_get():
    var counter = Counter[String]()
    counter["a"] = 1
    counter["b"] = 2

    var a: Int = counter.get("a").value()
    var b: Int = counter.get("b").value()
    var c: Int = counter.get("c", 3)

    var d: Optional[Int] = counter.get("d")

    assert_equal(a, 1)
    assert_equal(b, 2)
    assert_equal(c, 3)
    assert_false(d)


def test_iter():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var keys = String("")
    for key in c:
        keys += key[]

    assert_equal(keys, "ab")


def test_iter_keys():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var keys = String("")
    for key in c.keys():
        keys += key[]

    assert_equal(keys, "ab")


def test_iter_values():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var sum = 0
    for value in c.values():
        sum += value[]

    assert_equal(sum, 3)


def test_iter_values_mut():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    for value in c.values():
        value[] += 1

    assert_equal(2, c["a"])
    assert_equal(3, c["b"])
    assert_equal(2, len(c))


def test_iter_items():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var keys = String("")
    var sum = 0
    for entry in c.items():
        keys += entry[].key
        sum += entry[].value

    assert_equal(keys, "ab")
    assert_equal(sum, 3)


def test_len():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    assert_equal(len(c), 2)
    c.pop("a")
    assert_equal(len(c), 1)
    c.clear()
    assert_equal(len(c), 0)


def test_total():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    assert_equal(c.total(), 3)


def test_most_common():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3

    var most_common = c.most_common(2)
    assert_equal(len(most_common), 2)
    assert_equal(most_common[0][0][String], "c")
    assert_equal(most_common[0][1][Int], 3)
    assert_equal(most_common[1][0][String], "b")
    assert_equal(most_common[1][1][Int], 2)


def test_eq_and_ne():
    var c1 = Counter[String]()
    c1["a"] = 1
    c1["b"] = 2
    c1["d"] = 0

    var c2 = Counter[String]()
    c2["a"] = 1
    c2["b"] = 2
    c2["c"] = 0

    assert_true(c1.__eq__(c2))
    assert_false(c1.__ne__(c2))

    c2["b"] = 3
    assert_false(c1.__eq__(c2))
    assert_true(c1.__ne__(c2))


def test_lt_le_gt_and_ge():
    var c1 = Counter[String]()
    c1["a"] = 1
    c1["b"] = 2
    c1["d"] = 0

    var c2 = Counter[String]()
    c2["a"] = 1
    c2["b"] = 2
    c2["c"] = 0

    assert_false(c1.__lt__(c2))
    assert_true(c1.__le__(c2))
    assert_false(c1.__gt__(c2))
    assert_true(c1.__ge__(c2))

    c2["b"] = 3
    assert_true(c1.__lt__(c2))
    assert_true(c1.__le__(c2))
    assert_false(c1.__gt__(c2))
    assert_true(c2.__gt__(c1))
    assert_false(c1.__ge__(c2))


def test_elements():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3

    var elements = c.elements()

    assert_equal(len(elements), 6)
    assert_equal(elements[0], "a")
    assert_equal(elements[1], "b")
    assert_equal(elements[2], "b")
    assert_equal(elements[3], "c")
    assert_equal(elements[4], "c")
    assert_equal(elements[5], "c")


def test_update():
    var c1 = Counter[String]()
    c1["a"] = 1
    c1["b"] = 2

    var c2 = Counter[String]()
    c2["b"] = 3
    c2["c"] = 4

    c1.update(c2)

    assert_equal(c1["a"], 1)
    assert_equal(c1["b"], 5)
    assert_equal(c1["c"], 4)


def test_add():
    var c1 = Counter[String]()
    c1["a"] = 3
    c1["b"] = 2
    c1["d"] = -1  # should be ignored

    var c2 = Counter[String]()
    c2["a"] = -2
    c2["b"] = 3
    c2["c"] = 4
    c2["e"] = 0  # should be ignored

    var c3 = c1 + c2

    assert_equal(c3["a"], 1)
    assert_equal(c3["b"], 5)
    assert_equal(c3["c"], 4)
    # Check that the original counters are not modified
    assert_equal(c1["a"], 3)
    assert_equal(c1["b"], 2)
    assert_equal(c1["d"], -1)

    c2 += c1

    assert_equal(c2["a"], 1)
    assert_equal(c2["b"], 5)
    assert_equal(c2["c"], 4)


def test_substract():
    var c1 = Counter[String]()
    c1["a"] = 4
    c1["b"] = 2
    c1["c"] = 0

    var c2 = Counter[String]()
    c2["a"] = 1
    c2["b"] = -2
    c2["c"] = 3

    c1.subtract(c2)

    assert_equal(c1["a"], 3)
    assert_equal(c1["b"], 4)
    assert_equal(c1["c"], -3)


def test_sub():
    var c1 = Counter[String]()
    c1["a"] = 4
    c1["b"] = 2
    c1["c"] = 0

    var c2 = Counter[String]()
    c2["a"] = 1
    c2["b"] = -2
    c2["c"] = 3

    var c3 = c1 - c2

    assert_equal(c3["a"], 3)
    assert_equal(c3["b"], 4)
    assert_equal(c3["c"], -3)
    # Check that the original counters are not modified
    assert_equal(c1["a"], 4)
    assert_equal(c1["b"], 2)
    assert_equal(c1["c"], 0)

    c2 -= c1

    assert_equal(c2["a"], -3)
    assert_equal(c2["b"], -4)
    assert_equal(c2["c"], 3)


def test_counter_setitem():
    c = Counter[Int]()
    c[1] = 1
    c[2] = 2
    assert_equal(c[1], 1)
    assert_equal(c[2], 2)
    assert_equal(c[3], 0)


def test_neg():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = -2
    c["c"] = 3

    var neg = -c

    assert_equal(neg["a"], 0)
    assert_equal(neg["b"], 2)
    assert_equal(neg["c"], 0)


def test_or():
    var c1 = Counter[String]()
    c1["a"] = 1
    c1["b"] = 2

    var c2 = Counter[String]()
    c2["b"] = 3
    c2["c"] = 4
    c2["d"] = -1

    var c3 = c1 | c2

    assert_equal(c3["a"], 1)
    assert_equal(c3["b"], 3)
    assert_equal(c3["c"], 4)
    assert_equal(c3["d"], 0)

    c1 |= c2

    assert_equal(c1["a"], 1)
    assert_equal(c1["b"], 3)
    assert_equal(c1["c"], 4)
    assert_equal(c1["d"], 0)


def test_pop():
    var counter = Counter[String]()
    counter["a"] = 1
    counter["b"] = 2

    var a = counter.pop("a")
    var b = counter.pop("b")
    var c = counter.pop("c", 3)

    assert_equal(a, 1)
    assert_equal(b, 2)
    assert_equal(c, 3)


def test_popitem():
    var counter = Counter[String]()
    counter["a"] = 1
    counter["b"] = 2

    var item = counter.popitem()
    assert_equal(item[0][String], "b")
    assert_equal(item[1][Int], 2)

    item = counter.popitem()
    assert_equal(item[0][String], "a")
    assert_equal(item[1][Int], 1)

    with assert_raises():
        counter.popitem()


def main():
    test_add()
    test_and()
    test_bool()
    test_clear()
    test_contains()
    test_copy()
    test_counter_construction()
    test_counter_getitem()
    test_counter_setitem()
    test_elements()
    test_eq_and_ne()
    test_fromkeys()
    test_get()
    test_iter()
    test_iter_keys()
    test_iter_items()
    test_iter_values()
    test_iter_values_mut()
    test_len()
    test_lt_le_gt_and_ge()
    test_most_common()
    test_neg()
    test_or()
    test_pop()
    test_popitem()
    test_substract()
    test_total()
    test_update()
