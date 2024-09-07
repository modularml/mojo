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
# RUN: %mojo --debug-level full %s

from collections import List

from memory import Arc
from testing import assert_equal, assert_false, assert_true


def test_basic():
    var p = Arc(4)
    var p2 = p
    p2[] = 3
    assert_equal(3, p[])


@value
struct ObservableDel(CollectionElement):
    var target: UnsafePointer[Bool]

    @no_inline
    fn touch(inout self):
        var b = self.target[]
        print(b)

    fn __init__(inout self, *, other: Self):
        self = other

    fn __del__(owned self):
        self.target.init_pointee_move(True)


def test_deleter_not_called_until_no_references():
    var deleted = False
    var p = Arc(ObservableDel(UnsafePointer.address_of(deleted)))
    var p2 = p
    _ = p^
    assert_false(deleted)

    var vec = List[Arc[ObservableDel]]()
    vec.append(p2)
    _ = p2^
    assert_false(deleted)
    _ = vec^
    assert_true(deleted)


def test_deleter_not_called_until_no_references_explicit_copy():
    var deleted = False
    var p = Arc(ObservableDel(UnsafePointer.address_of(deleted)))
    var p2 = Arc(other=p)
    _ = p^
    assert_false(deleted)

    var vec = List[Arc[ObservableDel]]()
    vec.append(Arc(other=p2))
    _ = p2^
    assert_false(deleted)
    _ = vec^
    assert_true(deleted)


def test_weak_upgradeable_when_strong_live():
    var deleted = UnsafePointer[Bool].alloc(1)
    deleted.init_pointee_explicit_copy(False)
    var p = Arc[ObservableDel, enable_weak = True](ObservableDel(deleted))
    assert_false(deleted[])

    var w = p.downgrade()
    var s_o = w.upgrade()
    assert_true(s_o)
    assert_false(deleted[])

    var s = s_o.take()
    _ = p^
    assert_false(deleted[])

    s[].touch()
    _ = s^
    assert_true(deleted[])

    deleted.free()

    # put access of a strong after check so that the
    # strong doesn't drop before we can upgrade


def test_weak_dies_when_strong_dies():
    var deleted = False
    var p = Arc[ObservableDel, enable_weak = True](ObservableDel(UnsafePointer.address_of(deleted)))
    assert_false(deleted)

    var w = p.downgrade()

    _ = p^

    var s_o = w.upgrade()

    assert_true(deleted)
    assert_false(s_o)


def main():
    test_basic()
    test_deleter_not_called_until_no_references()
    test_deleter_not_called_until_no_references_explicit_copy()
    test_weak_upgradeable_when_strong_live()
    test_weak_dies_when_strong_dies()
