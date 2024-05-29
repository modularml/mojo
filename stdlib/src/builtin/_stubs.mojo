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

from builtin.range import _StridedRangeIterator

# ===----------------------------------------------------------------------===#
# __MLIRType
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct __MLIRType[T: AnyTrivialRegType](Movable, Copyable):
    var value: T


# ===----------------------------------------------------------------------===#
# parameter_for
# ===----------------------------------------------------------------------===#


trait _IntNext(Copyable):
    fn __next__(inout self) -> Int:
        ...


trait _IntIter(_IntNext):
    fn __len__(self) -> Int:
        ...


trait _IntIterable(_IntIter):
    fn __iter__(self) -> Self:
        ...


trait _StridedIterable(_IntIter):
    fn __iter__(self) -> _StridedRangeIterator:
        ...


struct _ParamForIterator[IteratorT: Copyable]:
    var next_it: IteratorT
    var value: Int
    var stop: Bool

    fn __init__(inout self, next_it: IteratorT, value: Int, stop: Bool):
        self.next_it = next_it
        self.value = value
        self.stop = stop


fn declval[T: AnyType]() -> T:
    constrained[False, "should only be used inside __type_of"]()
    while True:
        pass


fn parameter_for_generator[
    T: _IntIterable,
](range: T) -> _ParamForIterator[__type_of(declval[T]().__iter__())]:
    return _generator(range.__iter__())


fn parameter_for_generator[
    T: _StridedIterable,
](range: T) -> _ParamForIterator[__type_of(declval[T]().__iter__())]:
    return _generator(range.__iter__())


fn _generator[
    IteratorT: _IntIter
](it: IteratorT) -> _ParamForIterator[IteratorT]:
    if it.__len__() == 0:
        return _ParamForIterator[IteratorT](
            __mlir_attr[`#kgen.unknown : !kgen.paramref<`, IteratorT, `>`],
            0,
            True,
        )
    var next_it = it
    var value = next_it.__next__()
    return _ParamForIterator(next_it, value, False)
