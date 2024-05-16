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
struct __MLIRType[T: AnyRegType](Movable, Copyable):
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


@value
struct _NextEval[T: _IntNext]:
    var iter: T
    var i: Int


fn _next_eval[T: _IntNext](iter: T) -> _NextEval[T]:
    var copy = iter
    var next = copy.__next__()
    return _NextEval(copy, next)


@always_inline
fn _gen_next[
    inferred T: _IntIter, iter: T, f: fn[i: Int] () capturing -> None
]():
    @parameter
    if iter.__len__() == 0:
        return
    else:
        alias next = _next_eval(iter)
        f[next.i]()
        _gen_next[next.iter, f]()


@always_inline
fn parameter_for[
    inferred T: _IntIterable, iter: T, f: fn[i: Int] () capturing -> None
]():
    _gen_next[iter.__iter__(), f]()


@always_inline
fn parameter_for[
    inferred T: _StridedIterable, iter: T, f: fn[i: Int] () capturing -> None
]():
    _gen_next[iter.__iter__(), f]()
