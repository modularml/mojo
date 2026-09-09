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

from std.os.os import _abort_base

# ===-----------------------------------------------------------------------===#
# __MLIRType
# ===-----------------------------------------------------------------------===#


struct __MLIRType[T: __mlir_type.`!kgen.non_struct_type`](
    TrivialRegisterPassable
):
    var value: Self.T
    comptime __del__is_trivial = True
    comptime __move_ctor_is_trivial = True
    comptime __copy_ctor_is_trivial = True


# ===-----------------------------------------------------------------------===#
# comptime for (was @parameter for) implementation details
# ===-----------------------------------------------------------------------===#


def paramfor_has_next[
    IteratorType: Iterator & Copyable
](it: IteratorType) -> Bool where conforms_to(
    IteratorType.Element,
    Movable & Deinitable,
):
    var result = it.copy()
    try:
        var elem = result.__next__()
        _ = elem^
        return True
    except:
        return False


def paramfor_next_iter[
    IteratorType: Iterator & Copyable
](it: IteratorType) -> IteratorType where conforms_to(
    IteratorType.Element,
    Movable & Deinitable,
):
    # NOTE: This function is called by the compiler's elaborator only when
    # paramfor_has_next will return true. This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    var result = it.copy()
    # This intentionally discards the value, but this only happens at comptime,
    # so recomputing it in the body of the loop is fine.
    try:
        var elem = result.__next__()
        _ = elem^
        return result.copy()
    except:
        _abort_base()


def paramfor_next_value[
    IteratorType: Iterator & Copyable
](it: IteratorType) -> IteratorType.Element:
    # NOTE: This function is called by the compiler's elaborator only when
    # paramfor_has_next will return true. This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    try:
        var result = it.copy()
        return result.__next__()
    except:
        _abort_base()
