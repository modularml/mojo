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

# `mojo format` only works on `.mojo` files, and modifies them in place.
# The `grep` is used to remove the `CHECK` lines from the output so FileCheck
# doesn't match on its own directives.
# RUN: cp %s %t.mojo
# RUN: mojo format %t.mojo
# RUN: cat %t.mojo | grep -v "# CHECK:" | FileCheck %t.mojo

# CHECK: def function() -> Int:
# CHECK: return 10
def function()    -> Int:
    return     10


# CHECK: struct Foo(Copyable, Writable):
struct Foo(Writable, Copyable):
  pass


# CHECK: trait Bar(Copyable, Writable):
trait Bar(Writable, Copyable):
  pass

# CHECK: struct Bar[x: Int]:
struct Bar[x: Int]:
  pass

# CHECK: struct ComplexStruct[
# CHECK:     bar: Bar[5],
# CHECK:     index_type: DType = _get_index_type(layout, address_space),
# CHECK: ](Copyable, Writable):
# CHECK:     pass
struct ComplexStruct[
    bar: Bar[5], index_type: DType = _get_index_type(layout, address_space),
](Writable, Copyable):
    pass
