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
# Regression test for MOCO-3440: --debug-level=line-tables must not crash when
# a struct field's type involves a @__parameter function that is referenced only
# in debug info and not in regular code.
#
# The mechanism: `c_long_long` in stdlib ffi is `comptime c_long_long =
# Scalar[_c_long_long_dtype()]`. When a function returns a concrete
# Pointer to a struct that (transitively) has a c_long_long field,
# the debug info for that function's return type embeds the un-concretized
# type !kgen.scalar<*apply(@_c_long_long_dtype)>. If the returned pointer is
# assigned to a local variable whose contents are never accessed, the only
# reference to @_c_long_long_dtype is in that debug info. LiftAndFoldApply
# then lifts the apply from the location into a kgen.param.apply op, and the
# elaborator crashes in instantiateGeneratorReference because
# @_c_long_long_dtype is absent from concreteNodes (it was never imported into
# the user module's symbol table).
#
# RUN: kgen -emit=llvm --debug-level line-tables %s | FileCheck %s

from std.ffi import c_int, c_long_long
from std.ffi import external_call


@fieldwise_init
struct AVOptionDefaultVal2(Movable, Writable):
    var value: c_long_long


@fieldwise_init
struct AVOption(Movable, Writable):
    var offset: c_int
    var default_val: AVOptionDefaultVal2


@fieldwise_init
struct AVClass(Movable, Writable):
    var option: Pointer[AVOption, ImmUntrackedOrigin]


def sws_get_class() -> Pointer[AVClass, ImmUntrackedOrigin]:
    return external_call[
        "sws_get_class", Pointer[AVClass, ImmUntrackedOrigin]
    ]()


def main():
    # Assign the return value but never access the nested fields.
    # The type Pointer[AVClass] appears in regular code (return type),
    # but the nested c_long_long field is only referenced in debug info.
    var cls = sws_get_class()
    _ = cls


# CHECK: define {{.*}}main{{.*}}
