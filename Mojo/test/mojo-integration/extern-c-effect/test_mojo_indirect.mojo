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
# Tests indirect calls through both plain Mojo and abi("C") function pointers
# using the same struct type, verifying that CallIndirectOpConversion applies
# the right calling convention in each case.
#
# FloatPair ({float, float}, 8 bytes) is SSE-classified on x86-64 SysV
# (returned packed in XMM0) and HFA on ARM64 AAPCS.  Calling through the
# wrong ABI pointer silently corrupts field values, so this test catches
# any mixing of the two paths.
#
# Four functions (two Mojo-ABI, two abi("C")) are called indirectly through
# matching pointer types to exercise both branches of CallIndirectOpConversion.
#
# RUN: %mojo %s | FileCheck %s


@fieldwise_init
struct FloatPair(TrivialRegisterPassable):
    var x: Float32
    var y: Float32


# --- Mojo ABI ---


def mojo_add_one(p: FloatPair) -> FloatPair:
    return FloatPair(p.x + 1.0, p.y + 1.0)


def mojo_add_ten(p: FloatPair) -> FloatPair:
    return FloatPair(p.x + 10.0, p.y + 10.0)


def pick_mojo(use_add_one: Bool) -> def(FloatPair) thin -> FloatPair:
    if use_add_one:
        return mojo_add_one
    return mojo_add_ten


# --- C ABI ---


def c_add_one(p: FloatPair) abi("C") -> FloatPair:
    return FloatPair(p.x + 1.0, p.y + 1.0)


def c_add_ten(p: FloatPair) abi("C") -> FloatPair:
    return FloatPair(p.x + 10.0, p.y + 10.0)


def pick_c(use_add_one: Bool) -> def(FloatPair) thin abi("C") -> FloatPair:
    if use_add_one:
        return c_add_one
    return c_add_ten


def main():
    var input = FloatPair(1.5, 2.5)

    var r1 = pick_mojo(True)(input)
    print(r1.x, r1.y)

    var r2 = pick_mojo(False)(input)
    print(r2.x, r2.y)

    var r3 = pick_c(True)(input)
    print(r3.x, r3.y)

    var r4 = pick_c(False)(input)
    print(r4.x, r4.y)


# CHECK: 2.5 3.5
# CHECK: 11.5 12.5
# CHECK: 2.5 3.5
# CHECK: 11.5 12.5
