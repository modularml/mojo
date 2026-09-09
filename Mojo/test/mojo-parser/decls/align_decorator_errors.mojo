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

# RUN: %parse-mojo-isolated -verify-diagnostics %s

# Test error cases for @align decorator.


# expected-error @+1 {{@align value must be a positive power of 2}}
@align(0)
struct ZeroAlign(Movable where False):
    var x: Int


# -1 is parsed as unary negation, not an integer literal
# expected-error @+1 {{@align value must be a positive power of 2}}
@align(-1)
struct NegativeAlign(Movable where False):
    var x: Int


# expected-error @+1 {{@align value must be a positive power of 2}}
@align(3)
struct NotPowerOfTwo1(Movable where False):
    var x: Int


# expected-error @+1 {{@align value must be a positive power of 2}}
@align(5)
struct NotPowerOfTwo2(Movable where False):
    var x: Int


# expected-error @+1 {{@align value must be a positive power of 2}}
@align(6)
struct NotPowerOfTwo3(Movable where False):
    var x: Int


# expected-error @+1 {{@align value must be a positive power of 2}}
@align(7)
struct NotPowerOfTwo4(Movable where False):
    var x: Int


# expected-error @+1 {{@align value must be a positive power of 2}}
@align(100)
struct NotPowerOfTwo5(Movable where False):
    var x: Int


# expected-error @+1 {{@align requires exactly one argument}}
@align
struct MissingArgument(Movable where False):
    var x: Int


# expected-error @+1 {{@align requires exactly one argument}}
@align()
struct EmptyArguments(Movable where False):
    var x: Int


# expected-error @+1 {{@align requires exactly one argument}}
@align(64, 128)
struct TooManyArguments(Movable where False):
    var x: Int


# expected-error @+1 {{'StringLiteral["64"]' does not implement the '__mlir_index__' method}}
@align("64")
struct StringArgument(Movable where False):
    var x: Int


# expected-error @+1 {{@align value exceeds maximum alignment (2^29)}}
@align(1073741824)  # 2^30, exceeds maximum
struct ExcessiveAlignment(Movable where False):
    var x: Int


# expected-error @+1 {{@align value exceeds maximum alignment (2^29)}}
@align(4294967296)  # 2^32, exceeds maximum
struct ExcessiveAlignment2(Movable where False):
    var x: Int


# @align(1) is allowed without warning - useful for parametric alignment fallback
@align(1)
struct AlignOne(Movable where False):
    var x: Int


# Test referencing a non-existent parameter
# expected-error @+1 {{use of unknown declaration 'nonexistent'}}
@align(nonexistent)
struct NonExistentParam(Movable where False):
    var x: Int


# Test referencing a non-existent parameter
# expected-error @+1 {{use of unknown declaration 'missing'}}
@align(missing)
struct NonExistentParamExpr(Movable where False):
    var x: Int


# Test using a non-Int typed parameter
# expected-error @+1 {{'String' does not implement the '__mlir_index__' method}}
@align(s)
struct WrongTypeParam[s: String](Movable where False):
    var x: Int


# Test using a Self. parameter
# expected-error @+1 {{'Self' type is not available in this context}}
@align(Self.x)
struct SelfParam[x: Int](Movable where False):
    pass
