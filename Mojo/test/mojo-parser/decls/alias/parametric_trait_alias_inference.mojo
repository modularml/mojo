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

# Regression test for parameter inference through a parametric associated
# comptime type (`Self.S[...]`).
#
# When the base type is abstract (e.g. a trait's `Self`), `Self.S[x]` is
# represented as a `bind_params` over an unfoldable `get_witness` generator.
# Forwarding such a value from one method to another (passing `s` from `load`
# to `load_at`) requires inferring the callee's parameter from the argument's
# type.  `ParamMatcher::matchParams` previously had no case for `bind_params`,
# so the two structurally-identical types failed to match and inference
# reported a bogus "depends on an unresolved parameter" error.  See the
# `BindParamsAttr` case added to `Mojo/lib/MojoParser/ParamMatcher.cpp`.

# RUN: %parse-mojo-isolated %s | FileCheck %s


# A single Int parameter is the minimal trigger.
trait IntStorage:
    comptime StorageType[x: Int]: TrivialRegisterPassable

    # The default method `load` emits a body that forwards `s` to `load_at`,
    # inferring `x` from `s`'s type.  Both the function and its forwarding call
    # must be emitted.
    # CHECK: lit.fn @"load[::SIMD[DType.int, 1]]
    @staticmethod
    def load(s: Self.StorageType[...]):
        # CHECK: lit.call{{.*}}load_at[::SIMD[DType.int, 1]]
        Self.load_at(s, 0)

    @staticmethod
    def load_at(s: Self.StorageType[...], offset: Int):
        ...


# The originally-reported case: an inferred `mut` parameter decomposed out of a
# dependent `origin` parameter, producing a multi-value `bind_params`.
trait TensorStorage:
    comptime StorageType[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ]: TrivialRegisterPassable

    # CHECK: lit.fn @"load[::Bool,LITOrigin
    @staticmethod
    def load(storage: Self.StorageType[...]):
        # CHECK: lit.call{{.*}}load_at[::Bool,LITOrigin
        Self.load_at(storage, 0)

    @staticmethod
    def load_at(storage: Self.StorageType[...], offset: Int):
        ...
