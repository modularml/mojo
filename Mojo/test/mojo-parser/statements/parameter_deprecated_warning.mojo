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

# RUN: %parse-mojo-isolated %s 2>&1 | FileCheck %s

# Test that '@parameter if', '@parameter for', and the '@parameter' function
# decorator issue deprecation warnings.


@fieldwise_init
struct IterRange(ImplicitlyCopyable, Iterator):
    comptime Element = Int

    var value: Int

    def __iter__(self) -> Self:
        return self

    def __next__(mut self) raises StopIteration -> Int:
        if self.value <= 0:
            raise StopIteration()
        return self.value


def test_parameter_if[a: __mlir_type.`!kgen.scalar<bool>`]():
    comptime if a:
        var inside: Int


def test_parameter_for[a: Int]():
    comptime for i in IterRange(a):
        pass


def test_parameter_decorator():
    # CHECK: warning: '@parameter' is deprecated; use '@__parameter'
    @parameter
    def nested() -> Int:
        return 1

    # Preferred spelling must not warn.
    @__parameter
    def preferred() -> Int:
        return 2

    _ = nested()
    _ = preferred()


# Test that 'comptime if' and 'comptime for' do NOT issue deprecation warnings,
# and that `@__parameter` does not inherit the `@parameter` deprecation warning.
# CHECK-NOT: '@parameter
def test_comptime_if[a: __mlir_type.`!kgen.scalar<bool>`]():
    comptime if a:
        var inside: Int


def test_comptime_for[a: Int]():
    comptime for i in IterRange(a):
        pass
