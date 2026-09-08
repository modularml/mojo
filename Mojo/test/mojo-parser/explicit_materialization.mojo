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

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct NonEM(Movable where False):
    def __init__(out self):
        pass

    def method(self):
        pass


struct Foo[v: NonEM](Movable where False):
    def __init__(out self):
        pass

    def method(self):
        pass


def func(x: NonEM = NonEM()):
    pass


def main():
    # This should not raise a warning.
    func()

    comptime x = NonEM()
    # This should.
    # expected-error@+2 {{cannot materialize comptime value of type 'NonEM' to runtime because it is not 'ImplicitlyCopyable'}}
    # expected-note@+1 {{use 'materialize' to explicitly materialize the value}}
    func(x)
