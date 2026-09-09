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

# RUN: %parse-mojo-isolated -verify-diagnostics -split-input-file %s


struct a(TrivialRegisterPassable):
    @always_inline("builtin")
    def b(c, d: a):
        c & d

    @always_inline("builtin")
    def __rand__(c, e: a):
        # expected-error @below {{'@always_inline("builtin")' does not support recursion}}
        e & c


# // -----


struct S(TrivialRegisterPassable):
    @always_inline("builtin")
    def f(self, x: S):
        # expected-error @below {{'@always_inline("builtin")' does not support recursion}}
        self.f(x)


# // -----


struct S(TrivialRegisterPassable):
    @always_inline("builtin")
    def f(self, x: S):
        self.g(x)

    @always_inline("builtin")
    def g(self, x: S):
        # expected-error @below {{'@always_inline("builtin")' does not support recursion}}
        self.f(x)
