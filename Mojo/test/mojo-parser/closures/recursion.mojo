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

# RUN: %parse-mojo-isolated --verify-diagnostics %s

def closure_holder_1() -> Int:
    # expected-note @below {{nested function declared here}}
    def bar(n: Int) -> Int:
        if n == 0:
            return 0
        # expected-error @below {{recursive references to nested functions are not supported; define 'bar' at file scope}}
        return n + bar(n - 1)

    return bar(10)


def closure_holder_2() -> Int:
    # expected-note @below {{nested function declared here}}
    def foo(n: Int) -> Int:
        def bar() {imm n} -> Int:
            if n <= 0:
                return 0
            # expected-error @below {{recursive references to nested functions are not supported; define 'foo' at file scope}}
            return foo(n - 1)
        return bar()

    return foo(10)
