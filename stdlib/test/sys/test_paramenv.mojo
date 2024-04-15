# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo -D bar=99 -D baz=hello %s | FileCheck %s
# RUN: %mojo -D bar=99 -D baz=hello -D foo=11 %s | FileCheck %s --check-prefix=FOO

from sys import env_get_int, env_get_string, is_defined


fn main():
    # CHECK-LABEL: === test_env
    print("=== test_env")

    # CHECK: is_defined(foo) False
    print("is_defined(foo)", is_defined["foo"]())
    # CHECK: is_defined(bar) True
    print("is_defined(bar)", is_defined["bar"]())
    # CHECK: env_get_int(bar) 99
    print("env_get_int(bar)", env_get_int["bar"]())
    # CHECK: env_get_string(baz) hello
    print("env_get_string(baz)", env_get_string["baz"]())

    # CHECK: env_get_int_or(foo, 42) 42
    # FOO: env_get_int_or(foo, 42) 11
    print("env_get_int_or(foo, 42)", env_get_int["foo", 42]())
    # CHECK: env_get_int_or(bar, 42) 99
    print("env_get_int_or(bar, 42)", env_get_int["bar", 42]())
