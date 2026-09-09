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
#
# A thin function pointer whose argument list is a `TypeList.splat[1, T]()`
# variadic pack is conceptually identical to one written with a single
# non-variadic argument of type `T`. This checks that a plain, non-variadic
# function is assignable to both forms.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


struct Memory:
    def __init__(out self):
        pass


def takes_one(arg: Memory):
    print("yes")


def takes_one_with_kwargs(arg: Memory, var **kwargs: Int):
    print(len(kwargs))


def main():
    var f1: def(Memory) thin = takes_one
    var f2: def(* args: * TypeList.splat[1, Memory]()) thin = takes_one
    var f3: def(
        * args: * TypeList.splat[1, Memory](), var ** kwargs: Int
    ) thin = takes_one_with_kwargs

    # CHECK: yes
    f1(Memory())
    # CHECK: yes
    f2(Memory())
    # CHECK: 1
    f3(Memory(), value=1)
