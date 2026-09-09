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

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s

# Ensure that recursive signature resolution reached through implicit
# conversion during constraint checking produces a normal cycle diagnostic
# instead of crashing.

# CHECK: error: attempt to resolve a recursive reference to declaration 'MyInt.__init__'
# CHECK: note: referenced from here
# CHECK: note: referenced through this use
# CHECK: note: by declaration 'MyInt.__init__'

@fieldwise_init
struct Wrapper(ImplicitlyCopyable, RegisterPassable):
    var value: Int


struct MyInt(TrivialRegisterPassable):
    var value: Int

    def __init__(out self):
        self.value = 0

    @implicit
    def __init__(out self, arg: Wrapper)
        where orig():
        self.value = arg.value


def take_myint(arg: MyInt):
    pass


def orig() -> Bool
    where take_myint(Wrapper(0)):
    return True
