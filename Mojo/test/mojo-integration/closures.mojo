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

# RUN: %mojo -debug-level full %s 2 3 | FileCheck %s
from std.sys import argv


from std.builtin._coroutine import Coroutine
from std.runtime._asyncrt import _run


@no_inline
def takeClosure(var writer: Coroutine[Int, ...]) -> Int:
    return _run(writer^)


@no_inline
def makeClosure(x: Int) -> Coroutine[Int, origin_of()._mlir_origin]:
    var z = x * x

    @__copy_capture(z)
    @__parameter
    async def writer() -> Int:
        return z

    return writer()


def main():
    try:
        var x = atol(argv()[1])
        var y = atol(argv()[2])

        var writer = makeClosure(x)
        var w = takeClosure(writer^)
        # CHECK: 4
        print(w)
    except e:
        print(e)
