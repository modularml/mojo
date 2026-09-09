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


# Always raises, never returns — return type is Never
def panic(msg: String) raises -> Never:
    raise Error(msg)


# Never raises, always returns — equivalent to: def safe_add(a: Int, b: Int) -> Int
def safe_add(a: Int, b: Int) raises Never -> Int:
    return a + b


def get_value_or_panic(maybe: Optional[Int]) raises -> Int:
    if maybe:
        return maybe.value()
    # Never substitutes for Int in this branch
    panic("value is missing")


def main():
    # A function that always raises
    try:
        panic("something went wrong")
    except e:
        print("Caught:", e)

    # Never substituting for return type in control flow
    try:
        var value = get_value_or_panic(Optional(42))
        print("Got value:", value)
    except e:
        print("Failed:", e)

    try:
        var value = get_value_or_panic(Optional[Int]())
        print("Got value:", value)
    except e:
        print("Failed:", e)

    # A function that never raises — no try block needed
    var result = safe_add(3, 4)
    print("safe_add(3, 4):", result)
