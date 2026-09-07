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


# start-legacy-capturing-closure
def use_closure[func: def(Int) capturing[_] -> Int](num: Int) -> Int:
    return func(num)


def create_closure():
    var x = 1

    @__parameter
    def add(i: Int) -> Int:
        return x + i

    var y = use_closure[add](2)
    print(y)


def main():
    create_closure()
    # end-legacy-capturing-closure
