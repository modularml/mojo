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


@always_inline
def modify(mut x: Int):
    x = 42


def use_ints(x: Int, y: Int):
    pass


@fieldwise_init
struct MyPair(TrivialRegisterPassable):
    var x: Int
    var y: Int


def main():
    var p = MyPair(3, 4)
    modify(p.x)
    use_ints(p.x, p.y)  # breakpoint
