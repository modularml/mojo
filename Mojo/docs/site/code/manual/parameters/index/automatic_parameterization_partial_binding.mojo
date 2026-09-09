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


@fieldwise_init
struct Fudge[sugar: Int, cream: Int, chocolate: Int = 7](Writable):
    pass


def eat(f: Fudge[5, ...]):
    print("Ate", f)


def devour(f: Fudge[_, 6, _]):
    print("Devoured", String(f))


def devour2(f: Fudge[_, chocolate=_, cream=6]):
    print("Devoured", String(f))


def main():
    eat(Fudge[5, 5, 7]())
    eat(Fudge[5, 8, 9]())
    # eat(Fudge[12, 5, 7]())  # invalid call to 'eat': failed to infer implicit
    # parameter 'cream' of argument 'f' type 'Fudge
    devour(Fudge[3, 6, 9]())
    devour(Fudge[4, 6, 8]())
