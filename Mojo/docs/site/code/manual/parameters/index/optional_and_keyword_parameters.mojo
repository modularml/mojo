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


def speak[a: Int = 3, msg: String = "woof"]():
    print(msg, a)


def use_defaults():
    speak()  # prints 'woof 3'
    speak[5]()  # prints 'woof 5'
    speak[7, "meow"]()  # prints 'meow 7'
    speak[msg="baaa"]()  # prints 'baaa 3'


def main():
    use_defaults()
