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


# start-parametric-function
def repeat[count: Int](msg: String):
    # evaluate the following for loop at compile time
    comptime for i in range(count):
        print(msg)
        # end-parametric-function


# start-call-parametric-function
def call_repeat():
    repeat[3]("Hello")
    # Prints "Hello" 3 times
    # end-call-parametric-function


def main():
    call_repeat()
