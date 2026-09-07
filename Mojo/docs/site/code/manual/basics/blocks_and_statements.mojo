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


# start-loop
def loop():
    for x in range(5):
        if x % 2 == 0:
            print(x)
            # end-loop


def matrix_multiply(a: Int, b: Int, c: Int):
    pass


# fmt: off

def linebreaks():
    var matrix_a = 10
    var matrix_b = 20
    var result_matrix = 5
    # start-linebreaks-in-function
    matrix_multiply(
        matrix_a,
        matrix_b,
        result_matrix
    )
    # end-linebreaks-in-function

# fmt: on


def linebreaks2():
    # start-multiline-string
    var long_text = (
        "This is a long line of text that is a lot easier to read if"
        " it is broken up across two lines instead of one long line."
    )
    # end-multiline-string
    _ = long_text


def main():
    loop()
    linebreaks()
    linebreaks2()
