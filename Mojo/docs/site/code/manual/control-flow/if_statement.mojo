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


def main():
    # Simple if statement
    var temp_celsius = Float64(25)
    if temp_celsius > 20:
        print("It is warm.")
        print("The temperature is", temp_celsius * 9 / 5 + 32, "Fahrenheit.")

    # Single-line if statements
    temp_celsius = 22
    # fmt: off
    if temp_celsius < 15:  print("It is cool.")  # Skipped because condition is False
    if temp_celsius > 20:  print("It is warm.")
    # fmt: on

    # Else and elif statements
    temp_celsius = 25
    if temp_celsius <= 0:
        print("It is freezing.")
    elif temp_celsius < 20:
        print("It is cool.")
    elif temp_celsius < 30:
        print("It is warm.")
    else:
        print("It is hot.")
