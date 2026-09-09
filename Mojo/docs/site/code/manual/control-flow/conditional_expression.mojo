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
    # Conditional expression
    var temp_celsius = 15
    var forecast = "warm" if temp_celsius > 20 else "cool"
    print("The forecast for today is", forecast)

    # Equivalent if-else statement
    if temp_celsius > 20:
        forecast = "warm"
    else:
        forecast = "cool"
    print("The forecast for today is", forecast)
