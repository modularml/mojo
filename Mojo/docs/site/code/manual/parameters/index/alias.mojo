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
struct Sentiment(Equatable, ImplicitlyCopyable):
    var _value: Int

    comptime NEGATIVE = Sentiment(0)
    comptime NEUTRAL = Sentiment(1)
    comptime POSITIVE = Sentiment(2)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


def is_happy(s: Sentiment):
    if s == Sentiment.POSITIVE:
        print("Yes. 😀")
    else:
        print("No. ☹️")


def main():
    var mood = Sentiment.POSITIVE
    is_happy(mood)
