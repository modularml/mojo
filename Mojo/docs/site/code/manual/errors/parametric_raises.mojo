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
struct NetworkError(Copyable, Writable):
    var code: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("NetworkError(code=", self.code, ")")


@fieldwise_init
struct ParseError(Copyable, Writable):
    var position: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ParseError(position=", self.position, ")")


def run_action[
    ErrorType: AnyType
](action: def() thin raises ErrorType -> Int) raises ErrorType -> Int:
    return action()


def fetch_data() raises NetworkError -> Int:
    raise NetworkError(code=404)


def parse_config() raises ParseError -> Int:
    raise ParseError(position=42)


def get_value() -> Int:
    return 99


def main():
    # ErrorType inferred as NetworkError
    try:
        _ = run_action(fetch_data)
    except e:
        print("Network failure:", e)

    # ErrorType inferred as ParseError
    try:
        _ = run_action(parse_config)
    except e:
        print("Parse failure:", e)

    # ErrorType inferred as Never — no try block needed
    var result = run_action(get_value)
    print("Got value:", result)
