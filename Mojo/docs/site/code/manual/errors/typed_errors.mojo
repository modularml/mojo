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
struct ValidationError(Copyable, Writable):
    var field: String
    var reason: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ValidationError(", self.field, "): ", self.reason)


def validate_username(username: String) raises ValidationError -> String:
    if username.byte_length() == 0:
        raise ValidationError(field="username", reason="cannot be empty")
    if username.count_codepoints() < 3:
        raise ValidationError(
            field="username", reason="must be at least 3 characters"
        )
    return username


def main():
    # Valid username
    try:
        print(validate_username("alice"))
    except e:
        print("Error in field '" + e.field + "': " + e.reason)

    # Error: empty username
    try:
        print(validate_username(""))
    except e:
        print("Error in field '" + e.field + "': " + e.reason)

    # Error: too-short username
    try:
        print(validate_username("ab"))
    except e:
        print("Error in field '" + e.field + "': " + e.reason)
