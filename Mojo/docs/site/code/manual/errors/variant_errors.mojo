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

from std.utils import Variant


@fieldwise_init
struct NotFoundError(Copyable, Writable):
    var path: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write("file not found: ", self.path)


@fieldwise_init
struct PermissionError(Copyable, Writable):
    var path: String
    var required_role: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "permission denied on ",
            self.path,
            " (requires ",
            self.required_role,
            ")",
        )


comptime FileError = Variant[NotFoundError, PermissionError]


def open_file(path: String) raises FileError -> String:
    if not path:
        raise FileError(NotFoundError(""))
    if path == "/secret":
        raise FileError(PermissionError("/secret", "admin"))
    return "Contents of " + path


def handle_file_error(e: FileError):
    if e.isa[NotFoundError]():
        print("Not found:", e[NotFoundError])
    elif e.isa[PermissionError]():
        print("Access denied:", e[PermissionError])


def main():
    try:
        print(open_file("/data"))
    except e:
        handle_file_error(e)

    try:
        print(open_file(""))
    except e:
        handle_file_error(e)

    try:
        print(open_file("/secret"))
    except e:
        handle_file_error(e)
