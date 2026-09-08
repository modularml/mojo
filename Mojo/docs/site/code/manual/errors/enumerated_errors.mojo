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
struct FileError(Equatable, ImplicitlyCopyable, Writable):
    var _variant: Int

    # Compile-time constant variants
    comptime not_found = FileError(_variant=1)
    comptime permission_denied = FileError(_variant=2)
    comptime already_exists = FileError(_variant=3)

    def variant_name(self) -> String:
        if self._variant == 1:
            return "not_found"
        elif self._variant == 2:
            return "permission_denied"
        elif self._variant == 3:
            return "already_exists"
        return "unknown"

    def write_to(self, mut writer: Some[Writer]):
        writer.write("FileError.", self.variant_name())


def open_file(path: String) raises FileError -> String:
    if not path:
        raise FileError.not_found
    if path == "/secret":
        raise FileError.permission_denied
    if path == "/existing":
        raise FileError.already_exists
    return "Contents of " + path


def handle_file_error(e: FileError):
    if e == FileError.not_found:
        print("Not found:", e)
    elif e == FileError.permission_denied:
        print("Permission denied:", e)
    elif e == FileError.already_exists:
        print("Already exists:", e)


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
