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

from std.ffi import _Global
from std.sys._io import stdout


def _is_stdout_tty() -> Bool:
    return stdout.isatty()


comptime _USE_COLOR = _Global["IS_STDOUT_TTY", _is_stdout_tty]


def _use_color() -> Bool:
    try:
        return _USE_COLOR.get_or_create_ptr()[]
    except:
        return False


@fieldwise_init
struct Color(ImplicitlyCopyable, Writable):
    """ANSI colors for terminal output."""

    var color: StaticString

    comptime NONE = Self("")
    comptime RED = Self("\033[91m")
    comptime GREEN = Self("\033[92m")
    comptime YELLOW = Self("\033[93m")
    comptime BLUE = Self("\033[94m")
    comptime MAGENTA = Self("\033[95m")
    comptime CYAN = Self("\033[96m")
    comptime BOLD_WHITE = Self("\033[1;97m")
    comptime END = Self("\033[0m")

    def write_to(self, mut writer: Some[Writer]):
        if _use_color():
            writer.write(self.color)


@fieldwise_init
struct Text[W: Writable, origin: ImmOrigin, //, color: Color](Writable):
    """Colors the given writable with the given `Color`."""

    var writable: Pointer[Self.W, Self.origin]

    def __init__(out self, ref[Self.origin] w: Self.W):
        self.writable = Pointer(to=w)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.color)
        self.writable[].write_to(writer)
        writer.write(Color.END)
