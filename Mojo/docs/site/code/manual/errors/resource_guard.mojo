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
from std.reflection import *


@fieldwise_init
struct ConnectionError(Copyable, Writable):
    var message: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ConnectionError: ", self.message)


struct ResourceGuard(ImplicitlyCopyable):
    var name: String
    var suppress_errors: Bool

    def __init__(out self, name: String, suppress_errors: Bool = False):
        self.name = name
        self.suppress_errors = suppress_errors

    def __enter__(self) -> Self:
        print("Acquiring:", self.name)
        return self

    def __exit__(self):
        print("Releasing:", self.name, "(no error)")

    def __exit__[ErrType: AnyType](self, err: ErrType) -> Bool:
        comptime type_name = reflect[ErrType].name()
        print("Releasing:", self.name)
        print("  Error type:", type_name)

        comptime if conforms_to(ErrType, Writable):
            print("  Message:", err)

        return self.suppress_errors


def use_connection() raises ConnectionError:
    raise ConnectionError("connection timed out")


def main() raises:
    # No error — calls __exit__(self)
    print("--- No error ---")
    with ResourceGuard("database"):
        print("Working...")

    # Typed error, not suppressed — calls __exit__[ErrType]
    print("\n--- Error, not suppressed ---")
    try:
        with ResourceGuard("network"):
            use_connection()
    except e:
        print("Caught:", e)

    # Typed error, suppressed — __exit__[ErrType] returns True
    print("\n--- Error, suppressed ---")
    with ResourceGuard("cache", suppress_errors=True):
        use_connection()
    print("Continued after suppressed error")
