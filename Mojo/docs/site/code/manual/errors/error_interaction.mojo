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


def validate_with_error(value: Int) raises -> Int:
    if value < 0:
        raise "value cannot be negative"
    return value


def validate_typed(value: Int) raises ValidationError -> Int:
    if value < 0:
        raise ValidationError(field="value", reason="cannot be negative")
    return value


# Pattern 1: Wrap an Error into a typed error at the boundary
def wrapped_validate(value: Int) raises ValidationError -> Int:
    try:
        return validate_with_error(value)
    except e:
        raise ValidationError(field="value", reason=String(e))


# Anti-pattern: bare raises erases type info at compile time
def validate_bare_raises(value: Int) raises -> Int:
    return validate_typed(value)


def main():
    # Pattern 1: Wrapping Error into typed errors
    print("--- Wrapping Error into typed errors ---")
    try:
        _ = wrapped_validate(-5)
    except e:
        print("Field:", e.field, "Reason:", e.reason)

    # Pattern 2: Catch typed errors with field access
    print("--- Catching typed errors ---")
    try:
        _ = validate_typed(-5)
    except e:
        print("Field:", e.field, "Reason:", e.reason)

    # Anti-pattern: bare raises loses compile-time type info
    print("--- Bare raises (type erased) ---")
    try:
        _ = validate_bare_raises(-5)
    except e:
        # e is typed as Error, not ValidationError
        # e.field would not compile here
        print("Caught (no field access):", e)
