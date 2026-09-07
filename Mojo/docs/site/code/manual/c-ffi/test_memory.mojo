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
# test_memory.mojo
# Tests for c-ffi.mdx: "Memory management", covering malloc with an Optional
# return, the CBuffer context manager, null handling with getenv, and
# origins keeping Mojo memory alive.
from std.ffi import OwnedDLHandle, c_char, c_int, c_size_t, external_call
from std.testing import assert_equal, assert_true


# --- Allocate C memory with malloc, then free it ---


def create_buffer(
    n: c_size_t,
) -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
    return external_call[
        "malloc", Optional[Pointer[UInt8, MutUntrackedOrigin]]
    ](n)


def test_malloc_free() raises:
    var allocated = create_buffer(c_size_t(16))
    assert_true(Bool(allocated), "malloc failed")
    var buf = allocated.value()
    buf[unsafe_offset=0] = 42
    buf[unsafe_offset=1] = 7
    assert_equal(buf[unsafe_offset=0], UInt8(42))
    assert_equal(buf[unsafe_offset=1], UInt8(7))
    external_call["free", NoneType](buf.unsafe_bitcast[NoneType]())


# --- Context manager that frees C memory automatically ---


struct CBuffer:
    var ptr: Pointer[UInt8, MutUntrackedOrigin]
    var size: c_size_t

    def __init__(out self, n: c_size_t) raises:
        self.size = n
        var allocated = external_call[
            "malloc", Optional[Pointer[UInt8, MutUntrackedOrigin]]
        ](n)
        if not allocated:
            raise Error("malloc failed")
        self.ptr = allocated.value()

    def __enter__(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        return self.ptr

    def __exit__(self):
        external_call["free", NoneType](self.ptr.unsafe_bitcast[NoneType]())


def test_context_manager() raises:
    with CBuffer(c_size_t(1024)) as buf:
        buf[unsafe_offset=0] = 42
        assert_equal(buf[unsafe_offset=0], UInt8(42))


# --- Null-able C pointer wrapped in Optional (getenv) ---


def test_getenv_optional() raises:
    # Set our own variable so the test doesn't depend on the ambient
    # environment (a bazel sandbox may strip PATH).
    var key: String = "MOJO_FFI_TEST_VAR"
    var value: String = "present"
    _ = external_call["setenv", c_int](
        key.as_c_string_slice(),
        value.as_c_string_slice(),
        c_int(1),  # overwrite
    )

    var wrapped = external_call[
        "getenv", Optional[Pointer[c_char, MutUntrackedOrigin]]
    ](key.as_c_string_slice())
    assert_true(Bool(wrapped))
    assert_equal(
        String(unsafe_from_utf8_ptr=wrapped.value()), String("present")
    )

    var missing: String = "MOJO_FFI_DEFINITELY_NOT_SET_98765"
    var empty = external_call[
        "getenv", Optional[Pointer[c_char, MutUntrackedOrigin]]
    ](missing.as_c_string_slice())
    assert_true(not empty)


# --- Origins keep Mojo memory alive across a C call ---


def test_origin_keeps_alive() raises:
    var proc = OwnedDLHandle()  # No path: opens the current process.
    var c_strlen = proc.get_function[c_size_t]("strlen")

    var line = String("Hello")
    assert_equal(Int(c_strlen(line.as_c_string_slice())), 5)

    # Refill the same variable; the pointer's origin still holds.
    line = "Hello, Mojo!"
    assert_equal(Int(c_strlen(line.as_c_string_slice())), 12)


def main() raises:
    test_malloc_free()
    test_context_manager()
    test_getenv_optional()
    test_origin_keeps_alive()
