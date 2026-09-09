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

from std.pathlib import Path
from std.ffi import OwnedDLHandle

from std.sys.info import CompilationTarget
from std.testing import assert_equal, assert_raises, assert_true
from std.testing import TestSuite


def _load_libc() raises -> OwnedDLHandle:
    """Loads libc from the standard location for this platform.

    Selects platform-appropriate paths up front so that a failure on
    (say) Linux doesn't propagate a confusing macOS-path error message.
    """
    comptime if CompilationTarget.is_linux():
        try:
            return OwnedDLHandle("libc.so.6")  # glibc
        except:
            pass
        return OwnedDLHandle("libc.so")  # musl / BSD
    elif CompilationTarget.is_macos():
        return OwnedDLHandle("/usr/lib/system/libsystem_c.dylib")
    else:
        comptime assert False, "libc discovery not implemented for platform"


def _load_libm() raises -> OwnedDLHandle:
    """Loads libm (math functions) from the standard location for this
    platform.

    On glibc, math lives in libm.so.6. On macOS and musl, math symbols
    are folded into libc, so we fall back to `_load_libc` there.
    """
    comptime if CompilationTarget.is_linux():
        try:
            return OwnedDLHandle("libm.so.6")  # glibc
        except:
            pass
        # musl folds math into libc.
        return _load_libc()
    else:
        # On macOS, math symbols are available through libSystem / libc.
        return _load_libc()


# ===----------------------------------------------------------------------=== #
# OwnedDLHandle tests
# ===----------------------------------------------------------------------=== #


def test_owned_dlhandle_invalid_path() raises:
    with assert_raises(contains="dlopen failed"):
        _ = OwnedDLHandle("/an/invalid/library")


def test_owned_dlhandle_invalid_path_obj() raises:
    with assert_raises(contains="dlopen failed"):
        _ = OwnedDLHandle(Path("/an/invalid/library"))


def test_owned_dlhandle_load_valid_library() raises:
    try:
        # Try common locations for libc
        var lib = OwnedDLHandle("libc.so.6")  # Linux
        assert_true(lib.__bool__(), "Library handle should be valid")
    except:
        try:
            var lib = OwnedDLHandle("libc.so")  # Some Linux systems
            assert_true(lib.__bool__(), "Library handle should be valid")
        except:
            try:
                var lib = OwnedDLHandle(
                    "/usr/lib/system/libsystem_c.dylib"
                )  # macOS
                assert_true(lib.__bool__(), "Library handle should be valid")
            except:
                # If none work, skip this test
                print(
                    "Warning: Could not find a standard C library to test with"
                )


def test_owned_dlhandle_check_symbol() raises:
    try:
        var lib = OwnedDLHandle("libc.so.6")
        # Common C library functions that should exist
        assert_true(lib.check_symbol("printf"), "printf should exist in libc")
        assert_true(lib.check_symbol("malloc"), "malloc should exist in libc")
    except:
        try:
            var lib = OwnedDLHandle("libc.so")
            assert_true(
                lib.check_symbol("printf"), "printf should exist in libc"
            )
        except:
            # Skip if we can't load libc
            print("Warning: Could not load libc for symbol test")


def test_owned_dlhandle_borrow() raises:
    """Test that borrow() returns a valid DLHandle reference."""
    try:
        var lib = OwnedDLHandle("libc.so.6")
        var borrowed = lib.borrow()
        # borrowed should be a valid DLHandle
        assert_true(borrowed.__bool__(), "Borrowed handle should be valid")
        assert_true(
            borrowed.check_symbol("printf"),
            "Borrowed handle should access symbols",
        )
    except:
        try:
            var lib = OwnedDLHandle("libc.so")
            var borrowed = lib.borrow()
            assert_true(borrowed.__bool__(), "Borrowed handle should be valid")
        except:
            # Skip if we can't load libc
            print("Warning: Could not load libc for borrow test")


def test_owned_dlhandle_global_symbols() raises:
    """Test loading global symbols from current process."""
    try:
        # Load symbols from the current process
        var lib = OwnedDLHandle()
        assert_true(lib.__bool__(), "Global symbol handle should be valid")
    except:
        # This might fail on some systems
        print("Warning: Could not load global symbols")


def test_owned_dlhandle_get_symbol_missing() raises:
    """Test that get_symbol returns None for a nonexistent symbol."""

    def _test_with_lib(lib: OwnedDLHandle) raises:
        var result = lib.get_symbol[NoneType](
            "this_symbol_does_not_exist_xyz_42"
        )
        assert_true(not result, "Missing symbol should return None")

    try:
        _test_with_lib(OwnedDLHandle("libc.so.6"))
    except:
        try:
            _test_with_lib(OwnedDLHandle("libc.so"))
        except:
            try:
                _test_with_lib(
                    OwnedDLHandle("/usr/lib/system/libsystem_c.dylib")
                )
            except:
                print(
                    "Warning: Could not load a standard C library to test with"
                )


def test_owned_dlhandle_get_symbol_found() raises:
    """Test that get_symbol returns a value for an existing symbol."""

    def _test_with_lib(lib: OwnedDLHandle) raises:
        var result = lib.get_symbol[NoneType]("printf")
        assert_true(Bool(result), "Existing symbol should return a value")

    try:
        _test_with_lib(OwnedDLHandle("libc.so.6"))
    except:
        try:
            _test_with_lib(OwnedDLHandle("libc.so"))
        except:
            try:
                _test_with_lib(
                    OwnedDLHandle("/usr/lib/system/libsystem_c.dylib")
                )
            except:
                print(
                    "Warning: Could not load a standard C library to test with"
                )


def test_owned_dlhandle_get_function_keepalive() raises:
    """Inline resolve and call with no later use of the handle."""
    var lib = _load_libc()
    # Inline resolve + call, no subsequent use of `lib`.
    var pid = lib.get_function[Int32]("getpid")()
    assert_true(pid > 0, "getpid should return a positive pid")


def test_owned_dlhandle_get_function_stored_callable() raises:
    var lib = _load_libc()
    var getpid_fn = lib.get_function[Int32]("getpid")
    assert_true(getpid_fn() > 0, "call 1")
    assert_true(getpid_fn() > 0, "call 2")
    assert_true(getpid_fn() > 0, "call 3")


def test_owned_dlhandle_get_function_multiple_inline_calls() raises:
    """Resolves and calls the same symbol inline, without binding the returned
    callable to a variable."""
    var lib = _load_libc()
    _ = lib.get_function[Int32]("getpid")()
    _ = lib.get_function[Int32]("getpid")()
    _ = lib.get_function[Int32]("getpid")()
    _ = lib.get_function[Int32]("getpid")()


def test_owned_dlhandle_get_function_with_args() raises:
    """Exercises the variadic argument-forwarding path with a scalar-in,
    scalar-out function through the C ABI."""
    var lib = _load_libc()
    var abs_fn = lib.get_function[Int32]("abs")
    assert_equal(abs_fn(Int32(-5)), Int32(5), "abs(-5) should return 5")
    assert_equal(abs_fn(Int32(42)), Int32(42), "abs(42) should return 42")
    assert_equal(abs_fn(Int32(0)), Int32(0), "abs(0) should return 0")


def test_owned_dlhandle_get_function_multiple_args() raises:
    """Exercises forwarding of more than one argument. A single argument does
    not distinguish per-argument lowering from a one-element pack."""
    var lib = _load_libm()
    var pow_fn = lib.get_function[Float64]("pow")
    assert_equal(pow_fn(Float64(2.0), Float64(10.0)), Float64(1024.0), "2^10")
    assert_equal(pow_fn(Float64(2.0), Float64(-2.0)), Float64(0.25), "2^-2")


def test_owned_dlhandle_get_function_missing_symbol_raises() raises:
    """A missing symbol raises `Error`, so callers that probe for optional
    symbols can recover."""
    var lib = _load_libc()
    with assert_raises(contains="symbol not found"):
        _ = lib.get_function[Int32]("this_symbol_does_not_exist_xyz_42")


def test_owned_dlhandle_get_function_float64_return() raises:
    """Exercises the `Float64` return-type path to match the docstring
    example and ensure non-`Int32` scalars round-trip through the C-ABI
    forwarding correctly."""
    var lib = _load_libm()
    var sqrt_fn = lib.get_function[Float64]("sqrt")
    assert_equal(sqrt_fn(Float64(4.0)), Float64(2.0), "sqrt(4.0)")
    assert_equal(sqrt_fn(Float64(0.0)), Float64(0.0), "sqrt(0.0)")
    assert_equal(sqrt_fn(Float64(1.0)), Float64(1.0), "sqrt(1.0)")


def test_owned_dlhandle_get_function_default_return_type() raises:
    """Exercises the default `NoneType` return type (omitted type param)
    against a void-returning libc function. `srand(unsigned)` takes a
    scalar and returns void."""
    var lib = _load_libc()
    var srand_fn = lib.get_function("srand")
    srand_fn(UInt32(42))
    srand_fn(UInt32(0))


def test_owned_dlhandle_get_function_explicit_nonetype_return() raises:
    """Same as the default-return-type test, but with `NoneType` stated
    explicitly — covers the shape used by `TVMFFIErrorMoveFromRaised`
    call sites."""
    var lib = _load_libc()
    var srand_fn = lib.get_function[NoneType]("srand")
    srand_fn(UInt32(7))


def test_owned_dlhandle_get_function_pointer_arg() raises:
    """Exercises a pointer argument through the C ABI: `atoi(const char*)`.
    A pointer is a scalar-class argument, distinct from the integer and float
    scalars covered elsewhere."""
    var lib = _load_libc()
    var atoi_fn = lib.get_function[Int32]("atoi")
    var s = String("123")
    assert_equal(atoi_fn(s.as_c_string_slice()), Int32(123), "atoi(123)")
    var s2 = String("-42")
    assert_equal(atoi_fn(s2.as_c_string_slice()), Int32(-42), "atoi(-42)")


@fieldwise_init
struct _DivT(TrivialRegisterPassable):
    """Matches C `div_t` = `{ int quot; int rem; }`."""

    var quot: Int32
    var rem: Int32


def test_owned_dlhandle_get_function_struct_return() raises:
    """Returns an aggregate by value through the C ABI:
    `div(int, int) -> div_t`."""
    var lib = _load_libc()
    var div_fn = lib.get_function[_DivT]("div")
    var r = div_fn(Int32(17), Int32(5))
    assert_equal(r.quot, Int32(3), "17 / 5 quot")
    assert_equal(r.rem, Int32(2), "17 % 5 rem")
    var r2 = div_fn(Int32(-17), Int32(5))
    assert_equal(r2.quot, Int32(-3), "-17 / 5 quot")
    assert_equal(r2.rem, Int32(-2), "-17 % 5 rem")


@fieldwise_init
struct _CDouble(TrivialRegisterPassable):
    """Same by-value ABI as C `double _Complex`: a homogeneous `{f64, f64}`
    aggregate (SSE-pair on x86-64, HFA-2 on ARM64)."""

    var re: Float64
    var im: Float64


def test_owned_dlhandle_get_function_struct_arg() raises:
    """Passes an aggregate by value through the C ABI:
    `cabs(double _Complex)`."""
    var lib = _load_libm()
    var cabs_fn = lib.get_function[Float64]("cabs")
    assert_equal(cabs_fn(_CDouble(3.0, 4.0)), Float64(5.0), "cabs(3+4i)")
    assert_equal(cabs_fn(_CDouble(5.0, 12.0)), Float64(13.0), "cabs(5+12i)")


def test_owned_dlhandle_call_struct_by_value() raises:
    """Passes and returns an aggregate by value through `OwnedDLHandle.call`,
    the other public entry point to the same forwarding."""
    var libm = _load_libm()
    assert_equal(
        libm.call["cabs", Float64](_CDouble(3.0, 4.0)),
        Float64(5.0),
        "call cabs(3+4i)",
    )
    var libc = _load_libc()
    var r = libc.call["div", _DivT](Int32(17), Int32(5))
    assert_equal(r.quot, Int32(3), "call div quot")
    assert_equal(r.rem, Int32(2), "call div rem")


def test_owned_dlhandle_automatic_cleanup() raises:
    # This test primarily verifies that the code compiles and runs
    # without crashes. The actual cleanup happens automatically.

    @always_inline
    def create_and_destroy_handle():
        try:
            var lib = OwnedDLHandle("libc.so.6")
            _ = lib.check_symbol("printf")
            # lib will be automatically closed here when it goes out of scope
        except:
            pass

    # Call the function multiple times to ensure cleanup works
    create_and_destroy_handle()
    create_and_destroy_handle()
    create_and_destroy_handle()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
