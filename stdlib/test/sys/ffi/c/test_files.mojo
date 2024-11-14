# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s


from testing import assert_equal, assert_false, assert_raises, assert_true

from pathlib import _dir_of_current_file
from time import sleep
from memory import UnsafePointer, memset, memcpy, memcmp, stack_allocation
from sys.info import os_is_macos

from sys.ffi.c.types import C, char_ptr, FILE
from sys.ffi.c.libc import TryLibc, Libc
from sys.ffi.c.constants import *


def _test_open_close(libc: Libc, suffix: String):
    file = str(_dir_of_current_file() / ("dummy_test_open_close" + suffix))
    ptr = char_ptr(file)
    with TryLibc(libc):
        filedes = libc.open(ptr, O_RDWR | O_CREAT | O_TRUNC | O_NONBLOCK, 0o666)
        assert_true(filedes != -1)
        sleep(0.05)
        assert_true(libc.close(filedes) != -1)
        for s in List(O_RDONLY, O_WRONLY, O_RDWR):
            print(s[])
            # if os_is_macos() and s[] != O_RDONLY:  # Permission denied
            #     continue
            filedes = libc.open(ptr, s[] | O_NONBLOCK)
            assert_true(filedes != -1)
            sleep(0.05)
            assert_true(libc.close(filedes) != -1)

        assert_true(libc.remove(ptr) != -1)
    _ = file^


def test_dynamic_open_close():
    _test_open_close(Libc[static=False](), "_dynamic")


def test_static_open_close():
    _test_open_close(Libc[static=True](), "_static")


def _test_fopen_fclose(libc: Libc, suffix: String):
    file = str(_dir_of_current_file() / ("dummy_test_fopen_fclose" + suffix))
    ptr = char_ptr(file)
    with TryLibc(libc):
        filedes = libc.creat(ptr, 0o666)
        assert_true(filedes != -1)
        for s in List(
            FM_WRITE,
            FM_WRITE_READ_CREATE,
            FM_READ,
            FM_READ_WRITE,
            FM_APPEND,
            FM_APPEND_READ,
        ):
            stream = libc.fopen(ptr, char_ptr(s[]))
            assert_true(stream != C.NULL.bitcast[FILE]())
            sleep(0.05)
            assert_true(libc.fclose(stream) != EOF)

        assert_true(libc.remove(ptr) != -1)
    _ = file^


def test_dynamic_fopen_fclose():
    _test_fopen_fclose(Libc[static=False](), "_dynamic")


def test_static_fopen_fclose():
    _test_fopen_fclose(Libc[static=True](), "_static")


def _test_fdopen_fclose(libc: Libc, suffix: String):
    file = str(_dir_of_current_file() / ("dummy_test_fdopen_fclose" + suffix))
    ptr = char_ptr(file)
    with TryLibc(libc):
        filedes = libc.creat(ptr, 0o666)
        assert_true(filedes != -1)
        for s in List(
            FM_WRITE,
            FM_WRITE_READ_CREATE,
            FM_READ,
            FM_READ_WRITE,
            FM_APPEND,
            FM_APPEND_READ,
        ):
            stream = libc.fdopen(filedes, char_ptr(s[]))
            assert_true(stream != C.NULL.bitcast[FILE]())
            sleep(0.05)
            assert_true(libc.fclose(stream) != EOF)
            filedes = libc.open(ptr, O_RDWR)
            assert_true(filedes != -1)

        assert_true(libc.remove(ptr) != -1)
    _ = file^


def test_dynamic_fdopen_fclose():
    _test_fdopen_fclose(Libc[static=False](), "_dynamic")


def test_static_fdopen_fclose():
    _test_fdopen_fclose(Libc[static=True](), "_static")


def _test_creat_openat(libc: Libc, suffix: String):
    file = str(_dir_of_current_file() / ("dummy_test_creat_openat" + suffix))
    ptr = char_ptr(file)
    with TryLibc(libc):
        filedes = libc.creat(ptr, 0o666)
        assert_true(filedes != -1)
        filedes = libc.openat(filedes, ptr, O_RDWR)
        assert_true(filedes != -1)
        sleep(0.05)
        assert_true(libc.close(filedes) != -1)
        assert_true(libc.remove(ptr) != -1)
    _ = file^


def test_dynamic_creat_openat():
    _test_creat_openat(Libc[static=False](), "_dynamic")


def test_static_creat_openat():
    _test_creat_openat(Libc[static=True](), "_static")


def _test_freopen(libc: Libc, suffix: String):
    file = str(_dir_of_current_file() / ("dummy_test_freopen" + suffix))
    ptr = char_ptr(file)
    with TryLibc(libc):
        filedes = libc.creat(ptr, 0o666)
        assert_true(filedes != -1)
        stream = libc.fopen(ptr, char_ptr(FM_READ_WRITE))
        assert_true(stream != C.NULL.bitcast[FILE]())
        sleep(0.05)
        stream = libc.freopen(ptr, char_ptr(FM_READ_WRITE), stream)
        assert_true(stream != C.NULL.bitcast[FILE]())
        sleep(0.05)
        assert_true(libc.close(filedes) != -1)
        assert_true(libc.remove(ptr) != -1)
    _ = file^


def test_dynamic_freopen():
    _test_freopen(Libc[static=False](), "_dynamic")


def test_static_freopen():
    _test_freopen(Libc[static=True](), "_static")


def _test_fmemopen_fprintf(libc: Libc, suffix: String):
    file = str(
        _dir_of_current_file() / ("dummy_test_fmemopen_fprintf" + suffix)
    )
    ptr = char_ptr(file)
    with TryLibc(libc):
        filedes = libc.creat(ptr, 0o666)
        assert_true(filedes != -1)

        # test print to file
        stream = libc.fopen(ptr, char_ptr(FM_WRITE))
        assert_true(stream != C.NULL.bitcast[FILE]())
        size = 1000
        a = UnsafePointer[Byte].alloc(size)
        memset(a, ord("a"), size - 1)
        a[size - 1] = 0
        num_bytes = libc.fprintf(stream, char_ptr(a))
        assert_equal(num_bytes, size - 1)
        assert_true(libc.fclose(stream) != EOF)

        # test print to buffer
        p = UnsafePointer[Byte].alloc(size)
        memset(p, 0, size)
        stream = libc.fmemopen(p.bitcast[C.void](), size, char_ptr(FM_WRITE))
        assert_true(stream != C.NULL.bitcast[FILE]())
        num_bytes = libc.fprintf(stream, char_ptr(a))
        assert_equal(num_bytes, size - 1)

        assert_true(libc.fclose(stream) != EOF)  # flush stream
        assert_true(libc.remove(ptr) != -1)
        assert_equal(0, memcmp(p, a, size - 1))  # compare buffer
        a.free()
        p.free()
    _ = file^


def test_dynamic_fmemopen_fprintf():
    _test_fmemopen_fprintf(Libc[static=False](), "_dynamic")


def test_static_fmemopen_fprintf():
    _test_fmemopen_fprintf(Libc[static=True](), "_static")


def _test_fseek_ftell(libc: Libc, suffix: String):
    file = str(_dir_of_current_file() / ("dummy_test_fseek_ftell" + suffix))
    ptr = char_ptr(file)
    with TryLibc(libc):
        filedes = libc.creat(ptr, 0o666)
        assert_true(filedes != -1)

        # print to file
        stream = libc.fopen(ptr, char_ptr(FM_WRITE))
        assert_true(stream != C.NULL.bitcast[FILE]())
        size = ord("~") - ord(" ")
        a = UnsafePointer[C.char].alloc(size)
        idx = 0
        for i in range(ord(" "), ord("~")):

            @parameter
            if os_is_macos():
                # MacOS is not actually compliant with ANSI C89, doesn't print
                # '%'. I think it triggers format specifier if it's not '%%'
                if i == ord("%"):
                    a[idx] = i + 1
                else:
                    a[idx] = i
            else:
                a[idx] = i
            idx += 1
        a[size - 1] = 0
        num_bytes = libc.fprintf(stream, a)
        assert_equal(num_bytes, size - 1)

        assert_true(libc.fflush(stream) != EOF)  # flush stream

        # test seek
        stream = libc.fopen(ptr, char_ptr(FM_WRITE))
        assert_true(stream != C.NULL.bitcast[FILE]())
        assert_equal(libc.fseek(stream, 10, SEEK_SET), 0)
        assert_equal(libc.ftell(stream), 10)
        assert_equal(libc.fseeko(stream, 10, SEEK_CUR), 0)
        assert_equal(libc.ftello(stream), 20)

        assert_true(libc.fclose(stream) != EOF)
        assert_true(libc.remove(ptr) != -1)
        a.free()
    _ = file^


def test_dynamic_fseek_ftell():
    _test_fseek_ftell(Libc[static=False](), "_dynamic")


def test_static_fseek_ftell():
    _test_fseek_ftell(Libc[static=True](), "_static")


def _test_fput_fget(libc: Libc, suffix: String):
    file = str(_dir_of_current_file() / ("dummy_test_fput_fget" + suffix))
    ptr = char_ptr(file)
    with TryLibc(libc):
        filedes = libc.creat(ptr, 0o666)
        assert_true(filedes != -1)

        # write
        filedes = libc.open(ptr, O_RDWR)
        assert_true(filedes != -1)
        stream = libc.fdopen(filedes, char_ptr(FM_READ_WRITE))
        assert_true(stream != C.NULL.bitcast[FILE]())
        size = 255
        for i in range(size - 1):
            assert_equal(libc.fputc(i + 1, stream), i + 1)

        assert_true(libc.fflush(stream) != EOF)  # flush stream
        stream = libc.fopen(ptr, char_ptr(FM_READ_WRITE))

        # read and compare
        for i in range(size - 1):
            assert_equal(libc.fgetc(stream), i + 1)

        a = UnsafePointer[C.char].alloc(size)
        memset(a, ord("a"), size - 1)
        a[size - 1] = 0

        # write
        stream = libc.fopen(ptr, char_ptr(FM_READ_WRITE))
        assert_true(libc.fputs(a, stream) != EOF)
        assert_true(libc.fclose(stream) != EOF)

        # read and compare
        stream = libc.fopen(ptr, char_ptr(FM_READ_WRITE))
        b = UnsafePointer[C.char].alloc(size)
        p = libc.fgets(b, size, stream)
        assert_equal(b, p)
        assert_true(libc.fflush(stream) != EOF)  # flush stream
        assert_equal(0, memcmp(p, a, size))

        # cleanup
        assert_true(libc.fclose(stream) != EOF)
        assert_true(libc.remove(ptr) != -1)
        a.free()
        b.free()
    _ = file^


def test_dynamic_fput_fget():
    _test_fput_fget(Libc[static=False](), "_dynamic")


def test_static_fput_fget():
    _test_fput_fget(Libc[static=True](), "_static")


def _test_dprintf(libc: Libc, suffix: String):
    file = str(_dir_of_current_file() / ("dummy_test_dprintf" + suffix))
    ptr = char_ptr(file)
    with TryLibc(libc):
        filedes = libc.creat(ptr, 0o666)
        assert_true(filedes != -1)

        # setup
        alias `%` = C.char(ord("%"))
        alias `d` = C.char(ord("d"))
        alias `1` = C.char(ord("1"))
        a = UnsafePointer[C.char].alloc(9)
        b = UnsafePointer[C.char].alloc(5)
        c = UnsafePointer[C.char].alloc(5)

        # print
        filedes = libc.open(ptr, O_RDWR)
        assert_true(filedes != -1)
        a[0], a[1], a[2], a[3], a[4] = `%`, `d`, `%`, `d`, C.char(0)
        num_bytes = libc.dprintf(filedes, a, C.int(1), C.int(1))
        assert_equal(num_bytes, 2)
        assert_true(libc.close(filedes) != -1)

        # read and compare
        stream = libc.fopen(ptr, char_ptr(FM_READ_WRITE))
        p = libc.fgets(b, 3, stream)
        assert_equal(b, p)
        assert_true(libc.fflush(stream) != EOF)  # flush stream
        c[0], c[1], c[2] = `1`, `1`, C.char(0)
        assert_equal(0, memcmp(p, c, 3))

        # print
        filedes = libc.open(ptr, O_RDWR)
        assert_true(filedes != -1)
        a[4], a[5], a[6], a[7], a[8] = `%`, `d`, `%`, `d`, C.char(0)
        num_bytes = libc.dprintf(
            filedes, a, C.int(1), C.int(1), C.int(1), C.int(1)
        )
        assert_equal(num_bytes, 4)
        assert_true(libc.close(filedes) != -1)

        # read and compare
        stream = libc.fopen(ptr, char_ptr(FM_READ_WRITE))
        p = libc.fgets(b, 5, stream)
        assert_equal(b, p)
        assert_true(libc.fflush(stream) != EOF)  # flush stream
        c[0], c[1], c[2], c[3], c[4] = `1`, `1`, `1`, `1`, C.char(0)
        assert_equal(0, memcmp(p, c, 5))

        # cleanup
        assert_true(libc.fclose(stream) != EOF)
        assert_true(libc.remove(ptr) != -1)
        a.free()
        b.free()
        c.free()
    _ = file^


def test_dynamic_dprintf():
    _test_dprintf(Libc[static=False](), "_dynamic")


def test_static_dprintf():
    _test_dprintf(Libc[static=True](), "_static")


def _test_printf(libc: Libc, suffix: String):
    with TryLibc(libc):
        # setup
        alias `%` = C.char(ord("%"))
        alias `d` = C.char(ord("d"))
        alias `1` = C.char(ord("1"))
        a = UnsafePointer[C.char].alloc(9)
        b = UnsafePointer[C.char].alloc(5)
        c = UnsafePointer[C.char].alloc(5)

        # print
        a[0], a[1], a[2], a[3], a[4] = `%`, `d`, `%`, `d`, C.char(0)
        num_bytes = libc.printf(a, C.int(1), C.int(1))
        assert_equal(num_bytes, 2)

        # print
        a[4], a[5], a[6], a[7], a[8] = `%`, `d`, `%`, `d`, C.char(0)
        num_bytes = libc.printf(a, C.int(1), C.int(1), C.int(1), C.int(1))
        assert_equal(num_bytes, 4)

        # cleanup
        a.free()
        b.free()
        c.free()


def test_dynamic_printf():
    _test_printf(Libc[static=False](), "_dynamic")


def test_static_printf():
    _test_printf(Libc[static=True](), "_static")


def _test_snprintf(libc: Libc, suffix: String):
    with TryLibc(libc):
        # setup
        alias `%` = C.char(ord("%"))
        alias `d` = C.char(ord("d"))
        alias `1` = C.char(ord("1"))
        a = UnsafePointer[C.char].alloc(9)
        b = UnsafePointer[C.char].alloc(5)
        c = UnsafePointer[C.char].alloc(5)

        # print
        a[0], a[1], a[2], a[3], a[4] = `%`, `d`, `%`, `d`, C.char(0)
        num_bytes = libc.snprintf(b, 3, a, C.int(1), C.int(1))
        assert_equal(num_bytes, 2)

        # read and compare
        c[0], c[1], c[2] = `1`, `1`, C.char(0)
        assert_equal(0, memcmp(b, c, 3))

        # print
        a[4], a[5], a[6], a[7], a[8] = `%`, `d`, `%`, `d`, C.char(0)
        num_bytes = libc.snprintf(
            b, 5, a, C.int(1), C.int(1), C.int(1), C.int(1)
        )
        assert_equal(num_bytes, 4)

        # read and compare
        c[2], c[3], c[4] = `1`, `1`, C.char(0)
        assert_equal(0, memcmp(b, c, 5))

        # cleanup
        a.free()
        b.free()
        c.free()


def test_dynamic_snprintf():
    _test_snprintf(Libc[static=False](), "_dynamic")


def test_static_snprintf():
    _test_snprintf(Libc[static=True](), "_static")


def _test_fscanf(libc: Libc, suffix: String):
    file = str(_dir_of_current_file() / ("dummy_test_fscanf" + suffix))
    ptr = char_ptr(file)
    with TryLibc(libc):
        filedes = libc.creat(ptr, 0o666)
        assert_true(filedes != -1)

        # setup
        alias `1` = C.char(ord("1"))
        a = UnsafePointer[C.char].alloc(2)

        filedes = libc.open(ptr, O_RDWR)
        assert_true(filedes != -1)
        a[0], a[1] = `1`, C.char(0)

        stream = libc.fdopen(filedes, char_ptr(FM_READ_WRITE))

        # print
        num_bytes = libc.fputs(a, stream)

        # read and compare
        value = stack_allocation[1, C.int]()
        value[0] = 0
        assert_true(libc.fseek(stream, 0) != -1)
        scanned = libc.fscanf(stream, char_ptr("%d"), value)
        assert_true(libc.fflush(stream) != EOF)
        assert_equal(num_bytes, 1)
        assert_equal(scanned, 1)
        assert_equal(value[0], 1)

        # cleanup
        assert_true(libc.close(filedes) != -1)
        assert_true(libc.remove(ptr) != -1)
        a.free()
    _ = file^


def test_dynamic_fscanf():
    _test_fscanf(Libc[static=False](), "_dynamic")


def test_static_fscanf():
    _test_fscanf(Libc[static=True](), "_static")


def _test_fcntl(libc: Libc, suffix: String):
    file = str(_dir_of_current_file() / ("dummy_test_fcntl" + suffix))
    ptr = char_ptr(file)
    with TryLibc(libc):
        filedes = libc.creat(ptr, 0o666)
        assert_true(filedes != -1)
        filedes = libc.fcntl(filedes, F_GETFD)
        assert_true(filedes != -1)
        filedes = libc.openat(filedes, ptr, O_RDWR)
        assert_true(filedes != -1)
        sleep(0.05)
        assert_true(libc.close(filedes) != -1)
        assert_true(libc.remove(ptr) != -1)
    _ = file^


def test_dynamic_fcntl():
    _test_fcntl(Libc[static=False](), "_dynamic")


def test_static_fcntl():
    _test_fcntl(Libc[static=True](), "_static")


def _test_ioctl(libc: Libc, suffix: String):
    file = str(_dir_of_current_file() / ("dummy_test_ioctl" + suffix))
    ptr = char_ptr(file)
    with TryLibc(libc):
        # TODO: a thorough test using the most often used functionality
        # see https://man7.org/linux/man-pages/man3/ioctl.3p.html
        ...
    _ = file^


def test_dynamic_ioctl():
    _test_ioctl(Libc[static=False](), "_dynamic")


def test_static_ioctl():
    _test_ioctl(Libc[static=True](), "_static")


def main():
    test_dynamic_open_close()
    test_static_open_close()
    test_dynamic_fopen_fclose()
    test_static_fopen_fclose()
    test_dynamic_fdopen_fclose()
    test_static_fdopen_fclose()
    test_dynamic_creat_openat()
    test_static_creat_openat()
    test_dynamic_freopen()
    test_static_freopen()
    test_dynamic_fmemopen_fprintf()
    test_static_fmemopen_fprintf()
    test_dynamic_fseek_ftell()
    test_static_fseek_ftell()
    test_dynamic_fput_fget()
    test_static_fput_fget()
    test_dynamic_dprintf()
    test_static_dprintf()
    test_dynamic_printf()
    test_static_printf()
    test_dynamic_printf()
    test_static_printf()
    test_dynamic_snprintf()
    test_static_snprintf()
    test_dynamic_fscanf()
    test_static_fscanf()
    test_dynamic_fcntl()
    test_static_fcntl()
    test_dynamic_ioctl()
    test_static_ioctl()
