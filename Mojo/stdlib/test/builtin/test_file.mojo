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

from std.os import remove
from std.pathlib import Path, _dir_of_current_file
from std.stat import S_ISFIFO
from std.subprocess import run
from std.tempfile import gettempdir
from std.time import sleep

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

comptime DUMMY_FILE_SIZE: Int = 954


def test_file_read() raises:
    var path = _dir_of_current_file() / "test_file_dummy_input.txt"
    with open(path, "r") as f:
        assert_true(
            f.read().startswith(
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            )
        )


def test_file_read_multi() raises:
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        assert_equal(f.read(12), "Lorem ipsum ")
        assert_equal(f.read(6), "dolor ")
        assert_true(
            f.read().startswith("sit amet, consectetur adipiscing elit.")
        )


def test_file_read_bytes_multi() raises:
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var bytes1 = f.read_bytes(12)
        assert_equal(len(bytes1), 12, "12 bytes")
        var string1 = String(unsafe_from_utf8=bytes1)
        assert_equal(string1.byte_length(), 12, "12 chars")
        assert_equal(string1, "Lorem ipsum ")

        var bytes2 = f.read_bytes(6)
        assert_equal(len(bytes2), 6, "6 bytes")
        var string2 = String(unsafe_from_utf8=bytes2)
        assert_equal(string2.byte_length(), 6, "6 chars")
        assert_equal(string2, "dolor ")

        # Read where N is greater than the number of bytes in the file.
        var s: String = f.read(1_000_000_000)

        assert_equal(s.byte_length(), 936)
        assert_true(s.startswith("sit amet, consectetur adipiscing elit."))


def test_file_read_bytes_all() raises:
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var bytes_all = f.read_bytes(-1)
        assert_equal(len(bytes_all), Int(DUMMY_FILE_SIZE))


def test_file_read_bytes_zero() raises:
    """Test reading 0 bytes returns empty list."""
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var bytes_zero = f.read_bytes(0)
        assert_equal(len(bytes_zero), 0)


def test_file_read_bytes_empty_file() raises:
    """Test reading from empty file returns empty list."""
    var temp_file = Path(gettempdir().value()) / "test_file_read_bytes_empty"

    # Create empty file
    with open(temp_file, "w"):
        pass

    # Read all bytes from empty file
    with open(temp_file, "r") as f:
        var bytes_all = f.read_bytes(-1)
        assert_equal(len(bytes_all), 0)

    # Read specific size from empty file
    with open(temp_file, "r") as f:
        var bytes_sized = f.read_bytes(10)
        assert_equal(len(bytes_sized), 0)


def test_file_read_bytes_large_with_resizing() raises:
    """Test read_bytes() with size=-1 triggers buffer doubling for large files.

    The DUMMY_FILE_SIZE is 954 bytes, which exceeds the initial 256 byte buffer,
    so this tests the exponential growth logic (256 -> 512 -> 1024).
    """
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var all_bytes = f.read_bytes()  # size=-1 default
        assert_equal(len(all_bytes), Int(DUMMY_FILE_SIZE))
        # Verify content is correct
        var content = String(unsafe_from_utf8=all_bytes)
        assert_true(content.startswith("Lorem ipsum"))


def test_file_read_bytes_from_write_only() raises:
    """Test that read_bytes from write-only file raises error."""
    var temp_file = (
        Path(gettempdir().value()) / "test_file_read_bytes_writeonly"
    )

    # Clean up any leftover file from previous runs
    try:
        remove(temp_file)
    except:
        pass

    var f = open(temp_file, "w")
    # Should raise error with errno message (EBADF - Bad file descriptor)
    with assert_raises(contains="Bad file"):
        _ = f.read_bytes()
    f.close()


def test_file_read_bytes_sequential_small() raises:
    """Test multiple small sequential read_bytes() calls."""
    var temp_file = Path(gettempdir().value()) / "test_file_read_bytes_seq"

    # Create file with known content
    var content = "0123456789" * 10  # 100 bytes
    with open(temp_file, "w") as f:
        f.write(content)

    # Read in chunks of 10 bytes
    with open(temp_file, "r") as f:
        var total_read = 0
        for _ in range(10):
            var chunk = f.read_bytes(10)
            assert_equal(len(chunk), 10)
            total_read += len(chunk)

        # Try to read more, should get 0 bytes (EOF)
        var eof = f.read_bytes(10)
        assert_equal(len(eof), 0)

        assert_equal(total_read, 100)


def test_file_read_all() raises:
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var all = f.read(-1)
        assert_equal(all.byte_length(), Int(DUMMY_FILE_SIZE))


def test_file_read_path() raises:
    var file_path = _dir_of_current_file() / "test_file_dummy_input.txt"

    with open(file_path, "r") as f:
        assert_true(
            f.read().startswith(
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            )
        )


def test_file_path_direct_read() raises:
    var file_path = _dir_of_current_file() / "test_file_dummy_input.txt"
    assert_true(
        file_path.read_text().startswith(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        )
    )


def test_file_read_context() raises:
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        assert_true(
            f.read().startswith(
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            )
        )


def test_file_read_to_address() raises:
    comptime DUMMY_FILE_SIZE = 954
    # Test buffer size > file size
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer = Array[UInt8, length=1000](fill=0)
        assert_equal(f.read(buffer), DUMMY_FILE_SIZE)
        assert_equal(buffer[0], 76)  # L
        assert_equal(buffer[1], 111)  # o
        assert_equal(buffer[2], 114)  # r
        assert_equal(buffer[3], 101)  # e
        assert_equal(buffer[4], 109)  # m
        assert_equal(buffer[5], 32)  # <space>
        assert_equal(buffer[56], 10)  # <LF>

    # Test buffer size < file size
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer = Array[UInt8, length=500](fill=0)
        assert_equal(f.read(buffer), 500)

    # Test buffer size == file size
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer = Array[UInt8, length=DUMMY_FILE_SIZE](fill=0)
        assert_equal(f.read(buffer), DUMMY_FILE_SIZE)

    # Test buffer size 0
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer = List[UInt8]()
        assert_equal(f.read(buffer), 0)

    # Test sequential reads of different sizes
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer_30 = Array[UInt8, length=30](fill=0)
        var buffer_1 = Array[UInt8, length=1](fill=0)
        var buffer_2 = Array[UInt8, length=2](fill=0)
        var buffer_100 = Array[UInt8, length=100](fill=0)
        var buffer_1000 = Array[UInt8, length=1000](fill=0)
        assert_equal(f.read(buffer_30), 30)
        assert_equal(f.read(buffer_1), 1)
        assert_equal(f.read(buffer_2), 2)
        assert_equal(f.read(buffer_100), 100)
        assert_equal(f.read(buffer_1000), DUMMY_FILE_SIZE - (30 + 1 + 2 + 100))

    # Test read after EOF
    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var buffer_1000 = Array[UInt8, length=1000](fill=0)
        assert_equal(f.read(buffer_1000), DUMMY_FILE_SIZE)
        assert_equal(f.read(buffer_1000), 0)


def test_file_seek() raises:
    import std.os

    with open(
        _dir_of_current_file() / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var pos = f.seek(6)
        assert_equal(pos, 6)

        comptime expected_msg1 = (
            "ipsum dolor sit amet, consectetur adipiscing elit."
        )
        assert_equal(f.read(expected_msg1.byte_length()), expected_msg1)

        # Seek from the end of the file
        pos = f.seek(-16, std.os.SEEK_END)
        assert_equal(pos, 938)

        var neg_offset = -16
        pos = f.seek(neg_offset, std.os.SEEK_END)
        assert_equal(pos, 938)

        _ = f.read(6)

        # Seek from current position, skip the space
        pos = f.seek(1, std.os.SEEK_CUR)
        assert_equal(pos, 945)
        assert_equal(f.read(7), "rhoncus")

        try:
            _ = f.seek(-12)
        except e:
            comptime expected_msg = "Failed to seek"
            assert_equal(
                String(e)[byte = : expected_msg.byte_length()], expected_msg
            )


def test_file_open_nodir() raises:
    var f = open(Path("test_file_open_nodir"), "w")
    f.close()


def test_file_write() raises:
    var content: String = "The quick brown fox jumps over the lazy dog"
    var TEMP_FILE = Path(gettempdir().value()) / "test_file_write"
    with open(TEMP_FILE, "w") as f:
        f.write(content)

    with open(TEMP_FILE, "r") as read_file:
        assert_equal(read_file.read(), content)


def test_file_write_span() raises:
    var content: String = "The quick brown fox jumps over the lazy dog"
    var TEMP_FILE = Path(gettempdir().value()) / "test_file_write_span"

    # Clean up any leftover file from previous runs
    try:
        remove(TEMP_FILE)
    except:
        pass

    with open(TEMP_FILE, "w") as f:
        f.write_bytes(content.as_bytes())

    with open(TEMP_FILE, "r") as read_file:
        assert_equal(read_file.read(), content)


def test_file_write_again() raises:
    var unexpected_content: String = "foo bar baz"
    var expected_content: String = "foo bar"
    var TEMP_FILE = Path(gettempdir().value()) / "test_file_write_again"
    with open(TEMP_FILE, "w") as f:
        f.write(unexpected_content)

    with open(TEMP_FILE, "w") as f:
        f.write(expected_content)

    with open(TEMP_FILE, "r") as read_file:
        assert_equal(read_file.read(), expected_content)


def test_file_rw_mode_preserves_content() raises:
    """Test that opening a file in 'rw' mode does not truncate existing content.

    The FileHandle "rw" mode should not truncate file contents before reading,
    unlike "w" mode which should truncate.
    """
    var temp_file = Path(gettempdir().value()) / "test_file_rw_mode"

    # Clean up any leftover file from previous runs
    try:
        remove(temp_file)
    except:
        pass

    # First, create a file with some content using write mode
    var expected_content = "hello\nworld"
    with open(temp_file, "w") as f:
        f.write(expected_content)

    # Now open it in "rw" mode and verify we can read the existing content
    with open(temp_file, "rw") as f:
        _ = f.seek(0)
        var content = f.read()
        assert_equal(
            content,
            expected_content,
            "rw mode should preserve existing file content",
        )

        # Also verify we can write to it
        _ = f.seek(0)
        f.write("new content")

    # Verify the write succeeded
    with open(temp_file, "r") as f:
        assert_equal(f.read(), "new content")


def test_file_write_mode_truncates() raises:
    """Test that opening a file in 'w' mode truncates existing content."""
    var temp_file = Path(gettempdir().value()) / "test_file_write_mode"

    # Clean up any leftover file from previous runs
    try:
        remove(temp_file)
    except:
        pass

    # Create a file with some content
    with open(temp_file, "w") as f:
        f.write("initial content")

    # Open in write mode and write less content
    with open(temp_file, "w") as f:
        f.write("new")

    # Verify the file was truncated
    with open(temp_file, "r") as f:
        assert_equal(
            f.read(), "new", "w mode should truncate existing file content"
        )


def test_file_get_raw_fd() raises:
    # since JIT and build give different file descriptors, we test by checking
    # if we printed to the right file.
    # First, ensure the test files are empty by opening in write mode
    var temp1 = Path(gettempdir().value()) / "test_file_dummy_1"
    var temp2 = Path(gettempdir().value()) / "test_file_dummy_2"
    var temp3 = Path(gettempdir().value()) / "test_file_dummy_3"
    # Ensure the files are empty by doing this cleanup at the beginning of the
    # test
    with open(temp1, "w"):
        pass
    with open(temp2, "w"):
        pass
    with open(temp3, "w"):
        pass

    var f1 = open(temp1, "rw")
    var f2 = open(temp2, "rw")
    var f3 = open(temp3, "rw")

    print(
        "test from file 1",
        file=FileDescriptor(f1._get_raw_fd()),
        flush=True,
        end="",
    )
    _ = f1.seek(0)
    assert_equal(f1.read(), "test from file 1")
    assert_equal(f2.read(), "")
    assert_equal(f3.read(), "")

    _ = f1.seek(0)
    _ = f2.seek(0)
    _ = f3.seek(0)

    print(
        "test from file 2",
        file=FileDescriptor(f2._get_raw_fd()),
        flush=True,
        end="",
    )
    print(
        "test from file 3",
        file=FileDescriptor(f3._get_raw_fd()),
        flush=True,
        end="",
    )

    _ = f2.seek(0)
    _ = f3.seek(0)

    assert_equal(f3.read(), "test from file 3")
    assert_equal(f2.read(), "test from file 2")
    assert_equal(f1.read(), "test from file 1")

    f1.close()
    f2.close()
    f3.close()


def test_file_append_mode() raises:
    """Test that opening a file in 'a' mode appends to existing content."""
    var temp_file = Path(gettempdir().value()) / "test_file_append_mode"

    # Clean up any leftover file from previous runs
    try:
        remove(temp_file)
    except:
        pass

    # Create a file with initial content
    var initial_content = "initial content"
    with open(temp_file, "w") as f:
        f.write(initial_content)

    # Open in append mode and add more content
    var appended_text = " appended"
    with open(temp_file, "a") as f:
        f.write(appended_text)

    # Verify the content was appended, not overwritten
    with open(temp_file, "r") as f:
        var content = f.read()
        assert_equal(
            content,
            initial_content + appended_text,
            "append mode should add to existing content",
        )

    # Test multiple append operations
    with open(temp_file, "a") as f:
        f.write(" more")
    with open(temp_file, "a") as f:
        f.write(" text")

    with open(temp_file, "r") as f:
        var final_content = f.read()
        assert_equal(
            final_content,
            initial_content + appended_text + " more text",
            "multiple appends should accumulate",
        )


def test_file_append_mode_creates_file() raises:
    """Test that append mode creates a new file if it doesn't exist."""
    var temp_file = Path(gettempdir().value()) / "test_file_append_new"

    # Delete the file if it exists
    try:
        remove(temp_file)
    except:
        pass

    # Open in append mode (should create the file)
    var content = "new file content"
    with open(temp_file, "a") as f:
        f.write(content)

    # Verify the file was created with the content
    with open(temp_file, "r") as f:
        assert_equal(
            f.read(), content, "append mode should create new file if missing"
        )


def test_file_append_mode_with_unicode() raises:
    """Test that append mode works correctly with Unicode characters."""
    var temp_file = Path(gettempdir().value()) / "test_file_append_unicode"

    # Clean up any leftover file from previous runs
    try:
        remove(temp_file)
    except:
        pass

    # Create a file with Unicode content
    with open(temp_file, "w") as f:
        f.write("Hello 🔥")

    # Append more Unicode content
    with open(temp_file, "a") as f:
        f.write(" World 🚀")

    # Verify both parts are present
    with open(temp_file, "r") as f:
        var content = f.read()
        assert_equal(
            content,
            "Hello 🔥 World 🚀",
            "append mode should handle Unicode correctly",
        )


def test_file_open_fifo() raises:
    """Test that opening a FIFO in write mode doesn't attempt to remove it.

    Regression test for bug where `FileHandle` should not try to remove
    special files (FIFOs, devices, sockets) when opening in write mode.
    Only regular files should be removed/truncated in write mode.

    This test creates a FIFO and verifies that attempting to open it doesn't
    raise the "unable to remove existing file" error. We use a background
    reader process to avoid blocking.
    """
    var fifo_path = Path(gettempdir().value()) / "test_file_fifo"

    # Clean up any existing FIFO from previous test runs
    try:
        remove(fifo_path)
    except:
        pass

    # Create a FIFO using mkfifo command. In the future, we should add a
    # `mkfifo` function in the stdlib itself.
    # Note that `mkfifo` is mandatory in POSIX which is all we currently
    # support, so no need to guard against availability.
    _ = run("mkfifo " + String(fifo_path))

    # Verify the FIFO was created
    assert_true(fifo_path.exists())

    # Start a background reader with explicit synchronization
    # Create a flag file that signals when the reader is ready
    var ready_flag = Path(gettempdir().value()) / "test_file_fifo_ready"
    try:
        remove(ready_flag)
    except:
        pass

    # Start the reader and signal when it's ready
    # The reader opens the FIFO first, then creates the ready flag
    var start_reader = (
        "sh -c '(cat "
        + String(fifo_path)
        + " > /dev/null & echo $! > "
        + String(gettempdir().value())
        + "/fifo_reader_pid; sleep 0.1; touch "
        + String(ready_flag)
        + ") &' >/dev/null 2>&1"
    )
    try:
        _ = run(start_reader)
    except:
        print("Warning: Could not start background reader, skipping test")
        try:
            remove(fifo_path)
        except:
            pass
        return

    # Wait for reader to signal it's ready (with timeout)
    var max_wait = 20  # 20 iterations * 0.1s = 2 seconds max
    var reader_ready = False
    for _ in range(max_wait):
        if ready_flag.exists():
            reader_ready = True
            break
        sleep(0.1)

    if not reader_ready:
        print("Warning: Reader not ready after timeout, skipping test")
        try:
            remove(fifo_path)
            remove(ready_flag)
        except:
            pass
        return

    # The key test: opening a FIFO in write mode should NOT raise
    # "unable to remove existing file" error. The bug was that `FileHandle`
    # tried to remove the FIFO before opening it, which failed.
    # If this raises an error, the test will fail.
    var f = open(fifo_path, "w")
    f.write("test data\n")
    f.close()

    # Clean up the FIFO and ready flag
    try:
        remove(fifo_path)
        remove(ready_flag)
    except:
        pass


def test_file_read_from_closed_file() raises:
    """Test that reading from a closed file raises an error with proper message.
    """
    var temp_file = Path(gettempdir().value()) / "test_file_read_closed"

    # Clean up any leftover file from previous runs
    try:
        remove(temp_file)
    except:
        pass

    # Create a file with some content
    with open(temp_file, "w") as f:
        f.write("test content")

    # Open and immediately close the file
    var f = open(temp_file, "r")
    f.close()

    # Trying to read from closed file should raise error with "invalid file handle"
    with assert_raises(contains="invalid file handle"):
        _ = f.read()


def test_file_read_from_write_only_file() raises:
    """Test that reading from a write-only file raises an error with errno."""
    var temp_file = Path(gettempdir().value()) / "test_file_read_writeonly"

    # Clean up any leftover file from previous runs
    try:
        remove(temp_file)
    except:
        pass

    # Open in write-only mode and try to read
    var f = open(temp_file, "w")

    # Should raise error with "Bad file" (EBADF) in the message
    with assert_raises(contains="Bad file"):
        _ = f.read()

    f.close()


def test_file_seek_invalid_file() raises:
    """Test that seeking on a closed file raises an error with proper message.
    """
    var temp_file = Path(gettempdir().value()) / "test_file_seek_closed"

    # Clean up any leftover file from previous runs
    try:
        remove(temp_file)
    except:
        pass

    with open(temp_file, "w") as f:
        f.write("test content")

    var f = open(temp_file, "r")
    f.close()

    # Trying to seek on closed file should raise error with "invalid file handle"
    with assert_raises(contains="invalid file handle"):
        _ = f.seek(0)


def test_file_read_bytes_to_span_from_closed() raises:
    """Test that reading bytes into a Span from a closed file raises an error.
    """
    var temp_file = Path(gettempdir().value()) / "test_file_read_span_closed"

    # Clean up any leftover file from previous runs
    try:
        remove(temp_file)
    except:
        pass

    with open(temp_file, "w") as f:
        f.write("test content")

    var f = open(temp_file, "r")
    f.close()

    # Try to read into a buffer from closed file - should get "invalid file handle"
    var buffer = Array[UInt8, length=10](fill=0)
    with assert_raises(contains="invalid file handle"):
        _ = f.read(buffer)


def test_file_multiple_close() raises:
    """Test that closing a file multiple times is safe."""
    var temp_file = Path(gettempdir().value()) / "test_file_multiple_close"

    # Clean up any leftover file from previous runs
    try:
        remove(temp_file)
    except:
        pass

    with open(temp_file, "w") as f:
        f.write("test")

    var f = open(temp_file, "r")

    # First close should succeed
    try:
        f.close()
    except e:
        assert_true(False, "First close should not raise: " + String(e))

    # Second close should also succeed (be a no-op)
    try:
        f.close()
    except e:
        assert_true(False, "Second close should not raise: " + String(e))


def test_file_invalid_mode() raises:
    """Test that invalid file mode strings raise proper error."""
    var temp_file = Path(gettempdir().value()) / "test_file_invalid_mode"

    # Test various invalid mode strings
    with assert_raises(contains="invalid mode:"):
        _ = open(temp_file, "x")

    with assert_raises(contains="invalid mode:"):
        _ = open(temp_file, "rb")

    with assert_raises(contains="invalid mode:"):
        _ = open(temp_file, "w+")

    with assert_raises(contains="invalid mode:"):
        _ = open(temp_file, "")


def test_file_read_nonexistent() raises:
    """Test that opening a non-existent file in read mode raises proper error.
    """
    var nonexistent_file = (
        Path(gettempdir().value()) / "test_file_does_not_exist_12345"
    )

    # Make sure the file doesn't exist
    try:
        remove(nonexistent_file)
    except:
        pass

    # Should raise error when file doesn't exist
    with assert_raises(contains="No such file or directory"):
        _ = open(nonexistent_file, "r")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
