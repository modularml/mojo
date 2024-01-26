# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -D CURRENT_DIR=%S -D TEMP_FILE_DIR=%T -debug-level full -I %kernels_test_root %s | FileCheck %s


from pathlib import Path
from sys.info import os_is_windows
from sys.param_env import env_get_string
from testing import assert_equal

from memory.buffer import Buffer, NDBuffer

from test_utils import linear_fill
from tensor import Tensor

alias CURRENT_DIR = env_get_string["CURRENT_DIR"]()
alias TEMP_FILE_DIR = env_get_string["TEMP_FILE_DIR"]()


# CHECK-LABEL: test_file_read
def test_file_read():
    print("== test_file_read")

    # CHECK: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    var f = open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    )
    print(f.read())
    f.close()


# CHECK-LABEL: test_file_read_multi
def test_file_read_multi():
    print("== test_file_read_multi")

    var f = open(
        (Path(CURRENT_DIR) / "test_file_dummy_input.txt"),
        "r",
    )

    # CHECK: Lorem ipsum
    print(f.read(12))

    # CHECK: dolor
    print(f.read(6))

    # CHECK: sit amet, consectetur adipiscing elit.
    print(f.read())

    f.close()


# CHECK-LABEL: test_file_read_bytes_multi
def test_file_read_bytes_multi():
    print("== test_file_read_bytes_multi")

    var f = open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    )

    # CHECK: Lorem ipsum
    let tensor1 = f.read_bytes(12)
    print(StringRef(tensor1.data(), tensor1.num_elements()))

    # CHECK: dolor
    let tensor2 = f.read_bytes(6)
    print(StringRef(tensor2.data(), tensor2.num_elements()))

    # Read where N is greater than the number of bytes in the file.
    let s: String = f.read(1e9)

    # CHECK: 936
    print(len(s))

    # CHECK: sit amet, consectetur adipiscing elit.
    print(s)

    _ = tensor2 ^
    _ = tensor1 ^

    f.close()


# CHECK-LABEL: test_file_read_path
def test_file_read_path():
    print("== test_file_read_path")

    let file_path = Path(CURRENT_DIR) / "test_file_dummy_input.txt"

    # CHECK: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    var f = open(file_path, "r")
    print(f.read())
    f.close()


# CHECK-LABEL: test_file_path_direct_read
def test_file_path_direct_read():
    print("== test_file_path_direct_read")

    let file_path = Path(CURRENT_DIR) / "test_file_dummy_input.txt"
    # CHECK: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    print(file_path.read_text())


# CHECK-LABEL: test_file_read_context
def test_file_read_context():
    print("== test_file_read_context")

    # CHECK: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    with open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    ) as f:
        print(f.read())


# CHECK-LABEL: test_file_seek
def test_file_seek():
    print("== test_file_seek")

    with open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    ) as f:
        let pos = f.seek(6)
        assert_equal(pos, 6)

        alias expected_msg1 = "ipsum dolor sit amet, consectetur adipiscing elit."
        assert_equal(f.read(len(expected_msg1)), expected_msg1)

        try:
            f.seek(-12)
        except e:
            alias expected_msg = "seek error"
            assert_equal(str(e)[: len(expected_msg)], expected_msg)


# CHECK-LABEL: test_file_write
def test_file_write():
    print("== test_file_write")

    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_file_write"
    var f = open(TEMP_FILE, "w")
    f.write("The quick brown fox jumps over the lazy dog")
    f.close()

    # CHECK: The quick brown fox jumps over the lazy dog
    var read_file = open(TEMP_FILE, "r")
    print(read_file.read())
    read_file.close()


# CHECK-LABEL: test_file_write_again
def test_file_write_again():
    print("== test_file_write_again")

    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_file_write_again"
    with open(TEMP_FILE, "w") as f:
        f.write("foo bar baz")

    with open(TEMP_FILE, "w") as f:
        f.write("foo bar")

    # CHECK: foo bar
    # CHECK-NOT: baz
    var read_file = open(TEMP_FILE, "r")
    print(read_file.read())
    read_file.close()


# CHECK-LABEL: test_buffer
def test_buffer():
    print("== test_buffer")
    let buf = Buffer[4, DType.float32].stack_allocation()
    buf.fill(2.0)
    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_buffer"
    buf.tofile(TEMP_FILE)

    with open(TEMP_FILE, "r") as f:
        let str = f.read()
        let buf_read = Buffer[4, DType.float32](
            str._as_ptr().bitcast[DType.float32]()
        )
        for i in range(4):
            # CHECK: 0.0
            print(buf[i] - buf_read[i])

        # Ensure string is not destroyed before the above check.
        _ = str[0]


# CHECK-LABEL: test_ndbuffer
def test_ndbuffer():
    print("== test_ndbuffer")
    let buf = NDBuffer[2, DimList(2, 2), DType.float32].stack_allocation()
    buf.fill(2.0)
    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_ndbuffer"
    buf.tofile(TEMP_FILE)

    with open(TEMP_FILE, "r") as f:
        let str = f.read()
        let buf_read = NDBuffer[2, DimList(2, 2), DType.float32](
            str._as_ptr().bitcast[DType.float32]()
        )
        for i in range(2):
            for j in range(2):
                # CHECK: 0.0
                print(buf[i, j] - buf_read[i, j])

        # Ensure string is not destroyed before the above check.
        _ = str[0]


# CHECK-LABEL: test_tensor
def test_tensor():
    print("== test_tensor")
    var tensor = Tensor[DType.int8](4)
    linear_fill(tensor, 1, 2, 3, 4)
    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_tensor"
    tensor.tofile(TEMP_FILE)
    let tensor_from_file = Tensor[DType.int8].fromfile(TEMP_FILE)

    # CHECK-NOT: False
    for i in range(tensor.num_elements()):
        print(tensor[i] == tensor_from_file[i])


# CHECK-LABEL: test_tensor_load_save
fn test_tensor_load_save() raises:
    print("== test_tensor_load_save")
    let tensor = Tensor[DType.int8](2, 2, 3)
    tensor._to_buffer().fill(2)

    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_tensor_load_save"
    tensor.save(TEMP_FILE)

    let tensor_loaded = Tensor[DType.int8].load(TEMP_FILE)

    # CHECK: True
    print(tensor == tensor_loaded)


def main():
    test_file_read()
    test_file_read_multi()
    test_file_read_bytes_multi()
    test_file_read_path()
    test_file_path_direct_read()
    test_file_read_context()
    test_file_seek()
    test_file_write()
    test_file_write_again()
    test_buffer()
    test_ndbuffer()
    test_tensor()
    test_tensor_load_save()
