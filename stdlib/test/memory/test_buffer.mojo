# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -D TEMP_FILE_DIR=%T -debug-level full %s | FileCheck %s

from memory.buffer import Buffer
from pathlib import Path
from sys.param_env import env_get_string
from utils.list import Dim

alias TEMP_FILE_DIR = env_get_string["TEMP_FILE_DIR"]()


# CHECK-LABEL: test_buffer
fn test_buffer():
    print("== test_buffer")

    alias vec_size = 4
    var data = Pointer[Float32].alloc(vec_size)

    var b1 = Buffer[DType.float32, 4](data)
    var b2 = Buffer[DType.float32, 4](data, 4)
    var b3 = Buffer[DType.float32](data, 4)

    # CHECK: 4 4 4
    print(len(b1), len(b2), len(b3))

    data.free()


# CHECK-LABEL: test_buffer
def test_buffer_tofile():
    print("== test_buffer")
    var buf = Buffer[DType.float32, 4].stack_allocation()
    buf.fill(2.0)
    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_buffer"
    buf.tofile(TEMP_FILE)

    with open(TEMP_FILE, "r") as f:
        var str = f.read()
        var buf_read = Buffer[DType.float32, 4](
            str._as_ptr().bitcast[DType.float32]()
        )
        for i in range(4):
            # CHECK: 0.0
            print(buf[i] - buf_read[i])

        # Ensure string is not destroyed before the above check.
        _ = str[0]


def main():
    test_buffer()
    test_buffer_tofile()
