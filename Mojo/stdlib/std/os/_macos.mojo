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

from std.collections import Array
from std.ffi import external_call
from std.time.time import _CTimeSpec

from .fstat import stat_result

comptime dev_t = Int32
comptime mode_t = UInt16
comptime nlink_t = UInt16

comptime __darwin_ino64_t = UInt64
comptime uid_t = UInt32
comptime gid_t = UInt32
comptime off_t = Int64
comptime blkcnt_t = Int64
comptime blksize_t = Int32


@fieldwise_init
struct _c_stat(Copyable, Defaultable, Writable):
    var st_dev: dev_t
    """ID of device containing file."""
    var st_mode: mode_t
    """Mode of file."""
    var st_nlink: nlink_t
    """Number of hard links."""
    var st_ino: __darwin_ino64_t
    """File serial number."""
    var st_uid: uid_t
    """User ID of the file."""
    var st_gid: gid_t
    """Group ID of the file."""
    var st_rdev: dev_t
    """Device ID."""
    var st_atimespec: _CTimeSpec
    """Time of last access."""
    var st_mtimespec: _CTimeSpec
    """Time of last data modification."""
    var st_ctimespec: _CTimeSpec
    """Time of last status change."""
    var st_birthtimespec: _CTimeSpec
    """Time of file creation(birth)."""
    var st_size: off_t
    """File size, in bytes."""
    var st_blocks: blkcnt_t
    """Blocks allocated for file."""
    var st_blksize: blksize_t
    """Optimal blocksize for I/O."""
    var st_flags: UInt32
    """User defined flags for file."""
    var st_gen: UInt32
    """File generation number."""
    var st_lspare: Int32
    """RESERVED: DO NOT USE!."""
    var st_qspare: Array[Int64, 2]
    """RESERVED: DO NOT USE!."""

    def __init__(out self):
        self.st_dev = 0
        self.st_mode = 0
        self.st_nlink = 0
        self.st_ino = 0
        self.st_uid = 0
        self.st_gid = 0
        self.st_rdev = 0
        self.st_atimespec = _CTimeSpec()
        self.st_mtimespec = _CTimeSpec()
        self.st_ctimespec = _CTimeSpec()
        self.st_birthtimespec = _CTimeSpec()
        self.st_size = 0
        self.st_blocks = 0
        self.st_blksize = 0
        self.st_flags = 0
        self.st_gen = 0
        self.st_lspare = 0
        self.st_qspare: Array[Int64, 2] = [0, 0]

    def write_to(self, mut writer: Some[Writer]):
        # fmt: off
        writer.write(
            "{\nst_dev: ", self.st_dev,
            ",\nst_mode: ", self.st_mode,
            ",\nst_nlink: ", self.st_nlink,
            ",\nst_ino: ", self.st_ino,
            ",\nst_uid: ", self.st_uid,
            ",\nst_gid: ", self.st_gid,
            ",\nst_rdev: ", self.st_rdev,
            ",\nst_atimespec: ", self.st_atimespec,
            ",\nst_mtimespec: ", self.st_mtimespec,
            ",\nst_ctimespec: ", self.st_ctimespec,
            ",\nst_birthtimespec: ", self.st_birthtimespec,
            ",\nst_size: ", self.st_size,
            ",\nst_blocks: ", self.st_blocks,
            ",\nst_blksize: ", self.st_blksize,
            ",\nst_flags: ", self.st_flags,
            "st_gen: ", self.st_gen,
            "\n}",
        )
        # fmt: on

    def _to_stat_result(self) -> stat_result:
        return stat_result(
            st_dev=Int(self.st_dev),
            st_mode=Int(self.st_mode),
            st_nlink=Int(self.st_nlink),
            st_ino=Int(self.st_ino),
            st_uid=Int(self.st_uid),
            st_gid=Int(self.st_gid),
            st_rdev=Int(self.st_rdev),
            st_atimespec=self.st_atimespec,
            st_ctimespec=self.st_ctimespec,
            st_mtimespec=self.st_mtimespec,
            st_birthtimespec=self.st_birthtimespec,
            st_size=Int(self.st_size),
            st_blocks=Int(self.st_blocks),
            st_blksize=Int(self.st_blksize),
            st_flags=Int(self.st_flags),
        )


@always_inline
def _stat(var path: String) raises -> _c_stat:
    var stat = _c_stat()
    var err = external_call["stat", Int32](
        path.as_c_string_slice(), Pointer(to=stat)
    )
    if err == -1:
        raise Error("unable to stat '", path, "'")
    return stat^


@always_inline
def _lstat(var path: String) raises -> _c_stat:
    var stat = _c_stat()
    var err = external_call["lstat", Int32](
        path.as_c_string_slice(), Pointer(to=stat)
    )
    if err == -1:
        raise Error("unable to lstat '", path, "'")
    return stat^
