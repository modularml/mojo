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

comptime dev_t = Int64
comptime mode_t = Int32
comptime nlink_t = Int64

comptime uid_t = Int32
comptime gid_t = Int32
comptime off_t = Int64
comptime blkcnt_t = Int64
comptime blksize_t = Int64


struct _c_stat(Copyable, Defaultable, Writable):
    var st_dev: dev_t
    """ID of device containing file."""
    var st_ino: Int64
    """File serial number."""
    var st_nlink: nlink_t
    """Number of hard links."""
    var st_mode: mode_t
    """Mode of file."""
    var st_uid: uid_t
    """User ID of the file."""
    var st_gid: gid_t
    """Group ID of the file."""
    var __pad0: Int32
    """Padding."""
    var st_rdev: dev_t
    """Device ID."""
    var st_size: off_t
    """File size, in bytes."""
    var st_blksize: blksize_t
    """Optimal blocksize for I/O."""
    var st_blocks: blkcnt_t
    """Blocks allocated for file."""
    var st_atimespec: _CTimeSpec
    """Time of last access."""
    var st_mtimespec: _CTimeSpec
    """Time of last data modification."""
    var st_ctimespec: _CTimeSpec
    """Time of last status change."""
    var st_birthtimespec: _CTimeSpec
    """Time of file creation(birth)."""
    var unused: Array[Int64, 3]
    """RESERVED: DO NOT USE!."""

    def __init__(out self):
        self.st_dev = 0
        self.st_mode = 0
        self.st_nlink = 0
        self.st_ino = 0
        self.st_uid = 0
        self.st_gid = 0
        self.__pad0 = 0
        self.st_rdev = 0
        self.st_size = 0
        self.st_blksize = 0
        self.st_blocks = 0
        self.st_atimespec = _CTimeSpec()
        self.st_mtimespec = _CTimeSpec()
        self.st_ctimespec = _CTimeSpec()
        self.st_birthtimespec = _CTimeSpec()
        self.unused: Array[Int64, 3] = [0, 0, 0]

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
            ",\nst_size: ", self.st_size,
            ",\nst_blksize: ", self.st_blksize,
            ",\nst_blocks: ", self.st_blocks,
            ",\nst_atimespec: ", self.st_atimespec,
            ",\nst_mtimespec: ", self.st_mtimespec,
            ",\nst_ctimespec: ", self.st_ctimespec,
            ",\nst_birthtimespec: ", self.st_birthtimespec,
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
            st_flags=0,
        )


@always_inline
def _stat(var path: String) raises -> _c_stat:
    var stat = _c_stat()
    var err = external_call["__xstat", Int32](
        Int32(0), path.as_c_string_slice(), Pointer(to=stat)
    )
    if err == -1:
        raise Error("unable to stat '", path, "'")
    return stat^


@always_inline
def _lstat(var path: String) raises -> _c_stat:
    var stat = _c_stat()
    var err = external_call["__lxstat", Int32](
        Int32(0), path.as_c_string_slice(), Pointer(to=stat)
    )
    if err == -1:
        raise Error("unable to lstat '", path, "'")
    return stat^
