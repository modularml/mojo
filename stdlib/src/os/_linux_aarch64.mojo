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

from time.time import _CTimeSpec

from utils import InlineArray

from .fstat import stat_result

alias dev_t = Int64
alias mode_t = Int32
alias nlink_t = Int32

alias uid_t = Int32
alias gid_t = Int32
alias off_t = Int64
alias blkcnt_t = Int64
alias blksize_t = Int32


@value
struct _c_stat(Stringable):
    var st_dev: dev_t  #  ID of device containing file
    var st_ino: Int64  # File serial number
    var st_mode: mode_t  # Mode of file
    var st_nlink: nlink_t  # Number of hard links
    var st_uid: uid_t  # User ID of the file
    var st_gid: gid_t  # Group ID of the file
    var st_rdev: dev_t  # Device ID
    var __pad0: dev_t  # Padding
    var st_size: off_t  # file size, in bytes
    var st_blksize: blksize_t  # optimal blocksize for I/O
    var __pad1: Int32  # Padding
    var st_blocks: blkcnt_t  #  blocks allocated for file
    var st_atimespec: _CTimeSpec  # time of last access
    var st_mtimespec: _CTimeSpec  # time of last data modification
    var st_ctimespec: _CTimeSpec  # time of last status change
    var st_birthtimespec: _CTimeSpec  # time of file creation(birth)
    var unused: InlineArray[Int64, 2]  # RESERVED: DO NOT USE!

    fn __init__(inout self):
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
        self.__pad1 = 0
        self.st_blocks = 0
        self.st_atimespec = _CTimeSpec()
        self.st_mtimespec = _CTimeSpec()
        self.st_ctimespec = _CTimeSpec()
        self.st_birthtimespec = _CTimeSpec()
        self.unused = InlineArray[Int64, 2](0, 0)

    fn __str__(self) -> String:
        var res = String("{\n")
        res += "st_dev: " + str(self.st_dev) + ",\n"
        res += "st_mode: " + str(self.st_mode) + ",\n"
        res += "st_nlink: " + str(self.st_nlink) + ",\n"
        res += "st_ino: " + str(self.st_ino) + ",\n"
        res += "st_uid: " + str(self.st_uid) + ",\n"
        res += "st_gid: " + str(self.st_gid) + ",\n"
        res += "st_rdev: " + str(self.st_rdev) + ",\n"
        res += "st_size: " + str(self.st_size) + ",\n"
        res += "st_blksize: " + str(self.st_blksize) + ",\n"
        res += "st_blocks: " + str(self.st_blocks) + ",\n"
        res += "st_atimespec: " + str(self.st_atimespec) + ",\n"
        res += "st_mtimespec: " + str(self.st_mtimespec) + ",\n"
        res += "st_ctimespec: " + str(self.st_ctimespec) + ",\n"
        res += "st_birthtimespec: " + str(self.st_birthtimespec) + "\n"
        return res + "}"

    fn _to_stat_result(self) -> stat_result:
        return stat_result(
            st_dev=int(self.st_dev),
            st_mode=int(self.st_mode),
            st_nlink=int(self.st_nlink),
            st_ino=int(self.st_ino),
            st_uid=int(self.st_uid),
            st_gid=int(self.st_gid),
            st_rdev=int(self.st_rdev),
            st_atimespec=self.st_atimespec,
            st_ctimespec=self.st_ctimespec,
            st_mtimespec=self.st_mtimespec,
            st_birthtimespec=self.st_birthtimespec,
            st_size=int(self.st_size),
            st_blocks=int(self.st_blocks),
            st_blksize=int(self.st_blksize),
            st_flags=0,
        )


@always_inline
fn _stat(path: String) raises -> _c_stat:
    var stat = _c_stat()
    var err = external_call["__xstat", Int32](
        Int32(0), path.unsafe_ptr(), UnsafePointer.address_of(stat)
    )
    if err == -1:
        raise "unable to stat '" + path + "'"
    return stat


@always_inline
fn _lstat(path: String) raises -> _c_stat:
    var stat = _c_stat()
    var err = external_call["__lxstat", Int32](
        Int32(0), path.unsafe_ptr(), UnsafePointer.address_of(stat)
    )
    if err == -1:
        raise "unable to lstat '" + path + "'"
    return stat
