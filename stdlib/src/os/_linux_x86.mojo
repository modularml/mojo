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

from utils.index import StaticIntTuple

from .fstat import stat_result

alias dev_t = Int64
alias mode_t = Int32
alias nlink_t = Int64

alias uid_t = Int32
alias gid_t = Int32
alias off_t = Int64
alias blkcnt_t = Int64
alias blksize_t = Int64


@value
@register_passable("trivial")
struct _c_stat(Stringable):
    var st_dev: dev_t  #  ID of device containing file
    var st_ino: Int64  # File serial number
    var st_nlink: nlink_t  # Number of hard links
    var st_mode: mode_t  # Mode of file
    var st_uid: uid_t  # User ID of the file
    var st_gid: gid_t  # Group ID of the file
    var __pad0: Int32  # Padding
    var st_rdev: dev_t  # Device ID
    var st_size: off_t  # file size, in bytes
    var st_blksize: blksize_t  # optimal blocksize for I/O
    var st_blocks: blkcnt_t  #  blocks allocated for file
    var st_atimespec: _CTimeSpec  # time of last access
    var st_mtimespec: _CTimeSpec  # time of last data modification
    var st_ctimespec: _CTimeSpec  # time of last status change
    var st_birthtimespec: _CTimeSpec  # time of file creation(birth)
    var unused: StaticTuple[Int64, 3]  # RESERVED: DO NOT USE!

    fn __init__() -> Self:
        return Self {
            st_dev: 0,
            st_mode: 0,
            st_nlink: 0,
            st_ino: 0,
            st_uid: 0,
            st_gid: 0,
            __pad0: 0,
            st_rdev: 0,
            st_size: 0,
            st_blksize: 0,
            st_blocks: 0,
            st_atimespec: _CTimeSpec(),
            st_mtimespec: _CTimeSpec(),
            st_ctimespec: _CTimeSpec(),
            st_birthtimespec: _CTimeSpec(),
            unused: StaticTuple[Int64, 3](0, 0, 0),
        }

    fn __str__(self) -> String:
        return "{\n" +
            "st_dev: {self.st_dev},\n" +
            "st_mode: {self.st_mode},\n" +
            "st_nlink: {self.st_nlink},\n" +
            "st_ino: {self.st_ino},\n" +
            "st_uid: {self.st_uid},\n" +
            "st_gid: {self.st_gid},\n" +
            "st_rdev: {self.st_rdev},\n" +
            "st_size: {self.st_size},\n" +
            "st_blksize: {self.st_blksize},\n" +
            "st_blocks: {self.st_blocks},\n" +
            "st_atimespec: {self.st_atimespec},\n" +
            "st_mtimespec: {self.st_mtimespec},\n" +
            "st_ctimespec: {self.st_ctimespec},\n" +
            "st_birthtimespec: {self.st_birthtimespec}\n" +
            "}"

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
        Int32(0), path._as_ptr(), Pointer.address_of(stat)
    )
    if err == -1:
        raise "unable to stat '" + path + "'"
    return stat


@always_inline
fn _lstat(path: String) raises -> _c_stat:
    var stat = _c_stat()
    var err = external_call["__lxstat", Int32](
        Int32(0), path._as_ptr(), Pointer.address_of(stat)
    )
    if err == -1:
        raise "unable to lstat '" + path + "'"
    return stat
