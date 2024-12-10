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
from sys.ffi import c_char, external_call

from memory import UnsafePointer

from .pwd import Passwd

alias uid_t = Int32
alias gid_t = Int32
alias char = UnsafePointer[c_char]


@register_passable("trivial")
struct _C_Passwd:
    var pw_name: char
    var pw_passwd: char
    var pw_uid: uid_t
    var pw_gid: gid_t
    var pw_gecos: char
    var pw_dir: char
    var pw_shell: char


fn _build_pw_struct(passwd_ptr: UnsafePointer[_C_Passwd]) raises -> Passwd:
    var c_pwuid = passwd_ptr[]
    return Passwd(
        pw_name=String(c_pwuid.pw_name),
        pw_passwd=String(c_pwuid.pw_passwd),
        pw_uid=int(c_pwuid.pw_uid),
        pw_gid=int(c_pwuid.pw_gid),
        pw_gecos=String(c_pwuid.pw_gecos),
        pw_dir=String(c_pwuid.pw_dir),
        pw_shell=String(c_pwuid.pw_shell),
    )


fn _getpw_linux(uid: UInt32) raises -> Passwd:
    var passwd_ptr = external_call["getpwuid", UnsafePointer[_C_Passwd]](uid)
    if not passwd_ptr:
        raise "user ID not found in the password database: " + str(uid)
    return _build_pw_struct(passwd_ptr)


fn _getpw_linux(name: String) raises -> Passwd:
    var passwd_ptr = external_call["getpwnam", UnsafePointer[_C_Passwd]](name)
    if not passwd_ptr:
        raise "user name not found in the password database: " + name
    return _build_pw_struct(passwd_ptr)
