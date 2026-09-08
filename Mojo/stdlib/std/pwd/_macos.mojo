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

from std.ffi import c_char, external_call

from .pwd import Passwd

comptime uid_t = Int32
comptime gid_t = Int32
comptime time_t = Int
comptime char = Pointer[c_char, MutUntrackedOrigin]


struct _C_Passwd(TrivialRegisterPassable):
    var pw_name: char
    var pw_passwd: char
    var pw_uid: uid_t
    var pw_gid: gid_t
    var pw_change: time_t  # Always 0
    var pw_class: char  # Always empty
    var pw_gecos: char
    var pw_dir: char
    var pw_shell: char
    var pw_expire: time_t  # Always 0


def _build_pw_struct(passwd_ptr: ImmPointer[_C_Passwd, _]) -> Passwd:
    var c_pwuid = passwd_ptr[]
    var passwd = Passwd(
        pw_name=String(unsafe_from_utf8_ptr=c_pwuid.pw_name),
        pw_passwd=String(unsafe_from_utf8_ptr=c_pwuid.pw_passwd),
        pw_uid=Int(c_pwuid.pw_uid),
        pw_gid=Int(c_pwuid.pw_gid),
        pw_gecos=String(unsafe_from_utf8_ptr=c_pwuid.pw_gecos),
        pw_dir=String(unsafe_from_utf8_ptr=c_pwuid.pw_dir),
        pw_shell=String(unsafe_from_utf8_ptr=c_pwuid.pw_shell),
    )
    return passwd^


def _getpw_macos(uid: UInt32) raises -> Passwd:
    var passwd_ptr = external_call[
        "getpwuid", OptionalPointer[_C_Passwd, UntrackedOrigin[mut=True]]
    ](uid)
    try:
        return _build_pw_struct(passwd_ptr[])
    except:
        raise Error("user ID not found in the password database: ", uid)


def _getpw_macos(var name: String) raises -> Passwd:
    var passwd_ptr = external_call[
        "getpwnam", OptionalPointer[_C_Passwd, UntrackedOrigin[mut=True]]
    ](name.as_c_string_slice())
    try:
        return _build_pw_struct(passwd_ptr[])
    except:
        raise Error("user name not found in the password database: ", name)
