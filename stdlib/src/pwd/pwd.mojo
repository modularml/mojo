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

# ===----------------------------------------------------------------------=== #
# Passwd
# ===----------------------------------------------------------------------=== #
from ._linux import _getpw_linux
from ._macos import _getpw_macos
from sys import os_is_windows, os_is_macos, os_is_linux


@value
struct Passwd(Stringable):
    """Represents user account information retrieved from the user password
    database related to a user ID."""

    var pw_name: String
    """User name."""
    var pw_passwd: String
    """User password."""
    var pw_uid: Int
    """User ID."""
    var pw_gid: Int
    """Group ID."""
    var pw_gecos: String
    """Real name or comment field."""
    var pw_dir: String
    """Home directory."""
    var pw_shell: String
    """Shell program."""

    fn format_to(self, inout writer: Formatter):
        """Formats this string to the provided formatter.

        Args:
            writer: The formatter to write to.
        """
        writer.write("pwd.struct_passwd(pw_name='", self.pw_name)
        writer.write("', pw_passwd='", self.pw_passwd)
        writer.write(", pw_uid=", self.pw_uid)
        writer.write(", pw_gid=", self.pw_gid)
        writer.write(", pw_gecos='", self.pw_gecos)
        writer.write("', pw_dir='", self.pw_dir)
        writer.write("', pw_shell='", self.pw_shell)
        writer.write("')")

    @no_inline
    fn __str__(self) -> String:
        """Gets the Passwd struct as a string.

        Returns:
          A compact string of the Passwd struct.
        """
        return String.format_sequence(self)

    @no_inline
    fn __repr__(self) -> String:
        """Gets the Passwd struct as a string.

        Returns:
          A compact string representation of Passwd struct.
        """
        return String.format_sequence(self)


fn getpwuid(uid: Int) raises -> Passwd:
    """Retrieve the password database entry for a given user ID.

    Args:
        uid: The user ID for which to retrieve the password database entry.

    Returns:
        An object containing the user's account information, including login
        name, encrypted password, user ID, group ID, real name, home directory,
        and shell program.

    Raises:
        If the user ID does not exist or there is an error retrieving the
        information.

    Constraints:
        This function is constrained to run on Linux or macOS operating systems
        only.
    """
    constrained[
        not os_is_windows(), "operating system must be Linux or macOS"
    ]()

    @parameter
    if os_is_macos():
        return _getpw_macos(uid)
    else:
        return _getpw_linux(uid)


fn getpwnam(name: String) raises -> Passwd:
    """
    Retrieves the user ID in the password database for the given user name.

    Args:
        name: The name of the user to retrieve the password entry for.

    Returns:
        An object containing the user's account information, including login
        name, encrypted password, user ID, group ID, real name, home directory,
        and shell program.

    Raises:
        If the user name does not exist or there is an error retrieving the
        information.

    Constraints:
        This function is constrained to run on Linux or macOS operating systems
        only.
    """
    constrained[
        not os_is_windows(), "operating system must be Linux or macOS"
    ]()

    @parameter
    if os_is_macos():
        return _getpw_macos(name)
    else:
        return _getpw_linux(name)
