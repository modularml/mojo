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
"""Provides a set of operating-system independent functions for manipulating
file system paths.

You can import these APIs from the `os.path` package. For example:

```mojo
from std.os.path import isdir
```
"""

from std.collections.string.string_span import _unsafe_strlen
from std.pwd import getpwuid, getpwnam
from std.stat import S_ISDIR, S_ISLNK, S_ISREG
from std.ffi import MAX_PATH, c_char, external_call, get_errno
from std.sys import CompilationTarget
from std.sys._libc import realpath as libc_realpath

from .. import PathLike as stdPathLike, getuid
from .._linux_aarch64 import _lstat as _lstat_linux_arm
from .._linux_aarch64 import _stat as _stat_linux_arm
from .._linux_x86 import _lstat as _lstat_linux_x86
from .._linux_x86 import _stat as _stat_linux_x86
from .._macos import _lstat as _lstat_macos
from .._macos import _stat as _stat_macos
from ..env import getenv
from ..fstat import stat
from ..os import sep


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
def _get_stat_st_mode(var path: String) raises -> Int:
    comptime if CompilationTarget.is_macos():
        return Int(_stat_macos(path^).st_mode)
    elif CompilationTarget.has_neon():
        return Int(_stat_linux_arm(path^).st_mode)
    else:
        return Int(_stat_linux_x86(path^).st_mode)


@always_inline
def _get_lstat_st_mode(var path: String) raises -> Int:
    comptime if CompilationTarget.is_macos():
        return Int(_lstat_macos(path^).st_mode)
    elif CompilationTarget.has_neon():
        return Int(_lstat_linux_arm(path^).st_mode)
    else:
        return Int(_lstat_linux_x86(path^).st_mode)


# ===----------------------------------------------------------------------=== #
# expanduser
# ===----------------------------------------------------------------------=== #


def _user_home_path(path: String) -> String:
    var user_end = path.find(sep, 1)
    if user_end < 0:
        user_end = path.byte_length()
    # Special POSIX syntax for ~[user-name]/path
    if path.byte_length() > 1 and user_end > 1:
        try:
            return getpwnam(String(path[byte=1:user_end])).pw_dir
        except:
            return ""
    else:
        var user_home = getenv("HOME")
        # Fallback to password database if `HOME` not set
        if not user_home:
            try:
                user_home = getpwuid(getuid()).pw_dir
            except:
                return ""
        return user_home


def expanduser[PathLike: stdPathLike, //](path: PathLike) raises -> String:
    """Expands a tilde "~" prefix in `path` to the user's home directory.

    For example, `~/folder` becomes `/home/current_user/folder`. On macOS and
    Linux a path starting with `~user/` will expand to the specified user's home
    directory, so `~user/folder` becomes `/home/user/folder`.

    If the home directory cannot be determined, or the `path` is not prefixed
    with "~", the original path is returned unchanged.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path that is being expanded.

    Returns:
        The expanded path.

    Raises:
        If the operation fails.
    """
    var fspath = path.__fspath__()
    if not fspath.startswith("~"):
        return fspath
    var userhome = _user_home_path(fspath)
    if not userhome:
        return fspath
    var path_split = fspath.split(sep, 1)
    # If there is a properly formatted separator, return expanded fspath.
    if len(path_split) == 2:
        return join(userhome, String(path_split[1]))
    # Path was a single `~` character, return home path
    return userhome


# ===----------------------------------------------------------------------=== #
# isdir
# ===----------------------------------------------------------------------=== #
def isdir[PathLike: stdPathLike, //](path: PathLike) -> Bool:
    """Return True if path is an existing directory. This follows
    symbolic links, so both islink() and isdir() can be true for the same path.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        True if the path is a directory or a link to a directory and
        False otherwise.

    Example:
    ```mojo
    from std.os.path import isdir

    print(isdir("/tmp")) # True
    print(isdir("noexist")) # False
    ```
    """
    var fspath = path.__fspath__()
    try:
        var st_mode = _get_stat_st_mode(fspath)
        if S_ISDIR(st_mode):
            return True
        return S_ISLNK(st_mode) and S_ISDIR(_get_lstat_st_mode(fspath^))
    except:
        return False


# ===----------------------------------------------------------------------=== #
# isfile
# ===----------------------------------------------------------------------=== #


def isfile[PathLike: stdPathLike, //](path: PathLike) -> Bool:
    """Test whether a path is a regular file.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        Returns True if the path is a regular file.

    Example:
    ```mojo
    from std.os.path import isfile

    print(isfile("/etc/hosts")) # True
    print(isfile("/tmp")) # False (it's a directory)
    ```
    """
    var fspath = path.__fspath__()
    try:
        var st_mode = _get_stat_st_mode(fspath)
        if S_ISREG(st_mode):
            return True
        return S_ISLNK(st_mode) and S_ISREG(_get_lstat_st_mode(fspath))
    except:
        return False


# ===----------------------------------------------------------------------=== #
# islink
# ===----------------------------------------------------------------------=== #
def islink[PathLike: stdPathLike, //](path: PathLike) -> Bool:
    """Return True if path refers to an existing directory entry that is a
    symbolic link.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        True if the path is a link to a directory and False otherwise.

    Example:
    ```mojo
    from std.os.path import islink

    print(islink("/etc/hosts"))  # False (regular file, not a symlink)
    ```
    """
    try:
        return S_ISLNK(_get_lstat_st_mode(path.__fspath__()))
    except:
        return False


# ===----------------------------------------------------------------------=== #
# dirname
# ===----------------------------------------------------------------------=== #


def dirname[PathLike: stdPathLike, //](path: PathLike) -> String:
    """Returns the directory component of a pathname.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to a file.

    Returns:
        The directory component of a pathname.

    Example:
    ```mojo
    from std.os.path import dirname

    print(dirname("/a/b/c.txt")) # "/a/b"
    print(dirname("c.txt")) # ""
    ```
    """
    var fspath = path.__fspath__()
    var i = fspath.rfind(sep) + 1
    var head = String(fspath[byte=:i])
    if head and head != sep * head.byte_length():
        return String(head.rstrip(sep))
    return head


# ===----------------------------------------------------------------------=== #
# realpath
# ===----------------------------------------------------------------------=== #


def realpath[
    PathLike: stdPathLike & Deinitable, //
](path: PathLike) raises -> String:
    """Expands all symbolic links and resolves references to /./, /../ and extra
    '/' characters in the null-terminated string named by path to produce a
    canonicalized absolute pathname.The resulting path will have no symbolic
    link, /./ or /../ components.

    Args:
        path: The path to resolve.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Raises:
       - Read or search permission was denied for a component of the path
       prefix.

       - path is NULL.

       - An I/O error occurred while reading from the filesystem.

       - Too many symbolic links were encountered in translating the pathname.

       - A component of a pathname exceeded NAME_MAX characters, or an entire
       pathname exceeded PATH_MAX characters.

       - The named file does not exist.

       - Out of memory.

       - A component of the path prefix is not a directory.

    Returns:
        A String of the resolved path.
    """
    var string = String(capacity_bytes=MAX_PATH)

    # Bind the fspath result to a variable so its buffer stays alive
    # through the libc_realpath call (avoids use-after-free).
    var fspath = path.__fspath__()
    var returned_path_ptr = libc_realpath(
        fspath.as_c_string_slice(),
        string.unsafe_as_bytes_mut().unsafe_ptr().unsafe_bitcast[c_char](),
    )
    if not returned_path_ptr:
        raise Error("realpath failed to resolve: ", get_errno())

    # We wrote the data directly into the String buffer
    # now we need to figure out the length
    string._set_byte_length(Int(_unsafe_strlen(string.as_bytes().unsafe_ptr())))
    string._set_nul_terminated()

    return string^


# ===----------------------------------------------------------------------=== #
# exists
# ===----------------------------------------------------------------------=== #


def exists[PathLike: stdPathLike, //](path: PathLike) -> Bool:
    """Return True if path exists.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        Returns True if the path exists and is not a broken symbolic link.

    Example:
    ```mojo
    from std.os.path import exists

    print(exists("/tmp")) # True
    print(exists("noexist")) # False
    ```
    """
    try:
        _ = _get_stat_st_mode(path.__fspath__())
        return True
    except:
        return False


# ===----------------------------------------------------------------------=== #
# lexists
# ===----------------------------------------------------------------------=== #


def lexists[PathLike: stdPathLike, //](path: PathLike) -> Bool:
    """Return True if path exists or is a broken symlink.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        Returns True if the path exists or is a broken symbolic link.
    """
    try:
        _ = _get_lstat_st_mode(path.__fspath__())
        return True
    except:
        return False


# ===----------------------------------------------------------------------=== #
# getsize
# ===----------------------------------------------------------------------=== #


def getsize[PathLike: stdPathLike, //](path: PathLike) raises -> Int:
    """Return the size, in bytes, of the specified path.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the file.

    Returns:
        The size of the path in bytes.

    Raises:
        If the operation fails.
    """
    return stat(path.__fspath__()).st_size


# ===----------------------------------------------------------------------=== #
# is_absolute
# ===----------------------------------------------------------------------=== #


def is_absolute[PathLike: stdPathLike, //](path: PathLike) -> Bool:
    """Return True if `path` is an absolute path name.
    On Unix, that means it begins with a slash.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to check.

    Returns:
        Return `True` if path is an absolute path name.

    Example:
    ```mojo
    from std.os.path import is_absolute

    print(is_absolute("/usr/bin")) # True
    print(is_absolute("relative")) # False
    ```
    """
    return path.__fspath__().startswith(sep)


# ===----------------------------------------------------------------------=== #
# join
# ===----------------------------------------------------------------------=== #


# TODO(MOCO-1532):
#   Use StringSlice here once param inference bug for empty variadic
#   list of parameterized types is fixed.
def join(var path: String, *paths: String) -> String:
    """Join two or more pathname components, inserting '/' as needed.
    If any component is an absolute path, all previous path components
    will be discarded.  An empty last part will result in a path that
    ends with a separator.

    Args:
        path: The path to join.
        paths: The paths to join.

    Returns:
        The joined path.

    Example:
    ```mojo
    from std.os.path import join

    print(join("a", "b", "c")) # "a/b/c"
    print(join("a", "/b", "c")) # "/b/c" (absolute resets)
    ```
    """
    var joined_path = path^

    for cur_path in paths:
        if cur_path.startswith(sep):
            joined_path = cur_path
        elif not joined_path or joined_path.endswith(sep):
            joined_path += cur_path
        else:
            joined_path += sep + cur_path

    return joined_path^


# ===----------------------------------------------------------------------=== #
# split
# ===----------------------------------------------------------------------=== #


def split[PathLike: stdPathLike, //](path: PathLike) -> Tuple[String, String]:
    """
    Split a given pathname into two components: head and tail. This is useful
    for separating the directory path from the filename. If the input path ends
    with a separator, the tail component will be empty. If there is no separator
    in the path, the head component will be empty, and the entire path will be
    considered the tail. Trailing separators in the head are stripped unless the
    head is the root directory.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to be split.

    Returns:
        A tuple containing two strings: (head, tail).

    Example:
    ```mojo
    from std.os.path import split

    print(split("/a/b/c.txt")) # ("/a/b", "c.txt")
    print(split("/a/b/")) # ("/a/b", "")
    ```
    """
    var fspath = path.__fspath__()
    var i = fspath.rfind(sep) + 1
    var head, tail = fspath[byte=:i], fspath[byte=i:]
    if head and head != String(sep) * head.byte_length():
        head = head.rstrip(sep)
    return String(head), String(tail)


def basename[PathLike: stdPathLike, //](path: PathLike) -> String:
    """Returns the tail section of a path.

    ```mojo
    from std.os.path import basename

    print(basename("a/path/foo.txt")) # "foo.txt"
    ```

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to retrieve the basename from.

    Returns:
        The basename from the path.
    """
    var fspath = path.__fspath__()
    var i = fspath.rfind(sep) + 1
    var head = fspath[byte=i:]
    if head and head != sep * head.byte_length():
        return String(head.rstrip(sep))
    return String(head)


# TODO uncomment this when unpacking is supported
# def join[PathLike: stdPathLike](path: PathLike, *paths: PathLike) -> String:
#     """Join two or more pathname components, inserting '/' as needed.
#     If any component is an absolute path, all previous path components
#     will be discarded.  An empty last part will result in a path that
#     ends with a separator.

#     Parameters:
#       PathLike: The type conforming to the os.PathLike trait.

#     Args:
#       path: The path to join.
#       paths: The paths to join.

#     Returns:
#       The joined path.
#     """
#     var paths_str= List[String]()

#     for cur_path in paths:
#         paths_str.append(cur_path[].__fspath__())

#     return join(path.__fspath__(), *paths_str)

# ===----------------------------------------------------------------------=== #
# split_extension
# ===----------------------------------------------------------------------=== #


# TODO: Move this to a generic path module when Windows is supported.
# As it can be used for both Windows and Unix-like systems.
def _split_extension(
    path: StringSlice,
    sep: StringSlice,
    alt_sep: StringSlice,
    extension_sep: StringSlice,
) raises -> Tuple[String, String]:
    """Splits `path` into the root and extension.

    Args:
        path: The path to be split.
        sep: The separator used in the path.
        alt_sep: The alternative separator used in the path.
        extension_sep: The extension separator used in the path.

    Returns:
        A tuple containing two strings: (root, extension).
    """
    # Find the last extension separator after the last separator.
    var head_end = path.rfind(sep)
    if alt_sep:
        head_end = max(head_end, path.rfind(alt_sep))

    var file_end = path.rfind(extension_sep)
    if file_end > head_end:
        # skip all leading dots
        var file_start = head_end + 1
        while file_start < file_end:
            if StringSlice(path[byte=file_start]) != extension_sep:
                return String(path[byte=:file_end]), String(
                    path[byte=file_end:]
                )
            file_start += 1

    return String(path), ""


def split_extension[
    PathLike: stdPathLike, //
](path: PathLike) raises -> Tuple[String, String]:
    """Splits `path` into the root and extension.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to be split.

    Returns:
        A tuple containing two strings: (root, extension).

    Raises:
        If the operation fails.

    Example:
    ```mojo
    from std.os.path import split_extension

    print(split_extension("foo.tar.gz")) # ("foo.tar", ".gz")
    print(split_extension("README")) # ("README", "")
    ```
    """
    return _split_extension(path.__fspath__(), sep, "", ".")


# ===----------------------------------------------------------------------=== #
# splitroot
# ===----------------------------------------------------------------------=== #


def splitroot[
    PathLike: stdPathLike, //
](path: PathLike) -> Tuple[String, String, String]:
    """Splits `path` into drive, root and tail. The tail contains anything after the root.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to be split.

    Returns:
        A tuple containing three strings: (drive, root, tail).
    """
    var p = path.__fspath__()
    comptime empty = ""
    var length = p.byte_length()

    # Relative path, e.g.: 'foo'
    if length < 1 or p[byte=:1] != StringSlice(sep):
        return empty, empty, p

    # Absolute path, e.g.: '/foo', '///foo', '////foo', etc.
    elif (
        length < 2
        or p[byte=1:2] != StringSlice(sep)
        or (length >= 3 and p[byte=2:3] == StringSlice(sep))
    ):
        return empty, String(sep), String(p[byte=1:])

    # Precisely two leading slashes, e.g.: '//foo'. Implementation defined per POSIX, see
    # https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_13
    else:
        return empty, String(p[byte=:2]), String(p[byte=2:])


# ===----------------------------------------------------------------------=== #
# expandvars
# ===----------------------------------------------------------------------=== #


def _is_shell_special_variable(byte: Byte) -> Bool:
    """Checks if `$` + `byte` identifies a special shell variable, such as `$@`.

    Args:
        byte: The byte to check.

    Returns:
        True if the byte is a special shell variable and False otherwise.
    """
    comptime shell_variables: Array[Int, 17] = [
        ord("*"),
        ord("#"),
        ord("$"),
        ord("@"),
        ord("!"),
        ord("?"),
        ord("-"),
        ord("0"),
        ord("1"),
        ord("2"),
        ord("3"),
        ord("4"),
        ord("5"),
        ord("6"),
        ord("7"),
        ord("8"),
        ord("9"),
    ]
    return Int(byte) in materialize[shell_variables]()


def _is_alphanumeric(byte: Byte) -> Bool:
    """Checks if `byte` is an ASCII letter, number, or underscore.

    Args:
        byte: The byte to check.

    Returns:
        True if the byte is an ASCII letter, number, or underscore and False otherwise.
    """
    var b = Int(byte)
    return (
        b == ord("_")
        or ord("0") <= b
        and b <= ord("9")
        or ord("a") <= b
        and b <= ord("z")
        or ord("A") <= b
        and b <= ord("Z")
    )


def _parse_variable_name[
    immutable: ImmOrigin
](bytes: Span[Byte, immutable]) -> Tuple[StringSlice[immutable], Int]:
    """Returns the environment variable name and the byte count required to extract it.
    For `${}` expansions, two additional bytes are added to the byte count to
    account for the braces, unless the closing brace is missing, in which
    case only the opening brace is accounted for.

    Args:
        bytes: The bytes to extract the environment variable name from.

    Returns:
        The environment variable name and the byte count required to extract it.
    """
    if bytes[0] == UInt8(ord("{")):
        if (
            len(bytes) > 2
            and _is_shell_special_variable(bytes[1])
            and bytes[2] == UInt8(ord("}"))
        ):
            return StringSlice(unsafe_from_utf8=bytes[1:2]), 3

        # Scan until the closing brace or the end of the bytes.
        var i = 1
        while i < len(bytes):
            if bytes[i] == UInt8(ord("}")):
                return StringSlice(unsafe_from_utf8=bytes[1:i]), i + 1
            i += 1
        # No closing brace found: `i == len(bytes)`, so the whole remainder
        # was consumed with no closing-brace byte to account for.
        return StringSlice(unsafe_from_utf8=bytes[1:i]), i
    elif _is_shell_special_variable(bytes[0]):
        return StringSlice(unsafe_from_utf8=bytes[0:1]), 1

    # Scan until we hit an invalid character in environment variable names.
    var i = 0
    while i < len(bytes) and _is_alphanumeric(bytes[i]):
        i += 1

    return StringSlice(unsafe_from_utf8=bytes[:i]), i


def expandvars[PathLike: stdPathLike, //](path: PathLike) -> String:
    """Replaces `${var}` or `$var` in the path with values from the current environment variables.
    Malformed variable names and references to non-existing variables are left unchanged.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path that is being expanded.

    Returns:
        The expanded path.
    """
    var path_str = path.__fspath__()
    var bytes = path_str.as_bytes()
    var buf = String()

    # Byte scanning should be fine, ${} is ASCII.
    var i = 0
    var j = 0
    while j < len(bytes):
        if bytes[j] == UInt8(ord("$")) and j + 1 < len(bytes):
            if not buf:
                buf.reserve_bytes(2 * len(bytes))
            buf.write_string(path_str[byte=i:j])

            var name, length = _parse_variable_name(bytes[j + 1 :])

            # Invalid syntax (`${}` or `${`) or $ was not followed by a name; write as is.
            if name.startswith("{") or name == "":
                buf.write_string(path_str[byte = j : j + length + 1])
            # Shell variable (eg `$@` or `$*`); write as is.
            elif _is_shell_special_variable(name.as_bytes()[0]):
                buf.write_string(path_str[byte = j : j + 2])
            # Environment variable; expand it. If no value, write as is.
            else:
                var value = getenv(String(name))
                if value != "":
                    buf.write(value)
                else:
                    buf.write_string(path_str[byte = j : j + length + 1])

            j += length
            i = j + 1
        j += 1

    if not buf:
        return path_str

    buf.write_string(path_str[byte=i:])
    return buf
