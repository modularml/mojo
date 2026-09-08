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

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   src/mblib2to3/pgen2/literals.py
#
# ===----------------------------------------------------------------------=== #

# Copyright 2004-2005 Elemental Security, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Safely evaluate Python string literals without using eval()."""

import re
from re import Match

simple_escapes: dict[str, str] = {
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "'": "'",
    '"': '"',
    "\\": "\\",
}


def escape(m: Match[str]) -> str:
    all, tail = m.group(0, 1)
    assert all.startswith("\\")
    esc = simple_escapes.get(tail)
    if esc is not None:
        return esc
    if tail.startswith("x"):
        hexes = tail[1:]
        if len(hexes) < 2:
            raise ValueError(f"invalid hex string escape ('\\{tail}')")
        try:
            i = int(hexes, 16)
        except ValueError:
            raise ValueError(
                f"invalid hex string escape ('\\{tail}')"
            ) from None
    else:
        try:
            i = int(tail, 8)
        except ValueError:
            raise ValueError(
                f"invalid octal string escape ('\\{tail}')"
            ) from None
    return chr(i)


def evalString(s: str) -> str:
    assert s.startswith(("'", '"')), repr(s[:1])
    q = s[0]
    if s[:3] == q * 3:
        q = q * 3
    assert s.endswith(q), repr(s[-len(q) :])
    assert len(s) >= 2 * len(q)
    s = s[len(q) : -len(q)]
    return re.sub(r"\\(\'|\"|\\|[abfnrtv]|x.{0,2}|[0-7]{1,3})", escape, s)


def test() -> None:
    for i in range(256):
        c = chr(i)
        s = repr(c)
        e = evalString(s)
        if e != c:
            print(i, c, s, e)


if __name__ == "__main__":
    test()
