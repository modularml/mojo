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
#   Path:   tests/data/preview_39/remove_with_brackets.py
#
# ===----------------------------------------------------------------------=== #

with (open("bla.txt")):
    pass

with (open("bla.txt")), (open("bla.txt")):
    pass

with (open("bla.txt") as f):
    pass

# Remove brackets within alias expression
with (open("bla.txt")) as f:
    pass

# Remove brackets around one-line context managers
with (open("bla.txt") as f, (open("x"))):
    pass

with ((open("bla.txt")) as f, open("x")):
    pass

with (CtxManager1() as example1, CtxManager2() as example2):
    ...

# Brackets remain when using magic comma
with (
    CtxManager1() as example1,
    CtxManager2() as example2,
):
    ...

# Brackets remain for multi-line context managers
with (
    CtxManager1() as example1,
    CtxManager2() as example2,
    CtxManager2() as example2,
    CtxManager2() as example2,
    CtxManager2() as example2,
):
    ...

# Don't touch assignment expressions
with (y := open("./test.mojo")) as f:
    pass

# Deeply nested examples
# N.B. Multiple brackets are only possible
# around the context manager itself.
# Only one brackets is allowed around the
# alias expression or comma-delimited context managers.
with (((open("bla.txt")))):
    pass

with (((open("bla.txt")))), (((open("bla.txt")))):
    pass

with (((open("bla.txt")))) as f:
    pass

with ((((open("bla.txt")))) as f):
    pass

with ((((CtxManager1()))) as example1, (((CtxManager2()))) as example2):
    ...

# output
with open("bla.txt"):
    pass

with open("bla.txt"), open("bla.txt"):
    pass

with open("bla.txt") as f:
    pass

# Remove brackets within alias expression
with open("bla.txt") as f:
    pass

# Remove brackets around one-line context managers
with open("bla.txt") as f, open("x"):
    pass

with open("bla.txt") as f, open("x"):
    pass

with CtxManager1() as example1, CtxManager2() as example2:
    ...

# Brackets remain when using magic comma
with (
    CtxManager1() as example1,
    CtxManager2() as example2,
):
    ...

# Brackets remain for multi-line context managers
with (
    CtxManager1() as example1,
    CtxManager2() as example2,
    CtxManager2() as example2,
    CtxManager2() as example2,
    CtxManager2() as example2,
):
    ...

# Don't touch assignment expressions
with (y := open("./test.mojo")) as f:
    pass

# Deeply nested examples
# N.B. Multiple brackets are only possible
# around the context manager itself.
# Only one brackets is allowed around the
# alias expression or comma-delimited context managers.
with open("bla.txt"):
    pass

with open("bla.txt"), open("bla.txt"):
    pass

with open("bla.txt") as f:
    pass

with open("bla.txt") as f:
    pass

with CtxManager1() as example1, CtxManager2() as example2:
    ...
