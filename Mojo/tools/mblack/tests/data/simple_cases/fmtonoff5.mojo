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
#   Path:   tests/data/simple_cases/fmtonoff5.py
#
# ===----------------------------------------------------------------------=== #

# Regression test for https://github.com/psf/black/issues/3129.
setup(
    entry_points={
        # fmt: off
        "console_scripts": [
            "foo-bar"
            "=foo.bar.:main",
        # fmt: on
            ]  # Includes an formatted indentation.
    },
)


# Regression test for https://github.com/psf/black/issues/2015.
run(
    # fmt: off
    [
        "ls",
        "-la",
    ]
    # fmt: on
    + path,
    check=True,
)


# Regression test for https://github.com/psf/black/issues/3026.
def test_func():
    # yapf: disable
    if  unformatted(  args  ):
        return True
    # yapf: enable
    elif b:
        return True

    return False


# Regression test for https://github.com/psf/black/issues/2567.
if True:
    # fmt: off
    for _ in range( 1 ):
    # fmt: on
        print ( "This won't be formatted" )
    print ( "This won't be formatted either" )
else:
    print ( "This will be formatted" )


# Regression test for https://github.com/psf/black/issues/3184.
class A:
    async def call(param):
        if param:
            # fmt: off
            if param[0:4] in (
                "ABCD", "EFGH"
            )  :
                # fmt: on
                print ( "This won't be formatted" )

            elif param[0:4] in ("ZZZZ",):
                print ( "This won't be formatted either" )

        print ( "This will be formatted" )


# Regression test for https://github.com/psf/black/issues/2985
class Named(t.Protocol):
    # fmt: off
    @property
    def  this_wont_be_formatted ( self ) -> str: ...

class Factory(t.Protocol):
    def  this_will_be_formatted ( self, **kwargs ) -> Named: ...
    # fmt: on


# output


# Regression test for https://github.com/psf/black/issues/3129.
setup(
    entry_points={
        # fmt: off
        "console_scripts": [
            "foo-bar"
            "=foo.bar.:main",
        # fmt: on
            ]  # Includes an formatted indentation.
    },
)


# Regression test for https://github.com/psf/black/issues/2015.
run(
    # fmt: off
    [
        "ls",
        "-la",
    ]
    # fmt: on
    + path,
    check=True,
)


# Regression test for https://github.com/psf/black/issues/3026.
def test_func():
    # yapf: disable
    if  unformatted(  args  ):
        return True
    # yapf: enable
    elif b:
        return True

    return False


# Regression test for https://github.com/psf/black/issues/2567.
if True:
    # fmt: off
    for _ in range( 1 ):
    # fmt: on
        print ( "This won't be formatted" )
    print ( "This won't be formatted either" )
else:
    print("This will be formatted")


# Regression test for https://github.com/psf/black/issues/3184.
class A:
    async def call(param):
        if param:
            # fmt: off
            if param[0:4] in (
                "ABCD", "EFGH"
            )  :
                # fmt: on
                print ( "This won't be formatted" )

            elif param[0:4] in ("ZZZZ",):
                print ( "This won't be formatted either" )

        print("This will be formatted")


# Regression test for https://github.com/psf/black/issues/2985
class Named(t.Protocol):
    # fmt: off
    @property
    def  this_wont_be_formatted ( self ) -> str: ...


class Factory(t.Protocol):
    def this_will_be_formatted(self, **kwargs) -> Named:
        ...

    # fmt: on
