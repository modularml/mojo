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
#   Path:   tests/data/simple_cases/fmtskip8.py
#
# ===----------------------------------------------------------------------=== #

# Make sure a leading comment is not removed.
def some_func(  unformatted,  args  ):  # fmt: skip
    print("I am some_func")
    return 0
    # Make sure this comment is not removed.


# Make sure a leading comment is not removed.
async def some_async_func(  unformatted,   args):  # fmt: skip
    print("I am some_async_func")
    await asyncio.sleep(1)


# Make sure a leading comment is not removed.
class SomeClass(  Unformatted,  SuperClasses  ):  # fmt: skip
    def some_method(  self,  unformatted,  args  ):  # fmt: skip
        print("I am some_method")
        return 0

    async def some_async_method(  self,  unformatted,  args  ):  # fmt: skip
        print("I am some_async_method")
        await asyncio.sleep(1)


# Make sure a leading comment is not removed.
if  unformatted_call(  args  ):  # fmt: skip
    print("First branch")
    # Make sure this is not removed.
elif  another_unformatted_call(  args  ):  # fmt: skip
    print("Second branch")
else  :  # fmt: skip
    print("Last branch")


while  some_condition(  unformatted,  args  ):  # fmt: skip
    print("Do something")


for  i  in  some_iter(  unformatted,  args  ):  # fmt: skip
    print("Do something")


async def test_async_for():
    async  for  i  in  some_async_iter(  unformatted,  args  ):  # fmt: skip
        print("Do something")


try  :  # fmt: skip
    some_call()
except  UnformattedError  as  ex:  # fmt: skip
    handle_exception()
finally  :  # fmt: skip
    finally_call()


with  give_me_context(  unformatted,  args  ):  # fmt: skip
    print("Do something")


async def test_async_with():
    async  with  give_me_async_context(  unformatted,  args  ):  # fmt: skip
        print("Do something")
