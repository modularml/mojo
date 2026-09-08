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
#   Path:   tests/data/py_37/python37.py
#
# ===----------------------------------------------------------------------=== #

#!/usr/bin/env python3.7


def f():
    return (i * 2 async for i in arange(42))


def g():
    return (
        something_long * something_long
        async for something_long in async_generator(with_an_argument)
    )


async def func():
    if test:
        out_batched = [
            i
            async for i in aitertools._async_map(
                self.async_inc, arange(8), batch_size=3
            )
        ]


def awaited_generator_value(n):
    return (await awaitable for awaitable in awaitable_list)


def make_arange(n):
    return (i * 2 for i in range(n) if await wrap(i))


# output


#!/usr/bin/env python3.7


def f():
    return (i * 2 async for i in arange(42))


def g():
    return (
        something_long * something_long
        async for something_long in async_generator(with_an_argument)
    )


async def func():
    if test:
        out_batched = [
            i
            async for i in aitertools._async_map(
                self.async_inc, arange(8), batch_size=3
            )
        ]


def awaited_generator_value(n):
    return (await awaitable for awaitable in awaitable_list)


def make_arange(n):
    return (i * 2 for i in range(n) if await wrap(i))
