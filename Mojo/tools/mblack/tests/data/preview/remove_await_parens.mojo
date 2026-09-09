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
#   Path:   tests/data/preview/remove_await_parens.py
#
# ===----------------------------------------------------------------------=== #

import asyncio

# Control example
async def main():
    await asyncio.sleep(1)


# Remove brackets for short coroutine/task
async def main():
    await (asyncio.sleep(1))


async def main():
    await (asyncio.sleep(1))


async def main():
    await (asyncio.sleep(1))


# Check comments
async def main():
    await (asyncio.sleep(1))  # Hello


async def main():
    await (asyncio.sleep(1))  # Hello


async def main():
    await (asyncio.sleep(1))  # Hello


# Long lines
async def main():
    await asyncio.gather(
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
    )


# Same as above but with magic trailing comma in function
async def main():
    await asyncio.gather(
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
    )


# Cr@zY Br@ck3Tz
async def main():
    await (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        (
                                                                            (
                                                                                (
                                                                                    (
                                                                                        (
                                                                                            (
                                                                                                (
                                                                                                    (
                                                                                                        (
                                                                                                            (
                                                                                                                (
                                                                                                                    (
                                                                                                                        (
                                                                                                                            (
                                                                                                                                (
                                                                                                                                    (
                                                                                                                                        (
                                                                                                                                            (
                                                                                                                                                (
                                                                                                                                                    (
                                                                                                                                                        (
                                                                                                                                                            (
                                                                                                                                                                (
                                                                                                                                                                    black(
                                                                                                                                                                        1
                                                                                                                                                                    )
                                                                                                                                                                )
                                                                                                                                                            )
                                                                                                                                                        )
                                                                                                                                                    )
                                                                                                                                                )
                                                                                                                                            )
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )


# Keep brackets around non power operations and nested awaits
async def main():
    await (set_of_tasks | other_set)


async def main():
    await (await asyncio.sleep(1))


# It's awaits all the way down...
async def main():
    await (await x)


async def main():
    await (yield x)


async def main():
    await (await (asyncio.sleep(1)))


async def main():
    await (await (await (await (await (asyncio.sleep(1))))))


# output
import asyncio


# Control example
async def main():
    await asyncio.sleep(1)


# Remove brackets for short coroutine/task
async def main():
    await asyncio.sleep(1)


async def main():
    await asyncio.sleep(1)


async def main():
    await asyncio.sleep(1)


# Check comments
async def main():
    await asyncio.sleep(1)  # Hello


async def main():
    await asyncio.sleep(1)  # Hello


async def main():
    await asyncio.sleep(1)  # Hello


# Long lines
async def main():
    await asyncio.gather(
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
    )


# Same as above but with magic trailing comma in function
async def main():
    await asyncio.gather(
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
    )


# Cr@zY Br@ck3Tz
async def main():
    await black(1)


# Keep brackets around non power operations and nested awaits
async def main():
    await (set_of_tasks | other_set)


async def main():
    await (await asyncio.sleep(1))


# It's awaits all the way down...
async def main():
    await (await x)


async def main():
    await (yield x)


async def main():
    await (await asyncio.sleep(1))


async def main():
    await (await (await (await (await asyncio.sleep(1)))))
