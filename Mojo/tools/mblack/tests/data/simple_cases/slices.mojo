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
#   Path:   tests/data/simple_cases/slices.py
#
# ===----------------------------------------------------------------------=== #

slice[a.b : c.d]
slice[d :: d + 1]
slice[d + 1 :: d]
slice[d::d]
slice[0]
slice[-1]
slice[:-1]
slice[::-1]
slice[:c, c - 1]
slice[c, c + 1, d::]
slice[ham[c::d] :: 1]
slice[ham[cheese**2 : -1] : 1 : 1, ham[1:2]]
slice[:-1:]
slice[lambda: None : lambda: None]
slice[lambda x, y, *args, really=2, **kwargs: None :, None::]
slice[1 or 2 : True and False]
slice[not so_simple : 1 < val <= 10]
slice[(1 for i in range(42)) : x]
slice[:: [i for i in range(42)]]


async def f():
    slice[await x : [i async for i in arange(42)] : 42]


# These are from PEP-8:
ham[1:9], ham[1:9:3], ham[:9:3], ham[1::3], ham[1:9:]
ham[lower:upper], ham[lower:upper:], ham[lower::step]
# ham[lower+offset : upper+offset]
ham[: upper_fn(x) : step_fn(x)], ham[:: step_fn(x)]
ham[lower + offset : upper + offset]
