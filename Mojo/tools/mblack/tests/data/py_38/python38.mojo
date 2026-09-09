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
#   Path:   tests/data/py_38/python38.py
#
# ===----------------------------------------------------------------------=== #

#!/usr/bin/env python3.8


def starred_return():
    my_list = ["value2", "value3"]
    return "value1", *my_list


def starred_yield():
    my_list = ["value2", "value3"]
    yield "value1", *my_list


# all right hand side expressions allowed in regular assignments are now also allowed in
# annotated assignments
a: Tuple[str, int] = "1", 2
a: Tuple[int, ...] = b, *c, d


def t():
    a: str = yield "a"


# output


#!/usr/bin/env python3.8


def starred_return():
    my_list = ["value2", "value3"]
    return "value1", *my_list


def starred_yield():
    my_list = ["value2", "value3"]
    yield "value1", *my_list


# all right hand side expressions allowed in regular assignments are now also allowed in
# annotated assignments
a: Tuple[str, int] = "1", 2
a: Tuple[int, ...] = b, *c, d


def t():
    a: str = yield "a"
