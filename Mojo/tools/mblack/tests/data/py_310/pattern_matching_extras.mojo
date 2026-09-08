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
#   Path:   tests/data/py_310/pattern_matching_extras.py
#
# ===----------------------------------------------------------------------=== #

import match

match something:
    case [a as b]:
        print(b)
    case [a as b, c, d, e as f]:
        print(f)
    case Point(a as b):
        print(b)
    case Point(int() as x, int() as y):
        print(x, y)


match = 1
case: int = re.match(something)

match re.match(case):
    case type("match", match):
        pass
    case match:
        pass


def func(match: case, case: match) -> case:
    match Something():
        case func(match, case):
            ...
        case another:
            ...


match maybe, multiple:
    case perhaps, 5:
        pass
    case perhaps, 6,:
        pass


match more := (than, one), indeed,:
    case _, (5, 6):
        pass
    case [[5], (6)], [7],:
        pass
    case _:
        pass


match a, *b, c:
    case [...]:
        assert "seq" == _
    case {}:
        assert "map" == b


match match(
    case,
    match(
        match,
        case,
        match,
        looooooooooooooooooooooooooooooooooooong,
        match,
        case,
        match,
    ),
    case,
):
    case case(
        match=case,
        case=re.match(
            loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong
        ),
    ):
        pass

    case [a as match]:
        pass

    case case:
        pass


match match:
    case case:
        pass


match a, *b(), c:
    case d, *f, g:
        pass


match something:
    case {
        "key": key as key_1,
        "password": PASS.ONE | PASS.TWO | PASS.THREE as password,
    }:
        pass
    case {"maybe": something(complicated as this) as that}:
        pass


match something:
    case 1 as a:
        pass

    case 2 as b, 3 as c:
        pass

    case 4 as d, (5 as e), (6 | 7 as g), *h:
        pass


match bar1:
    case Foo(aa=Callable() as aa, bb=int()):
        print(bar1.aa, bar1.bb)
    case _:
        print("no match", "\n")


match bar1:
    case Foo(
        normal=x,
        perhaps=[list, {an: d, dict: 1.0}] as y,
        otherwise=something,
        q=t as u,
    ):
        pass
