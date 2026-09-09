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

# RUN: %mojo -debug-level full %s | FileCheck %s


def make_int_list() -> List[Int]:
    var l: List = [1, 2, 3]
    return l^


def make_flat() -> List[StaticString]:
    var l = List[StaticString]()
    l.append("source")
    l.append("target")
    return l^


def make_nested() -> List[List[StaticString]]:
    var outer = List[List[StaticString]]()
    var inner1 = List[StaticString]()
    inner1.append("alpha")
    inner1.append("beta")
    var inner2 = List[StaticString]()
    inner2.append("gamma")
    inner2.append("epsilon")
    outer.append(inner1^)
    outer.append(inner2^)
    return outer^


# Wrapping the strings in a struct is the interesting part: the wrapper is
# written to constant-global memory as a string-bearing aggregate, which
# re-enters constant-global mapping for each embedded string (MOCO-3738).
struct Flat(Copyable, Movable):
    var items: List[StaticString]

    def __init__(out self, items: List[StaticString]):
        self.items = items.copy()


struct Inner(Copyable, Movable):
    var pattern: StaticString
    var leading: Array[StaticString, 1]
    var count: Int

    def __init__(
        out self,
        pattern: StaticString,
        leading: Array[StaticString, 1],
        count: Int,
    ):
        self.pattern = pattern
        self.leading = leading.copy()
        self.count = count


struct Outer(Copyable, Movable):
    var id: StaticString
    var items: List[Inner]

    def __init__(out self, id: StaticString, items: List[Inner]):
        self.id = id
        self.items = items.copy()


comptime WRAPPED: Flat = Flat(items=["foo", "bar"])

comptime REGIONS: Dict[String, Outer] = {
    "ES": Outer(
        id="ES",
        items=[Inner(pattern="(\\d{4})", leading=["905"], count=1)],
    ),
}


def main() raises:
    comptime lst = make_int_list()
    var dyn_lst = materialize[lst]()
    # CHECK: [1, 2, 3]
    print(dyn_lst)

    comptime flat = make_flat()
    var names = materialize[flat]()
    # CHECK: [source, target]
    print(names)

    comptime nested = make_nested()
    var nested_names = materialize[nested]()
    # CHECK-LITERAL: [[alpha, beta], [gamma, epsilon]]
    print(nested_names)

    # Compare rather than just print: a stale pointer in a materialized
    # StaticString only shows up once the bytes are dereferenced.
    var wrapped = materialize[WRAPPED]()
    # CHECK: wrapped: True True
    print("wrapped:", wrapped.items[0] == "foo", wrapped.items[1] == "bar")

    var regions = materialize[REGIONS]()
    ref region = regions["ES"]
    ref inner = region.items[0]
    # CHECK: region: True True True
    print(
        "region:",
        region.id == "ES",
        inner.pattern == "(\\d{4})",
        inner.leading[0] == "905",
    )
