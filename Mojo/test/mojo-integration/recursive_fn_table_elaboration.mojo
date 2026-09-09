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

# RUN: %mojo %s | FileCheck %s
# RUN: %mojo --num-threads 1 %s | FileCheck %s

# The deadlock needs a single method whose elaboration acquires two blockers
# at once: a recursive self-edge and a comptime function-pointer table.
# See the two marked lines in `to_string`.

from std.memory import ArcPointer
from std.utils import Variant


@fieldwise_init
struct Arr(Movable):
    var values: List[ArcPointer[Value]]


@fieldwise_init
struct Str(Movable):
    var value: String


@fieldwise_init
struct IntVal(Movable):
    var value: Int


comptime Payload = Variant[Arr, Str, IntVal]


struct Value(Deinitable, Movable):
    comptime Kind = UInt8
    comptime ARRAY: Self.Kind = 0
    comptime STRING: Self.Kind = 1
    comptime INT: Self.Kind = 2

    var kind: Self.Kind
    var payload: Payload

    def __init__(out self, kind: Self.Kind, var payload: Payload):
        self.kind = kind
        self.payload = payload^

    def __deinit__(deinit self):
        pass

    def to_string(imm self) raises -> String:
        if self.kind == Self.ARRAY:
            ref arr = self.payload[Arr]
            var s = String("[")
            for i in range(len(arr.values)):
                if i > 0:
                    s += ","
                # Blocker 1: the recursive call makes elaborating `to_string`
                # wait on `to_string` itself (an SCC self-edge on the node).
                s += arr.values[i][].to_string()
            s += "]"
            return s^
        # Blocker 2, on the SAME node while blocker 1 is still pending:
        # indexing the comptime `SERIALIZERS` table forces its generator to
        # elaborate. Two concurrent blockers overflowed the old single slot.
        return materialize[SERIALIZERS]()[self.kind](self)


comptime Serializer = def(Value) raises thin -> String


@always_inline
def string_to_json(v: Value) raises -> String:
    return '"' + v.payload[Str].value + '"'


@always_inline
def int_to_json(v: Value) raises -> String:
    return String(v.payload[IntVal].value)


# The table must hold at least two distinct thin functions to trigger the bug.
@always_inline
def serializer_table() -> Array[Serializer, 3]:
    var table = Array[Serializer, 3](fill=int_to_json)
    table[Value.STRING] = string_to_json
    return table^


comptime SERIALIZERS = serializer_table()


def main() raises:
    # CHECK: "ok"
    print(Value(Value.STRING, Payload(Str("ok"))).to_string())

    # A nested array drives both the recursive branch and the table dispatch.
    var arr = Arr(
        [
            ArcPointer(Value(Value.INT, Payload(IntVal(1)))),
            ArcPointer(Value(Value.STRING, Payload(Str("x")))),
        ]
    )
    # CHECK: [1,"x"]
    print(Value(Value.ARRAY, Payload(arr^)).to_string())
