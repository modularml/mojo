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

from std.collections.type_dict import TypeDict
from std.testing import TestSuite, assert_equal


def test_type_dict_construction() raises:
    comptime td = TypeDict[
        T=Int,
        Trait=AnyType,
        [1, 2, 3],
        Int,
        String,
        Float64,
    ]
    _ = td


def test_type_dict_length() raises:
    comptime td = TypeDict[
        T=Int,
        Trait=AnyType,
        [1, 2, 3],
        Int,
        String,
        Float64,
    ]
    comptime assert td.length == 3


def test_type_dict_get() raises:
    comptime td = TypeDict[
        T=Int,
        Trait=AnyType,
        [1, 2, 3],
        Int,
        String,
        Float64,
    ]
    comptime assert td.get[1] == Int
    comptime assert td.get[2] == String
    comptime assert td.get[3] == Float64


def test_type_dict_string_keys() raises:
    comptime td = TypeDict[
        T=String,
        Trait=AnyType,
        ["a", "b", "c"],
        Int,
        String,
        Float64,
    ]
    comptime assert td.get["a"] == Int
    comptime assert td.get["b"] == String
    comptime assert td.get["c"] == Float64


def test_type_dict_single_entry() raises:
    comptime td = TypeDict[
        T=Int,
        Trait=AnyType,
        [42],
        Float64,
    ]
    comptime assert td.get[42] == Float64
    comptime assert td.length == 1


# Complex use-cases:
# 1. Using a constrained trait (not just AnyType)
# 2. Creating an alias for a TypeDict and reusing it
# 3. Using TypeDict within a generic function to dispatch types
# 4. Using TypeDict with boolean keys


def test_type_dict_with_constrained_trait() raises:
    # Use a trait that only types with specific capabilities can satisfy
    comptime td = TypeDict[
        T=Int,
        Trait=Movable,
        [1, 2],
        Int,
        String,
    ]
    comptime assert td.get[1] == Int
    comptime assert td.get[2] == String


comptime MyTypeDict = TypeDict[
    T=Int,
    Trait=AnyType,
    [10, 20, 30],
    Int,
    String,
    Float64,
]


def test_type_dict_alias() raises:
    comptime assert MyTypeDict.length == 3
    comptime assert MyTypeDict.get[10] == Int
    comptime assert MyTypeDict.get[20] == String
    comptime assert MyTypeDict.get[30] == Float64


def _dispatch[key: Int]() -> Int:
    comptime td = TypeDict[
        T=Int,
        Trait=AnyType,
        [1, 2],
        Int,
        String,
    ]
    # Use comptime to return a different value based on the type
    comptime if td.get[key] == Int:
        return 42
    else:
        return 0


def test_type_dict_in_generic_function() raises:
    assert_equal(_dispatch[1](), 42)
    assert_equal(_dispatch[2](), 0)


def test_type_dict_boolean_keys() raises:
    comptime td = TypeDict[
        T=Bool,
        Trait=AnyType,
        [True, False],
        Int,
        String,
    ]
    comptime assert td.get[True] == Int
    comptime assert td.get[False] == String


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
