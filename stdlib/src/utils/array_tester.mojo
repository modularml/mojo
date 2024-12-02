# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Defines `Testee` and `Tester`. 

`Testee` can be used as the element type of an array or a similar datastructure.
`Tester` holds `Int` fields to count the amount of moves, copy, init, del.
An instance of `Testee` contains an `UnsafePointer` to an instance of `Tester`.
When multiple instances of Testee contains an `UnsafePointer` to the same instance of `Tester`,
Assertions can be created on the current amount of `init`, `copy`, `del`, `moves` a list does.

For example, if an instance of `List[Testee]` is created and appended 10 elements,
the `oninit` field of the instance of `Tester` should be 10.
If `pop()` is done of the list to move it's last element out,
the `onmove` field of the instance of `Testeer` should be 1.

Additionally, `Testee` have an `Int` field to store a value.
That value can be used to check the equality of the elements of two lists, for example. 

Example:
```mojo
def main():
    var _tester = Tester()
    var x = List[Testee]()
    for i in range(10):
        x.append(Testee(_tester, i))
    for i in x: print(i[].value)
    assert_equal(_tester.oninit, 10)
    assert_equal(_tester.ondelete, 0)
    assert_equal(_tester.oncopy, 0)
    y = x[0]
    assert_equal(y.value, 0)
    assert_equal(_tester.oncopy, 1)
    _ = x.pop()
    assert_equal(_tester.ondelete, 2)
    __type_of(x).__del__(x^)
    assert_equal(_tester.ondelete, 11)
    assert_equal(_tester.oncopy, 1)
```
"""
from memory import UnsafePointer

# ===----------------------------------------------------------------------===#
# Tester
# ===----------------------------------------------------------------------===#

struct Tester:
    """"Defines the Tester type."""
    # NOTE: Is non-movable (`__moveinit__` and `__copyinit__`)
    
    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#
    var oninit: Int
    """"Counts __init__ of `Testee` instances."""
    var ondelete: Int
    """"Counts `__del__` of `Testee` instances."""
    var oncopy: Int
    """"Counts __copyinit__ of `Testee` instances."""
    var onmove: Int
    """"Counts `__moveinit__` of `Testee` instances."""
    
    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#
    fn __init__(inout self):
        """Create an instance."""
        self.ondelete = 0
        self.onmove = 0
        self.oncopy = 0
        self.oninit = 0
    
    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#   
    fn reset_counters(inout self):
        """"Set the value of all counters to 0."""
        self.ondelete = 0
        self.onmove = 0
        self.oncopy = 0
        self.oninit = 0


# ===----------------------------------------------------------------------===#
# Testee
# ===----------------------------------------------------------------------===#

@value
struct Testee:
    """"Defines the Testee type."""
    
    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#
    var tester: UnsafePointer[Tester]
    """"UnsafePointer to a non-movable `Tester` instance."""
    var value: Int
    """A value for testing value equality between Testee instances."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#
    fn __init__(inout self, ref[_]tester: Tester, arg: Int):
        """Increments the `Tester.oninit` field on initialization."""
        self.tester = UnsafePointer.address_of(tester)
        self.value = arg
        self.tester[].oninit+=1
    fn __copyinit__(inout self, existing: Testee):
        """"Increments the `Tester.oncopy` field on __copyinit__."""
        self.tester = existing.tester
        self.value = existing.value
        self.tester[].oncopy+=1

    fn __moveinit__(inout self, owned other:Self):
        """Increments the `Tester.onmove` field on __moveinit__."""
        self.tester = other.tester
        self.value = other.value
        self.tester[].onmove+=1

    fn __del__(owned self):
        """Increments the `Tester.ondel` field on __del__."""
        self.tester[].ondelete += 1