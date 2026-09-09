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
from std.memory import ThinAllocation, dealloc

comptime Element = String  # Adapt for your type
comptime ListNode = Node[Element]  # Constructing a LinkedList


struct Node[T: ImplicitlyCopyable & Writable & Deinitable](Movable):
    comptime NodePointer = Pointer[Self, MutUntrackedOrigin]

    var value: Optional[Self.T]  # The `Node`'s value
    var next: Optional[Self.NodePointer]  # Pointer to the next `Node`

    # Uses an `Optional` value to allow 'empty' Node construction
    # that can be moved into newly allocated memory
    def __init__(out self, value: Optional[Self.T] = None):
        self.value = value
        self.next = {}

    # Constructs a `Node` with a `value` with heap allocation and
    # returns a pointer to the new `Node`.
    @staticmethod
    def make_node(value: Self.T) -> Self.NodePointer:
        var node_ptr = alloc[Self]({count = 1}).unsafe_leak()
        node_ptr.unsafe_write(Self(value))
        return node_ptr

    # Destroys the pointee, then releases the `Node`'s heap allocation by
    # pairing the leaked pointer back up with the layout it was allocated with.
    @staticmethod
    def free_node(var node_ptr: Self.NodePointer):
        node_ptr.unsafe_deinit_pointee()
        dealloc(
            ThinAllocation(unsafe_owned_ptr=node_ptr).unsafe_with_layout(
                {count = 1}
            )
        )

    # Constructs a `Node` with allocated memory, assigns a value, appends
    # the pointer to `self.next`. Replaces any existing `next`.
    def append(mut self, value: Self.T):
        # Free chain if replacing `next`
        if self.next:
            var next_ptr = self.next.value()
            next_ptr[].free_chain()
            Self.free_node(next_ptr)

        self.next = Self.make_node(value)

    # Prints the list starting at this pointer's pointee
    @staticmethod
    def print_list(node: Optional[Self.NodePointer]):
        if not node:
            print("Empty list")
            return

        var node_ptr = node.value()

        var current_value: Optional[Self.T] = node_ptr[].value
        if current_value:
            print(current_value.value(), end=" ")

        if node_ptr[].next:
            Self.print_list(node_ptr[].next)
        else:
            print()

    # Releases all successively allocated `Node` pointees. Does not release self.
    def free_chain(self):
        var current = self.next
        while current:
            var current_ptr = current.value()
            var next_node = current_ptr[].next
            Self.free_node(current_ptr)
            current = next_node


def main():
    var values: List[Element] = ["one", "one", "two", "three", "five", "eight"]
    var list_head = ListNode.make_node(values[0])

    var current = list_head
    for idx in range(1, len(values), 1):
        current[].append(values[idx])
        current = current[].next.value()

    ListNode.print_list(list_head)

    # Demonstrates cleanup. In short-lived programs, the OS reclaims memory
    # at exit
    list_head[].free_chain()
    ListNode.free_node(list_head)
