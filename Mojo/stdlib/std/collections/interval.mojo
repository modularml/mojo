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

"""
A self-balancing interval tree is a specialized binary search tree designed to
efficiently store and query intervals.

It maintains intervals sorted by their low endpoints and augments each node with a
`max_high` attribute, representing the maximum high endpoint in its subtree. This
`max_high` value enables efficient overlap searching by pruning the search space.
Self-balancing mechanisms, such as Red-Black or AVL trees, ensure logarithmic time
complexity for operations.

Key Features:
  - Stores intervals (low, high).
  - Nodes ordered by `low` endpoints.
  - `max_high` attribute at each node for efficient overlap search.
  - Self-balancing (e.g., using Red-Black tree logic) for O(log n) operations.

Operations:
  - Insertion: O(log n) - Adds a new interval, maintaining balance and updating
    `max_high`.
  - Overlap Search: O(log n) - Finds intervals overlapping a query interval using
    `max_high` for pruning.
  - Deletion: O(log n) - Removes an interval, maintaining balance and updating
    `max_high`.

Space Complexity: O(n), where n is the number of intervals.

Use Cases:
  - Calendar scheduling
  - Computational geometry
  - Genomics
  - Database indexing
  - Resource allocation

In essence, this data structure provides a fast and efficient way to manage and
query interval data, particularly for finding overlaps.
"""


from std.builtin.string_literal import StaticString
from std.memory.alloc import alloc, dealloc, ThinAllocation, Layout

import std.format._utils as fmt

from .deque import Deque


trait IntervalElement(Comparable, Copyable, Deinitable, Intable, Writable):
    """The trait denotes a trait composition of the `Copyable`,
    `Writable`, `Intable`, and `Comparable` traits. Which is also subtractable.
    """

    def __sub__(self, rhs: Self) -> Self:
        """Subtracts rhs from self, must be implemented in concrete types.

        Args:
            rhs: The value to subtract from self.

        Returns:
            The result of subtracting rhs from self.
        """
        ...


struct Interval[T: IntervalElement](
    Boolable,
    Equatable,
    ImplicitlyCopyable,
    Sized,
    Writable,
):
    """A half-open interval [start, end) that represents a range of values.

    The interval includes the start value but excludes the end value.

    Parameters:
        T: The type of the interval bounds.
    """

    var start: Self.T
    """The inclusive start of the interval."""

    var end: Self.T
    """The exclusive end of the interval."""

    def __init__(out self, start: Self.T, end: Self.T):
        """Initialize an interval with start and end values.

        Args:
            start: The starting value of the interval.
            end: The ending value of the interval. Must be greater than or
              equal to start.
        """
        debug_assert(
            start <= end, "invalid interval '(", start, ", ", end, ")'"
        )
        self.start = start.copy()
        self.end = end.copy()

    def __init__(out self, interval: Tuple[Self.T, Self.T], /):
        """Initialize an interval with a tuple of start and end values.

        Args:
            interval: A tuple containing the start and end values.
        """
        self.start = interval[0].copy()
        self.end = interval[1].copy()

    def __init__(out self, *, copy: Self):
        """Create a new instance of the interval by copying the values
        from an existing one.

        Args:
            copy: The interval to copy values from.
        """
        self.start = copy.start.copy()
        self.end = copy.end.copy()

    def overlaps(self, other: Self) -> Bool:
        """Returns whether this interval overlaps with another interval.

        Args:
            other: The interval to check for overlap with.

        Returns:
            True if the intervals overlap, False otherwise.
        """
        return other.start < self.end and other.end > self.start

    def union(self, other: Self) -> Self:
        """Returns the union of this interval and another interval.

        Args:
            other: The interval to union with.

        Returns:
            The union of this interval and the other interval.
        """
        debug_assert(
            self.overlaps(other),
            "intervals do not overlap when computing the union of '",
            self,
            "' and '",
            other,
            "'",
        )
        var start = (
            self.start.copy() if self.start
            < other.start else other.start.copy()
        )
        var end = self.end.copy() if self.end > other.end else other.end.copy()
        return Self(start, end)

    def intersection(self, other: Self) -> Self:
        """Returns the intersection of this interval and another interval.

        Args:
            other: The interval to intersect with.

        Returns:
            The intersection of this interval and the other interval.
        """
        debug_assert(
            self.overlaps(other),
            "intervals do not overlap when computing the intersection of '",
            self,
            "' and '",
            other,
            "'",
        )
        var start = (
            self.start.copy() if self.start
            > other.start else other.start.copy()
        )
        var end = self.end.copy() if self.end < other.end else other.end.copy()
        return Self(start, end)

    def __contains__(self, other: Self.T) -> Bool:
        """Returns whether a value is contained within this interval.

        Args:
            other: The value to check.

        Returns:
            True if the value is within the interval bounds, False otherwise.
        """
        return self.start <= other.copy() < self.end.copy()

    def __contains__(self, other: Self) -> Bool:
        """Returns whether another interval is fully contained within this
        interval.

        Args:
            other: The interval to check.

        Returns:
            True if the other interval is fully contained within this interval,
            False otherwise.
        """
        return self.start <= other.start and self.end >= other.end

    def __le__(self, other: Self) -> Bool:
        """Returns whether this interval is less than or equal to another
        interval.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval's start is less than or equal to the other interval's start.
        """
        return self.start <= other.start

    def __ge__(self, other: Self) -> Bool:
        """Returns whether this interval is greater than or equal to another
        interval.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval's end is greater than or equal to the other interval's end.
        """
        return self.end >= other.end

    def __lt__(self, other: Self) -> Bool:
        """Returns whether this interval is less than another interval.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval's start is less than the other interval's start.
        """
        return self.start < other.start

    def __gt__(self, other: Self) -> Bool:
        """Returns whether this interval is greater than another interval.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval's end is greater than the other interval's end.
        """
        return self.end > other.end

    def __len__(self) -> Int:
        """Returns the length of this interval.

        Returns:
            The difference between end and start values as an integer.
        """
        assert Bool(self), "interval is empty"
        return Int(self.end - self.start)

    def __bool__(self) -> Bool:
        """Returns whether this interval is empty.

        Returns:
            True if the interval is not empty (start < end), False otherwise.
        """
        return self.start < self.end

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes this interval to a writer in the format '(start, end)'.

        Args:
            writer: The writer to write the interval to.
        """
        writer.write("(", self.start, ", ", self.end, ")")


struct _IntervalNode[
    T: IntervalElement,
    U: Copyable & Comparable & Deinitable & Writable,
](Copyable, Writable):
    """A node containing an interval and associated data.

    Parameters:
        T: The type of the interval bounds, must support subtraction, integer
          conversion, string conversion, comparison and collection operations.
        U: The type of the associated data, must support string conversion
          and collection operations.
    """

    comptime _PointerType = Optional[Pointer[Self, MutUntrackedOrigin]]

    var interval: Interval[Self.T]
    """The interval contained in this node."""

    var data: Self.U
    """The data associated with this interval."""

    var max_end: Self.T
    """The maximum end value of this node."""

    var _left: Self._PointerType
    """The left child of this node."""

    var _right: Self._PointerType
    """The right child of this node."""

    var _parent: Self._PointerType
    """The parent of this node."""

    var _is_red: Bool
    """Red-black node color."""

    def left(
        ref self,
    ) -> ref[self._left] Optional[Pointer[Self, MutUntrackedOrigin]]:
        """Returns a reference to the left child pointer."""
        return self._left

    def right(
        ref self,
    ) -> ref[self._right] Optional[Pointer[Self, MutUntrackedOrigin]]:
        """Returns a reference to the right child pointer."""
        return self._right

    def parent(
        ref self,
    ) -> ref[self._parent] Optional[Pointer[Self, MutUntrackedOrigin]]:
        """Returns a reference to the parent pointer."""
        return self._parent

    def __init__(
        out self,
        start: Self.T,
        end: Self.T,
        data: Self.U,
        *,
        left: Self._PointerType = {},
        right: Self._PointerType = {},
        parent: Self._PointerType = {},
        is_red: Bool = True,
    ):
        """Creates a new interval node.

        Args:
            start: The start value of the interval.
            end: The end value of the interval.
            data: The data to associate with this interval.
            left: The left child of this node.
            right: The right child of this node.
            parent: The parent of this node.
            is_red: Whether this node is red in the red-black tree.
        """
        self = Self(
            Interval(start, end),
            data,
            left=left,
            right=right,
            parent=parent,
            is_red=is_red,
        )

    def __init__(
        out self,
        interval: Interval[Self.T],
        data: Self.U,
        *,
        left: Self._PointerType = {},
        right: Self._PointerType = {},
        parent: Self._PointerType = {},
        is_red: Bool = True,
    ):
        """Creates a new interval node.

        Args:
            interval: The interval to associate with this node.
            data: The data to associate with this interval.
            left: The left child of this node.
            right: The right child of this node.
            parent: The parent of this node.
            is_red: Whether this node is red in the red-black tree.
        """
        self.interval = interval
        self.max_end = interval.end.copy()
        self.data = data.copy()
        self._left = left
        self._right = right
        self._parent = parent
        self._is_red = is_red

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes this interval node to a writer in the format
        '(start, end): data'.

        Args:
            writer: The writer to write the interval node to.
        """
        writer.write(self.interval, "=", self.data)

    def depth(self) -> Int:
        """Returns the depth of this interval node.

        Returns:
            The depth of this interval node.
        """
        var left_depth = self.left().value()[].depth() if self.left() else 0
        var right_depth = self.right().value()[].depth() if self.right() else 0
        return 1 + max(left_depth, right_depth)

    def __bool__(self) -> Bool:
        """Returns whether this interval node is empty.

        Returns:
            True if the interval node is empty, False otherwise.
        """
        return Bool(self.interval)

    def __eq__(self, other: Self) -> Bool:
        """Returns whether this interval node is equal to another interval node.

        Args:
            other: The interval node to compare with.

        Returns:
            True if the interval nodes are equal, False otherwise.
        """
        return self.interval == other.interval and self.data == other.data

    def __lt__(self, other: Self) -> Bool:
        return self.interval < other.interval

    def __gt__(self, other: Self) -> Bool:
        return self.interval > other.interval


struct IntervalTree[
    T: IntervalElement,
    U: Copyable & Comparable & Deinitable & Writable,
](Defaultable, Writable):
    """An interval tree data structure for efficient range queries.

    Parameters:
        T: The type of the interval bounds, must support subtraction, integer
          conversion, string conversion, comparison and collection operations.
        U: The type of the associated data, must support string conversion
          and collection operations.
    """

    comptime _IntervalNodePointer = Optional[
        Pointer[_IntervalNode[Self.T, Self.U], MutUntrackedOrigin]
    ]

    var _root: Self._IntervalNodePointer
    """The root node of the interval tree."""

    var _len: Int
    """The number of elements in the interval tree."""

    def __init__(out self):
        """Initializes an empty IntervalTree."""
        self._root = Self._IntervalNodePointer()
        self._len = 0

    def __deinit__(deinit self):
        """Destructor that frees the interval tree's memory."""
        if self._root:
            Self._del_helper(self._root.value())

    @staticmethod
    def _del_helper(
        node: Pointer[_IntervalNode[Self.T, Self.U], MutUntrackedOrigin],
    ):
        if node[].left():
            Self._del_helper(node[].left().value())
        if node[].right():
            Self._del_helper(node[].right().value())
        node.unsafe_deinit_pointee()
        dealloc(
            ThinAllocation(unsafe_owned_ptr=node).unsafe_with_layout(
                {count = 1}
            )
        )

    def _left_rotate(mut self, rotation_node: Self._IntervalNodePointer):
        """Performs a left rotation around node x in the red-black tree.

        This method performs a left rotation around the given node x, which is a
        standard operation in red-black trees used to maintain balance. The rotation
        preserves the binary search tree property while changing the structure.

        Before:          After:
             x            y
            / \\         / \\
           a   y   =>   x   c
              / \\      / \\
             b   c    a   b

        Args:
            rotation_node: A pointer to the node around which to perform the left rotation.

        Note:
            The rotation assumes that x has a right child. The method will assert if
            either the root or x's right child is not set.
        """
        assert Bool(self._root), "node is not set"
        var rotation_node_nn = rotation_node.value()
        var rotation_right_child = rotation_node_nn[].right()
        assert Bool(rotation_right_child), "right child is not set"
        var rotation_right_child_nn = rotation_right_child.value()
        rotation_node_nn[].right() = rotation_right_child_nn[].left()

        if rotation_right_child_nn[].left():
            rotation_right_child_nn[].left().value()[].parent() = rotation_node

        rotation_right_child_nn[].parent() = rotation_node_nn[].parent()

        if not rotation_node_nn[].parent():
            self._root = rotation_right_child
        elif rotation_node == rotation_node_nn[].parent().value()[].left():
            rotation_node_nn[].parent().value()[].left() = rotation_right_child
        else:
            rotation_node_nn[].parent().value()[].right() = rotation_right_child

        rotation_right_child_nn[].left() = rotation_node
        rotation_node_nn[].parent() = rotation_right_child

        rotation_node_nn[].max_end = rotation_node_nn[].interval.end.copy()
        if rotation_node_nn[].left():
            rotation_node_nn[].max_end = max(
                rotation_node_nn[].max_end,
                rotation_node_nn[].left().value()[].max_end,
            )
        if rotation_node_nn[].right():
            rotation_node_nn[].max_end = max(
                rotation_node_nn[].max_end,
                rotation_node_nn[].right().value()[].max_end,
            )

        rotation_right_child_nn[].max_end = (
            rotation_right_child_nn[].interval.end.copy()
        )
        if rotation_right_child_nn[].left():
            rotation_right_child_nn[].max_end = max(
                rotation_right_child_nn[].max_end,
                rotation_right_child_nn[].left().value()[].max_end,
            )
        if rotation_right_child_nn[].right():
            rotation_right_child_nn[].max_end = max(
                rotation_right_child_nn[].max_end,
                rotation_right_child_nn[].right().value()[].max_end,
            )

    def _right_rotate(mut self, rotation_node: Self._IntervalNodePointer):
        """Performs a right rotation around node y in the red-black tree.

        This method performs a right rotation around the given node y, which is a
        standard operation in red-black trees used to maintain balance. The rotation
        preserves the binary search tree property while changing the structure.

        Before:          After:
             y            x
            / \\         / \\
           x   c   =>   a   y
          / \\          / \\
         a   b        b   c

        Args:
            rotation_node: A pointer to the node around which to perform the right rotation.

        Note:
            The rotation assumes that y has a left child. The method will assert if
            either the root or y's left child is not set.
        """
        assert Bool(self._root), "root node is not set"
        var rotation_node_nn = rotation_node.value()
        var rotation_left_child = rotation_node_nn[].left()
        assert Bool(rotation_left_child), "left child node is not set"
        var rotation_left_child_nn = rotation_left_child.value()
        rotation_node_nn[].left() = rotation_left_child_nn[].right()

        if rotation_left_child_nn[].right():
            rotation_left_child_nn[].right().value()[].parent() = rotation_node

        rotation_left_child_nn[].parent() = rotation_node_nn[].parent()

        if not rotation_node_nn[].parent():
            self._root = rotation_left_child
        elif rotation_node == rotation_node_nn[].parent().value()[].right():
            rotation_node_nn[].parent().value()[].right() = rotation_left_child
        else:
            rotation_node_nn[].parent().value()[].left() = rotation_left_child

        rotation_left_child_nn[].right() = rotation_node
        rotation_node_nn[].parent() = rotation_left_child

        rotation_node_nn[].max_end = rotation_node_nn[].interval.end.copy()
        if rotation_node_nn[].left():
            rotation_node_nn[].max_end = max(
                rotation_node_nn[].max_end,
                rotation_node_nn[].left().value()[].max_end,
            )
        if rotation_node_nn[].right():
            rotation_node_nn[].max_end = max(
                rotation_node_nn[].max_end,
                rotation_node_nn[].right().value()[].max_end,
            )

        rotation_left_child_nn[].max_end = (
            rotation_left_child_nn[].interval.end.copy()
        )
        if rotation_left_child_nn[].left():
            rotation_left_child_nn[].max_end = max(
                rotation_left_child_nn[].max_end,
                rotation_left_child_nn[].left().value()[].max_end,
            )

    def insert(mut self, interval: Tuple[Self.T, Self.T], data: Self.U):
        """Insert a new interval into the tree using a tuple representation.

        Args:
            interval: A tuple containing the start and end values of the interval.
            data: The data value to associate with this interval.
        """
        self.insert(Interval(interval[0], interval[1]), data)

    def insert(mut self, interval: Interval[Self.T], data: Self.U):
        """Insert a new interval into the tree.

        This method inserts a new interval and its associated data into the interval tree.
        It maintains the binary search tree property based on interval start times and
        updates the tree structure to preserve red-black tree properties.

        Args:
            interval: The interval to insert into the tree.
            data: The data value to associate with this interval.
        """
        # Allocate memory for a new node and initialize it with the interval
        # and data
        var new_node: Pointer[
            _IntervalNode[Self.T, Self.U], MutUntrackedOrigin
        ] = alloc(Layout[_IntervalNode[Self.T, Self.U]].single()).unsafe_leak()
        new_node.unsafe_write(_IntervalNode(interval, data))
        self._len += 1

        # If the tree is empty, set the root to the new node and color it black.
        if not self._root:
            self._root = new_node
            self._root.value()[]._is_red = False
            return

        # Find the insertion point by traversing down the tree
        # parent_node tracks the parent of the current node
        var parent_node = Self._IntervalNodePointer()
        # current_node traverses down the tree until we find an empty spot
        var current_node = self._root
        while current_node:
            parent_node = current_node
            if new_node[] < current_node.value()[]:
                current_node = current_node.value()[].left()
            else:
                current_node = current_node.value()[].right()
            parent_node.value()[].max_end = max(
                parent_node.value()[].max_end,
                new_node[].interval.end,
            )

        new_node[].parent() = parent_node
        if not parent_node:
            self._root = new_node
        elif new_node[] < parent_node.value()[]:
            parent_node.value()[].left() = new_node
        else:
            parent_node.value()[].right() = new_node

        self._insert_fixup(new_node)

    def _insert_fixup(mut self, current_node0: Self._IntervalNodePointer):
        """Fixes up the red-black tree properties after an insertion.

        This method restores the red-black tree properties that may have been violated
        during insertion of a new node. It performs rotations and color changes to
        maintain the balance and color properties of the red-black tree.

        Args:
            current_node0: A pointer to the newly inserted node that may violate red-black
                properties.
        """
        var current_node = current_node0

        # While the parent of the current node is red, we need to fix violations
        while (
            current_node != self._root
            and current_node.value()[].parent().value()[]._is_red
        ):
            var node = current_node.value()
            var parent = node[].parent().value()
            var grandparent = parent[].parent().value()
            if node[].parent() == grandparent[].left():
                # Get uncle node (parent's sibling)
                var uncle_node = grandparent[].right()
                if uncle_node and uncle_node.value()[]._is_red:
                    # Case 1: Uncle is red - recolor parent, uncle and grandparent
                    parent[]._is_red = False
                    uncle_node.value()[]._is_red = False
                    grandparent[]._is_red = True
                    current_node = parent[].parent()
                else:
                    # Case 2: Uncle is black and node is a right child
                    if current_node == parent[].right():
                        current_node = node[].parent()
                        self._left_rotate(current_node)
                        node = current_node.value()
                        parent = node[].parent().value()
                        grandparent = parent[].parent().value()
                    # Case 3: Uncle is black and node is a left child
                    parent[]._is_red = False
                    grandparent[]._is_red = True
                    self._right_rotate(parent[].parent())
            else:
                # Mirror case - parent is right child of grandparent
                var uncle_node = grandparent[].left()
                if uncle_node and uncle_node.value()[]._is_red:
                    # Case 1: Uncle is red - recolor
                    parent[]._is_red = False
                    uncle_node.value()[]._is_red = False
                    grandparent[]._is_red = True
                    current_node = parent[].parent()
                else:
                    # Case 2: Uncle is black and node is a left child
                    if current_node == parent[].left():
                        current_node = node[].parent()
                        self._right_rotate(current_node)
                        node = current_node.value()
                        parent = node[].parent().value()
                        grandparent = parent[].parent().value()
                    # Case 3: Uncle is black and node is a right child
                    parent[]._is_red = False
                    grandparent[]._is_red = True
                    self._left_rotate(parent[].parent())

        # Ensure root is black to maintain red-black tree properties
        self._root.value()[]._is_red = False

    def write_to(self, mut writer: Some[Writer]):
        """Writes the interval tree to a writer.

        Args:
            writer: The writer to write the interval tree to.
        """
        self._draw(writer)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the repr of this `IntervalTree` to a writer.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._draw(w)

        fmt.FormatStruct(writer, "IntervalTree").params(
            fmt.TypeNames[Self.T, Self.U](),
        ).fields(write_fields)

    @no_inline
    def _draw[w: Writer](self, mut writer: w):
        """Draws the interval tree in a simple ASCII tree format.

        Creates a text representation of the tree using ASCII characters, with each node
        indented according to its depth. Uses '├─' and '└─' characters to show the tree
        structure.

        Parameters:
            w: The writer type that implements the Writer trait.

        Args:
            writer: The writer to output the tree visualization to.
        """
        self._draw_helper(writer, self._root, "", True)

    @no_inline
    def _draw_helper[
        w: Writer
    ](
        self,
        mut writer: w,
        node: Self._IntervalNodePointer,
        indent: String,
        is_last: Bool,
    ):
        """Helper function to recursively draw the interval tree.

        Recursively traverses the tree and draws each node with proper indentation
        and branch characters to show the tree structure.

        Parameters:
            w: The writer type that implements the Writer trait.

        Args:
            writer: The writer to output the tree visualization to.
            node: The current node being drawn.
            indent: The current indentation string.
            is_last: Whether this node is the last child of its parent.
        """
        # Handle empty tree case
        if not node:
            return

        writer.write(indent)
        var next_indent = indent
        if is_last:
            writer.write("├─")
            next_indent += "  "
        else:
            writer.write("└─")
            next_indent += "| "
        writer.write(node.value()[], "\n")
        # Recursively draw left and right subtrees
        self._draw_helper(writer, node.value()[].left(), next_indent, False)
        self._draw_helper(writer, node.value()[].right(), next_indent, True)

    @no_inline
    def _draw3[w: Writer](self, mut writer: w) raises:
        """Draws the interval tree in a simple ASCII tree format.

        Creates a text representation of the tree using ASCII characters, with each node
        indented according to its depth. Uses '├─' and '└─' characters to show the tree
        structure.

        Parameters:
            w: The writer type that implements the Writer trait.

        Args:
            writer: The writer to output the tree visualization to.
        """
        # Handle empty tree case
        if not self._root:
            return writer.write("Empty")

        var work_list = Deque[Tuple[Self._IntervalNodePointer, String, Bool]]()
        work_list.append((self._root, String(), True))

        while work_list:
            var node, indent, is_last = work_list.pop()
            if not node:
                continue
            writer.write(indent)
            if is_last:
                writer.write("├─ ")
                indent += "   "
            else:
                writer.write("└─ ")
                indent += "|  "
            writer.write(node.value()[], "\n")
            work_list.append((node.value()[].left(), indent, False))
            work_list.append((node.value()[].right(), indent, True))

    @no_inline
    def _draw2[w: Writer](self, mut writer: w) raises:
        """Draws the interval tree in a visual ASCII art format.

        Creates a grid representation of the tree with nodes and connecting branches.
        Each level of the tree is separated by 3 rows vertically.
        Nodes are connected by '/' and '\' characters for left and right children.

        Parameters:
            w: The writer type that implements the Writer trait.

        Args:
            writer: The writer to output the tree visualization to.
        """
        # Handle empty tree case
        if not self._root:
            return writer.write("Empty")

        # Calculate dimensions needed for the grid
        var height = self._root.value()[].depth()
        var width = 2**height - 1

        # Create 2D grid of spaces to hold the tree visualization
        # Each row is a list of single character strings
        var grid = List[List[String]]()
        for _ in range(3 * height):
            var row = List[String]()
            for _ in range(4 * width):
                row.append(" ")  # Initialize with spaces
            grid.append(row^)

        var work_list = Deque[Tuple[Self._IntervalNodePointer, Int, Int, Int]]()
        work_list.append((self._root, 0, 0, width))

        while work_list:
            # Recursively fills the grid with node values and connecting branches.
            var node, level, left, right = work_list.pop()
            if not node:
                continue

            # Calculate position for current node
            var mid = (left + right) // 2  # Center point between boundaries
            var pos_x = mid * 4  # Scale x position for readability
            var pos_y = level * 3  # Scale y position for branch drawing

            # Draw the current node's value
            var node_str = String(node.value()[])
            var start_pos = max(
                0, pos_x - node_str.byte_length() // 2
            )  # Center the node text
            var i = 0
            for char in node_str.codepoints():
                grid[pos_y][start_pos + i] = String(char)
                i += 1

            # Add drawing left branch to the worklist.
            if node.value()[].left():
                for y in range(1, 3):
                    grid[pos_y + y][pos_x - 2 * y + 1] = "/"  # Draw left branch
                work_list.append((node.value()[].left(), level + 1, left, mid))

            # Add drawing right branch to the worklist.
            if node.value()[].right():
                for y in range(1, 3):
                    grid[pos_y + y][pos_x + 2 * y] = "\\"  # Draw right branch
                work_list.append(
                    (node.value()[].right(), level + 1, mid, right)
                )

        # Output the completed grid row by row
        for row in grid:
            var row_str = String(StaticString("").join(row).rstrip())
            if row_str:
                writer.write(row_str, "\n")

    def depth(self) -> Int:
        """Returns the depth of the interval tree.

        Returns:
            The depth of the interval tree.
        """
        if not self._root:
            return 0

        return self._root.value()[].depth()

    def transplant(
        mut self,
        mut u: Self._IntervalNodePointer,
        mut v: Self._IntervalNodePointer,
    ):
        """Transplants the subtree rooted at node u with the subtree rooted at node v.

        Args:
            u: The node to transplant.
            v: The node to transplant to.
        """
        if not u.value()[].parent():
            self._root = v
        elif u == u.value()[].parent().value()[].left():
            u.value()[].parent().value()[].left() = v
        else:
            u.value()[].parent().value()[].right() = v

        if v:
            v.value()[].parent() = u.value()[].parent()

    def search(self, interval: Tuple[Self.T, Self.T]) raises -> List[Self.U]:
        """Searches for intervals overlapping with the given tuple.

        Args:
            interval: The interval tuple (start, end).

        Returns:
            A list of data associated with overlapping intervals.

        Raises:
            If the operation fails.
        """
        return self.search(Interval(interval[0], interval[1]))

    def search(self, interval: Interval[Self.T]) raises -> List[Self.U]:
        """Searches for intervals overlapping with the given interval.

        Args:
            interval: The interval to search.

        Returns:
            A list of data associated with overlapping intervals.

        Raises:
            If the operation fails.
        """
        return self._search_helper(self._root, interval)

    def _search_helper(
        self, node: Self._IntervalNodePointer, interval: Interval[Self.T]
    ) raises -> List[Self.U]:
        var result = List[Self.U]()
        var work_list = Deque[Self._IntervalNodePointer]()
        work_list.append(node)

        while work_list:
            var current_node = work_list.pop()
            if not current_node:
                continue
            if current_node.value()[].interval.overlaps(interval):
                result.append(current_node.value()[].data.copy())
            if (
                current_node.value()[].left()
                and current_node.value()[].left().value()[].interval.start
                <= interval.end
            ):
                work_list.append(current_node.value()[].left())
            if (
                current_node.value()[].right()
                and current_node.value()[].right().value()[].max_end
                >= interval.start
            ):
                work_list.append(current_node.value()[].right())

        return result^
