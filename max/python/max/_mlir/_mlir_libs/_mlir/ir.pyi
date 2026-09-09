# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

"""MLIR IR Bindings"""

import enum
import types
from collections.abc import Callable, Iterator, Sequence
from typing import ClassVar, Final, Generic, Self, TypeVar, overload

import _mlir
from typing_extensions import Buffer

class DiagnosticSeverity(enum.Enum):
    ERROR = 0

    WARNING = 1

    NOTE = 2

    REMARK = 3

class WalkOrder(enum.Enum):
    PRE_ORDER = 0

    POST_ORDER = 1

class OperationEquivalenceFlags(enum.IntFlag):
    __str__ = __repr__

    def __repr__(self, /):
        """Return repr(self)."""

    NONE = 0

    IGNORE_LOCATIONS = 1

    IGNORE_DISCARDABLE_ATTRS = 2

    IGNORE_PROPERTIES = 4

    IGNORE_COMMUTATIVITY = 8

class WalkResult(enum.Enum):
    ADVANCE = 0

    INTERRUPT = 1

    SKIP = 2

class Diagnostic:
    @property
    def severity(self) -> DiagnosticSeverity:
        """Returns the severity of the diagnostic."""

    @property
    def location(self) -> Location:
        """Returns the location associated with the diagnostic."""

    @property
    def message(self) -> str:
        """Returns the message text of the diagnostic."""

    @property
    def notes(self) -> tuple[Diagnostic]:
        """Returns a tuple of attached note diagnostics."""

class DiagnosticInfo:
    def __init__(self, diag: Diagnostic) -> None:
        """Creates a DiagnosticInfo from a Diagnostic."""

    @property
    def severity(self) -> DiagnosticSeverity:
        """The severity level of the diagnostic."""

    @property
    def location(self) -> Location:
        """The location associated with the diagnostic."""

    @property
    def message(self) -> str:
        """The message text of the diagnostic."""

    @property
    def notes(self) -> list[DiagnosticInfo]:
        """List of attached note diagnostics."""

class DiagnosticHandler:
    def detach(self) -> None:
        """Detaches the diagnostic handler from the context."""

    @property
    def attached(self) -> bool:
        """Returns True if the handler is attached to a context."""

    @property
    def had_error(self) -> bool:
        """Returns True if an error was encountered during diagnostic handling."""

    def __enter__(self, /) -> DiagnosticHandler:
        """Enters the diagnostic handler as a context manager."""

    def __exit__(
        self,
        exc_type: object | None,
        exc_value: object | None,
        traceback: object | None,
    ) -> None:
        """Exits the diagnostic handler context manager."""

class ThreadPool:
    def __init__(self) -> None:
        """Creates a new thread pool with default concurrency."""

    def get_max_concurrency(self) -> int:
        """Returns the maximum number of threads in the pool."""

class Context:
    def __init__(self) -> None:
        """
        Creates a new MLIR context.

        The context is the top-level container for all MLIR objects. It owns the storage
        for types, attributes, locations, and other core IR objects. A context can be
        configured to allow or disallow unregistered dialects and can have dialects
        loaded on-demand.
        """

    def __enter__(self, /) -> Context:
        """Enters the context as a context manager."""

    def __exit__(
        self,
        exc_type: object | None,
        exc_value: object | None,
        traceback: object | None,
    ) -> None:
        """Exits the context manager."""

    current: ClassVar[Final[Context | None]] = ...
    """
    Gets the Context bound to the current thread or returns None if no context is set.
    """

    @property
    def dialects(self) -> Dialects:
        """Gets a container for accessing dialects by name."""

    @property
    def d(self) -> Dialects:
        """Alias for `dialects`."""

    def get_dialect_descriptor(self, dialect_name: str) -> DialectDescriptor:
        """Gets or loads a dialect by name, returning its descriptor object."""

    def is_dialect_loaded(self, dialect_name: str) -> bool:
        """Checks if a dialect is loaded in the context."""

    @property
    def allow_unregistered_dialects(self) -> bool:
        """Controls whether unregistered dialects are allowed in this context."""

    @allow_unregistered_dialects.setter
    def allow_unregistered_dialects(self, arg: bool, /) -> None: ...
    def attach_diagnostic_handler(self, callback: object) -> object:
        """Attaches a diagnostic handler that will receive callbacks."""

    def enable_multithreading(self, enable: bool) -> None:
        """
        Enables or disables multi-threading support in the context.

        Args:
          enable: Whether to enable (True) or disable (False) multi-threading.
        """

    def set_thread_pool(self, arg: ThreadPool, /) -> None:
        """
        Sets a custom thread pool for the context to use.

        Args:
          pool: A ThreadPool object to use for parallel operations.

        Note:
          Multi-threading is automatically disabled before setting the thread pool.
        """

    def get_num_threads(self) -> int:
        """Gets the number of threads in the context's thread pool."""

    def is_registered_operation(self, operation_name: str) -> bool:
        """
        Checks whether an operation with the given name is registered.

        Args:
          operation_name: The fully qualified name of the operation (e.g., `arith.addf`).

        Returns:
          True if the operation is registered, False otherwise.
        """

    def append_dialect_registry(self, registry: DialectRegistry) -> None:
        """
        Appends the contents of a dialect registry to the context.

        Args:
          registry: A DialectRegistry containing dialects to append.
        """

    @property
    def emit_error_diagnostics(self) -> bool:
        """
        Controls whether error diagnostics are emitted to diagnostic handlers.

        By default, error diagnostics are captured and reported through MLIRError exceptions.
        """

    @emit_error_diagnostics.setter
    def emit_error_diagnostics(self, arg: bool, /) -> None: ...
    def load_all_available_dialects(self) -> None:
        """
        Loads all dialects available in the registry into the context.

        This eagerly loads all dialects that have been registered, making them
        immediately available for use.
        """

    def begin_transient_scope(self) -> None:
        """
        Begins a transient scope on the context, freezing the base layer.

        All subsequently allocated types, attributes, and unregistered operations
        are treated as transient and will be deallocated with end_transient_scope().
        Raises a ValueError if the context is already in a transient scope.
        """

    def end_transient_scope(self) -> None:
        """
        Ends the transient scope and resets the context to the base state.

        Prunes all transient types, attributes, affine expressions, distinct
        attributes, and unregistered operations added during the transient scope.

        Note: Any Python objects referencing transient IR entities become invalid
        after this call and must not be accessed.
        """

    @property
    def is_in_transient_scope(self) -> bool:
        """Returns whether the context is currently in a transient scope."""

class DialectDescriptor:
    @property
    def namespace(self) -> str:
        """Returns the namespace of the dialect."""

class Dialects:
    def __getitem__(self, arg: str, /) -> object:
        """Gets a dialect by name using subscript notation."""

    def __getattr__(self, arg: str, /) -> object:
        """Gets a dialect by name using attribute notation."""

class Dialect:
    def __init__(self, descriptor: object) -> None:
        """Creates a Dialect from a DialectDescriptor."""

    @property
    def descriptor(self) -> object:
        """Returns the DialectDescriptor for this dialect."""

class DialectRegistry:
    def __init__(self) -> None:
        """Creates a new empty dialect registry."""

class Location:
    def __enter__(self, /) -> Location:
        """Enters the location as a context manager."""

    def __exit__(
        self,
        exc_type: object | None,
        exc_value: object | None,
        traceback: object | None,
    ) -> None:
        """Exits the location context manager."""

    @overload
    def __eq__(self, arg: Location, /) -> bool:
        """Compares two locations for equality."""

    @overload
    def __eq__(self, arg: object, /) -> bool:
        """Compares location with non-location object (always returns False)."""

    current: ClassVar[Final[Location | None]] = ...
    """Gets the Location bound to the current thread or raises ValueError."""

    @staticmethod
    def from_attr(
        attribute: Attribute, context: _mlir.ir.Context | None = None
    ) -> Location:
        """Gets a Location from a `LocationAttr`."""

    @staticmethod
    def unknown(context: _mlir.ir.Context | None = None) -> UnknownLoc:
        """Alias for `UnknownLoc.get()`."""

    @overload
    @staticmethod
    def file(
        filename: str,
        line: int,
        col: int,
        context: _mlir.ir.Context | None = None,
    ) -> FileLineColLoc:
        """Alias for `FileLineColLoc.get()`."""

    @overload
    @staticmethod
    def file(
        filename: str,
        start_line: int,
        start_col: int,
        end_line: int,
        end_col: int,
        context: _mlir.ir.Context | None = None,
    ) -> FileLineColLoc:
        """Alias for `FileLineColLoc.get()` over a range."""

    @staticmethod
    def name(
        name: str,
        childLoc: Location | None = None,
        context: _mlir.ir.Context | None = None,
    ) -> NameLoc:
        """Alias for `NameLoc.get()`."""

    @staticmethod
    def callsite(
        callee: Location,
        frames: Sequence[Location],
        context: _mlir.ir.Context | None = None,
    ) -> CallSiteLoc:
        """Alias for `CallSiteLoc.get()`."""

    @staticmethod
    def fused(
        locations: Sequence[Location],
        metadata: Attribute | None = None,
        context: _mlir.ir.Context | None = None,
    ) -> Location:
        """Alias for `FusedLoc.get()` (may collapse to a non-fused location)."""

    @property
    def context(self) -> Context:
        """Context that owns the `Location`."""

    @property
    def attr(self) -> Attribute:
        """Get the underlying `LocationAttr`."""

    @property
    def typeid(self) -> TypeID:
        """Gets the `TypeID` of the underlying LocationAttr."""

    def emit_error(self, message: str) -> None:
        """
        Emits an error diagnostic at this location.

        Args:
          message: The error message to emit.
        """

class UnknownLoc(Location):
    def __init__(self, cast_from_loc: Location) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """
    (arg: object, /) -> mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyTypeID
    """

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> UnknownLoc:
        """Gets a Location representing an unknown location."""

class FileLineColLoc(Location):
    def __init__(self, cast_from_loc: Location) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """
    (arg: object, /) -> mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyTypeID
    """

    @overload
    @staticmethod
    def get(
        filename: str,
        line: int,
        col: int,
        context: _mlir.ir.Context | None = None,
    ) -> FileLineColLoc:
        """Gets a FileLineColLoc for a file, line, and column."""

    @overload
    @staticmethod
    def get(
        filename: str,
        start_line: int,
        start_col: int,
        end_line: int,
        end_col: int,
        context: _mlir.ir.Context | None = None,
    ) -> FileLineColLoc:
        """Gets a FileLineColLoc spanning a file and line/column range."""

    @property
    def filename(self) -> str:
        """Gets the filename from a `FileLineColLoc`."""

    @property
    def start_line(self) -> int:
        """Gets the start line number from a `FileLineColLoc`."""

    @property
    def start_col(self) -> int:
        """Gets the start column number from a `FileLineColLoc`."""

    @property
    def end_line(self) -> int:
        """Gets the end line number from a `FileLineColLoc`."""

    @property
    def end_col(self) -> int:
        """Gets the end column number from a `FileLineColLoc`."""

class NameLoc(Location):
    def __init__(self, cast_from_loc: Location) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """
    (arg: object, /) -> mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyTypeID
    """

    @staticmethod
    def get(
        name: str,
        child_loc: Location | None = None,
        context: _mlir.ir.Context | None = None,
    ) -> NameLoc:
        """Gets a NameLoc with an optional child location."""

    @property
    def name_str(self) -> str:
        """Gets the name string from a `NameLoc`."""

    @property
    def child_loc(self) -> Location:
        """Gets the child location from a `NameLoc`."""

class CallSiteLoc(Location):
    def __init__(self, cast_from_loc: Location) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """
    (arg: object, /) -> mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyTypeID
    """

    @staticmethod
    def get(
        callee: Location,
        frames: Sequence[Location],
        context: _mlir.ir.Context | None = None,
    ) -> CallSiteLoc:
        """Gets a CallSiteLoc chaining a callee and one or more caller frames."""

    @property
    def callee(self) -> Location:
        """Gets the callee location from a `CallSiteLoc`."""

    @property
    def caller(self) -> Location:
        """Gets the caller location from a `CallSiteLoc`."""

class FusedLoc(Location):
    def __init__(self, cast_from_loc: Location) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """
    (arg: object, /) -> mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyTypeID
    """

    @staticmethod
    def get(
        locations: Sequence[Location],
        metadata: Attribute | None = None,
        context: _mlir.ir.Context | None = None,
    ) -> FusedLoc:
        """
        Gets a FusedLoc from an array of locations and optional metadata. Raises if the fuse would collapse to a non-fused location; use `Location.fused(...)` for the permissive variant.
        """

    @property
    def locations(self) -> list[object]:
        """Gets the list of locations from a `FusedLoc`."""

    @property
    def metadata(self) -> Attribute | None:
        """Gets the metadata attribute from a `FusedLoc`, or None if absent."""

class Module:
    @overload
    @staticmethod
    def parse(asm: str, context: _mlir.ir.Context | None = None) -> Module: ...
    @overload
    @staticmethod
    def parse(asm: bytes, context: _mlir.ir.Context | None = None) -> Module:
        """
        Parses a module's assembly format from a string.

        Returns a new MlirModule or raises an MLIRError if the parsing fails.

        See also: https://mlir.llvm.org/docs/LangRef/
        """

    @staticmethod
    def parseFile(path: str, context: _mlir.ir.Context | None = None) -> Module:
        """
        Parses a module's assembly format from a string.

        Returns a new MlirModule or raises an MLIRError if the parsing fails.

        See also: https://mlir.llvm.org/docs/LangRef/
        """

    @staticmethod
    def create(loc: Location | None = None) -> Module:
        """Creates an empty module."""

    @property
    def context(self) -> Context:
        """Context that created the `Module`."""

    @property
    def operation(self) -> Operation:
        """Accesses the module as an operation."""

    @property
    def body(self) -> Block:
        """Return the block for this module."""

    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    def __eq__(self, other: Module) -> bool:
        """Compares two modules for equality."""

    def __hash__(self) -> int:
        """Returns the hash value of the module."""

class Operation(_OperationBase):
    @staticmethod
    def create(
        name: str,
        results: Sequence[Type] | None = None,
        operands: Sequence[Value] | None = None,
        attributes: dict[str, Attribute] | None = None,
        successors: Sequence[Block] | None = None,
        regions: int = 0,
        loc: Location | None = None,
        ip: object | None = None,
        infer_type: bool = False,
    ) -> Operation:
        """
        Creates a new operation.

        Args:
          name: Operation name (e.g. `dialect.operation`).
          results: Optional sequence of Type representing op result types.
          operands: Optional operands of the operation.
          attributes: Optional Dict of {str: Attribute}.
          successors: Optional List of Block for the operation's successors.
          regions: Number of regions to create (default = 0).
          location: Optional Location object (defaults to resolve from context manager).
          ip: Optional InsertionPoint (defaults to resolve from context manager or set to False to disable insertion, even with an insertion point set in the context manager).
          infer_type: Whether to infer result types (default = False).
        Returns:
          A new detached Operation object. Detached operations can be added to blocks, which causes them to become attached.
        """

    @staticmethod
    def parse(
        source: str,
        *,
        source_name: str = "",
        context: _mlir.ir.Context | None = None,
    ) -> OpView:
        """
        Parses an operation. Supports both text assembly format and binary bytecode format.
        """

    @property
    def operation(self) -> Operation:
        """Returns self (the operation)."""

    @property
    def opview(self) -> OpView:
        """
        Returns an OpView of this operation.

        Note:
          If the operation has a registered and loaded dialect then this OpView will
          be concrete wrapper class.
        """

    @property
    def block(self) -> Block:
        """Returns the block containing this operation."""

    @property
    def successors(self) -> OpSuccessors:
        """Returns the list of Operation successors."""

    def replace_uses_of_with(self, of: Value, with_: Value) -> None:
        """
        Replaces uses of the 'of' value with the 'with' value inside the operation.
        """

class OpView(_OperationBase):
    @overload
    def __init__(self, operation: Operation) -> None: ...
    @overload
    def __init__(
        self,
        name: str,
        opRegionSpec: tuple[int, bool],
        operandSegmentSpecObj: object | None = None,
        resultSegmentSpecObj: object | None = None,
        results: Sequence | None = None,
        operands: Sequence | None = None,
        attributes: dict[str, Attribute] | None = None,
        successors: Sequence[Block] | None = None,
        regions: int | None = None,
        loc: Location | None = None,
        ip: object | None = None,
    ) -> None: ...
    @property
    def operation(self) -> Operation: ...
    @property
    def opview(self) -> OpView: ...
    @property
    def successors(self) -> OpSuccessors:
        """Returns the list of Operation successors."""

    @classmethod
    def build_generic(
        cls,
        results: Sequence[Type] | None = None,
        operands: Sequence[Value] | None = None,
        attributes: dict[str, Attribute] | None = None,
        successors: Sequence[Block] | None = None,
        regions: int | None = None,
        loc: Location | None = None,
        ip: InsertionPoint | None = None,
    ) -> Self:
        """Builds a specific, generated OpView based on class level attributes."""

    @classmethod
    def parse(
        cls,
        source: str,
        *,
        source_name: str = "",
        context: Context | None = None,
    ) -> Self:
        """Parses a specific, generated OpView based on class level attributes."""

    @classmethod
    def has_trait(
        cls, trait_cls: type, context: _mlir.ir.Context | None = None
    ) -> bool:
        """Checks if the operation has a given trait."""

class OpAdaptor:
    @overload
    def __init__(
        self, operands: list[Value], attributes: OpAttributeMap
    ) -> None:
        """Creates an OpAdaptor with the given operands and attributes."""

    @overload
    def __init__(self, operands: list[Value], opview: OpView) -> None:
        """Creates an OpAdaptor with the given operands and operation view."""

    @property
    def operands(self) -> list:
        """Returns the operands of the adaptor."""

    @property
    def attributes(self) -> OpAttributeMap:
        """Returns the attributes of the adaptor."""

class Region:
    @property
    def blocks(self) -> BlockList:
        """Returns a forward-optimized sequence of blocks."""

    @property
    def owner(self) -> OpView:
        """Returns the operation owning this region."""

    def __iter__(self) -> BlockIterator:
        """Iterates over blocks in the region."""

    @overload
    def __eq__(self, arg: Region, /) -> bool:
        """Compares two regions for pointer equality."""

    @overload
    def __eq__(self, arg: object, /) -> bool:
        """Compares region with non-region object (always returns False)."""

class Block:
    @property
    def owner(self) -> OpView:
        """Returns the owning operation of this block."""

    @property
    def region(self) -> Region:
        """Returns the owning region of this block."""

    @property
    def arguments(self) -> BlockArgumentList:
        """Returns a list of block arguments."""

    def add_argument(self, type: Type, loc: Location) -> BlockArgument:
        """
        Appends an argument of the specified type to the block.

        Args:
          type: The type of the argument to add.
          loc: The source location for the argument.

        Returns:
          The newly added block argument.
        """

    def erase_argument(self, index: int) -> None:
        """
        Erases the argument at the specified index.

        Args:
          index: The index of the argument to erase.
        """

    @property
    def operations(self) -> OperationList:
        """Returns a forward-optimized sequence of operations."""

    @staticmethod
    def create_at_start(
        parent: Region,
        arg_types: Sequence[Type] = [],
        arg_locs: Sequence[Location] | None = None,
    ) -> Block:
        """
        Creates and returns a new Block at the beginning of the given region (with given argument types and locations).
        """

    def append_to(self, region: Region) -> None:
        """
        Appends this block to a region.

        Transfers ownership if the block is currently owned by another region.

        Args:
          region: The region to append the block to.
        """

    def create_before(
        self, *arg_types, arg_locs: Sequence[Location] | None = None
    ) -> Block:
        """
        Creates and returns a new Block before this block (with given argument types and locations).
        """

    def create_after(
        self, *arg_types, arg_locs: Sequence[Location] | None = None
    ) -> Block:
        """
        Creates and returns a new Block after this block (with given argument types and locations).
        """

    def __iter__(self) -> OperationIterator:
        """Iterates over operations in the block."""

    @overload
    def __eq__(self, arg: Block, /) -> bool:
        """Compares two blocks for pointer equality."""

    @overload
    def __eq__(self, arg: object, /) -> bool:
        """Compares block with non-block object (always returns False)."""

    def __hash__(self) -> int:
        """Returns the hash value of the block."""

    def append(self, operation: _OperationBase) -> None:
        """
        Appends an operation to this block.

        If the operation is currently in another block, it will be moved.

        Args:
          operation: The operation to append to the block.
        """

    @property
    def successors(self) -> BlockSuccessors:
        """Returns the list of Block successors."""

    @property
    def predecessors(self) -> BlockPredecessors:
        """Returns the list of Block predecessors."""

class InsertionPoint:
    @overload
    def __init__(self, block: Block) -> None:
        """Inserts after the last operation but still inside the block."""

    @overload
    def __init__(self, beforeOperation: _OperationBase) -> None:
        """Inserts before a referenced operation."""

    def __enter__(self, /) -> InsertionPoint:
        """Enters the insertion point as a context manager."""

    def __exit__(
        self,
        exc_type: object | None,
        exc_value: object | None,
        traceback: object | None,
    ) -> None:
        """Exits the insertion point context manager."""

    current: ClassVar[Final[InsertionPoint]] = ...
    """
    Gets the InsertionPoint bound to the current thread or raises ValueError if none has been set.
    """

    @staticmethod
    def at_block_begin(block: Block) -> InsertionPoint:
        """
        Creates an insertion point at the beginning of a block.

        Args:
          block: The block at whose beginning operations should be inserted.

        Returns:
          An InsertionPoint at the block's beginning.
        """

    @staticmethod
    def at_block_terminator(block: Block) -> InsertionPoint:
        """
        Creates an insertion point before a block's terminator.

        Args:
          block: The block whose terminator to insert before.

        Returns:
          An InsertionPoint before the terminator.

        Raises:
          ValueError: If the block has no terminator.
        """

    @staticmethod
    def after(operation: _OperationBase) -> InsertionPoint:
        """
        Creates an insertion point immediately after an operation.

        Args:
          operation: The operation after which to insert.

        Returns:
          An InsertionPoint after the operation.
        """

    def insert(self, operation: _OperationBase) -> None:
        """
        Inserts an operation at this insertion point.

        Args:
          operation: The operation to insert.
        """

    @property
    def block(self) -> Block:
        """Returns the block that this `InsertionPoint` points to."""

    @property
    def ref_operation(self) -> Operation | None:
        """
        The reference operation before which new operations are inserted, or None if the insertion point is at the end of the block.
        """

class Attribute:
    def __init__(self, cast_from_type: Attribute) -> None:
        """Casts the passed attribute to the generic `Attribute`."""

    @staticmethod
    def parse(asm: str, context: _mlir.ir.Context | None = None) -> Attribute:
        """
        Parses an attribute from an assembly form. Raises an `MLIRError` on failure.
        """

    @property
    def context(self) -> Context:
        """Context that owns the `Attribute`."""

    @property
    def type(self) -> Type:
        """Returns the type of the `Attribute`."""

    def get_named(self, arg: str, /) -> NamedAttribute:
        """
        Binds a name to the attribute, creating a `NamedAttribute`.

        Args:
          name: The name to bind to the `Attribute`.

        Returns:
          A `NamedAttribute` with the given name and this attribute.
        """

    @overload
    def __eq__(self, arg: Attribute, /) -> bool:
        """Compares two attributes for equality."""

    @overload
    def __eq__(self, arg: object, /) -> bool:
        """Compares attribute with non-attribute object (always returns False)."""

    def __hash__(self) -> int:
        """Returns the hash value of the attribute."""

    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    @property
    def typeid(self) -> TypeID:
        """Returns the `TypeID` of the attribute."""

    def maybe_downcast(self) -> Attribute:
        """Downcasts the attribute to a more specific attribute if possible."""

class NamedAttribute:
    @property
    def name(self) -> str:
        """The name of the `NamedAttribute` binding."""

    @property
    def attr(self) -> Attribute:
        """The underlying generic attribute of the `NamedAttribute` binding."""

class Type:
    def __init__(self, cast_from_type: Type) -> None:
        """Casts the passed type to the generic `Type`."""

    @staticmethod
    def parse(asm: str, context: _mlir.ir.Context | None = None) -> Type:
        """
        Parses the assembly form of a type.

        Returns a Type object or raises an `MLIRError` if the type cannot be parsed.

        See also: https://mlir.llvm.org/docs/LangRef/#type-system
        """

    @property
    def context(self) -> Context:
        """Context that owns the `Type`."""

    @overload
    def __eq__(self, arg: Type, /) -> bool:
        """Compares two types for equality."""

    @overload
    def __eq__(self, other: object | None) -> bool:
        """Compares type with non-type object (always returns False)."""

    def __hash__(self) -> int:
        """Returns the hash value of the `Type`."""

    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    def maybe_downcast(self) -> Type:
        """Downcasts the Type to a more specific `Type` if possible."""

    @property
    def typeid(self) -> TypeID:
        """
        Returns the `TypeID` of the `Type`, or raises `ValueError` if `Type` has no `TypeID`.
        """

class TypeID:
    @overload
    def __eq__(self, arg: TypeID, /) -> bool:
        """Compares two `TypeID`s for equality."""

    @overload
    def __eq__(self, arg: object, /) -> bool:
        """Compares TypeID with non-TypeID object (always returns False)."""

    def __hash__(self) -> int:
        """Returns the hash value of the `TypeID`."""

_T = TypeVar("_T", bound=Type)

class Value(Generic[_T]):
    def __init__(self, value: Value) -> None:
        """Creates a Value reference from another `Value`."""

    @property
    def context(self) -> Context:
        """Context in which the value lives."""

    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    @property
    def owner(self) -> OpView | Block:
        """
        Returns the owner of the value (`Operation` for results, `Block` for arguments).
        """

    @property
    def uses(self) -> OpOperandIterator:
        """Returns an iterator over uses of this value."""

    @overload
    def __eq__(self, arg: Value, /) -> bool:
        """Compares two values for pointer equality."""

    @overload
    def __eq__(self, arg: object, /) -> bool:
        """Compares value with non-value object (always returns False)."""

    def __hash__(self) -> int:
        """Returns the hash value of the value."""

    @overload
    def get_name(
        self,
        use_local_scope: bool = False,
        use_name_loc_as_prefix: bool = False,
    ) -> str:
        """
        Returns the string form of value as an operand.

        Args:
          use_local_scope: Whether to use local scope for naming.
          use_name_loc_as_prefix: Whether to use the location attribute (NameLoc) as prefix.

        Returns:
          The value's name as it appears in IR (e.g., `%0`, `%arg0`).
        """

    @overload
    def get_name(self, state: AsmState) -> str:
        """Returns the string form of value as an operand (i.e., the ValueID)."""

    @property
    def type(self) -> _T:
        """Returns the type of the value."""

    def set_type(self, type: _T):
        """Sets the type of the value."""

    def replace_all_uses_with(self, arg: Value, /) -> None:
        """
        Replace all uses of value with the new value, updating anything in the IR that uses `self` to use the other value instead.
        """

    @overload
    def replace_all_uses_except(
        self, with_: Value, exceptions: Operation
    ) -> None: ...
    @overload
    def replace_all_uses_except(
        self, with_: Value, exceptions: Sequence[Operation]
    ) -> None:
        """
        Replace all uses of this value with the `with` value, except for those
        in `exceptions`. `exceptions` can be either a single operation or a list of
        operations.
        """

    def maybe_downcast(self) -> BlockArgument | OpResult | Value:
        """Downcasts the `Value` to a more specific kind if possible."""

    @property
    def location(self) -> Location:
        """Returns the source location of the value."""

class BlockArgument(Value[_T]):
    def __init__(self, value: Value) -> None: ...
    def maybe_downcast(self) -> BlockArgument: ...
    @property
    def owner(self) -> Block:
        """Returns the block that owns this argument."""

    @property
    def arg_number(self) -> int:
        """Returns the position of this argument in the block's argument list."""

    def set_type(self, type: Type) -> None:
        """Sets the type of this block argument."""

    def set_location(self, loc: Location) -> None:
        """Sets the location of this block argument."""

class OpResult(Value[_T]):
    def __init__(self, value: Value) -> None: ...
    def maybe_downcast(self) -> OpResult: ...
    @property
    def owner(self) -> OpView:
        """Returns the operation that produces this result."""

    @property
    def result_number(self) -> int:
        """Returns the position of this result in the operation's result list."""

class OpOperand:
    @property
    def owner(self) -> OpView:
        """Returns the operation that owns this operand."""

    @property
    def operand_number(self) -> int:
        """Returns the operand number in the owning operation."""

class AsmState:
    @overload
    def __init__(self, value: Value, use_local_scope: bool = False) -> None:
        """
        Creates an `AsmState` for consistent SSA value naming.

        Args:
          value: The value to create state for.
          use_local_scope: Whether to use local scope for naming.
        """

    @overload
    def __init__(
        self, op: _OperationBase, use_local_scope: bool = False
    ) -> None:
        """
        Creates an AsmState for consistent SSA value naming.

        Args:
          op: The operation to create state for.
          use_local_scope: Whether to use local scope for naming.
        """

class SymbolTable:
    def __init__(self, arg: _OperationBase, /) -> None:
        """
        Creates a symbol table for an operation.

        Args:
          operation: The `Operation` that defines a symbol table (e.g., a `ModuleOp`).

        Raises:
          TypeError: If the operation is not a symbol table.
        """

    def __getitem__(self, arg: str, /) -> OpView:
        """
        Looks up a symbol by name in the symbol table.

        Args:
          name: The name of the symbol to look up.

        Returns:
          The operation defining the symbol.

        Raises:
          KeyError: If the symbol is not found.
        """

    def insert(self, operation: _OperationBase) -> StringAttr:
        """
        Inserts a symbol operation into the symbol table.

        Args:
          operation: An operation with a symbol name to insert.

        Returns:
          The symbol name attribute of the inserted operation.

        Raises:
          ValueError: If the operation does not have a symbol name.
        """

    def erase(self, operation: _OperationBase) -> None:
        """
        Erases a symbol operation from the symbol table.

        Args:
          operation: The symbol operation to erase.

        Note:
          The operation is also erased from the IR and invalidated.
        """

    def __delitem__(self, arg: str, /) -> None:
        """Deletes a symbol by name from the symbol table."""

    def __contains__(self, arg: str, /) -> bool:
        """Checks if a symbol with the given name exists in the table."""

    @staticmethod
    def set_symbol_name(symbol: _OperationBase, name: str) -> None:
        """Sets the symbol name for a symbol operation."""

    @staticmethod
    def get_symbol_name(symbol: _OperationBase) -> StringAttr:
        """Gets the symbol name from a symbol operation."""

    @staticmethod
    def get_visibility(symbol: _OperationBase) -> StringAttr:
        """Gets the visibility attribute of a symbol operation."""

    @staticmethod
    def set_visibility(symbol: _OperationBase, visibility: str) -> None:
        """Sets the visibility attribute of a symbol operation."""

    @staticmethod
    def replace_all_symbol_uses(
        old_symbol: str, new_symbol: str, from_op: _OperationBase
    ) -> None:
        """
        Replaces all uses of a symbol with a new symbol name within the given operation.
        """

    @staticmethod
    def walk_symbol_tables(
        from_op: _OperationBase, all_sym_uses_visible: bool, callback: object
    ) -> None:
        """
        Walks symbol tables starting from an operation with a callback function.
        """

class BlockArgumentList(Sequence[BlockArgument]):
    def __getitem__(self, key, /):
        """Return self[key]."""

    def __len__(self, /):
        """Return len(self)."""

    def __add__(self, arg: BlockArgumentList, /) -> list[BlockArgument]: ...
    @property
    def types(self) -> list[Type]:
        """Returns a list of types for all arguments in this argument list."""

class BlockIterator:
    def __iter__(self) -> BlockIterator:
        """Returns an iterator over the blocks in the operation's region."""

    def __next__(self) -> Block:
        """Returns the next block in the iteration."""

class BlockList:
    def __getitem__(self, arg: int, /) -> Block:
        """Returns the block at the specified index."""

    def __iter__(self) -> BlockIterator:
        """Returns an iterator over blocks in the operation's region."""

    def __len__(self) -> int:
        """Returns the number of blocks in the operation's region."""

    def append(self, *args, arg_locs: Sequence | None = None) -> Block:
        """
        Appends a new block, with argument types as positional args.

        Returns:
          The created block.
        """

class BlockSuccessors(Sequence[Block]):
    def __getitem__(self, key, /):
        """Return self[key]."""

    def __len__(self, /):
        """Return len(self)."""

    def __add__(self, arg: BlockSuccessors, /) -> list[Block]: ...

class BlockPredecessors(Sequence[Block]):
    def __getitem__(self, key, /):
        """Return self[key]."""

    def __len__(self, /):
        """Return len(self)."""

    def __add__(self, arg: BlockPredecessors, /) -> list[Block]: ...

class OperationIterator:
    def __iter__(self) -> OperationIterator:
        """Returns an iterator over the operations in an operation's block."""

    def __next__(self) -> OpView:
        """Returns the next operation in the iteration."""

class OperationList:
    def __getitem__(self, arg: int, /) -> OpView:
        """Returns the operation at the specified index."""

    def __iter__(self) -> OperationIterator:
        """Returns an iterator over operations in the list."""

    def __len__(self) -> int:
        """Returns the number of operations in the list."""

class OpAttributeMap:
    def __contains__(self, name: str) -> bool:
        """Checks if an attribute with the given name exists in the map."""

    def __len__(self) -> int:
        """Returns the number of attributes in the map."""

    @overload
    def __getitem__(self, name: str) -> Attribute:
        """Gets an attribute by name."""

    @overload
    def __getitem__(self, index: int) -> NamedAttribute:
        """Gets a named attribute by index."""

    def __setitem__(self, name: str, attr: Attribute) -> None:
        """Sets an attribute with the given name."""

    def __delitem__(self, name: str) -> None:
        """Deletes an attribute with the given name."""

    def get(self, key: str, default: object | None = None) -> Attribute | None:
        """Gets an attribute by name or the default value, if it does not exist."""

    def __iter__(self) -> Iterator[str]:
        """Iterates over attribute names."""

    def keys(self) -> list[str]:
        """Returns a list of attribute names."""

    def values(self) -> list[Attribute]:
        """Returns a list of attribute values."""

    def items(self) -> list[tuple[str, Attribute]]:
        """Returns a list of `(name, attribute)` tuples."""

class OpOperandIterator:
    def __iter__(self) -> OpOperandIterator:
        """Returns an iterator over operands."""

    def __next__(self) -> OpOperand:
        """Returns the next operand in the iteration."""

class OpOperandList(Sequence[Value], Generic[_T]):
    def __getitem__(self, key, /):
        """Return self[key]."""

    def __len__(self, /):
        """Return len(self)."""

    def __add__(self, arg: OpOperandList, /) -> list[Value]: ...
    def __setitem__(self, index: int, value: Value) -> None:
        """Sets the operand at the specified index to a new value."""

class OpOperands(Sequence[OpOperand]):
    def __getitem__(self, key, /):
        """Return self[key]."""

    def __len__(self, /):
        """Return len(self)."""

    def __add__(self, arg: OpOperands, /) -> list[OpOperand]: ...

class OpResultList(Sequence[OpResult], Generic[_T]):
    def __getitem__(self, key, /):
        """Return self[key]."""

    def __len__(self, /):
        """Return len(self)."""

    def __add__(self, arg: OpResultList, /) -> list[OpResult]: ...
    @property
    def types(self) -> list[Type]:
        """Returns a list of types for all results in this result list."""

    @property
    def owner(self) -> OpView:
        """Returns the operation that owns this result list."""

class OpSuccessors(Sequence[Block]):
    def __getitem__(self, key, /):
        """Return self[key]."""

    def __len__(self, /):
        """Return len(self)."""

    def __add__(self, arg: OpSuccessors, /) -> list[Block]: ...
    def __setitem__(self, index: int, block: Block) -> None:
        """Sets the successor block at the specified index."""

class RegionSequence(Sequence[Region]):
    def __getitem__(self, key, /):
        """Return self[key]."""

    def __len__(self, /):
        """Return len(self)."""

    def __add__(self, arg: RegionSequence, /) -> list[Region]: ...

class AttrBuilder:
    @staticmethod
    def contains(attribute_kind: str) -> bool:
        """
        Checks whether an attribute builder is registered for the given attribute kind.
        """

    @staticmethod
    def get(attribute_kind: str) -> Callable:
        """Gets the registered attribute builder for the given attribute kind."""

    @staticmethod
    def insert(
        attribute_kind: str,
        attr_builder: Callable,
        replace: bool = False,
        allow_existing: bool = False,
    ) -> None:
        """
        Register an attribute builder for building MLIR attributes from Python values.
        """

class DynamicOpTrait:
    @classmethod
    def attach(
        cls,
        op_name: type | str,
        target: object | None = None,
        context: _mlir.ir.Context | None = None,
    ) -> bool:
        """Attach the dynamic op trait subclass to the given operation name."""

class IsTerminatorTrait(DynamicOpTrait):
    @classmethod
    def attach(
        cls, op_name: type | str, context: _mlir.ir.Context | None = None
    ) -> bool:
        """Attach IsTerminator trait to the given operation name."""

class NoTerminatorTrait(DynamicOpTrait):
    @classmethod
    def attach(
        cls, op_name: type | str, context: _mlir.ir.Context | None = None
    ) -> bool:
        """Attach NoTerminator trait to the given operation name."""

class IsIsolatedFromAboveTrait(DynamicOpTrait):
    @classmethod
    def attach(
        cls, op_name: type | str, context: _mlir.ir.Context | None = None
    ) -> bool:
        """Attach IsIsolatedFromAbove trait to the given operation name."""

class RecursiveMemoryEffectsTrait(DynamicOpTrait):
    @classmethod
    def attach(
        cls, op_name: type | str, context: _mlir.ir.Context | None = None
    ) -> bool:
        """Attach RecursiveMemoryEffects trait to the given operation name."""

class MLIRError(Exception):
    @property
    def message(self) -> str: ...
    @property
    def error_diagnostics(self) -> list[DiagnosticInfo]: ...

class AffineExpr:
    @overload
    def __add__(self, arg: AffineExpr, /) -> AffineAddExpr: ...
    @overload
    def __add__(self, arg: int, /) -> AffineAddExpr: ...
    def __radd__(self, arg: int, /) -> AffineAddExpr: ...
    @overload
    def __mul__(self, arg: AffineExpr, /) -> AffineMulExpr: ...
    @overload
    def __mul__(self, arg: int, /) -> AffineMulExpr: ...
    def __rmul__(self, arg: int, /) -> AffineMulExpr: ...
    @overload
    def __mod__(self, arg: AffineExpr, /) -> AffineModExpr: ...
    @overload
    def __mod__(self, arg: int, /) -> AffineModExpr: ...
    def __rmod__(self, arg: int, /) -> AffineModExpr: ...
    @overload
    def __sub__(self, arg: AffineExpr, /) -> AffineAddExpr: ...
    @overload
    def __sub__(self, arg: int, /) -> AffineAddExpr: ...
    def __rsub__(self, arg: int, /) -> AffineAddExpr: ...
    @overload
    def __eq__(self, arg: AffineExpr, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def context(self) -> Context: ...
    def compose(self, arg: AffineMap, /) -> AffineExpr: ...
    def maybe_downcast(self) -> AffineExpr: ...
    def shift_dims(
        self, num_dims: int, shift: int, offset: int = 0
    ) -> AffineExpr: ...
    def shift_symbols(
        self, num_symbols: int, shift: int, offset: int = 0
    ) -> AffineExpr: ...
    @staticmethod
    def simplify_affine_expr(
        expr: AffineExpr, num_dims: int, num_symbols: int
    ) -> AffineExpr:
        """
        Simplify an affine expression by flattening and some amount of simple analysis.
        """

    @overload
    @staticmethod
    def get_add(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineAddExpr:
        """Gets an affine expression containing a sum of two expressions."""

    @overload
    @staticmethod
    def get_add(arg0: int, arg1: AffineExpr, /) -> AffineAddExpr:
        """
        Gets an affine expression containing a sum of a constant and another expression.
        """

    @overload
    @staticmethod
    def get_add(arg0: AffineExpr, arg1: int, /) -> AffineAddExpr:
        """
        Gets an affine expression containing a sum of an expression and a constant.
        """

    @overload
    @staticmethod
    def get_mul(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineMulExpr:
        """Gets an affine expression containing a product of two expressions."""

    @overload
    @staticmethod
    def get_mul(arg0: int, arg1: AffineExpr, /) -> AffineMulExpr:
        """
        Gets an affine expression containing a product of a constant and another expression.
        """

    @overload
    @staticmethod
    def get_mul(arg0: AffineExpr, arg1: int, /) -> AffineMulExpr:
        """
        Gets an affine expression containing a product of an expression and a constant.
        """

    @overload
    @staticmethod
    def get_mod(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineModExpr:
        """
        Gets an affine expression containing the modulo of dividing one expression by another.
        """

    @overload
    @staticmethod
    def get_mod(arg0: int, arg1: AffineExpr, /) -> AffineModExpr:
        """
        Gets a semi-affine expression containing the modulo of dividing a constant by an expression.
        """

    @overload
    @staticmethod
    def get_mod(arg0: AffineExpr, arg1: int, /) -> AffineModExpr:
        """
        Gets an affine expression containing the module of dividingan expression by a constant.
        """

    @overload
    @staticmethod
    def get_floor_div(
        arg0: AffineExpr, arg1: AffineExpr, /
    ) -> AffineFloorDivExpr:
        """
        Gets an affine expression containing the rounded-down result of dividing one expression by another.
        """

    @overload
    @staticmethod
    def get_floor_div(arg0: int, arg1: AffineExpr, /) -> AffineFloorDivExpr:
        """
        Gets a semi-affine expression containing the rounded-down result of dividing a constant by an expression.
        """

    @overload
    @staticmethod
    def get_floor_div(arg0: AffineExpr, arg1: int, /) -> AffineFloorDivExpr:
        """
        Gets an affine expression containing the rounded-down result of dividing an expression by a constant.
        """

    @overload
    @staticmethod
    def get_ceil_div(
        arg0: AffineExpr, arg1: AffineExpr, /
    ) -> AffineCeilDivExpr:
        """
        Gets an affine expression containing the rounded-up result of dividing one expression by another.
        """

    @overload
    @staticmethod
    def get_ceil_div(arg0: int, arg1: AffineExpr, /) -> AffineCeilDivExpr:
        """
        Gets a semi-affine expression containing the rounded-up result of dividing a constant by an expression.
        """

    @overload
    @staticmethod
    def get_ceil_div(arg0: AffineExpr, arg1: int, /) -> AffineCeilDivExpr:
        """
        Gets an affine expression containing the rounded-up result of dividing an expression by a constant.
        """

    @staticmethod
    def get_constant(
        value: int, context: _mlir.ir.Context | None = None
    ) -> AffineConstantExpr:
        """Gets a constant affine expression with the given value."""

    @staticmethod
    def get_dim(
        position: int, context: _mlir.ir.Context | None = None
    ) -> AffineDimExpr:
        """Gets an affine expression of a dimension at the given position."""

    @staticmethod
    def get_symbol(
        position: int, context: _mlir.ir.Context | None = None
    ) -> AffineSymbolExpr:
        """Gets an affine expression of a symbol at the given position."""

    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

class AffineConstantExpr(AffineExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def get(
        value: int, context: _mlir.ir.Context | None = None
    ) -> AffineConstantExpr: ...
    @property
    def value(self) -> int: ...

class AffineDimExpr(AffineExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def get(
        position: int, context: _mlir.ir.Context | None = None
    ) -> AffineDimExpr: ...
    @property
    def position(self) -> int: ...

class AffineSymbolExpr(AffineExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def get(
        position: int, context: _mlir.ir.Context | None = None
    ) -> AffineSymbolExpr: ...
    @property
    def position(self) -> int: ...

class AffineBinaryExpr(AffineExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @property
    def lhs(self) -> AffineExpr: ...
    @property
    def rhs(self) -> AffineExpr: ...

class AffineAddExpr(AffineBinaryExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineAddExpr: ...

class AffineMulExpr(AffineBinaryExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineMulExpr: ...

class AffineModExpr(AffineBinaryExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineModExpr: ...

class AffineFloorDivExpr(AffineBinaryExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineFloorDivExpr: ...

class AffineCeilDivExpr(AffineBinaryExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineCeilDivExpr: ...

class AffineMap:
    @overload
    def __eq__(self, arg: AffineMap, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    @staticmethod
    def compress_unused_symbols(
        arg0: Sequence[AffineMap], arg1: _mlir.ir.Context, /
    ) -> list[AffineMap]: ...
    @property
    def context(self) -> Context:
        """Context that owns the Affine Map"""

    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    @staticmethod
    def get(
        dim_count: int,
        symbol_count: int,
        exprs: Sequence[AffineExpr],
        context: _mlir.ir.Context | None = None,
    ) -> AffineMap:
        """Gets a map with the given expressions as results."""

    @staticmethod
    def get_constant(
        value: int, context: _mlir.ir.Context | None = None
    ) -> AffineMap:
        """Gets an affine map with a single constant result"""

    @staticmethod
    def get_empty(context: _mlir.ir.Context | None = None) -> AffineMap:
        """Gets an empty affine map."""

    @staticmethod
    def get_identity(
        n_dims: int, context: _mlir.ir.Context | None = None
    ) -> AffineMap:
        """Gets an identity map with the given number of dimensions."""

    @staticmethod
    def get_minor_identity(
        n_dims: int, n_results: int, context: _mlir.ir.Context | None = None
    ) -> AffineMap:
        """
        Gets a minor identity map with the given number of dimensions and results.
        """

    @staticmethod
    def get_permutation(
        permutation: Sequence[int], context: _mlir.ir.Context | None = None
    ) -> AffineMap:
        """Gets an affine map that permutes its inputs."""

    def get_submap(self, result_positions: Sequence[int]) -> AffineMap: ...
    def get_major_submap(self, n_results: int) -> AffineMap: ...
    def get_minor_submap(self, n_results: int) -> AffineMap: ...
    def replace(
        self,
        expr: AffineExpr,
        replacement: AffineExpr,
        n_result_dims: int,
        n_result_syms: int,
    ) -> AffineMap: ...
    @property
    def is_permutation(self) -> bool: ...
    @property
    def is_projected_permutation(self) -> bool: ...
    @property
    def n_dims(self) -> int: ...
    @property
    def n_inputs(self) -> int: ...
    @property
    def n_symbols(self) -> int: ...
    @property
    def results(self) -> AffineExprList: ...

class AffineExprList(Sequence[AffineExpr]):
    def __getitem__(self, key, /):
        """Return self[key]."""

    def __len__(self, /):
        """Return len(self)."""

    def __add__(self, arg: AffineExprList, /) -> list[AffineExpr]: ...

class IntegerSet:
    @overload
    def __eq__(self, arg: IntegerSet, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def context(self) -> Context: ...
    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    @staticmethod
    def get(
        num_dims: int,
        num_symbols: int,
        exprs: Sequence[AffineExpr],
        eq_flags: Sequence[bool],
        context: _mlir.ir.Context | None = None,
    ) -> IntegerSet: ...
    @staticmethod
    def get_empty(
        num_dims: int, num_symbols: int, context: _mlir.ir.Context | None = None
    ) -> IntegerSet: ...
    def get_replaced(
        self,
        dim_exprs: Sequence[AffineExpr],
        symbol_exprs: Sequence[AffineExpr],
        num_result_dims: int,
        num_result_symbols: int,
    ) -> IntegerSet: ...
    @property
    def is_canonical_empty(self) -> bool: ...
    @property
    def n_dims(self) -> int: ...
    @property
    def n_symbols(self) -> int: ...
    @property
    def n_inputs(self) -> int: ...
    @property
    def n_equalities(self) -> int: ...
    @property
    def n_inequalities(self) -> int: ...
    @property
    def constraints(self) -> IntegerSetConstraintList: ...

class IntegerSetConstraint:
    @property
    def expr(self) -> AffineExpr: ...
    @property
    def is_eq(self) -> bool: ...

class IntegerSetConstraintList(Sequence[IntegerSetConstraint]):
    def __getitem__(self, key, /):
        """Return self[key]."""

    def __len__(self, /):
        """Return len(self)."""

    def __add__(
        self, arg: IntegerSetConstraintList, /
    ) -> list[IntegerSetConstraint]: ...

class AffineMapAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(affine_map: AffineMap) -> AffineMapAttr:
        """Gets an attribute wrapping an AffineMap."""

    @property
    def value(self) -> AffineMap:
        """Returns the value of the AffineMap attribute"""

class DenseBoolArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @staticmethod
    def get(
        values: Sequence, context: _mlir.ir.Context | None = None
    ) -> DenseBoolArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseBoolArrayIterator: ...
    def __add__(self, arg: Sequence, /) -> DenseBoolArrayAttr: ...

class DenseBoolArrayIterator:
    def __iter__(self) -> DenseBoolArrayIterator: ...
    def __next__(self) -> bool: ...

class DenseI8ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @staticmethod
    def get(
        values: Sequence[int], context: _mlir.ir.Context | None = None
    ) -> DenseI8ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseI8ArrayIterator: ...
    def __add__(self, arg: Sequence, /) -> DenseI8ArrayAttr: ...

class DenseI8ArrayIterator:
    def __iter__(self) -> DenseI8ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseI16ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @staticmethod
    def get(
        values: Sequence[int], context: _mlir.ir.Context | None = None
    ) -> DenseI16ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseI16ArrayIterator: ...
    def __add__(self, arg: Sequence, /) -> DenseI16ArrayAttr: ...

class DenseI16ArrayIterator:
    def __iter__(self) -> DenseI16ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseI32ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @staticmethod
    def get(
        values: Sequence[int], context: _mlir.ir.Context | None = None
    ) -> DenseI32ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseI32ArrayIterator: ...
    def __add__(self, arg: Sequence, /) -> DenseI32ArrayAttr: ...

class DenseI32ArrayIterator:
    def __iter__(self) -> DenseI32ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseI64ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @staticmethod
    def get(
        values: Sequence[int], context: _mlir.ir.Context | None = None
    ) -> DenseI64ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseI64ArrayIterator: ...
    def __add__(self, arg: Sequence, /) -> DenseI64ArrayAttr: ...

class DenseI64ArrayIterator:
    def __iter__(self) -> DenseI64ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseF32ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @staticmethod
    def get(
        values: Sequence[float], context: _mlir.ir.Context | None = None
    ) -> DenseF32ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> float: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseF32ArrayIterator: ...
    def __add__(self, arg: Sequence, /) -> DenseF32ArrayAttr: ...

class DenseF32ArrayIterator:
    def __iter__(self) -> DenseF32ArrayIterator: ...
    def __next__(self) -> float: ...

class DenseF64ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @staticmethod
    def get(
        values: Sequence[float], context: _mlir.ir.Context | None = None
    ) -> DenseF64ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> float: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseF64ArrayIterator: ...
    def __add__(self, arg: Sequence, /) -> DenseF64ArrayAttr: ...

class DenseF64ArrayIterator:
    def __iter__(self) -> DenseF64ArrayIterator: ...
    def __next__(self) -> float: ...

class ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        attributes: Sequence[Attribute], context: _mlir.ir.Context | None = None
    ) -> ArrayAttr:
        """Gets a uniqued Array attribute"""

    def __getitem__(self, arg: int, /) -> Attribute: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> ArrayAttributeIterator: ...
    def __add__(self, arg: Sequence[Attribute], /) -> ArrayAttr: ...

class ArrayAttributeIterator:
    def __iter__(self) -> ArrayAttributeIterator: ...
    def __next__(self) -> Attribute: ...

class BoolAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @staticmethod
    def get(value: bool, context: _mlir.ir.Context | None = None) -> BoolAttr:
        """Gets an uniqued bool attribute"""

    @property
    def value(self) -> bool:
        """Returns the value of the bool attribute"""

    def __bool__(self) -> bool:
        """Converts the value of the bool attribute to a Python bool"""

class DenseElementsAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __buffer__(self, flags, /):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """

    def __release_buffer__(self, buffer, /):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """

    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __len__(self) -> int: ...
    @overload
    @staticmethod
    def get(
        array: Buffer,
        signless: bool = True,
        type: Type | None = None,
        shape: Sequence[int] | None = None,
        context: Context | None = None,
    ) -> DenseElementsAttr:
        """
        Gets a DenseElementsAttr from a Python buffer or array.

        When `type` is not provided, then some limited type inferencing is done based
        on the buffer format. Support presently exists for 8/16/32/64 signed and
        unsigned integers and float16/float32/float64. DenseElementsAttrs of these
        types can also be converted back to a corresponding buffer.

        For conversions outside of these types, a `type=` must be explicitly provided
        and the buffer contents must be bit-castable to the MLIR internal
        representation:

          * Integer types: the buffer must be byte aligned to the next byte boundary.
          * Floating point types: Must be bit-castable to the given floating point
            size.
          * i1 (bool): Each boolean value is stored as a single byte (0 or 1).

        If a single element buffer is passed, then a splat will be created.

        Args:
          array: The array or buffer to convert.
          signless: If inferring an appropriate MLIR type, use signless types for
            integers (defaults True).
          type: Skips inference of the MLIR element type and uses this instead. The
            storage size must be consistent with the actual contents of the buffer.
          shape: Overrides the shape of the buffer when constructing the MLIR
            shaped type. This is needed when the physical and logical shape differ.
          context: Explicit context, if not from context manager.

        Returns:
          DenseElementsAttr on success.

        Raises:
          ValueError: If the type of the buffer or array cannot be matched to an MLIR
            type or if the buffer does not meet expectations.
        """

    @overload
    @staticmethod
    def get(
        attrs: Sequence[Attribute],
        type: Type | None = None,
        context: Context | None = None,
    ) -> DenseElementsAttr:
        """
        Gets a DenseElementsAttr from a Python list of attributes.

        Note that it can be expensive to construct attributes individually.
        For a large number of elements, consider using a Python buffer or array instead.

        Args:
          attrs: A list of attributes.
          type: The desired shape and type of the resulting DenseElementsAttr.
            If not provided, the element type is determined based on the type
            of the 0th attribute and the shape is `[len(attrs)]`.
          context: Explicit context, if not from context manager.

        Returns:
          DenseElementsAttr on success.

        Raises:
          ValueError: If the type of the attributes does not match the type
            specified by `shaped_type`.
        """

    @staticmethod
    def get_splat(
        shaped_type: Type, element_attr: Attribute
    ) -> DenseElementsAttr:
        """Gets a DenseElementsAttr where all values are the same"""

    @property
    def is_splat(self) -> bool: ...
    def get_splat_value(self) -> Attribute: ...

class DenseFPElementsAttr(DenseElementsAttr):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @overload
    @staticmethod
    def get(
        array: Buffer,
        signless: bool = True,
        type: Type | None = None,
        shape: Sequence[int] | None = None,
        context: Context | None = None,
    ) -> DenseFPElementsAttr:
        """
        Gets a DenseElementsAttr from a Python buffer or array.

        When `type` is not provided, then some limited type inferencing is done based
        on the buffer format. Support presently exists for 8/16/32/64 signed and
        unsigned integers and float16/float32/float64. DenseElementsAttrs of these
        types can also be converted back to a corresponding buffer.

        For conversions outside of these types, a `type=` must be explicitly provided
        and the buffer contents must be bit-castable to the MLIR internal
        representation:

          * Integer types: the buffer must be byte aligned to the next byte boundary.
          * Floating point types: Must be bit-castable to the given floating point
            size.
          * i1 (bool): Each boolean value is stored as a single byte (0 or 1).

        If a single element buffer is passed, then a splat will be created.

        Args:
          array: The array or buffer to convert.
          signless: If inferring an appropriate MLIR type, use signless types for
            integers (defaults True).
          type: Skips inference of the MLIR element type and uses this instead. The
            storage size must be consistent with the actual contents of the buffer.
          shape: Overrides the shape of the buffer when constructing the MLIR
            shaped type. This is needed when the physical and logical shape differ.
          context: Explicit context, if not from context manager.

        Returns:
          DenseElementsAttr on success.

        Raises:
          ValueError: If the type of the buffer or array cannot be matched to an MLIR
            type or if the buffer does not meet expectations.
        """

    @overload
    @staticmethod
    def get(
        attrs: Sequence[Attribute],
        type: Type | None = None,
        context: Context | None = None,
    ) -> DenseFPElementsAttr:
        """
        Gets a DenseElementsAttr from a Python list of attributes.

        Note that it can be expensive to construct attributes individually.
        For a large number of elements, consider using a Python buffer or array instead.

        Args:
          attrs: A list of attributes.
          type: The desired shape and type of the resulting DenseElementsAttr.
            If not provided, the element type is determined based on the type
            of the 0th attribute and the shape is `[len(attrs)]`.
          context: Explicit context, if not from context manager.

        Returns:
          DenseElementsAttr on success.

        Raises:
          ValueError: If the type of the attributes does not match the type
            specified by `shaped_type`.
        """

    @staticmethod
    def get_splat(
        shaped_type: Type, element_attr: Attribute
    ) -> DenseFPElementsAttr:
        """Gets a DenseFPElementsAttr where all values are the same"""

    def __getitem__(self, arg: int, /) -> float: ...

class DenseIntElementsAttr(DenseElementsAttr):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @overload
    @staticmethod
    def get(
        array: Buffer,
        signless: bool = True,
        type: Type | None = None,
        shape: Sequence[int] | None = None,
        context: Context | None = None,
    ) -> DenseIntElementsAttr:
        """
        Gets a DenseElementsAttr from a Python buffer or array.

        When `type` is not provided, then some limited type inferencing is done based
        on the buffer format. Support presently exists for 8/16/32/64 signed and
        unsigned integers and float16/float32/float64. DenseElementsAttrs of these
        types can also be converted back to a corresponding buffer.

        For conversions outside of these types, a `type=` must be explicitly provided
        and the buffer contents must be bit-castable to the MLIR internal
        representation:

          * Integer types: the buffer must be byte aligned to the next byte boundary.
          * Floating point types: Must be bit-castable to the given floating point
            size.
          * i1 (bool): Each boolean value is stored as a single byte (0 or 1).

        If a single element buffer is passed, then a splat will be created.

        Args:
          array: The array or buffer to convert.
          signless: If inferring an appropriate MLIR type, use signless types for
            integers (defaults True).
          type: Skips inference of the MLIR element type and uses this instead. The
            storage size must be consistent with the actual contents of the buffer.
          shape: Overrides the shape of the buffer when constructing the MLIR
            shaped type. This is needed when the physical and logical shape differ.
          context: Explicit context, if not from context manager.

        Returns:
          DenseElementsAttr on success.

        Raises:
          ValueError: If the type of the buffer or array cannot be matched to an MLIR
            type or if the buffer does not meet expectations.
        """

    @overload
    @staticmethod
    def get(
        attrs: Sequence[Attribute],
        type: Type | None = None,
        context: Context | None = None,
    ) -> DenseIntElementsAttr:
        """
        Gets a DenseElementsAttr from a Python list of attributes.

        Note that it can be expensive to construct attributes individually.
        For a large number of elements, consider using a Python buffer or array instead.

        Args:
          attrs: A list of attributes.
          type: The desired shape and type of the resulting DenseElementsAttr.
            If not provided, the element type is determined based on the type
            of the 0th attribute and the shape is `[len(attrs)]`.
          context: Explicit context, if not from context manager.

        Returns:
          DenseElementsAttr on success.

        Raises:
          ValueError: If the type of the attributes does not match the type
            specified by `shaped_type`.
        """

    @staticmethod
    def get_splat(
        shaped_type: Type, element_attr: Attribute
    ) -> DenseIntElementsAttr:
        """Gets a DenseIntElementsAttr where all values are the same"""

    def __getitem__(self, arg: int, /) -> int: ...

class DenseResourceElementsAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get_from_buffer(
        array: Buffer,
        name: str,
        type: Type,
        alignment: int | None = None,
        is_mutable: bool = False,
        context: Context | None = None,
    ) -> DenseResourceElementsAttr:
        """
        Gets a DenseResourceElementsAttr from a Python buffer or array.

        This function does minimal validation or massaging of the data, and it is
        up to the caller to ensure that the buffer meets the characteristics
        implied by the shape.

        The backing buffer and any user objects will be retained for the lifetime
        of the resource blob. This is typically bounded to the context but the
        resource can have a shorter lifespan depending on how it is used in
        subsequent processing.

        Args:
          buffer: The array or buffer to convert.
          name: Name to provide to the resource (may be changed upon collision).
          type: The explicit ShapedType to construct the attribute with.
          context: Explicit context, if not from context manager.

        Returns:
          DenseResourceElementsAttr on success.

        Raises:
          ValueError: If the type of the buffer or array cannot be matched to an MLIR
            type or if the buffer does not meet expectations.
        """

class DictAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    def __contains__(self, arg: str, /) -> bool: ...
    def __len__(self) -> int: ...
    @staticmethod
    def get(
        value: dict[str, Attribute] = {},
        context: _mlir.ir.Context | None = None,
    ) -> DictAttr:
        """Gets an uniqued dict attribute"""

    @overload
    def __getitem__(self, arg: str, /) -> Attribute: ...
    @overload
    def __getitem__(self, arg: int, /) -> NamedAttribute: ...

class SymbolRefAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        symbols: Sequence[str], context: _mlir.ir.Context | None = None
    ) -> SymbolRefAttr:
        """Gets a uniqued SymbolRef attribute from a list of symbol names"""

    @property
    def value(self) -> list[str]:
        """Returns the value of the SymbolRef attribute as a list[str]"""

class FlatSymbolRefAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        value: str, context: _mlir.ir.Context | None = None
    ) -> FlatSymbolRefAttr:
        """Gets a uniqued FlatSymbolRef attribute"""

    @property
    def value(self) -> str:
        """Returns the value of the FlatSymbolRef attribute as a string"""

class OpaqueAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        dialect_namespace: str,
        buffer: Buffer,
        type: Type,
        context: Context | None = None,
    ) -> OpaqueAttr:
        """Gets an Opaque attribute."""

    @property
    def dialect_namespace(self) -> str:
        """Returns the dialect namespace for the Opaque attribute as a string"""

    @property
    def data(self) -> bytes:
        """Returns the data for the Opaqued attributes as `bytes`"""

class FloatAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        type: Type, value: float, loc: _mlir.ir.Location | None = None
    ) -> FloatAttr:
        """Gets an uniqued float point attribute associated to a type"""

    @staticmethod
    def get_unchecked(
        type: Type, value: float, context: _mlir.ir.Context | None = None
    ) -> FloatAttr:
        """Gets an uniqued float point attribute associated to a type"""

    @staticmethod
    def get_f32(
        value: float, context: _mlir.ir.Context | None = None
    ) -> FloatAttr:
        """Gets an uniqued float point attribute associated to a f32 type"""

    @staticmethod
    def get_f64(
        value: float, context: _mlir.ir.Context | None = None
    ) -> FloatAttr:
        """Gets an uniqued float point attribute associated to a f64 type"""

    @property
    def value(self) -> float:
        """Returns the value of the float attribute"""

    def __float__(self) -> float:
        """Converts the value of the float attribute to a Python float"""

class IntegerAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(type: Type, value: object) -> IntegerAttr:
        """Gets an uniqued integer attribute associated to a type"""

    @property
    def value(self) -> int:
        """Returns the value of the integer attribute"""

    def __int__(self) -> int:
        """Converts the value of the integer attribute to a Python int"""

    def __index__(self) -> int:
        """Converts the value of the integer attribute to a Python int"""

class IntegerSetAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(integer_set: IntegerSet) -> IntegerSetAttr:
        """Gets an attribute wrapping an IntegerSet."""

class StringAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @overload
    @staticmethod
    def get(
        value: str, context: _mlir.ir.Context | None = None
    ) -> StringAttr: ...
    @overload
    @staticmethod
    def get(
        value: bytes, context: _mlir.ir.Context | None = None
    ) -> StringAttr:
        """Gets a uniqued string attribute"""

    @staticmethod
    def get_typed(type: Type, value: str) -> StringAttr:
        """Gets a uniqued string attribute associated to a type"""

    @property
    def value(self) -> str:
        """Returns the value of the string attribute"""

    @property
    def value_bytes(self) -> bytes:
        """Returns the value of the string attribute as `bytes`"""

class TypeAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(value: Type, context: _mlir.ir.Context | None = None) -> TypeAttr:
        """Gets a uniqued Type attribute"""

    @property
    def value(self) -> Type: ...

class UnitAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> UnitAttr:
        """Create a Unit attribute."""

class StridedLayoutAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    attr_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        offset: int,
        strides: Sequence[int],
        context: _mlir.ir.Context | None = None,
    ) -> StridedLayoutAttr:
        """Gets a strided layout attribute."""

    @staticmethod
    def get_fully_dynamic(
        rank: int, context: _mlir.ir.Context | None = None
    ) -> StridedLayoutAttr:
        """
        Gets a strided layout attribute with dynamic offset and strides of a given rank.
        """

    @property
    def offset(self) -> int:
        """Returns the value of the float point attribute"""

    @property
    def strides(self) -> list[int]:
        """Returns the value of the float point attribute"""

class DynamicAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @property
    def type(self) -> Type: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @staticmethod
    def get(
        full_attr_name: str,
        attributes: Sequence[Attribute],
        context: _mlir.ir.Context | None = None,
    ) -> DynamicAttr:
        """Create a dynamic attribute."""

    @property
    def params(self) -> list[Attribute]:
        """
        Returns the parameters of the dynamic attribute as a list of attributes.
        """

    @property
    def attr_name(self) -> str: ...
    @staticmethod
    def lookup_typeid(
        full_attr_name: str, context: _mlir.ir.Context | None = None
    ) -> TypeID:
        """Look up the TypeID for the given dynamic attribute name."""

class Speculatability(enum.Enum):
    NotSpeculatable = 0

    Speculatable = 1

    RecursivelySpeculatable = 2

class MemoryEffect:
    """A memory effect."""

    def __eq__(self, arg: MemoryEffect, /) -> bool:
        """Compares two memory effects for equality."""

    Allocate: ClassVar[Final[MemoryEffect]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.MemoryEffect"""

    Free: ClassVar[Final[MemoryEffect]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.MemoryEffect"""

    Read: ClassVar[Final[MemoryEffect]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.MemoryEffect"""

    Write: ClassVar[Final[MemoryEffect]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.MemoryEffect"""

class SideEffectResource:
    """A side effect resource."""

    Default: ClassVar[Final[SideEffectResource]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.SideEffectResource"""

class MemoryEffectInstance:
    """A concrete instance of a memory effect."""

    def __init__(
        self,
        effect: MemoryEffect,
        target: OpOperand
        | OpResult
        | BlockArgument
        | SymbolRefAttr
        | FlatSymbolRefAttr
        | None = None,
        *,
        parameters: Attribute | None = None,
        stage: int = 0,
        effect_on_full_region: bool = False,
        resource: SideEffectResource = ...,
    ) -> None:
        """
        Creates a memory effect instance. The target may be an OpOperand, OpResult, BlockArgument, SymbolRefAttr, or None.
        """

    @property
    def effect(self) -> MemoryEffect:
        """Returns the kind of memory effect."""

    @property
    def resource(self) -> SideEffectResource:
        """Returns the affected side effect resource."""

    @property
    def stage(self) -> int:
        """Returns the stage at which the effect occurs."""

    @property
    def effect_on_full_region(self) -> bool:
        """Returns whether the effect applies to the full resource."""

    @property
    def parameters(self) -> Attribute | None:
        """Returns the effect parameters, if any."""

    @property
    def value(self) -> OpResult | BlockArgument | None:
        """Returns the affected value, if any."""

    @property
    def symbol_ref(self) -> SymbolRefAttr | FlatSymbolRefAttr | None:
        """Returns the affected symbol reference, if any."""

class ConditionallySpeculatable:
    def __init__(
        self, object: object, context: _mlir.ir.Context | None = None
    ) -> None:
        """
        Creates an interface from a given operation/opview object or from a subclass of OpView. Raises ValueError if the operation does not implement the interface.
        """

    @property
    def operation(self) -> Operation:
        """Returns an Operation for which the interface was constructed."""

    @property
    def opview(self) -> OpView:
        """
        Returns an OpView subclass _instance_ for which the interface was constructed
        """

    def getSpeculatability(self) -> Speculatability:
        """Returns the speculatability of the given operation."""

    @classmethod
    def attach(
        cls,
        op_name: object,
        *,
        target: object | None = None,
        context: _mlir.ir.Context | None = None,
    ) -> None:
        """Attach the interface subclass to the given operation name."""

class InferShapedTypeOpInterface:
    def __init__(
        self, object: object, context: _mlir.ir.Context | None = None
    ) -> None:
        """
        Creates an interface from a given operation/opview object or from a subclass of OpView. Raises ValueError if the operation does not implement the interface.
        """

    @property
    def operation(self) -> Operation:
        """Returns an Operation for which the interface was constructed."""

    @property
    def opview(self) -> OpView:
        """
        Returns an OpView subclass _instance_ for which the interface was constructed
        """

    def inferReturnTypeComponents(
        self,
        operands: Sequence | None = None,
        attributes: Attribute | None = None,
        regions: types.CapsuleType | None = None,
        properties: Sequence[Region] | None = None,
        context: _mlir.ir.Context | None = None,
        loc: _mlir.ir.Location | None = None,
    ) -> list[ShapedTypeComponents]:
        """
        Given the arguments required to build an operation, attempts to infer
        its return shaped type components. Raises ValueError on failure.
        """

class InferTypeOpInterface:
    def __init__(
        self, object: object, context: _mlir.ir.Context | None = None
    ) -> None:
        """
        Creates an interface from a given operation/opview object or from a subclass of OpView. Raises ValueError if the operation does not implement the interface.
        """

    @property
    def operation(self) -> Operation:
        """Returns an Operation for which the interface was constructed."""

    @property
    def opview(self) -> OpView:
        """
        Returns an OpView subclass _instance_ for which the interface was constructed
        """

    def inferReturnTypes(
        self,
        operands: Sequence | None = None,
        attributes: Attribute | None = None,
        properties: types.CapsuleType | None = None,
        regions: Sequence[Region] | None = None,
        context: _mlir.ir.Context | None = None,
        loc: _mlir.ir.Location | None = None,
    ) -> list[Type]:
        """
        Given the arguments required to build an operation, attempts to infer
        its return types. Raises ValueError on failure.
        """

class MemoryEffectsOpInterface:
    def __init__(
        self, object: object, context: _mlir.ir.Context | None = None
    ) -> None:
        """
        Creates an interface from a given operation/opview object or from a subclass of OpView. Raises ValueError if the operation does not implement the interface.
        """

    @property
    def operation(self) -> Operation:
        """Returns an Operation for which the interface was constructed."""

    @property
    def opview(self) -> OpView:
        """
        Returns an OpView subclass _instance_ for which the interface was constructed
        """

    def get_effects(self) -> list[MemoryEffectInstance]:
        """Returns the memory effects of the operation."""

    @classmethod
    def attach(
        cls,
        op_name: object,
        *,
        target: object | None = None,
        context: _mlir.ir.Context | None = None,
    ) -> None:
        """Attach the interface subclass to the given operation name."""

class ShapedTypeComponents:
    @property
    def element_type(self) -> Type:
        """Returns the element type of the shaped type components."""

    @overload
    @staticmethod
    def get(element_type: Type) -> ShapedTypeComponents:
        """Create an shaped type components object with only the element type."""

    @overload
    @staticmethod
    def get(shape: list[int], element_type: Type) -> ShapedTypeComponents:
        """Create a ranked shaped type components object."""

    @overload
    @staticmethod
    def get(
        shape: list[int], element_type: Type, attribute: Attribute
    ) -> ShapedTypeComponents:
        """Create a ranked shaped type components object with attribute."""

    @property
    def has_rank(self) -> bool:
        """Returns whether the given shaped type component is ranked."""

    @property
    def rank(self) -> int | None:
        """
        Returns the rank of the given ranked shaped type components. If the shaped type components does not have a rank, None is returned.
        """

    @property
    def shape(self) -> list | None:
        """
        Returns the shape of the ranked shaped type components as a list of integers. Returns none if the shaped type component does not have a rank.
        """

class IntegerType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    class Signedness(enum.Enum):
        SIGNLESS = 0

        SIGNED = 1

        UNSIGNED = 2

    SIGNLESS: Signedness = ...

    SIGNED: Signedness = ...

    UNSIGNED: Signedness = ...

    @staticmethod
    def get_signless(
        width: int, context: _mlir.ir.Context | None = None
    ) -> IntegerType:
        """Create a signless integer type"""

    @staticmethod
    def get_signed(
        width: int, context: _mlir.ir.Context | None = None
    ) -> IntegerType:
        """Create a signed integer type"""

    @staticmethod
    def get_unsigned(
        width: int, context: _mlir.ir.Context | None = None
    ) -> IntegerType:
        """Create an unsigned integer type"""

    @staticmethod
    def get(
        width: int,
        signedness: Signedness = Signedness.SIGNLESS,
        context: _mlir.ir.Context | None = None,
    ) -> IntegerType:
        """Create an integer type"""

    @property
    def signedness(self) -> IntegerType.Signedness: ...
    @property
    def width(self) -> int:
        """Returns the width of the integer type"""

    @property
    def is_signless(self) -> bool:
        """Returns whether this is a signless integer"""

    @property
    def is_signed(self) -> bool:
        """Returns whether this is a signed integer"""

    @property
    def is_unsigned(self) -> bool:
        """Returns whether this is an unsigned integer"""

class FloatType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @property
    def width(self) -> int:
        """Returns the width of the floating-point type"""

class IndexType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> IndexType:
        """Create a index type."""

class Float4E2M1FNType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float4E2M1FNType:
        """Create a float4_e2m1fn type."""

class Float6E2M3FNType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float6E2M3FNType:
        """Create a float6_e2m3fn type."""

class Float6E3M2FNType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float6E3M2FNType:
        """Create a float6_e3m2fn type."""

class Float8E4M3FNType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float8E4M3FNType:
        """Create a float8_e4m3fn type."""

class Float8E5M2Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float8E5M2Type:
        """Create a float8_e5m2 type."""

class Float8E4M3Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float8E4M3Type:
        """Create a float8_e4m3 type."""

class Float8E4M3FNUZType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float8E4M3FNUZType:
        """Create a float8_e4m3fnuz type."""

class Float8E4M3B11FNUZType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float8E4M3B11FNUZType:
        """Create a float8_e4m3b11fnuz type."""

class Float8E5M2FNUZType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float8E5M2FNUZType:
        """Create a float8_e5m2fnuz type."""

class Float8E3M4Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float8E3M4Type:
        """Create a float8_e3m4 type."""

class Float8E8M0FNUType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float8E8M0FNUType:
        """Create a float8_e8m0fnu type."""

class Float8E5M3FNUType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> Float8E5M3FNUType:
        """Create a float8_e5m3fnu type."""

class BF16Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> BF16Type:
        """Create a bf16 type."""

class F16Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> F16Type:
        """Create a f16 type."""

class FloatTF32Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> FloatTF32Type:
        """Create a tf32 type."""

class F32Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> F32Type:
        """Create a f32 type."""

class F64Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> F64Type:
        """Create a f64 type."""

class NoneType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: _mlir.ir.Context | None = None) -> NoneType:
        """Create a none type."""

class ComplexType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(arg: Type, /) -> ComplexType:
        """Create a complex type"""

    @property
    def element_type(self) -> Type:
        """Returns element type."""

class ShapedType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @property
    def element_type(self) -> Type:
        """Returns the element type of the shaped type."""

    @property
    def has_rank(self) -> bool:
        """Returns whether the given shaped type is ranked."""

    @property
    def rank(self) -> int:
        """Returns the rank of the given ranked shaped type."""

    @property
    def has_static_shape(self) -> bool:
        """Returns whether the given shaped type has a static shape."""

    def is_dynamic_dim(self, dim: int) -> bool:
        """
        Returns whether the dim-th dimension of the given shaped type is dynamic.
        """

    def is_static_dim(self, dim: int) -> bool:
        """
        Returns whether the dim-th dimension of the given shaped type is static.
        """

    def get_dim_size(self, dim: int) -> int:
        """Returns the dim-th dimension of the given ranked shaped type."""

    @staticmethod
    def is_dynamic_size(dim_size: int) -> bool:
        """
        Returns whether the given dimension size indicates a dynamic dimension.
        """

    @staticmethod
    def is_static_size(dim_size: int) -> bool:
        """Returns whether the given dimension size indicates a static dimension."""

    def is_dynamic_stride_or_offset(self, dim_size: int) -> bool:
        """
        Returns whether the given value is used as a placeholder for dynamic strides and offsets in shaped types.
        """

    def is_static_stride_or_offset(self, dim_size: int) -> bool:
        """
        Returns whether the given shaped type stride or offset value is statically-sized.
        """

    @property
    def shape(self) -> list[int]:
        """Returns the shape of the ranked shaped type as a list of integers."""

    @staticmethod
    def get_dynamic_size() -> int:
        """Returns the value used to indicate dynamic dimensions in shaped types."""

    @staticmethod
    def get_dynamic_stride_or_offset() -> int:
        """
        Returns the value used to indicate dynamic strides or offsets in shaped types.
        """

class VectorType(ShapedType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        shape: Sequence[int],
        element_type: Type,
        *,
        scalable: Sequence | None = None,
        scalable_dims: Sequence[int] | None = None,
        loc: _mlir.ir.Location | None = None,
    ) -> VectorType:
        """Create a vector type"""

    @staticmethod
    def get_unchecked(
        shape: Sequence[int],
        element_type: Type,
        *,
        scalable: Sequence | None = None,
        scalable_dims: Sequence[int] | None = None,
        context: _mlir.ir.Context | None = None,
    ) -> VectorType:
        """Create a vector type"""

    @property
    def scalable(self) -> bool: ...
    @property
    def scalable_dims(self) -> list[bool]: ...

class RankedTensorType(ShapedType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        shape: Sequence[int],
        element_type: Type,
        encoding: Attribute | None = None,
        loc: _mlir.ir.Location | None = None,
    ) -> RankedTensorType:
        """Create a ranked tensor type"""

    @staticmethod
    def get_unchecked(
        shape: Sequence[int],
        element_type: Type,
        encoding: Attribute | None = None,
        context: _mlir.ir.Context | None = None,
    ) -> RankedTensorType:
        """Create a ranked tensor type"""

    @property
    def encoding(self) -> Attribute | None: ...

class UnrankedTensorType(ShapedType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        element_type: Type, loc: _mlir.ir.Location | None = None
    ) -> UnrankedTensorType:
        """Create a unranked tensor type"""

    @staticmethod
    def get_unchecked(
        element_type: Type, context: _mlir.ir.Context | None = None
    ) -> UnrankedTensorType:
        """Create a unranked tensor type"""

class MemRefType(ShapedType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        shape: Sequence[int],
        element_type: Type,
        layout: Attribute | None = None,
        memory_space: Attribute | None = None,
        loc: _mlir.ir.Location | None = None,
    ) -> MemRefType:
        """Create a memref type"""

    @staticmethod
    def get_unchecked(
        shape: Sequence[int],
        element_type: Type,
        layout: Attribute | None = None,
        memory_space: Attribute | None = None,
        context: _mlir.ir.Context | None = None,
    ) -> MemRefType:
        """Create a memref type"""

    @property
    def layout(self) -> Attribute:
        """The layout of the MemRef type."""

    def get_strides_and_offset(self) -> tuple[list[int], int]:
        """The strides and offset of the MemRef type."""

    @property
    def affine_map(self) -> AffineMap:
        """The layout of the MemRef type as an affine map."""

    @property
    def memory_space(self) -> Attribute | None:
        """Returns the memory space of the given MemRef type."""

class UnrankedMemRefType(ShapedType):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        element_type: Type,
        memory_space: Attribute | None,
        loc: _mlir.ir.Location | None = None,
    ) -> UnrankedMemRefType:
        """Create a unranked memref type"""

    @staticmethod
    def get_unchecked(
        element_type: Type,
        memory_space: Attribute | None,
        context: _mlir.ir.Context | None = None,
    ) -> UnrankedMemRefType:
        """Create a unranked memref type"""

    @property
    def memory_space(self) -> Attribute | None:
        """Returns the memory space of the given Unranked MemRef type."""

class TupleType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get_tuple(
        elements: Sequence[Type], context: _mlir.ir.Context | None = None
    ) -> TupleType:
        """Create a tuple type"""

    def get_type(self, pos: int) -> Type:
        """Returns the pos-th type in the tuple type."""

    @property
    def num_types(self) -> int:
        """Returns the number of types contained in a tuple."""

class FunctionType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        inputs: Sequence[Type],
        results: Sequence[Type],
        context: _mlir.ir.Context | None = None,
    ) -> FunctionType:
        """Gets a FunctionType from a list of input and result types"""

    @property
    def inputs(self) -> list[Type]:
        """Returns the list of input types in the FunctionType."""

    @property
    def results(self) -> list[Type]:
        """Returns the list of result types in the FunctionType."""

class OpaqueType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...

    type_name: ClassVar[Final[str]] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(
        dialect_namespace: str,
        buffer: str,
        context: _mlir.ir.Context | None = None,
    ) -> OpaqueType:
        """Create an unregistered (opaque) dialect type."""

    @property
    def dialect_namespace(self) -> str:
        """Returns the dialect namespace for the Opaque type as a string."""

    @property
    def data(self) -> str:
        """Returns the data for the Opaque type as a string."""

class DynamicType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...

    static_typeid: ClassVar[Final[TypeID]] = ...
    """(arg: object, /) -> max._mlir._mlir_libs._mlir.ir.TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    @staticmethod
    def get(
        full_type_name: str,
        attributes: Sequence[Attribute],
        context: _mlir.ir.Context | None = None,
    ) -> DynamicType:
        """Create a dynamic type."""

    @property
    def params(self) -> list[Attribute]:
        """Returns the parameters of the dynamic type as a list of attributes."""

    @property
    def type_name(self) -> str: ...
    @staticmethod
    def lookup_typeid(
        full_type_name: str, context: _mlir.ir.Context | None = None
    ) -> TypeID:
        """Look up the TypeID for the given dynamic type name."""
