# Mojo🔥 Compile Time Code Evaluation

## Introduction

The Mojo programming language provides a rich metaprogramming model. Part of
that model is the ability to evaluate code at compile-time. KGEN provides
parameters, the fundamental element of Mojo's metaprogramming system, and
parameter expressions, the fundamental blocks of building compile-time
expressions. Originally, KGEN parameter expressions were limited to
`ParamOperatorAttr`, a fixed and non-extensible "operator" set that are
evaluated at compile-time. These operators include arithmetic operations on
the MLIR builtin `index` type to `get_sizeof`, which returns the size of a type,
and `in`, which checks whether a value is contained in a set of values.

This was great for bootstrapping KGEN and the metaprogramming system but quickly
falls short with an extensible type system due to its non-modular design: other
dialects can't add operators. Operators also duplicate code: the arithmetic
operators on `index` are copies of the MLIR operations from the `index` dialect.
Furthermore, Mojo only has one IR: MLIR operations. It follows that KGEN needs
the ability to execute user-written code at compile-time.

KGEN already uses a JIT compiler as part of elaboration. One option to execute
user-written code at compile time is to just JIT it. This approach has a variety
of problems:

1. It won't work in environments that forbid JITs: iOS, for example.
2. The compiler itself needs to understand the ABI of the host system.
3. We may grow some operations that can only be executed at compile-time.

The other option is to write an IR interpreter. That is the approach KGEN took
and that is what this document will describe.

## Fold All the Ops

MLIR already has the infrastructure for evaluating operations at compile-time:
operation fold hooks. They are simple to write and provide multiple other
benefits: constant folding makes all code faster, and MLIR's SCCP optimization
builds on top of fold hooks. The main downside to using fold hooks is they
represent constant values as MLIR attributes, which are slow to create and
consume lots of memory since they are uniqued (and therefore immortal) in the
MLIR context. We have decided to stick to this representation for the
foreseeable future despite this downside, and we can upgrade to a more
aggressive solution if necessary.

The low-level IR for Mojo is the POP dialect. In order to interpret Mojo code,
we need to fold all the POP operations. This requires constant data types for
all POP dialect types: SIMD vectors, arrays, structs, and variants. These
attributes are defined in `POPAttrs.td`. Folders for most POP operations are
written using these attributes. They are found in `POPOpsFolders.cpp`.

Operations which define folders that return constant values given constant
inputs are handled trivially by the interpreter.

## Interpreter Memory Model

The interpreter has a fully-featured memory model that supports loads, stores,
get-element-pointer (GEP) on aggregate types, and bitcasting. It does so by
emulating real memory. Operations can opt-in to accessing the memory model
by implementing an interface instead of an operation folder:
`InterpreterOpInterface`. This interface requires the operation to implement one
function, `interpret`, which takes as one of its arguments a reference to the
`InterpreterState`. This also allows operations access to the target
configuration and data layout, which is needed to perform memory operations.

The interpreter emulates its own address space. Allocations, reads, and writes
use integer addresses to point to its internal memory. These addresses are
wrapped in a `PointerAttr` so they can be passed between operations and their
folders. The API for the memory model is:

```C++
intptr_t allocateMemory(size_t size);

ErrorOrSuccess writeAttributeToMemory(intptr_t addr, TypedAttr value);

ErrorOr<TypedAttr> readAttributeFromMemory(intptr_t addr, Type type);

ErrorOr<void *> getMemory(intptr_t addr, size_t size);
```

The first function `allocateMemory` tells the interpreter to create `size` bytes
of memory and return an address to the start of that memory. To store constant
values to memory, operations can call `writeAttributeToMemory` and
`readAttributeFromMemory`. In order to write and type values of a certain type,
that type must implement `MemoryableTypeInterface` which has two methods:
`writeTo` and `readFrom`. Note that this interface requires tight coupling
between the type and the attribute used to represent values of that type. These
methods require the type to interpret the data contained in the attribute as if
it was stored in memory during runtime. This is what allows bitcasting to work.
In many cases, the type needs to compute its runtime size using the compilation
target's data layout information. Once computed, the type should call
`getMemory` to interact directly with the interpreter's memory.

Integer addresses allow the interpreter to validate memory accesses. The
interpreter maintains a virtual memory space offset that can be used to identify
accesses to null and out-of-range addresses. These errors are returned to the
caller, allowing the interpreter to gracefully handle memory errors during
compile-time evaluation of code.

## Interpreter Control-Flow Model

The interpreter has an extremely simple control-flow model that allows it to
implement loops, branches, and control-flow graphs. The interpreter keeps a
"program counter", which is just an `Operation *`. When the interpreter runs
without branches, the current operation is evaluated using a folder or interface
and then the program counter advances to the next operation using
`Operation::getNextNode`. Operations that implement `InterpreterOpInterface` can
alter control-flow by using the API:

```C++
void transferControlFlowTo(Block *target, ArrayRef<Attribute> arguments);

void transferControlFlowTo(Operation *target);
```

The first version of `transferControlFlowTo` sets the program counter to the
first operation of the specified block, provided the constant values to use for
the block arguments. This operation can be used by control-flow entry
operations, such as `HLCF::IfOp` and `HLCF::LoopOp`, to branch to one of its
regions. It can also be used by operations with successors to branch to another
block in a CFG region. For example, this is the `interpret` method for
`HLCF::IfOp`:

```C++
ErrorTreeOrSuccess IfOp::interpret(ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  auto cond = dyn_cast_if_present<BoolAttr>(operands[0]);
  if (!cond)
    return ErrorTree(getLoc(), "non-constant condition");

  state.transferControlFlowTo(
      &(cond.getValue() ? getThenRegion() : getElseRegion()).front(), {});
  return success();
}
```

The second version models functional control-flow, both returning from a
function call and branching out of an `if` or `loop` region back to the parent
operation. When this function is called on a target operation, the return values
as passed are taken as the results of the operation and the program counter
advances to the operation after the target operation. For example, this is the
`interpret` method for `HLCF::BreakOp`:

```C++
ErrorTreeOrSuccess BreakOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  auto loop = getOperation()->getParentOfType<LoopOp>();
  state.transferControlFlowTo(loop, operands);
  return success();
}
```

The interpreter maintains a callstack. A stack frame contains the callsite
operation and a value map (SSA value to constant value map) for this particular
invocation of the function, as well as some extra information for generating a
stack trace when the interpreter fails. Call operations should get the body of
their callee, using `lookupFunctionBody` if the callee is a symbol. This is
important for the KGEN elaborator since it "deflates" functions by storing their
bodies in a cache. Call operations should then push a stack frame and transfer
control-flow to the function body.

To return from a function call, the return operation can pop the current stack
frame to retrieve the callsite operation and then transfer control-flow to that
operation. The interpreter will resume at the next operation.

The interpreter does not maintain a frame for operations with regions if they
are not function calls, and instead it allows value mappings to be overwritten
if a non-function region (e.g. a loop body) is executed more than once. A region
that represents a function body can have overlapping execution, so its value
mappings must be maintained on a stack. However, a loop body is only executed
within a call frame from beginning to end, so overwriting values will always
ensure the correct value bindings.

## Compile-Time Code Evaluation in KGEN

The representation of a function call in a parameter context is the `apply`
parameter operator in KGEN. For example:

```mlir
kgen.func @call_me() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}

kgen.func @call_it() {
  kgen.param.constant = <apply(:() -> index @call_me)>
  kgen.return
}
```

Parameter operator expressions fold and canonicalize on construction if all
their inputs are simple constants and the operators can be evaluated locally. In
this case, the operands are simple constants but `apply` cannot be evaluated
locally because it requires the definition of `@call_me`. KGEN leaves it up to
the Elaborator to handle "symbolic parameter expressions" like these via the
`IREvaluator` subclass of `ParameterEvaluator`. It is through this class that
the Elaborator invokes the interpreter.

The interpreter, however, can only run on concrete functions. When the
`IREvaluator` encounters an `apply` operator, it first asks the Elaborator for
all candidates of its callee, which concretizes the callee if it was not
concrete. It picks the first valid candidate if there are multiple. The `apply`
operator can then be collapsed into the result value.

## Compile-Time Function Calls in Mojo

In Mojo, executing code at compile-time is similar to C++ `constexpr` functions.
Right now, the semantics of evaluating a function call at compile-time are
simple: when calling a function inside a "parameter context", i.e. a parameter
value or alias value, the parser emits an `apply` operator. Elsewhere, the
parser emits a `kgen.call` operation.

```python
# Call `foo` in a parameter context. This emits an `apply`.
func_with_params[foo(5)]()

# Call `foo` in an alias context. This emits an `apply`.
alias value = foo(5)

# Call `foo` in an RValue context. This emits a `kgen.call`.
let value = foo(5)
```
