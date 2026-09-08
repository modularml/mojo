//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef COOPS_TD
#define COOPS_TD

include "Mojo/CODialect/CODialect.td"
include "Mojo/CODialect/COTypes.td"
include "Mojo/KGENDialect/KGENInterfaces.td"
include "Support/Compiler/AttrBase.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class COOp<string mnemonic, list<Trait> traits = []> :
    Op<CODialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// HandleOp
//===----------------------------------------------------------------------===//

def CO_HandleOp : COOp<"handle"> {
  let summary = "Get the handle to the current coroutine.";
  let description = [{
    The `co.handle` operation returns the handle to the current coroutine.

    Example:

    ```mlir
    %hdl = co.handle : i32, index
    ```
  }];

  let arguments = (ins M_TypeArrayAttr:$types);
  let results = (outs CoroutineType:$coroutine);
  let assemblyFormat = "attr-dict custom<CoroutineTypes>($types)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SuspendOp
//===----------------------------------------------------------------------===//

def CO_SuspendOp : COOp<"suspend", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>]> {
  let summary = "Suspend the current coroutine and await resumption.";
  let description = [{
    The `co.suspend` operation suspends the current coroutine function
    at the site of the operation and executes a region. The code in the region
    must resume the coroutine at some point. This operation can only be used
    inside an async function.

    Example:

    ```mlir
    co.suspend (%hdl) {
      co.resume %hdl
      co.suspend.end
    }
    ```
  }];

  let regions = (region AnyRegion:$body);
  let assemblyFormat = "custom<SuspendBody>($body) attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SuspendEndOp
//===----------------------------------------------------------------------===//

def CO_SuspendEndOp : COOp<"suspend.end", [Pure, Terminator,
    HasParent<"SuspendOp">]> {
  let summary = "terminator for `co.suspend`";
  let description = [{
    The `co.suspend.end` operation is the terminator for the
    `co.suspend` operation. The operation signifies the end of the
    coroutine suspension function denoted by the surrounding
    `co.suspend` operation. In an async function, it is semantically
    a fallthrough.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

def CO_InvokeOp : COOp<"invoke", [
    DeclareOpInterfaceMethods<KGEN_CallOpInterface>,
    DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>]> {
  let summary = "invoke an async function, forming a coroutine";
  let description = [{
    The `co.invoke` operation calls an `async` parametric callable. It returns
    a cold-started `!co.routine` from the callee and the bound arguments.

    Example:

    ```mlir
    %coro = co.invoke[:() -> index: @foo]()
    ```
  }];

  let arguments = (ins
    AttrOfType<KGEN_FuncTypeGeneratorType>:$callee,
    Variadic<AnyType>:$operands
  );
  let results = (outs CoroutineType:$coroutine);

  let assemblyFormat = [{
  `` custom<AsyncParametricCallee>($callee, $operands, type($operands))
  attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExecuteOp
//===----------------------------------------------------------------------===//

def CO_ExecuteOp : COOp<"execute", [SingleBlock,
    DeclareOpInterfaceMethods<FunctionLike>,
    DebugInfo_InlinedSubprogramScoped]> {
  let summary = "an inline async region";
  let description = [{
    The `co.execute` operation declares an inline async region and returns a
    coroutine handle to the region. This operation can represent an inlined
    post-elaboration async function. The region also carries the byref result
    slots required by the callee.

    Example:

    ```mlir
    %0 = kgen.param.constant: i32 = <3>
    %coro = co.execute : i32, i64 {
      %1 = kgen.param.constant: i64 = <5>
      kgen.return %0, %1 : i32, i64
    }
    ```
  }];

  let arguments = (ins M_TypeArrayAttr:$types);
  let results = (outs Res<CoroutineType, "", [MemAlloc]>:$coroutine);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    `` custom<CoroutineTypes>($types) custom<ExecuteBody>($bodyRegion) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SetResultsOp
//===----------------------------------------------------------------------===//

def CO_SetResultsOp : COOp<"set_results"> {
  let summary = "Set the direct results of the coroutine.";
  let description = [{
    The `co.set_results` operation sets the direct SSA result values of the
    async function from which the coroutine was formed.

    Example:

    ```mlir
    co.set_results %hdl(%0, %1) : i32, i64
    ```
  }];

  let arguments = (ins Arg<CoroutineType, "", [MemWrite]>:$coroutine,
                       Variadic<AnyType>:$values);
  let assemblyFormat = [{
    $coroutine `(` $values `)` attr-dict (`:` type($values)^)?
  }];
}

//===----------------------------------------------------------------------===//
// GetResultsOp
//===----------------------------------------------------------------------===//

def CO_GetResultsOp : COOp<"get_results"> {
  let summary = "Get the direct results of the coroutine.";
  let description = [{
    The `co.get_results` operation reads from the coroutine object the SSA
    value results of the async function from which the coroutine was formed. For
    example, given the following async function:

    ```mlir
    kgen.func @async_fn() async -> (i32, i64)
    ```

    The coroutine formed from this function will contain the `i32` and `i64`
    SSA value results of the function, and they can be read from the coroutine:

    ```mlir
    %coro = co.invoke[() async -> (i32, i64): @async_fn]()
    %0, %1 = co.get_results %coro : i32, i64
    ```
  }];

  let arguments = (ins Arg<CoroutineType, "", [MemRead]>:$coroutine);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$coroutine attr-dict (`:` type($results)^)?";
}

//===----------------------------------------------------------------------===//
// GetCallbackPtrOp
//===----------------------------------------------------------------------===//

def CO_GetCallbackPtrOp : COOp<"get_callback_ptr", [Pure]> {
  let summary = "Get a pointer to the coroutine callback.";
  let description = [{
    The `co.get_callback_ptr` takes a coroutine and returns a pointer to its
    callback closure. Every coroutine has a callback closure that is invoked
    when the coroutine exits. This operation can be used to access the callback
    closure value and modify it. It is by default a garbage value. The callback
    closure is exactly twice the pointer width, enough to fit a function pointer
    and a single data field. Attempting to access data beyond the pointee is
    undefined behaviour.

    Example:

    ```mlir
    %ptr = co.get_callback_ptr %coro : <struct<(index, pointer<none>)>>
    ```
  }];

  let arguments = (ins CoroutineType:$coroutine);
  let results = (outs PointerType:$callback);
  let assemblyFormat = "$coroutine attr-dict `:` type($callback)";
}

//===----------------------------------------------------------------------===//
// ResumeOp
//===----------------------------------------------------------------------===//

def CO_ResumeOp : COOp<"resume"> {
  let summary = "Get a coroutine's resume function.";
  let description = [{
    The `co.resume` operation returns a function that resumes execution of a
    suspended coroutine.  Resuming a coroutine that is not suspended or that is
    uninitialized is undefined behaviour.

    Example:

    ```mlir
    %fn = co.resume %hdl : <(!co.routine) -> ()>
    kgen.call_indirect %fn(%hdl) : (!co.routine) -> ()
    ```
  }];

  let arguments = (ins CoroutineType:$coroutine);
  let results = (outs KGEN_FuncTypeGeneratorType:$result);
  let assemblyFormat = "$coroutine attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// AwaitOp
//===----------------------------------------------------------------------===//

def CO_AwaitOp : COOp<"await"> {
  let summary = "await the results of another coroutine";
  let description = [{
    The `co.await` operation starts and awaits the results of another coroutine.
    It pushes a frame on the dynamic call stack formed between coroutines.
  }];

  let arguments = (ins CoroutineType:$coroutine, Variadic<AnyType>:$slots);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// DestroyOp
//===----------------------------------------------------------------------===//

def CO_DestroyOp : COOp<"destroy"> {
  let summary = "Destroy the memory associated with a coroutine.";
  let description = [{
    The `co.destroy` operation destroys any memory associated with
    a coroutine handle. A coroutine handle may or may not have any dynamic
    memory associated with it. A call to a coroutine function returns a
    coroutine handle and that handle must be deallocated with this operation.
    Doing anything with a coroutine handle after it is destroyed is undefined
    behaviour.

    Example:

    ```mlir
    co.destroy %hdl
    ```
  }];

  let arguments = (ins Res<CoroutineType, "", [MemFree]>:$coroutine);
  let assemblyFormat = "$coroutine attr-dict";
}

//===----------------------------------------------------------------------===//
// SetByRefErrorAndResultOp
//===----------------------------------------------------------------------===//

def CO_SetByRefErrorAndResultOp : COOp<"set_byref_error_result"> {
  let summary = "Set the by ref arguments of the coroutine.";
  let description = [{
    The `co.set_byref_error_result` operation sets the by reference result
    value and the by reference error value of the async function.

    Example:

    ```mlir
    co.set_byref_error_result %hdl(%0, %1) :
      !kgen.pointer<index>, !kgen.pointer<index>
    ```
  }];

  let arguments = (ins Arg<CoroutineType, "", [MemWrite]>:$coroutine,
                       PointerType:$result, Optional<PointerType>:$error);
  let assemblyFormat = [{
    $coroutine `(` $result (`,` $error^)? `)` attr-dict `:` type(operands)
  }];
}

//===----------------------------------------------------------------------===//
// GetByRefErrorAndResultOp
//===----------------------------------------------------------------------===//

// TODO: Delete this op. It's only needed for staging the old coroutine lowering
// between LowerClosures and LowerKGENCoroutines.
def CO_GetByRefErrorAndResultOp : COOp<"get_byref_error_result"> {
  let arguments = (ins Arg<CoroutineType, "", [MemRead]>:$coroutine);
  let results = (outs PointerType:$result, Optional<PointerType>:$error);
  let assemblyFormat = "$coroutine attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// HotInvokeOp
//===----------------------------------------------------------------------===//

def CO_HotInvokeOp : COOp<"hot_invoke", [
    DeclareOpInterfaceMethods<KGEN_CallOpInterface>,
    DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>]> {
  let summary = "invoke an async function, forming a coroutine";
  let description = [{
    The `co.hot_invoke` operation calls an `async` parametric callable. It returns
    a hot-started `!co.routine` from the callee and the bound arguments.

    Example:

    ```mlir
    %coro = co.hot_invoke[() async -> index: @foo](%arg1)
    ```
  }];

  let arguments = (ins
    AttrOfType<KGEN_FuncTypeGeneratorType>:$callee,
    Variadic<AnyType>:$calleeOperands
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
  `` custom<HotAsyncParametricCallee>(
       $callee, $calleeOperands, type($calleeOperands), type($results))
    attr-dict
  }];

  let hasVerifier = 1;
}

#endif // COOPS_TD
