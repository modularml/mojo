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

#ifndef DEBUGINFOOPS_TD
#define DEBUGINFOOPS_TD

include "Support/DebugInfoDialect/IR/DebugInfoAttrs.td"
include "Support/DebugInfoDialect/IR/DebugInfoExprAttrs.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DebugInfoOp
//===----------------------------------------------------------------------===//

/// Base class for DebugInfo dialect operations.
class DebugInfoOp<string mnemonic, list<Trait> traits = []>
    : Op<DebugInfoDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// LineTableLocOp
//===----------------------------------------------------------------------===//

def DebugInfo_LineTableLocOp : DebugInfoOp<"line_table_loc"> {
  let summary = "explicitly mark a source location in IR";
  let description = [{
    The `debuginfo.line_table_loc` operation ensures a line-table entry for a
    source location is generated at a specific program point. This allows places
    that are not normally break-able to become break-able by giving a line-table
    entry.

    This may come with a performance cost as it is lowered into a nop or
    equivalent instruction.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// KillOp
//===----------------------------------------------------------------------===//

def DebugInfo_KillOp : DebugInfoOp<"kill"> {
  let summary = "debug variable kill definition";
  let description = [{
    The `debuginfo.kill` operation indicates when a debug info variable does
    not have a value. The variable may not have been initialized, or may have
    been destroyed.
  }];

  let arguments = (ins DebugInfo_DILocalVariableAttr:$valueInfo);
  let assemblyFormat = "$valueInfo attr-dict";
}

//===----------------------------------------------------------------------===//
// ValueOp
//===----------------------------------------------------------------------===//

def DebugInfo_ValueOp : DebugInfoOp<"value"> {
  let summary = "debug variable value definition";
  let description = [{
    The `debuginfo.value` operation provides information when a debug info
    variable is set to a new value. Note that this operation describes the value
    of a variable directly, not its address.

    The location expression specifies how this value relates to a source
    language variable by specifying a DI Expression that can convert the operand
    IR value into the source language variable.

    Example:

    ```mlir
    #location_expr = #debuginfo.expr.ir_value : i32

    %0 = constant 0 : i32
    debuginfo.value #debuginfo.local_variable<...> #location_expr = %0 : i32
    ```
  }];

  let arguments = (ins
    AnyType:$value,
    DebugInfo_DILocalVariableAttr:$valueInfo,
    DebugInfo_DIExprAttrInterface:$conversionExpr
  );

  let builders = [
    OpBuilder<(ins "mlir::Value":$value, "DILocalVariableAttr":$valueInfo), [{
      build($_builder, $_state, value, valueInfo,
            DIIRValueExprAttr::get(value.getType()));
    }]>
  ];

  let assemblyFormat = [{
    $valueInfo custom<ValueOpOperand>($conversionExpr, $value, type($value)) attr-dict
  }];

  let hasVerifier = 1;
}

#endif // DEBUGINFOOPS_TD
