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

#ifndef DEBUGINFOEXPRATTRS_TD
#define DEBUGINFOEXPRATTRS_TD

include "Support/DebugInfoDialect/IR/DebugInfoAttrs.td"
include "Support/DebugInfoDialect/IR/DebugInfoExprAttrInterfaces.td"

//===----------------------------------------------------------------------===//
// DIExprAttrBase
//===----------------------------------------------------------------------===//

class DIExprAttrBase<string attrName, string attrMnemonic,
  list<Trait> traits = []>
    : DIAttr<attrName, attrMnemonic,
             traits # [DebugInfo_DIExprAttrInterface]> {
}

//===----------------------------------------------------------------------===//
// DIAggregatesIntoExprAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DIAggregatesIntoExprAttr
  : DIExprAttrBase<"DIAggregatesIntoExpr", "expr.agg"> {
  let summary = "DI Expression fragment-to-aggregate operator.";
  let description = [{
    TODO: Create lower-level version with offset & size instead of field index.

    The `debuginfo.expr.agg` attribute represents a DI Expression operator
    that aggregates together a struct from a single field of that struct. Its
    operands are
      - A DI Expression of type T, and
      - A constant index i,
    and it has a result where the i-th field has type T.

    Example:

    ```mlir
    !m1 = !debuginfo.member<foo: T>
    #ir_value = #debuginfo.expr.ir_value : T
    #agg = #debuginfo.expr.agg<#ir_value, 1> : !debuginfo.ti.struct<S(!m0, !m1)>
    ```
  }];
  let parameters = (ins
    "DIExprAttr":$fieldExpr,
    "unsigned":$fieldIndex,
    AttributeSelfTypeParameter<"">:$type
  );
  let assemblyFormat = "`<` $fieldExpr `,` $fieldIndex `>`";
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "DIExprAttr":$fieldExpr,
                                        "unsigned":$fieldIndex,
                                        "Type":$type), [{
      return get(type.getContext(), fieldExpr, fieldIndex, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// DIDerefExprAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DIDerefExprAttr : DIExprAttrBase<"DIDerefExpr", "expr.deref"> {
  let summary = "DI Expression dereference operator.";
  let description = [{
    The `#debuginfo.expr.deref` attribute represents a DI Expression operator
    that performs dereference on a pointer.

    Example:

    ```mlir
    #ir_value = #debuginfo.expr.ir_value : !debuginfo.ti.ptr<T>
    #deref = #debuginfo.expr.deref<#ir_value>
    ```
  }];
  let parameters = (ins
    "DIExprAttr":$ptrExpr,
    AttributeSelfTypeParameter<"">:$type
  );
  let assemblyFormat = "`<` $ptrExpr `>`";
  let genVerifyDecl = 1;
  let builders = [TypeBuilderWithInferredContext<
    (ins "DIExprAttr":$ptrExpr, "Type":$type), "", "DIExprAttr">];
}

//===----------------------------------------------------------------------===//
// DIIRValueExprAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DIIRValueExprAttr
  : DIExprAttrBase<"DIIRValueExpr", "expr.irvalue"> {
  let summary = "DI Expression IR Value operator.";
  let description = [{
    The `#debuginfo.expr.ir_value` attribute represents a DI Expression operator
    that functions as a placeholder argument. During execution of the DI
    Expression, this value will be filled in based on the context. For example,
    when the surrounding DI Expression is used by a DebugInfo ValueOp, this
    attr represents the operand of that ValueOp.

    Example:

    ```mlir
    #location_expr = #debuginfo.expr.ir_value : index

    %0 = ... : index
    %1 = debuginfo.value #var #location_expr = %0 : index
    ```
  }];
  let parameters = (ins
    AttributeSelfTypeParameter<"">:$type
  );
  let assemblyFormat = "";

  let builders = [TypeBuilderWithInferredContext<(ins "Type":$type), [{
    return get(type.getContext(), type);
  }]>];
}

//===----------------------------------------------------------------------===//
// DIRefOfExprAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DIRefOfExprAttr : DIExprAttrBase<"DIRefOfExpr", "expr.refof"> {
  let summary = "DI Expression reference-of operator.";
  let description = [{
    The `#debuginfo.expr.refof` attribute represents a DI Expression operator
    that gets a reference from a value.

    Example:

    ```mlir
    #ir_value = #debuginfo.expr.ir_value : T
    #deref = #debuginfo.expr.refof<#ir_value> : !debuginfo.ti.ptr<T>
    ```
  }];
  let parameters = (ins
    "DIExprAttr":$valueExpr,
    AttributeSelfTypeParameter<"">:$type
  );
  let assemblyFormat = "`<` $valueExpr `>`";

  let builders = [TypeBuilderWithInferredContext<
    (ins "DIExprAttr":$valueExpr, "Type":$type), "", "DIExprAttr">];
}

#endif // DEBUGINFOEXPRATTRS_TD
