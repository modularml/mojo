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

#ifndef KGEN_KGENDIALECT_KGENATTRINTERFACES_TD
#define KGEN_KGENDIALECT_KGENATTRINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ParameterAttr
//===----------------------------------------------------------------------===//

def ParameterAttrInterface : AttrInterface<"ParameterAttr"> {
  let description = [{
    Any attribute that implements `TypedAttr` can be used as a parameter
    attribute in KGEN, but this interface allows parameter attributes to plug
    into specific parts of the KGEN parameter system.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Returns whether the attribute can be considered a simple constant.
    }], "bool", "isConstant">,

    InterfaceMethod<[{
      Compare this parameter attribute with another attribute of the same kind,
      returning true if this parameter should be considered "less" than the
      other attribute.
    }], "bool", "isLessThan", (ins "Attribute":$rhs), [{}], [{
      return ParameterAttr::compareSubElements($_attr, rhs);
    }]>,

    InterfaceMethod<[{
      Returns a custom error message for this parameter attribute, or success if
      there are no errors. This allows parameter attributes to implement their
      own validation logic and error reporting, producing better elaborator
      error messages when they cannot be simplified then.
    }], "ErrorOrSuccess", "validateForElaborator", (ins), [{}], [{
      return success();
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Return true if the parameter can be considered a "simple" constant. A
    /// parameter is a constant if it contains no nested expressions that have
    /// not been evaluated.
    static bool isSimpleConstant(Attribute attr);

    /// Compare two parameter constants, returning true if the `lhs` parameter
    /// should be considered "less" than the `rhs` parameter.
    static bool compare(Attribute lhs, Attribute rhs);

    // Compare sub-elements of a type or attribute lexicographically (first all
    // the nested attributes, then all the nested types).
    static bool compareSubElements(Attribute lhs, Attribute rhs);
    static bool compareSubElements(Type lhs, Type rhs);
  }];
}

//===----------------------------------------------------------------------===//
// ParameterScopeAttrInterface
//===----------------------------------------------------------------------===//

def ParameterScopeAttrInterface : AttrInterface<"ParameterScopeAttrInterface"> {
  let description = [{
    The `ParameterScopeAttrInterface` describes an attribute that declares a
    nested parameter scope within a parameter expression. It enables
    `ParamIndexRefAttr` values inside the attribute to reference parameters
    declared in a scope.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Return the types of the nested input parameter declarations.
    }], "ArrayRef<Type>", "getInputParamTypes">,
  ];
}

//===----------------------------------------------------------------------===//
// ContextuallyEvaluatedAttrInterface
//===----------------------------------------------------------------------===//

def ContextuallyEvaluatedAttrInterface :
    AttrInterface<"ContextuallyEvaluatedAttrInterface"> {
  let description = [{
    This interface describes parameter attributes whose evaluation may require
    additional context. This is in contrast to "simple" parameter attributes
    that can be simplified context-free at construction time.

    Attributes implementing this interface can provide an `evaluateWithContext`
    method that performs context-dependent evaluation using the provided
    `ParameterEvaluationContext`. The context provides hooks for struct
    resolution, conformance lookup, and error reporting.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Evaluate this attribute using the provided context. Returns the evaluated
      result on success, or failure() if evaluation cannot proceed (e.g., if
      required information is not yet available). A null TypedAttr result with
      success indicates async evaluation was triggered and the caller should
      retry later.

      The default implementation returns failure(), indicating the attribute
      does not support generalized contextual evaluation.
    }],
    "FailureOr<TypedAttr>", "evaluateWithContext",
    (ins "ParameterEvaluationContext &":$context), [{}], [{
      return failure();
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// IndexRefAttrInterface
//===----------------------------------------------------------------------===//

def IndexRefAttrInterface : AttrInterface<"IndexRefAttrInterface"> {
  let description = [{
    Index-based parameter references are a relative parameter referencing scheme
    that uses a pair of integers to reference parameters in a way that doesn't
    involve names. This is useful for later knowing if two types are equal, even
    if they have different parameter names.

    For example, these two aliases have equal types:

    ```mojo
    comptime A: def[T: AnyType](x: T)->None = ...
    comptime B: def[Y: AnyType](x: Y)->None = ...
    ```

    ...if those param-refs use indexes instead of names like:

    ```mojo
    comptime A: def[_: AnyType](x: *(0,0))->None = ...
    comptime B: def[_: AnyType](x: *(0,0))->None = ...
    ```

    All types in Mojo use `IndexRefAttrInterface` instead of parameter names.
    The above are `ParamIndexRefAttr` specifically.

    All `IndexRefAttrInterface` have two fields: a depth and an index.

    - depth: Which containing signature contains the parameter we're referring
      to. Non-negative integer. Zero means the nearest containing signature
      (like above), one means the signature containing that one, etc.
      Note they cannot refer to any op's parameter-decls, and you cannot always
      use a depth to refer to surrounding scopes, see DCRTODS.
    - index: index of the parameter decl in that signature (non-negative integer).

    See IRAIDAI for more details, context, and examples.

    These depths must be carefully handled and adjusted when dealing with
    multiple signatures or scopes, see STCHDDDOS.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Get the depth of the index reference.
    }], "size_t", "getDepth">,
    InterfaceMethod<[{
      Get the index of the referenced parameter.
    }], "size_t", "getIndex">,
    InterfaceMethod<[{
      Change the depth and index of the parameter without changing the semantics
      of the parameter it refers to. This method also supplies the new
      subelements of the attribute.
    }], "IndexRefAttrInterface", "replace",
      (ins "size_t":$depth, "size_t":$index, "ArrayRef<Attribute>":$attrs,
           "ArrayRef<Type>":$types)>
  ];
}

//===----------------------------------------------------------------------===//
// DeclRefAttrInterface
//===----------------------------------------------------------------------===//

def DeclRefAttrInterface : AttrInterface<"DeclRefAttrInterface"> {
  let description = [{
    This interface provides a mechanism for attributes of `!lit.struct` type
    to implement a verifier hook. `StructType` is a reference to an out-of-line
    type definition and values of this type require a symbol table to lookup the
    definition for verification.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Verify whether the value of the attribute matches the type. A symbol table
      evaluation handler is provided to lookup the type definition and perform
      any parameter evaluation necessary.
    }], "LogicalResult", "verifySymbolUses",
      (ins "SymTabEvaluationContext &":$evaluationContext,
           "Location":$loc)>
  ];
}

//===----------------------------------------------------------------------===//
// CallableSymbolAttrInterface
//===----------------------------------------------------------------------===//

def CallableSymbolAttrInterface : AttrInterface<"CallableSymbolAttrInterface"> {
  let description = [{
    This interface describes typed attributes that refer to a concrete callable
    symbol. The underlying symbol constant keeps the declaration identity and
    true signature, while `getType()` returns the callable type presented by the
    attribute at the use site.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Return the symbol for this callable reference.
    }], "SymbolRefAttr", "getSymbol">,
    InterfaceMethod<[{
      Return the parameter bindings for this callable reference.
    }], "ArrayRef<TypedAttr>", "getParamValues">,
    InterfaceMethod<[{
      Return the callable type presented at this use site.
    }], "Type", "getType">,
  ];
}

//===----------------------------------------------------------------------===//
// FnMetadataAttrInterface
//===----------------------------------------------------------------------===//

def FnMetadataAttrInterface : AttrInterface<"FnMetadataAttrInterface"> {
  let description = [{
    This interface describes attributes that are attached to a `!kgen.func`
    type. Function metadata attributes carry additional information about a
    callable on top of the information in the base `FuncType`. This
    interface defines the required methods for this metadata attribute,
    including verification and print hooks.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Verify the FuncType parameters using the current metadata.
    }], "LogicalResult", "verifyFuncType", (ins
      "function_ref<InFlightDiagnostic()>":$emitError,
      "FunctionType":$values,
      "ArrayRef<ArgConvention>":$argConventions,
      "FnEffects":$effects)>,
    InterfaceMethod<[{
      Request the custom metadata attribute print the provided FuncType.
    }], "void", "printFuncTypeBody", (ins "AsmPrinter &":$p,
                                           "FuncType":$sig)>,
    InterfaceMethod<[{
      Request the custom metadata attribute print the provided FuncType.
    }], "void", "printFuncType", (ins "AsmPrinter &":$p, "FuncType":$sig)>,
    InterfaceMethod<[{
      Print the provided FuncType inline, without any dialect prefix or
      angle-bracket wrap. Used when the FuncType appears as the body of an
      already-wrapped generator type, where the prefix and wrapping would be
      redundant.
    }], "void", "printFuncTypeInline", (ins "AsmPrinter &":$p,
                                            "FuncType":$sig)>,
    InterfaceMethod<[{
      Check if this metadata attribute is equal to another metadata attribute.
      Returns true if they are equal.
    }], "bool", "equals", (ins "FnMetadataAttrInterface":$other)>,
  ];
}

#endif // KGEN_KGENDIALECT_KGENATTRINTERFACES_TD
