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

#ifndef KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD
#define KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ParameterTypeInterface
//===----------------------------------------------------------------------===//

def ParameterTypeInterface : TypeInterface<"ParameterTypeInterface"> {
  let description = [{
    The `ParameterTypeInterface` can be used by types to plug into various parts
    of the KGEN parameter system, including pretty parsing of attribute values.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Parse an attribute of this type, using pretty syntax if desired. This
      method should return none if the syntax guard does not match the expected
      pretty syntax of an attribute of this type. The KGEN parameter parser
      reserves several tokens to guard generic parameter attributes:

      - `*` denotes the beginning of an escaped parameter reference.
      - `?` denotes an unknown parameter value
      - barewords are parameter references or parameter operators

      The type may use any other token to guard its attribute pretty parsing. If
      the attribute's pretty format contains commas (e.g. a sequence of
      elements), its format must be surrounded by a delimiter to disambiguate
      from operator operand lists.
    }], "OptionalParseResult", "parseValue", (ins "AsmParser &":$p,
                                                  "TypedAttr &":$value)>,
    InterfaceMethod<[{
      Print an attribute of this type, using pretty syntax if desired. Return
      false if the attribute is not one the type knows how to print.
    }], "LogicalResult", "printValue", (ins "AsmPrinter &":$p,
                                            "TypedAttr":$value)>,
    InterfaceMethod<[{
      Return whether the type is a metatype.
    }], "bool", "isMetaType", (ins), [{}], [{ return false; }]>,

    InterfaceMethod<[{
      Return the metatype of the type it has one.
    }], "Type", "getMetaType", (ins), [{}], [{ return Type(); }]>
  ];
}

//===----------------------------------------------------------------------===//
// ParameterScopeTypeInterface
//===----------------------------------------------------------------------===//

def ParameterScopeTypeInterface : TypeInterface<"ParameterScopeTypeInterface"> {
  let description = [{
    The `ParameterScopeTypeInterface` describes a type that declares a nested
    parameter scope within a type expression. It enables `ParamIndexRefAttr`
    values inside the type to reference parameters declared in a scope.

    For example, if we have this Mojo code:

    ```mojo
    def foo[T: AnyType]():
      comptime bork: def[
        T: AnyType,
        inner_f: def[Y: AnyType](t: T, y: Y) -> None
      ] -> None = ...
    ```

    The `def` after `bork:` is a `kgen.generator` which is a
    `ParameterScopeTypeInterface`.

    `ParameterScopeTypeInterface` also causes the `depth` fields of
    `ParamIndexRefAttr`/`ImplicitOriginRefAttr`/etc that are contained (even
    indirectly) within this object to be higher, see PSTIAIRAID.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Return the types of the nested input parameter declarations.
    }], "ArrayRef<Type>", "getInputParamTypes">,
  ];
}

//===----------------------------------------------------------------------===//
// StructTypeInterface
//===----------------------------------------------------------------------===//

def StructTypeInterface : TypeInterface<"StructTypeInterface"> {
  let description = [{
    This interface provides mechanisms for types that refer to a KGEN
    declaration to verify the referenced declaration.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Verify whether the symbol reference is valid. A symbol table evaluation
      handler is provided to lookup the type definition and perform any
      parameter evaluation necessary.
    }], "LogicalResult", "verifySymbolUses",
      (ins "SymTabEvaluationContext &":$evaluationContext,
           "Location":$loc)>,
    InterfaceMethod<[{
      Get the referenced symbol.
    }], "SymbolRefAttr", "getSymbolRef">
  ];
}

//===----------------------------------------------------------------------===//
// ContextuallyEvaluatedTypeInterface
//===----------------------------------------------------------------------===//

def ContextuallyEvaluatedTypeInterface :
    TypeInterface<"ContextuallyEvaluatedTypeInterface"> {
  let description = [{
    This interface describes types whose evaluation may require additional
    context. This is in contrast to "simple" types that can be simplified
    context-free at construction time.

    Types implementing this interface can provide an `evaluateWithContext`
    method that performs context-dependent evaluation using the provided
    `ParameterEvaluationContext`. The context provides hooks for struct
    resolution, conformance lookup, and error reporting.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Evaluate this type using the provided context. Returns the evaluated
      result on success, or failure() if evaluation cannot proceed (e.g., if
      required information is not yet available). A null Type result with
      success indicates async evaluation was triggered and the caller should
      retry later.

      The default implementation returns failure(), indicating the type
      does not support generalized contextual evaluation.
    }],
    "FailureOr<Type>", "evaluateWithContext",
    (ins "ParameterEvaluationContext &":$context), [{}], [{
      return failure();
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// TraitSymbolInterface
//===----------------------------------------------------------------------===//

def TraitSymbolInterface : TypeInterface<"TraitSymbolInterface"> {
  let description = [{
    Interface for types that carry a list of trait symbol references, such as
    the `!lit.trait` type, or a `!kgen.typevalue<trait_ref<...>>`.

    The practical reason why we need the interface is to avoid cyclic build
    dependencies.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Return the trait symbol references carried by this type.
    }], "ArrayRef<TraitSymbolAttr>", "getTraitSymbols">,
  ];
}

//===----------------------------------------------------------------------===//
// SugaredTypeInterface
//===----------------------------------------------------------------------===//

def SugaredTypeInterface : TypeInterface<"SugaredTypeInterface"> {
  let description = [{
    This interface can be used to customize SugarAttr behavior.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Returns whether the sugar should be elided for the given attribute. This
      returns AlwaysInlineBuiltin if it should only be elided when it is a
      builtin call, or Alias if it should always be elided.
    }], "std::optional<KGEN::SugarKind>", "canElideSugarFor",
        (ins "TypedAttr":$attr)>,

    InterfaceMethod<[{
      If this type has a cached canonical type, return it, otherwise return
      null.  The type canonicalizer will recursively canonicalize types without
      a cache.
    }], "Type", "getCachedCanonicalType", (ins "Type":$type)>,
  ];
}

#endif // KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD
