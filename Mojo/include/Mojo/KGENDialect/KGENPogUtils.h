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
//
// This file declares utility functions for parsing, printing and verifying
// POG (parameter/argument metadata) related concepts: passing kinds,
// variadicness, parameter signatures, and constraints. These helpers operate
// purely on KGEN-namespace attributes (PogListAttr/PogMetadataAttr/
// ConstraintAttr/PassingKind/VariadicKind/ArgConvention) and live in KGEN so
// they can be called from KGEN-side implementations without circular linkage
// back to libLITDialect.a.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENPOGUTILS_H
#define KGEN_KGENDIALECT_KGENPOGUTILS_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENEnums.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Parameter Mangling
//===----------------------------------------------------------------------===//

/// Demangle a mangled parameter name if it is has a "`" postfix and and
/// trailing depth and unique ID. If `forUser` is true, then any prefixes for
/// autoparameters are removed.  If not, only the `42 suffix is removed.  The
/// later is important when calculating the ASTDecl name for a parameter.  The
/// former is useful when printing the name.
StringRef demangleParameterName(StringRef name, bool forUser = false);

/// Return true if the parameter is not exposed to the user.
inline bool isHiddenGeneratorParam(PassingKind passingKind, StringRef name) {
  return passingKind == PassingKind::Implicit ||
         (passingKind == PassingKind::Inferred && name.contains('.'));
}

//===----------------------------------------------------------------------===//
// Parsing and Printing
//===----------------------------------------------------------------------===//

/// Parse an optional default value of the given type. `defaultVal` is not
/// modified if a default value was not present.
ParseResult parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                      Type type);
void printOptionalDefaultValue(AsmPrinter &p, TypedAttr defaultVal, Type type);

/// Parse a parameter specification in a lit op.
ParseResult parseOptionalParameterSpec(AsmParser &p,
                                       ParamDeclArrayAttr &inputParamDecls,
                                       PogListAttr &paramListAttr);

/// Print a parameter specification in a lit op. A ParameterEvaluator is
/// necessary to substitute parameters into parametric parameters.
void printOptionalParameterSpec(AsmPrinter &p,
                                ArrayRef<ParamDeclAttr> paramDecls,
                                PogListAttr paramListAttr,
                                ParameterEvaluator &evaluator);

/// Parse a parameter signature (input/result types with optional default
/// values) if present. If `parseBody` is provided, it will be called after
/// parsing the input parameter spec.
ParseResult parseOptionalParamSignature(AsmParser &p,
                                        SmallVectorImpl<Type> &inputParamTypes,
                                        PogListAttr &paramListAttr,
                                        function_ref<ParseResult()> parseBody);

/// Print the parameter type signature if there are any input or result types,
/// along with the default input parameter values.
void printOptionalParamSignature(AsmPrinter &p, ArrayRef<Type> inputParamTypes,
                                 PogListAttr paramListAttr,
                                 bool omitEmptyAngleBrackets = false);

/// Parse an optional parameter or argument name.
ParseResult parseOptionalName(AsmParser &p, StringAttr &name);

/// Parse an optional passing convention and variadicness. The the given index
/// will be added to the appropriate index array if a variadicness is present.
ParseResult parseConventionAndVariadicness(
    AsmParser &p, ArgConvention &convention, VariadicKind &variadic,
    std::optional<ArgConvention> &origVariadicConvention, size_t idx);

/// Print an optional passing convention and variadicness.
void printConventionAndVariadicness(AsmPrinter &p, ArgConvention convention,
                                    VariadicKind variadicness);

//===----------------------------------------------------------------------===//
// PassingKindParser / PassingKindPrinter
//===----------------------------------------------------------------------===//

/// Handles parsing '|' and '*' in lit IR and counts the number of arguments of
/// different passing kinds.
/// TODO(#23387): fix this when AsmParser can handle '/'.
class PassingKindParser {
public:
  enum Marker { PLUS, BAR, STAR, QUESTION, NUM_MARKERS };
  static constexpr std::array<char, NUM_MARKERS> markers{'+', '|', '*', '?'};

  PassingKindParser(AsmParser &parser) : parser(parser) {}

  /// Try to parse a single optional '*' or '|', and emit an error if a
  /// duplicate is found or a '|' comes after a '*'.
  OptionalParseResult parseOptionalStarSlash();

  /// Populate the parameter passing kinds.
  void populatePassingKinds(SmallVectorImpl<PassingKind> &kinds) const;

  /// Return true if the parser is currently parsing an implicit parameter.
  bool isCurrentImplicit() const { return foundMarkers[QUESTION]; }

  /// Return true if the parser is currently parsing a keyword-only parameter.
  bool isCurrentKwOnly() const {
    return foundMarkers[STAR] && !foundMarkers[QUESTION];
  }

private:
  AsmParser &parser;
  size_t idx = 0;
  std::array<bool, NUM_MARKERS> foundMarkers{};
  std::array<size_t, NUM_MARKERS> idxOfEach{};
};

/// Handles printing '/', '+', '?', and '*' in lit IR. Optionally, it allows
/// specifying a replacement to be used instead of '/' and '+'. It also allows
/// specifying a flag to suppress the '/' if it immediately follows the first
/// argument (useful if printing methods with mojo syntax).
class PassingKindPrinter {
public:
  PassingKindPrinter(raw_ostream &os, size_t numPogs,
                     std::function<PassingKind(size_t)> getPassingKind,
                     bool suppressSlashAfterSelf = false, char slash = '/',
                     StringRef plus = "+");
  PassingKindPrinter(raw_ostream &os, PogListAttr pogListAttr,
                     bool suppressSlashAfterSelf = false, char slash = '/',
                     StringRef plus = "+");
  PassingKindPrinter(AsmPrinter &printer, PogListAttr pogListAttr,
                     char slash = '/', StringRef plus = "+");

  /// Print a single '*' or '/' if needed, given the index of the passing kind.
  void printOptionalStarSlash(size_t idx);

  /// Print a single trailing '/' at the end of a signature if needed.
  void printOptionalTrailingSlash(size_t idx) const;

private:
  raw_ostream &os;
  size_t numPogs;
  std::function<PassingKind(size_t)> getPassingKind;
  PassingKind prevPassingKind;
  bool suppressSlashAfterSelf;
  char slash; // TODO: remove this when AsmParser can handle '/'.
  StringRef plus;
};

//===----------------------------------------------------------------------===//
// Verifier helpers
//===----------------------------------------------------------------------===//

/// Verify the the order of passing kinds, and that the number of defaults
/// doesn't exceed the number of corresponding passing kinds.
LogicalResult verifyPassingKinds(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<PogMetadataAttr> pogs,
                                 StringRef argOrParam);

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENPOGUTILS_H
