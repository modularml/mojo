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

#include "Mojo/TransformUtils/ManglingUtils.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace KGEN;

void KGEN::prettyPrintParameter(TypedAttr value, raw_ostream &os) {
  if (auto typeCst = dyn_cast<TypeParamAttr>(value)) {
    // Pretty print common type values.
    Type typeValue = typeCst.getTypeValue();
    if (auto structInst = dyn_cast<StructInstanceType>(typeValue)) {
      // Print full struct type name with its parameters recursively.
      os << structInst.getName().strref();
      if (!structInst.getParamValues().empty()) {
        os << '[';
        llvm::interleave(
            structInst.getParamValues(), os,
            [&](TypedAttr paramValue) { prettyPrintParameter(paramValue, os); },
            ",");
        os << ']';
      }
    } else if (auto typeValueType = dyn_cast<TypeValueType>(typeValue)) {
      // Print the wrapped type parameter.
      prettyPrintParameter(typeValueType.getTypeValue(), os);
    } else {
      os << getParamAsString(value);
    }
    return;
  }

  if (auto genref = dyn_cast<TypeGeneratorRefAttr>(value)) {
    // Print type symbol references with its name and its parameters
    // recursively.
    os << genref.getSymbol().getLeafReference().strref();
    if (!genref.getParamValues().empty()) {
      os << '[';
      llvm::interleave(
          genref.getParamValues(), os,
          [&](TypedAttr paramValue) { prettyPrintParameter(paramValue, os); },
          ",");
      os << ']';
    }
    return;
  }

  if (auto typeInstanceRef = dyn_cast<TypeInstanceRefAttr>(value)) {
    os << typeInstanceRef.getSymbol().getLeafReference().strref();
    return;
  }

  // Fallback to default format.
  os << getParamAsString(value);
}

//===----------------------------------------------------------------------===//
// mangleParameterValues
//===----------------------------------------------------------------------===//

namespace {

/// A mangling fragment collected from the parameter attr tree.
///
/// NotEscaped fragments come from freshly printed / source text and may
/// contain `@` or `~`. Escaped fragments are already-encoded names or type
/// prints that may embed them.
enum class FragmentKind { NotEscaped, Escaped };

struct Fragment {
  FragmentKind kind;
  std::string text;
};

/// Printable encoding of `@` / `"` for ELF-safe, printer-stable names.
///
/// `@` and `"` are invalid ELF symbols. Use a printable
/// scheme instead:
///   `@` -> `~A`
///   `"` -> `~Q`
///   `~` -> `~~`
void appendEncodedNotEscaped(StringRef text, std::string &out) {
  for (char c : text) {
    if (c == '~')
      out += "~~";
    else if (c == '@')
      out += "~A";
    else if (c == '"')
      out += "~Q";
    else
      out.push_back(c);
  }
}

/// Sanitize an already-escaped name: replace any lingering `@` / `"`, but do
/// not re-encode `~` (that would turn a prior `~A` / `~Q` into `~~A` / `~~Q`).
void appendSanitizedEscaped(StringRef text, std::string &out) {
  for (char c : text) {
    if (c == '@')
      out += "~A";
    else if (c == '"')
      out += "~Q";
    else
      out.push_back(c);
  }
}

std::string joinAndEncode(ArrayRef<Fragment> fragments) {
  size_t capacity = 0;
  for (const Fragment &fragment : fragments)
    capacity += fragment.text.size() + 1;
  std::string result;
  result.reserve(capacity * 2);
  for (const Fragment &fragment : fragments) {
    if (fragment.kind == FragmentKind::NotEscaped)
      appendEncodedNotEscaped(fragment.text, result);
    else
      appendSanitizedEscaped(fragment.text, result);
  }
  return result;
}

void collectParameterFragments(TypedAttr value, SmallVectorImpl<Fragment> &out);

void collectSymbolRef(SymbolRefAttr symbol, SmallVectorImpl<Fragment> &out) {
  auto appendEscapedName = [&](StringRef name) {
    if (name.starts_with("@"))
      name = name.drop_front();
    out.push_back({FragmentKind::Escaped, name.str()});
  };

  appendEscapedName(symbol.getRootReference().getValue());
  for (FlatSymbolRefAttr nested : symbol.getNestedReferences()) {
    out.push_back({FragmentKind::NotEscaped, "::"});
    appendEscapedName(nested.getValue());
  }
}

void collectPrintedParam(TypedAttr value, FragmentKind kind,
                         SmallVectorImpl<Fragment> &out) {
  std::string printed = getParamAsString(value);
  StringRef text = printed;
  if (text.starts_with("@"))
    text = text.drop_front();
  out.push_back({kind, text.str()});
}

void collectTypeForMangling(Type type, SmallVectorImpl<Fragment> &out);

/// Emit `#kgen.instref<sym>` / `#kgen.genref<sym<...>>` without going through
/// the asm printer.
void collectTypeRefAttr(TypeGeneratorRefAttr genref,
                        SmallVectorImpl<Fragment> &out) {
  out.push_back({FragmentKind::NotEscaped, "#kgen.genref<"});
  collectSymbolRef(genref.getSymbol(), out);
  if (!genref.getParamValues().empty()) {
    out.push_back({FragmentKind::NotEscaped, "<"});
    llvm::interleave(
        genref.getParamValues(),
        [&](TypedAttr paramValue) {
          collectParameterFragments(paramValue, out);
        },
        [&] { out.push_back({FragmentKind::NotEscaped, ", "}); });
    out.push_back({FragmentKind::NotEscaped, ">"});
  }
  out.push_back({FragmentKind::NotEscaped, ">"});
}

void collectTypeRefAttr(TypeInstanceRefAttr instref,
                        SmallVectorImpl<Fragment> &out) {
  out.push_back({FragmentKind::NotEscaped, "#kgen.instref<"});
  collectSymbolRef(instref.getSymbol(), out);
  out.push_back({FragmentKind::NotEscaped, ">"});
}

/// Print a type for mangling without feeding SymbolRefs through the asm
/// printer.
void collectTypeForMangling(Type type, SmallVectorImpl<Fragment> &out) {
  if (auto typeValueType = dyn_cast<TypeValueType>(type)) {
    out.push_back({FragmentKind::NotEscaped, "typevalue<"});
    collectParameterFragments(typeValueType.getTypeValue(), out);
    out.push_back({FragmentKind::NotEscaped, ">"});
    return;
  }

  if (auto structInst = dyn_cast<StructInstanceType>(type)) {
    // Struct instance names may already be encoded specializations.
    out.push_back({FragmentKind::Escaped, structInst.getName().str()});
    if (!structInst.getParamValues().empty()) {
      out.push_back({FragmentKind::NotEscaped, "["});
      llvm::interleave(
          structInst.getParamValues(),
          [&](TypedAttr paramValue) {
            collectParameterFragments(paramValue, out);
          },
          [&] { out.push_back({FragmentKind::NotEscaped, ","}); });
      out.push_back({FragmentKind::NotEscaped, "]"});
    }
    return;
  }

  // Fallback to asm printer; we have special cased all the types that introduce
  // escaped characters that can result in escape character bloat due to
  // nesting.
  std::string printed;
  {
    llvm::raw_string_ostream os(printed);
    printKGENType(os, type);
  }
  out.push_back({FragmentKind::Escaped, std::move(printed)});
}

void collectParameterFragments(TypedAttr value,
                               SmallVectorImpl<Fragment> &out) {
  // Flatten ParamListAttr / packs into a bracketed sequence of fragments.
  if (auto variadic = dyn_cast<ParamListAttr>(value)) {
    out.push_back({FragmentKind::NotEscaped, "["});
    llvm::interleave(
        variadic.getValues(),
        [&](TypedAttr paramValue) {
          collectParameterFragments(paramValue, out);
        },
        [&] { out.push_back({FragmentKind::NotEscaped, ","}); });
    out.push_back({FragmentKind::NotEscaped, "]"});
    return;
  }

  // Function / generator parameters print as a bare symbol name (plus optional
  // bound params). Collect that name directly so mangling stays `fn=sillyFn`.
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(value)) {
    collectSymbolRef(symbolCst.getSymbol(), out);
    if (!symbolCst.getParamValues().empty()) {
      out.push_back({FragmentKind::NotEscaped, "<"});
      llvm::interleave(
          symbolCst.getParamValues(),
          [&](TypedAttr paramValue) {
            collectParameterFragments(paramValue, out);
          },
          [&] { out.push_back({FragmentKind::NotEscaped, ", "}); });
      out.push_back({FragmentKind::NotEscaped, ">"});
    }
    return;
  }

  // Type refs: same as symbols — never asm-print (avoids `@"..."` / `\22`).
  if (auto genref = dyn_cast<TypeGeneratorRefAttr>(value)) {
    collectTypeRefAttr(genref, out);
    return;
  }
  if (auto instref = dyn_cast<TypeInstanceRefAttr>(value)) {
    collectTypeRefAttr(instref, out);
    return;
  }

  // Type parameters: mirror printSugaredTypeValue structurally so nested
  // typevalue / instref / genref paths stay quote-free.
  if (auto typeCst = dyn_cast<TypeParamAttr>(value)) {
    bool nonTrivial = !typeCst.hasIdenticalRepresentation();
    if (nonTrivial)
      out.push_back({FragmentKind::NotEscaped, "["});
    collectTypeForMangling(typeCst.getTypeValue(), out);
    if (nonTrivial) {
      out.push_back({FragmentKind::NotEscaped, ", "});
      collectTypeForMangling(typeCst.getMlirType(), out);
      out.push_back({FragmentKind::NotEscaped, "]"});
    }
    return;
  }

  // Strings and other source text: encode `@` / `"` / `~` at join time.
  collectPrintedParam(value, FragmentKind::NotEscaped, out);
}

} // namespace

std::string KGEN::mangleParameterValues(GeneratorOpInterface generator,
                                        ArrayRef<TypedAttr> inputParamValues) {
  SmallVector<Fragment, 16> fragments;
  // Generator name may already be an encoded specialization: treat as Escaped.
  fragments.push_back({FragmentKind::Escaped, generator.getName().str()});

  if (inputParamValues.empty())
    return joinAndEncode(fragments);

  auto inputParamDecls = generator.getInputParams();
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues)) {
    fragments.push_back({FragmentKind::NotEscaped, ","});
    fragments.push_back({FragmentKind::NotEscaped, inputDecl.getName().str()});
    fragments.push_back({FragmentKind::NotEscaped, "="});
    collectParameterFragments(value, fragments);
  }

  return joinAndEncode(fragments);
}
