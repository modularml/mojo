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
// Placeholder scheme for the `__mlir_op[`...`]` f-string. Users write only:
//   %{name}          the in-scope Mojo value `name`
//   %{type_of(name)}  the MLIR type of `name`
//
// `emitFStringSubscriptMLIROp` (the parser) rewrites each use into one of the
// internal placeholders below, chosen by what `name` resolves to:
//   %arg<N>          `name` is an SSA value -> operand N, referenced through a
//                    synthetic `kgen.func` entry block when parsed here.
//   %param<N>          `name` is a comptime/parametric value (no SSA form) ->
//                    entry N of the op's `fstring_params` array, stringified
//                    and inlined by the elaborator once parameters bind.
//   %type_of(arg<N>)  `%{type_of(name)}` where operand N's type is parametric
//   ->
//                    replaced with the concrete type here once known. (A
//                    concrete operand type is printed inline by the parser.)
//
// An op carries `%param<N>` / `%type_of(arg<N>)` only while parametric, so it
// is stashed on `kgen.deferred` for the elaborator; a fully concrete op is
// lowered immediately. Worked example (`T` and `tag` bound at the call site):
//
//   Mojo:
//     fn f[T: dtype](x: __mlir_type[`!kgen.scalar<`, T, `>`]):
//         comptime tag = __mlir_attr.`5 : i64`
//         __mlir_op[`d.op %{x} {t = %{tag}} : %{type_of(x)}`,
//                   _type=__mlir_deferred_type]
//
//   After the parser (on `kgen.deferred`, `T` still unbound). The angle
//   brackets bound each index so an adjacent template digit can't be absorbed:
//     opName           = "d.op %arg<0> {t = %param<0>} : %type_of(arg<0>)"
//     fstring_params = [#kgen<to_string_deferred(... tag ...)>]
//     operand 0        = %x
//
//   In the elaborator just before re-parse (T = si32; %param<0> inlined;
//   wrapped so %arg<0> -> %arg0 and %type_of(arg0) -> the concrete operand
//   type):
//     kgen.func @__mojo_inline_mlir_op(%arg0: !kgen.scalar<si32>) {
//       d.op %arg0 {t = 5} : !kgen.scalar<si32>
//       kgen.return
//     }
//
//   Resulting op (parsed, operands re-bound to the real SSA values):
//     d.op %x {t = 5} : !kgen.scalar<si32>
//
//===----------------------------------------------------------------------===//

#include "Mojo/ToolCommon/MLIROpFString.h"

#include "Mojo/KGENDialect/KGENOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace M::KGEN {

bool isFStringTemplate(llvm::StringRef tmpl) {
  size_t pos = 0;
  while ((pos = tmpl.find('%', pos)) != llvm::StringRef::npos) {
    llvm::StringRef rest = tmpl.drop_front(pos + 1);
    if (rest.starts_with("arg<") || rest.starts_with("param<") ||
        rest.starts_with("type_of("))
      return true;
    ++pos;
  }
  return false;
}

llvm::StringRef getFStringParamsAttrName() { return "fstring_params"; }

bool scanFStringTemplate(
    llvm::StringRef tmpl, llvm::raw_ostream &out,
    llvm::function_ref<int(llvm::StringRef, llvm::raw_ostream &)>
        onPlaceholder) {
  enum class State { Normal, String };
  State state = State::Normal;
  size_t i = 0;
  while (i < tmpl.size()) {
    char c = tmpl[i];
    if (state == State::String) {
      out << c;
      if (c == '\\' && i + 1 < tmpl.size()) {
        out << tmpl[i + 1];
        i += 2;
        continue;
      }
      if (c == '"')
        state = State::Normal;
      ++i;
      continue;
    }
    if (c == '"') {
      state = State::String;
      out << c;
      ++i;
      continue;
    }
    if (c == '%') {
      int consumed = onPlaceholder(tmpl.drop_front(i + 1), out);
      if (consumed < 0)
        return false;
      if (consumed == 0)
        out << c;
      i += consumed > 0 ? size_t(1 + consumed) : 1;
      continue;
    }
    out << c;
    ++i;
  }
  return true;
}

namespace {

/// If `rest` begins with `<N>`, set `n` and return the chars through the '>',
/// else 0. The '>' bounds the index so adjacent template digits aren't
/// absorbed.
size_t matchBracketIndex(llvm::StringRef rest, size_t &n) {
  if (!rest.starts_with("<"))
    return 0;
  size_t len = 1;
  while (len < rest.size() && llvm::isDigit(rest[len]))
    ++len;
  if (len == 1 || len >= rest.size() || rest[len] != '>')
    return 0;
  if (rest.slice(1, len).getAsInteger(10, n))
    return 0;
  return len + 1;
}

/// Lower the placeholders left behind: `%arg<N>` → the synthetic SSA name
/// `%argN` (resolved via the func entry block), `%type_of(arg<N>)` → operand
/// N's type text. `//` comments are not recognized.
LogicalResult substitutePlaceholders(llvm::StringRef tmpl,
                                     llvm::ArrayRef<Value> operands,
                                     llvm::raw_ostream &out,
                                     std::string &errorMsg) {
  bool ok = scanFStringTemplate(
      tmpl, out, [&](llvm::StringRef rest, llvm::raw_ostream &os) -> int {
        if (rest.consume_front("type_of(arg")) {
          size_t n = 0;
          size_t idxLen = matchBracketIndex(rest, n);
          if (idxLen == 0 || !rest.drop_front(idxLen).starts_with(")"))
            return 0;
          if (n >= operands.size()) {
            llvm::raw_string_ostream eos(errorMsg);
            eos << "%type_of(arg<" << n << ">) out of range (have "
                << operands.size() << " operands)";
            return -1;
          }
          operands[n].getType().print(os);
          return int(llvm::StringRef("type_of(arg").size() + idxLen + 1);
        }
        if (rest.consume_front("arg")) {
          size_t n = 0;
          size_t idxLen = matchBracketIndex(rest, n);
          if (idxLen == 0)
            return 0;
          if (n >= operands.size()) {
            llvm::raw_string_ostream eos(errorMsg);
            eos << "%arg<" << n << "> out of range (have " << operands.size()
                << " operands)";
            return -1;
          }
          os << "%arg" << n;
          return int(llvm::StringRef("arg").size() + idxLen);
        }
        if (rest.starts_with("type") &&
            (rest.size() == 4 || !llvm::isAlpha(rest[4]))) {
          errorMsg = "`%type` placeholder is no longer supported; inline the "
                     "result type text directly or use `%{type_of(name)}`";
          return -1;
        }
        return 0;
      });
  return success(ok);
}

} // namespace

Operation *lowerFStringMLIROp(OpBuilder &builder, Location loc,
                              llvm::StringRef tmpl,
                              llvm::ArrayRef<Value> operands,
                              llvm::ArrayRef<Type> resultTypes,
                              std::string &errorMsg, llvm::StringRef userTmpl) {
  MLIRContext *context = builder.getContext();

  // Names the offending f-string in every failure; prefer the original source.
  std::string inTmpl =
      (" in f-string `" + (userTmpl.empty() ? tmpl : userTmpl) + "`").str();

  std::string body;
  {
    llvm::raw_string_ostream os(body);
    if (failed(substitutePlaceholders(tmpl, operands, os, errorMsg))) {
      errorMsg += inTmpl;
      return nullptr;
    }
  }

  // Wrap in `kgen.func` (always loaded), not `func.func`: parsing the latter
  // lazily loads the `func` dialect, which crashes under a multi-threaded pass
  // manager (e.g. mo-opt).
  std::string wrapped;
  {
    llvm::raw_string_ostream os(wrapped);
    os << "kgen.func @__mojo_inline_mlir_op(";
    for (auto [i, v] : llvm::enumerate(operands)) {
      if (i)
        os << ", ";
      os << "%arg" << i << ": ";
      v.getType().print(os);
    }
    os << ") {\n  " << body << "\n  kgen.return\n}";
  }

  // `verifyAfterParse=false`: op verifiers that consult `getDefiningOp()`
  // would reject the synthetic `%arg<N>` block-arg operands. Verification
  // runs after RAUW'ing the real operand values below.
  std::string parseErrors;
  OwningOpRef<ModuleOp> modOr;
  {
    llvm::raw_string_ostream errsOS(parseErrors);
    ScopedDiagnosticHandler diagHandler(context, [&](Diagnostic &diag) {
      errsOS << diag.str() << "\n";
      for (auto &note : diag.getNotes())
        errsOS << "  note: " << note.str() << "\n";
    });
    ParserConfig config(context, /*verifyAfterParse=*/false);
    modOr = parseSourceString<ModuleOp>(wrapped, config);
  }
  if (!modOr) {
    errorMsg = "failed to parse f-string MLIR op" + inTmpl + ": " + parseErrors;
    return nullptr;
  }

  auto func = dyn_cast<FuncOp>(modOr->getBody()->front());
  if (!func || func.getBodyRegion().empty()) {
    errorMsg = "no op" + inTmpl;
    return nullptr;
  }
  // Expect exactly one user op plus the synthesized `kgen.return`; reject more
  // rather than silently keeping the first.
  size_t numOps = func.getBodyRegion().front().getOperations().size();
  if (numOps < 2) {
    errorMsg = "no op" + inTmpl;
    return nullptr;
  }
  if (numOps > 2) {
    errorMsg = "expected exactly one op" + inTmpl;
    return nullptr;
  }

  Operation *parsedOp = &func.getBodyRegion().front().front();
  if (parsedOp->hasTrait<OpTrait::IsTerminator>()) {
    errorMsg = "f-string MLIR op cannot be a terminator" + inTmpl;
    return nullptr;
  }

  for (auto [arg, v] :
       llvm::zip(func.getBodyRegion().front().getArguments(), operands))
    arg.replaceAllUsesWith(v);

  parsedOp->setLoc(loc);
  parsedOp->remove();
  builder.insert(parsedOp);

  // NOTE: this operation is not verified at this stage. KGENVerifier,
  // which is invoked after Elaborator, will verify all operations
  // from dialects used by Mojo: KGEN, LIT, POP, NVVM, HLCF, ROCDL.

  return parsedOp;
}

} // namespace M::KGEN
