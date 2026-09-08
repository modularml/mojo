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

#ifndef KGEN_MOJOPARSER_PARSERPARAMETEREVALUATOR_H
#define KGEN_MOJOPARSER_PARSERPARAMETEREVALUATOR_H

#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include <utility>

namespace M::KGEN::LIT {
class SharedState;

/// An evaluation context that uses the parser's SharedState to evaluate
/// expressions. Inherits common dispatch logic from the base class and
/// provides parser-specific struct resolution and context handling.
class ParserEvaluationContext : public ParameterEvaluationContext {
public:
  /// Creates an attribute and immediately tries to simplify it via
  /// evaluateExpression. Returns the simplified result if successful,
  /// otherwise returns the original attribute.
  template <typename AttrT, typename... Args>
  TypedAttr getAndFold(Args &&...args) {
    auto attr = AttrT::get(std::forward<Args>(args)...);
    if (auto toEval =
            dyn_cast_if_present<ContextuallyEvaluatedAttrInterface>(attr)) {
      auto simplified = evaluateExpression(toEval);
      if (succeeded(simplified) && simplified.value())
        return simplified.value();
    }
    return attr;
  }

protected:
  /// Resolve struct info using the parser's SharedState.
  FailureOr<ResolvedStructHandle> resolveStructOp(TypedAttr typeValue,
                                                  bool acceptAsync) override;

  /// Resolve a function symbol via the parser's `DeclResolver` to its
  /// underlying `lit.fn`.
  FuncInterface resolveFunctionDecl(SymbolRefAttr symbol) override;

  /// Resolve conformance using ASTDecl lookup in the parser context.
  Operation *resolveConformanceForStruct(ResolvedStructHandle resolved,
                                         TraitSymbolAttr traitSymbol) override;

  /// Provide a ParameterEvaluator configured for the struct parameters.
  void withEvaluator(
      ArrayRef<ParamDeclAttr> paramDecls, ArrayRef<TypedAttr> paramValues,
      llvm::function_ref<void(ParameterEvaluator &)> callback) override;

  /// Handle parser-specific attributes (TypeConformsToTrait, Downcast).
  FailureOr<TypedAttr>
  evaluateContextSpecific(ContextuallyEvaluatedAttrInterface attr) override;

private:
  friend class SharedState;
  ParserEvaluationContext(SharedState &shared) : shared(shared) {}

  SharedState &shared;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARSERPARAMETEREVALUATOR_H
