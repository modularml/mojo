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

#ifndef KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOUSEREXPRESSION_H
#define KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOUSEREXPRESSION_H

#include "JITUserExpression.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "lldb/Expression/LLVMUserExpression.h"

namespace M::KGEN::Mojo {
class MojoPersistentExpressionState;
class MojoTypeSystem;

//===----------------------------------------------------------------------===//
// MojoUserExpression
//===----------------------------------------------------------------------===//

/// MojoUserExpression encapsulates the objects needed to parse and interpret or
/// JIT an expression.
class MojoUserExpression : public JitUserExpression {
  static char ID;

public:
  MojoUserExpression(lldb_private::ExecutionContextScope &exeScope,
                     llvm::StringRef expr, llvm::StringRef prefix,
                     lldb_private::SourceLanguage language,
                     ResultType desiredType,
                     const lldb_private::EvaluateExpressionOptions &options);
  ~MojoUserExpression() override;

  //===--------------------------------------------------------------------===//
  // Expression parsing and execution
  //===--------------------------------------------------------------------===//

  /// Return the function name that should be used for executing the expression.
  const char *FunctionName() override;
  /// Set the function name that should be used for executing the expression.
  void setFunctionName(std::string exprFnName);

  /// Return the module name used to wrap the expression if it is a python
  /// expression. Returns nullopt if this is a pure mojo expression.
  const std::optional<std::string> &getPythonModuleName();

  /// Parse the expression.
  bool Parse(lldb_private::DiagnosticManager &diagnosticManager,
             lldb_private::ExecutionContext &exeCtx,
             lldb_private::ExecutionPolicy executionPolicy,
             bool keepResultInMemory, bool generateDebugInfo) override;

  /// Return the type system helper for this expression.
  lldb_private::ExpressionTypeSystemHelper *GetTypeSystemHelper() override;

  /// Return the result variable for this expression after dematerialization.
  lldb::ExpressionVariableSP GetResultAfterDematerialization(
      lldb_private::ExecutionContextScope *exeScope) override;

  /// Set the fixed expression text for this expression.
  void setFixedText(StringRef fixedText) { m_fixed_text = fixedText.str(); }

  //===--------------------------------------------------------------------===//
  // RTTI support
  //===--------------------------------------------------------------------===//

  bool isA(const void *classID) const override {
    return classID == &ID || JitUserExpression::isA(classID);
  }
  static bool classof(const Expression *obj) { return obj->isA(&ID); }

private:
  //===--------------------------------------------------------------------===//
  // Expression parsing and execution
  //===--------------------------------------------------------------------===//

  /// Add the function arguments used when invoking the wrapper function for the
  /// generated expression.
  bool
  addArguments(lldb_private::ExecutionContext &exeCtx,
               std::vector<lldb::addr_t> &args, lldb::addr_t structAddress,
               lldb_private::DiagnosticManager &diagnosticManager) override;

  /// Process and wrap the expression text, and then parse it.
  LogicalResult
  wrapTextAndParseExpression(lldb_private::DiagnosticManager &diagnosticManager,
                             lldb_private::ExecutionContext &exeCtx,
                             lldb_private::ExecutionContextScope *exeScope,
                             MojoPersistentExpressionState &state);

  /// Process and wrap the given expression text, which contains python
  /// expressions, and then parse it.
  LogicalResult wrapTextAndParsePythonExpression(
      StringRef pythonExpr, lldb_private::DiagnosticManager &diagnosticManager,
      lldb_private::ExecutionContext &exeCtx,
      lldb_private::ExecutionContextScope *exeScope,
      MojoPersistentExpressionState &state);

  /// Handle the execution of the expression.
  lldb::ExpressionResults
  DoExecute(lldb_private::DiagnosticManager &diagnosticManager,
            lldb_private::ExecutionContext &exeCtx,
            const lldb_private::EvaluateExpressionOptions &options,
            lldb::UserExpressionSP &sharedPtrToMe,
            lldb::ExpressionVariableSP &result) override;

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  struct Impl;

  std::unique_ptr<Impl> impl;
};

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOUSEREXPRESSION_H
