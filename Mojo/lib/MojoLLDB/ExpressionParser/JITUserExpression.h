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

#ifndef KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_JITUSEREXPRESSION_H
#define KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_JITUSEREXPRESSION_H

#include "lldb/Expression/UserExpression.h"
#include <string>
#include <vector>

namespace M::KGEN::Mojo {
class JITExecutionUnit;

/// This class encapsulates a one-time expression for use in lldb. LLDB uses
/// expressions for various purposes, notably to call functions and as a backend
/// for the expr command.
class JitUserExpression : public lldb_private::UserExpression {
  static char ID;

public:
  JitUserExpression(lldb_private::ExecutionContextScope &exeScope,
                    llvm::StringRef expr, llvm::StringRef prefix,
                    lldb_private::SourceLanguage language,
                    ResultType desiredType,
                    const lldb_private::EvaluateExpressionOptions &options);
  ~JitUserExpression() override;

  bool FinalizeJITExecution(
      lldb_private::DiagnosticManager &diagnosticManager,
      lldb_private::ExecutionContext &exeCtx,
      lldb::ExpressionVariableSP &result,
      lldb::addr_t functionStackBottom = LLDB_INVALID_ADDRESS,
      lldb::addr_t functionStackTop = LLDB_INVALID_ADDRESS) override;

  bool CanInterpret() override { return false; }

  lldb_private::Materializer *GetMaterializer() override {
    return materializer.get();
  }

  //--------------------------------------------------------------------------//
  // llvm casting support
  //--------------------------------------------------------------------------//

  bool isA(const void *classID) const override {
    return classID == &ID || UserExpression::isA(classID);
  }
  static bool classof(const Expression *obj) { return obj->isA(&ID); }

protected:
  lldb::ExpressionResults
  DoExecute(lldb_private::DiagnosticManager &diagnosticManager,
            lldb_private::ExecutionContext &exeCtx,
            const lldb_private::EvaluateExpressionOptions &options,
            lldb::UserExpressionSP &sharedPtrToMe,
            lldb::ExpressionVariableSP &result) override;

  bool prepareToExecuteJITExpression(
      lldb_private::DiagnosticManager &diagnosticManager,
      lldb_private::ExecutionContext &exeCtx, lldb::addr_t &structAddress);

  virtual bool
  addArguments(lldb_private::ExecutionContext &exeCtx,
               std::vector<lldb::addr_t> &args, lldb::addr_t structAddress,
               lldb_private::DiagnosticManager &diagnosticManager) = 0;

  /// The execution unit the expression is stored in.
  std::shared_ptr<JITExecutionUnit> executionUnit;
  /// The materializer to use when running the expression.
  std::unique_ptr<lldb_private::Materializer> materializer;
  /// The module containing the expression.
  lldb::ModuleWP jitModule;
  /// The target for storing persistent data like types and variables.
  lldb_private::Target *target = nullptr;

  /// The address at which the arguments to the expression have been
  /// materialized.
  lldb::addr_t materializedAddress = LLDB_INVALID_ADDRESS;
  /// The dematerializer.
  lldb_private::Materializer::DematerializerSP dematerializer;
};

} // namespace M::KGEN::Mojo
#endif
