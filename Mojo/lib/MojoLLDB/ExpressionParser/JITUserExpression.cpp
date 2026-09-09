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

#include "JITUserExpression.h"
#include "JITExecutionUnit.h"
#include "lldb/Core/Module.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Expression/IRInterpreter.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallUserExpression.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/ErrorMessages.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"

using namespace M::KGEN::Mojo;
using namespace lldb_private;

namespace {
//----------------------------------------------------------------------------//
// ThreadPlanCallMojoUserExpression
//----------------------------------------------------------------------------//

class ThreadPlanCallMojoUserExpression : public ThreadPlanCallUserExpression {
public:
  using ThreadPlanCallUserExpression::ThreadPlanCallUserExpression;

  bool ShouldAutoContinue(Event *event_ptr) override {
    Log *log = GetLog(LLDBLog::Expressions | LLDBLog::Step);

    if (auto realStopInfo = GetPrivateStopInfo()) {
      LLDB_LOG(log,
               "ThreadPlanCallMojoUserExpression::ShouldAutoContinue: Got stop "
               "reason - {0}.",
               Thread::StopReasonAsString(realStopInfo->GetStopReason()));

      switch (realStopInfo->GetStopReason()) {
      case lldb::eStopReasonFork:
      case lldb::eStopReasonVFork:
      case lldb::eStopReasonVForkDone:
      case lldb::eStopReasonExec:
        // On Linux, LLDB's user expression thread plans abruptly stop on forks
        // and execs by default, which leaves the expression unterminated.
        // However, that's not the behavior we want, especially because we rely
        // on the mojo invoking python for its own python setup.
        LLDB_LOG(log, "ThreadPlanCallMojoUserExpression::ShouldAutoContinue: "
                      "Will auto continue.");

        return true;
      default:
        return false;
      }
    }
    return false;
  }
};
} // namespace

//----------------------------------------------------------------------------//
// JitUserExpression
//----------------------------------------------------------------------------//

JitUserExpression::JitUserExpression(ExecutionContextScope &exeScope,
                                     llvm::StringRef expr,
                                     llvm::StringRef prefix,
                                     lldb_private::SourceLanguage language,
                                     ResultType desiredType,
                                     const EvaluateExpressionOptions &options)
    : UserExpression(exeScope, expr, prefix, language, desiredType, options) {}

JitUserExpression::~JitUserExpression() {
  if (!target)
    return;
  if (lldb::ModuleSP jitModuleSP = jitModule.lock())
    target->GetImages().Remove(jitModuleSP);
}

char JitUserExpression::ID;

//----------------------------------------------------------------------------//
// Execution
//----------------------------------------------------------------------------//

lldb::ExpressionResults JitUserExpression::DoExecute(
    DiagnosticManager &diagnosticManager, ExecutionContext &exeCtx,
    const EvaluateExpressionOptions &options,
    lldb::UserExpressionSP &sharedPtrToMe, lldb::ExpressionVariableSP &result) {
  Log *log = GetLog(LLDBLog::Expressions | LLDBLog::Step);

  if (m_jit_start_addr == LLDB_INVALID_ADDRESS) {
    diagnosticManager.PutString(
        lldb::eSeverityError,
        "Expression can't be run, because there is no JIT compiled function");
    return lldb::eExpressionSetupError;
  }

  lldb::addr_t structAddress = LLDB_INVALID_ADDRESS;
  if (!prepareToExecuteJITExpression(diagnosticManager, exeCtx,
                                     structAddress)) {
    diagnosticManager.Printf(
        lldb::eSeverityError,
        "errored out in %s, couldn't PrepareToExecuteJITExpression",
        __FUNCTION__);
    return lldb::eExpressionSetupError;
  }
  if (!exeCtx.HasThreadScope()) {
    diagnosticManager.Printf(lldb::eSeverityError,
                             "%s called with no thread selected", __FUNCTION__);
    return lldb::eExpressionSetupError;
  }

  // Store away the thread ID for error reporting, in case it exits
  // during execution:
  lldb::tid_t exprThreadId = exeCtx.GetThreadRef().GetID();

  lldb::addr_t functionStackBottom = LLDB_INVALID_ADDRESS;
  lldb::addr_t functionStackTop = LLDB_INVALID_ADDRESS;
  Address wrapperAddress(m_jit_start_addr);

  std::vector<lldb::addr_t> args;
  if (!addArguments(exeCtx, args, structAddress, diagnosticManager)) {
    diagnosticManager.Printf(lldb::eSeverityError,
                             "errored out in %s, couldn't AddArguments",
                             __FUNCTION__);
    return lldb::eExpressionSetupError;
  }

  lldb::ThreadPlanSP callPlan(new ThreadPlanCallMojoUserExpression(
      exeCtx.GetThreadRef(), wrapperAddress, args, options, sharedPtrToMe));

  StreamString ss;
  if (!callPlan || !callPlan->ValidatePlan(&ss)) {
    diagnosticManager.PutString(lldb::eSeverityError, ss.GetString());
    return lldb::eExpressionSetupError;
  }

  ThreadPlanCallMojoUserExpression *userExpressionPlan =
      static_cast<ThreadPlanCallMojoUserExpression *>(callPlan.get());

  lldb::addr_t functionStackPointer =
      userExpressionPlan->GetFunctionStackPointer();

  functionStackBottom = functionStackPointer - HostInfo::GetPageSize();
  functionStackTop = functionStackPointer;

  LLDB_LOGF(
      log, "-- [JITUserExpression::Execute] Execution of expression begins --");

  if (exeCtx.GetProcessPtr())
    exeCtx.GetProcessPtr()->SetRunningUserExpression(true);

  lldb::ExpressionResults executionResult =
      exeCtx.GetProcessRef().RunThreadPlan(exeCtx, callPlan, options,
                                           diagnosticManager);

  if (exeCtx.GetProcessPtr())
    exeCtx.GetProcessPtr()->SetRunningUserExpression(false);

  LLDB_LOGF(log, "-- [JITUserExpression::Execute] Execution of expression "
                 "completed --");

  if (executionResult == lldb::eExpressionInterrupted ||
      executionResult == lldb::eExpressionHitBreakpoint) {
    const char *errorDesc = nullptr;
    if (userExpressionPlan) {
      if (auto realStopInfoSp = userExpressionPlan->GetRealStopInfo())
        errorDesc = realStopInfoSp->GetDescription();
    }
    if (errorDesc)
      diagnosticManager.Printf(lldb::eSeverityError,
                               "Execution was interrupted, reason: %s.",
                               errorDesc);
    else
      diagnosticManager.PutString(lldb::eSeverityError,
                                  "Execution was interrupted.");

    if ((executionResult == lldb::eExpressionInterrupted &&
         options.DoesUnwindOnError()) ||
        (executionResult == lldb::eExpressionHitBreakpoint &&
         options.DoesIgnoreBreakpoints()))
      diagnosticManager.AppendMessageToDiagnostic(
          "The process has been returned to the state before expression "
          "evaluation.");
    else {
      if (executionResult == lldb::eExpressionHitBreakpoint)
        userExpressionPlan->TransferExpressionOwnership();
      diagnosticManager.AppendMessageToDiagnostic(
          "The process has been left at the point where it was "
          "interrupted, "
          "use \"thread return -x\" to return to the state before "
          "expression evaluation.");
    }

    return executionResult;
  }
  if (executionResult == lldb::eExpressionStoppedForDebug) {
    diagnosticManager.PutString(
        lldb::eSeverityInfo,
        "Execution was halted at the first instruction of the expression "
        "function because \"debug\" was requested.\n"
        "Use \"thread return -x\" to return to the state before expression "
        "evaluation.");
    return executionResult;
  }
  if (executionResult == lldb::eExpressionThreadVanished) {
    diagnosticManager.Printf(lldb::eSeverityError,
                             "Couldn't complete execution; the thread "
                             "on which the expression was being run: 0x%" PRIx64
                             " exited during its execution.",
                             exprThreadId);
    return executionResult;
  }
  if (executionResult != lldb::eExpressionCompleted) {
    diagnosticManager.Printf(lldb::eSeverityError,
                             "Couldn't execute function; result was %s",
                             toString(executionResult).c_str());
    return executionResult;
  }

  if (FinalizeJITExecution(diagnosticManager, exeCtx, result,
                           functionStackBottom, functionStackTop))
    return lldb::eExpressionCompleted;
  return lldb::eExpressionResultUnavailable;
}

bool JitUserExpression::FinalizeJITExecution(
    DiagnosticManager &diagnosticManager, ExecutionContext &exeCtx,
    lldb::ExpressionVariableSP &result, lldb::addr_t functionStackBottom,
    lldb::addr_t functionStackTop) {
  Log *log = GetLog(LLDBLog::Expressions);

  LLDB_LOGF(log, "-- [JITUserExpression::FinalizeJITExecution] Dematerializing "
                 "after execution --");

  if (!dematerializer) {
    diagnosticManager.Printf(lldb::eSeverityError,
                             "Couldn't apply expression side effects : no "
                             "dematerializer is present");
    return false;
  }

  Status dematerializeError;
  dematerializer->Dematerialize(dematerializeError, functionStackBottom,
                                functionStackTop);
  if (!dematerializeError.Success()) {
    diagnosticManager.Printf(lldb::eSeverityError,
                             "Couldn't apply expression side effects : %s",
                             dematerializeError.AsCString("unknown error"));
    return false;
  }

  result =
      GetResultAfterDematerialization(exeCtx.GetBestExecutionContextScope());
  if (result)
    result->TransferAddress();
  dematerializer.reset();
  return true;
}

bool JitUserExpression::prepareToExecuteJITExpression(
    DiagnosticManager &diagnosticManager, ExecutionContext &exeCtx,
    lldb::addr_t &structAddress) {
  lldb::TargetSP target;
  lldb::ProcessSP process;
  lldb::StackFrameSP frame;

  if (!LockAndCheckContext(exeCtx, target, process, frame)) {
    diagnosticManager.PutString(
        lldb::eSeverityError,
        "The context has changed before we could JIT the expression!");
    return false;
  }

  if (m_jit_start_addr != LLDB_INVALID_ADDRESS) {
    if (materializedAddress == LLDB_INVALID_ADDRESS) {
      llvm::Expected<lldb::addr_t> materializedAddressOr =
          executionUnit->Malloc(
              materializer->GetStructByteSize(),
              materializer->GetStructAlignment(),
              lldb::ePermissionsReadable | lldb::ePermissionsWritable,
              IRMemoryMap::eAllocationPolicyMirror, /*zero_memory=*/false,
              /*used_policy=*/nullptr);
      if (llvm::Error error = materializedAddressOr.takeError()) {
        std::string errorStr = llvm::toString(std::move(error));
        diagnosticManager.Printf(
            lldb::eSeverityError,
            "Couldn't allocate space for materialized struct: %s",
            errorStr.c_str());
        return false;
      }
      materializedAddress = std::move(materializedAddressOr.get());
    }
    structAddress = materializedAddress;

    Status materializeError;
    dematerializer = materializer->Materialize(frame, *executionUnit,
                                               structAddress, materializeError);
    if (!materializeError.Success()) {
      diagnosticManager.Printf(lldb::eSeverityError, "Couldn't materialize: %s",
                               materializeError.AsCString());
      return false;
    }
  }
  return true;
}
