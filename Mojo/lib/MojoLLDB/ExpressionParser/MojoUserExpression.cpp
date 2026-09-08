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

#include "MojoUserExpression.h"
#include "../Logging/MojoExpressionLogger.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "Logging.h"
#include "Mojo/MojoTooling/REPLPythonExprUtils.h"
#include "MojoDiagnostic.h"
#include "MojoExpressionParser.h"
#include "MojoExpressionVariable.h"
#include "Support/CrashReporting/CrashReporting.h"
#include "Support/FileSystemExtras.h"
#include "Support/Telemetry/Telemetry.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"

using namespace M;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

namespace {
//===----------------------------------------------------------------------===//
// MojoUserExpressionHelper
//===----------------------------------------------------------------------===//

/// An expression helper for Mojo expressions.
class MojoUserExpressionHelper
    : public llvm::RTTIExtends<MojoUserExpressionHelper,
                               ExpressionTypeSystemHelper> {
public:
  // LLVM RTTI support
  static char ID;

  MojoUserExpressionHelper(Target &) {}
};

char MojoUserExpressionHelper::ID;

//===----------------------------------------------------------------------===//
// ResultDelegate
//===----------------------------------------------------------------------===//

/// This class implements a variable delegate for the result of an expression.
class ResultDelegate : public Materializer::PersistentVariableDelegate {
public:
  ResultDelegate(lldb::TargetSP target) : target(std::move(target)) {}

  ConstString GetName() override {
    return persistentState->GetNextPersistentVariableName();
  }

  void DidDematerialize(lldb::ExpressionVariableSP &varArg) override {
    variable = varArg;
  }

  void RegisterPersistentState(PersistentExpressionState &persistentStateArg) {
    persistentState = &persistentStateArg;
  }

  lldb::ExpressionVariableSP &GetVariable() { return variable; }

private:
  lldb::TargetSP target;
  PersistentExpressionState *persistentState;
  lldb::ExpressionVariableSP variable;
};

//===----------------------------------------------------------------------===//
// PersistentVariableDelegate
//===----------------------------------------------------------------------===//

/// This class implements a variable delegate for persistent variables.
class PersistentVariableDelegate
    : public Materializer::PersistentVariableDelegate {
public:
  PersistentVariableDelegate() = default;
  ConstString GetName() override { return ConstString(); }
  void DidDematerialize(lldb::ExpressionVariableSP &variable) override {}
};
} // namespace

//===----------------------------------------------------------------------===//
// MojoUserExpression::Impl
//===----------------------------------------------------------------------===//

static MojoTypeSystem &getMojoTypeSystem(Target &target) {
  if (auto typeSystemOr =
          target.GetScratchTypeSystemForLanguage(lldb::eLanguageTypeMojo))
    return *llvm::cast<MojoTypeSystem>(typeSystemOr.get().get());
  llvm::report_fatal_error(
      "The Mojo type system plug-in must have already been registered.");
}

static MojoPersistentExpressionState &
getMojoPersistentState(MojoTypeSystem &typeSystem) {
  return *llvm::cast<MojoPersistentExpressionState>(
      (typeSystem.GetPersistentExpressionState()));
}

struct MojoUserExpression::Impl {
  Impl(ExecutionContextScope &exeScope, Target &target)
      : target(target), typeSystemHelper(target),
        resultDelegate(target.shared_from_this()),
        typeSystem(getMojoTypeSystem(target)),
        persistentState(getMojoPersistentState(typeSystem)),
        expressionLogger(MojoExpressionLogger::getLoggerForTarget(target)) {}

  /// The target associated with this expression.
  Target &target;

  /// The type system helper.
  MojoUserExpressionHelper typeSystemHelper;

  /// The various expression delegates.
  ResultDelegate resultDelegate;
  PersistentVariableDelegate persistentVariableDelegate;

  /// The name of the expression function once it has been set.
  std::optional<std::string> exprFnName;

  /// The name of the python module that wraps the expression, if the expression
  /// is a Python expression, nullopt otherwise.
  std::optional<std::string> pythonModuleName;

  /// The underlying expression parser.
  std::unique_ptr<MojoExpressionParser> parser;
  MojoTypeSystem &typeSystem;
  MojoPersistentExpressionState &persistentState;
  MojoExpressionLogger &expressionLogger;
};

//===----------------------------------------------------------------------===//
// MojoUserExpression
//===----------------------------------------------------------------------===//

MojoUserExpression::MojoUserExpression(ExecutionContextScope &exeScope,
                                       llvm::StringRef expr,
                                       llvm::StringRef prefix,
                                       lldb_private::SourceLanguage language,
                                       ResultType desiredType,
                                       const EvaluateExpressionOptions &options)
    : JitUserExpression(exeScope, expr, prefix, language, desiredType, options),
      impl(std::make_unique<Impl>(exeScope, *m_target_wp.lock())) {}

MojoUserExpression::~MojoUserExpression() = default;
char MojoUserExpression::ID;

//===----------------------------------------------------------------------===//
// Expression parsing and execution
//===----------------------------------------------------------------------===//

// Process cell magics within the given expression.
static LogicalResult processMagics(DiagnosticManager &diagnosticManager,
                                   std::string &exprText,
                                   MojoTypeSystem &typeSystem) {
  SmallVector<StringRef, 4> lines;
  llvm::SplitString(exprText, lines, "\n");

  // Process each line looking for a magic.
  SmallVector<unsigned> magicPositions;
  bool hadNonMagic = false;
  for (StringRef line : lines) {
    line = line.ltrim();
    if (line.empty())
      continue;
    if (!line.starts_with("%")) {
      hadNonMagic = !line.starts_with("#");
      continue;
    }
    bool isCellMagic = line.starts_with("%%");

    // Split the line into the magic name and the rest of the line.
    auto [magicName, magicArgs] = line.split(' ');
    magicName = magicName.drop_front(isCellMagic ? 2 : 1);
    magicArgs = magicArgs.trim();

    // Handle "hidden" magic first. These are a bit special because they can
    // appear anywhere in the cell.
    if (!isCellMagic && magicName == "#") {
      // We don't do anything special here, just move on, the processing of
      // this is handled elsewhere.
      continue;
    }

    // We want to let the user know if we see a magic incorrectly placed the
    // middle in the expression without triggering the actual expression
    // evaluation. After processing the magics that may appear in the middle
    // of an expression, error out on any others.
    if (hadNonMagic) {
      const char *errorFmt =
          "`%%{0}` can only be at the beginning of an expression.";
      diagnosticManager.AddDiagnostic(std::make_unique<MojoDiagnostic>(
          llvm::formatv(errorFmt, magicName).str(), lldb::eSeverityError,
          false));
      return failure();
    }

    auto reportUnknownMagic = [&, magicName = magicName] {
      diagnosticManager.PutString(
          lldb::eSeverityError,
          llvm::formatv("unknown magic: {0}", magicName).str());
      return failure();
    };

    if (isCellMagic) {
      // The processing of python magic is done separately, but we
      // can do some verification now.
      if (magicName == "python")
        continue;
      return reportUnknownMagic();
    }

    // Process a change of working directory.
    if (magicName == "cd") {
      if (magicArgs.empty()) {
        diagnosticManager.PutString(
            lldb::eSeverityError,
            "'%%cd' magic requires a directory argument, or '-' to pop the "
            "directory stack");
        return failure();
      }
      if (magicArgs == "-")
        typeSystem.popWorkingDirectory();
      else
        typeSystem.pushWorkingDirectory(magicArgs);
    } else {
      return reportUnknownMagic();
    }

    // Remove the magic from the expression.
    magicPositions.push_back(line.data() - exprText.data());
  }

  // Insert a comment at the start of each magic line so that the line numbers
  // in the error messages are correct.
  for (auto pos : llvm::reverse(magicPositions))
    exprText.insert(pos, "# ");

  return success();
}

bool MojoUserExpression::Parse(DiagnosticManager &diagnosticManager,
                               ExecutionContext &exeCtx,
                               ExecutionPolicy executionPolicy,
                               bool keepResultInMemory,
                               bool generateDebugInfo) {
  // Setup the execution context.
  InstallContext(exeCtx);

  // Initialize the persistent state.
  impl->resultDelegate.RegisterPersistentState(impl->persistentState);

  // Parse the expression text.
  Process *process = exeCtx.GetProcessPtr();
  // Check that there actually is a process that can parse the expression.
  if (!process) {
    diagnosticManager.AddDiagnostic(std::make_unique<MojoDiagnostic>(
        "target mojo process does not exist", lldb::eSeverityError, false));
    return false;
  }
  auto *exeScope = process ? (ExecutionContextScope *)process : &impl->target;

  // On exit, log all of the diagnostics that were collected.
  auto broadcastDiagnostics = llvm::scope_exit([&] {
    impl->expressionLogger.broadcastDiagnostics(diagnosticManager);
    diagnosticManager.Clear();
  });

  // Process any magics used in the cell.
  if (failed(processMagics(diagnosticManager, m_expr_text, impl->typeSystem)))
    return false;
  StringRef exprText(m_expr_text);

  // If the expression starts with `%%python`, the user wants to treat this as a
  // python expression. Otherwise, it should be treated as a Mojo expression.
  auto [firstLine, rest] = exprText.split('\n');
  if (firstLine.rtrim() == "%%python") {
    if (failed(wrapTextAndParsePythonExpression(rest, diagnosticManager, exeCtx,
                                                exeScope,
                                                impl->persistentState))) {
      return false;
    }
  } else if (failed(wrapTextAndParseExpression(
                 diagnosticManager, exeCtx, exeScope, impl->persistentState))) {
    return false;
  }

  // Prepare the output of the parser for execution, evaluating it statically if
  // possible.
  Status jitError = impl->parser->prepareForExecution(
      m_jit_start_addr, m_jit_end_addr, executionUnit, exeCtx, executionPolicy,
      keepResultInMemory);
  if (!jitError.Success()) {
    m_jit_start_addr = m_jit_end_addr = LLDB_INVALID_ADDRESS;

    const char *errorCStr = jitError.AsCString();
    if (errorCStr && errorCStr[0])
      diagnosticManager.PutString(lldb::eSeverityError, errorCStr);
    else
      diagnosticManager.PutString(lldb::eSeverityError,
                                  "expression can't be interpreted or run\n");
    return false;
  }

  if (process && m_jit_start_addr != LLDB_INVALID_ADDRESS)
    m_jit_process_wp = lldb::ProcessWP(process->shared_from_this());
  return true;
}

ExpressionTypeSystemHelper *MojoUserExpression::GetTypeSystemHelper() {
  return &impl->typeSystemHelper;
}

lldb::ExpressionVariableSP MojoUserExpression::GetResultAfterDematerialization(
    ExecutionContextScope *exeScope) {
  return impl->resultDelegate.GetVariable();
}

bool MojoUserExpression::addArguments(ExecutionContext &exeCtx,
                                      std::vector<lldb::addr_t> &args,
                                      lldb::addr_t structAddress,
                                      DiagnosticManager &diagnosticManager) {
  args.push_back(structAddress);
  return true;
}

static std::atomic_flag &getTraceDumpSignalRegisteredFlag() {
  static std::atomic_flag traceDumpSignalRegistered = ATOMIC_FLAG_INIT;
  return traceDumpSignalRegistered;
}

/// Signal handler that will dump the stack trace to the log. If we can't pull
/// out the mojo type system, we simply return because the only purpose of this
/// handler is to print a stack trace to the mojo log.
static void dumpTraceOnSignal(void *cookie) {
  // Signal handlers registered with llvm::sys::AddSignalHandler like this one
  // run only once.  Clear the flag so we re-register this handler next time
  // 'round.  (Note: We don't want to register the handler in here ourselves,
  // since that may cause llvm::sys::RunSignalHandlers to run us again
  // immediately without waiting for another signal.)
  getTraceDumpSignalRegisteredFlag().clear();

  auto *debugger = (Debugger *)cookie;

  // First simulate a crash for Crashpad.  This does not rely on the type
  // system and can report issues even if the other mechanisms below fail.
  generateNonFatalDump();

  // Pull the type system out of the current target.
  lldb::TargetSP currentTarget =
      debugger->GetSelectedExecutionContext(false).GetTargetSP();
  if (!currentTarget)
    return;

  // Great - now we can broadcast to it.
  std::string traceStr;
  llvm::raw_string_ostream trace(traceStr);
  llvm::sys::PrintStackTrace(trace);
  // This will also flush the debug logs.
  MojoExpressionLogger::getLoggerForTarget(*currentTarget)
      .errorLog("Backtrace:\n{0}", traceStr);
}

/// Register the trace dumping signal handler exactly once.
static void registerTraceDumpHandler(Debugger &debugger) {
  // N.B.: This is not really thread safe.  If two threads come through here,
  // one of them will register the signal handler, while the other one will see
  // that the flag was set and continue without waiting for the registration to
  // finish.  This is probably not a problem in practice --
  // wrapTextAndParseExpression is usually called only from a single thread,
  // and the worst case is execution continues without a trace-dump handler for
  // a little bit.
  if (!getTraceDumpSignalRegisteredFlag().test_and_set())
    llvm::sys::AddSignalHandler(dumpTraceOnSignal, (void *)&debugger);
}

LogicalResult MojoUserExpression::wrapTextAndParseExpression(
    DiagnosticManager &diagnosticManager, ExecutionContext &exeCtx,
    ExecutionContextScope *exeScope, MojoPersistentExpressionState &state) {
  // Parse the expression.
  materializer = std::make_unique<Materializer>();
  impl->parser =
      std::make_unique<MojoExpressionParser>(exeScope, *this, m_options);

  // Register the trace dump signal handler before we enable the
  // CrashRecoveryContext so it is picked up properly.
  registerTraceDumpHandler(exeCtx.GetTargetRef().GetDebugger());
  llvm::CrashRecoveryContext::Enable();

  // Disable the crash recovery context for the next time around.
  auto scopeExit =
      llvm::scope_exit([]() { llvm::CrashRecoveryContext::Disable(); });
  llvm::CrashRecoveryContext crc;

  // Signal handlers don't fire unless this flag is set.
  crc.DumpStackAndCleanupOnFailure = true;

  LogicalResult result = failure();
  if (!crc.RunSafelyOnThread(
          [&]() { result = impl->parser->parse(state, diagnosticManager); })) {
    impl->expressionLogger.errorLog(
        "Crash recovered: CrashRecoveryContext::RetCode (on POSIX: "
        "signal number + 128) = {0}",
        crc.RetCode);
    diagnosticManager.PutString(
        lldb::eSeverityError,
        "The Mojo REPL has crashed and attempted recovery. If the REPL "
        "behaves inconsistently, please restart to ensure correct behavior.");
    return failure();
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Python expression parsing and execution

const char *MojoUserExpression::FunctionName() {
  assert(impl->exprFnName && "expected a function name");
  return impl->exprFnName->c_str();
}

void MojoUserExpression::setFunctionName(std::string exprFnName) {
  assert(!impl->exprFnName && "unexpected function name");
  impl->exprFnName = std::move(exprFnName);
}

const std::optional<std::string> &MojoUserExpression::getPythonModuleName() {
  return impl->pythonModuleName;
}

/// Import the various top-level python symbols defined in the given python
/// expression into the current mojo context by emitting binding code to the
/// given stream.
static LogicalResult
importPythonSymbolsIntoMojo(Debugger &debugger, StringRef pythonExpr,
                            StringRef moduleName, raw_ostream &mojoExprOS,
                            DiagnosticManager &diagnosticManager) {
  ErrorOr<std::vector<std::unique_ptr<ExtractedPythonSymbol>>>
      extractedSymbolsOr = extractPythonSymbolsFromReplExpr(pythonExpr);
  if (failed(extractedSymbolsOr)) {
    diagnosticManager.PutString(lldb::eSeverityWarning,
                                extractedSymbolsOr.getError());
    return failure();
  }

  // Here we access the result, which is a serialized description of each
  // symbol to extract.
  for (auto &symbol : llvm::make_pointee_range(*extractedSymbolsOr)) {
    if (auto *decl = dyn_cast<ExtractedPythonDecl>(&symbol)) {
      mojoExprOS << llvm::formatv("var {0} = {1}.{0}\n", decl->getName(),
                                  moduleName);
    } else if (auto *import = dyn_cast<ExtractedPythonImport>(&symbol)) {
      mojoExprOS << llvm::formatv(
          "var {0} = __mojo_repl_Python.import_module(\"{1}\")\n",
          import->getName(), import->getModule());
    }
  }

  return success();
}

LogicalResult MojoUserExpression::wrapTextAndParsePythonExpression(
    StringRef pythonExpr, lldb_private::DiagnosticManager &diagnosticManager,
    lldb_private::ExecutionContext &exeCtx,
    lldb_private::ExecutionContextScope *exeScope,
    MojoPersistentExpressionState &state) {
  impl->expressionLogger.debugLog("Parsing the following python code:\n{0}",
                                  pythonExpr.data());

  // Generate a wrapper python expression that builds a new module from the
  // given source expression string.
  //   {0}: The escaped source expression string.
  //   {1}: The name of the module to create.
  const char *pythonWrapperExpr = R"(
import sys, types

code_string = "{0}"
expr_module = types.ModuleType('{1}')
exec(code_string, expr_module.__dict__)
sys.modules['{1}'] = expr_module
  )";

  // Generate an escaped version of the python expression to import, also taking
  // this time to add implicit imports for any previously defined modules.
  std::string escapedPythonExpr;
  llvm::raw_string_ostream escapedPythonExprOS(escapedPythonExpr);
  for (const auto &exprInst : state.getExpressionInstances()) {
    if (exprInst->pythonModuleName) {
      escapedPythonExprOS.write_escaped("try:\n");
      escapedPythonExprOS.write_escaped(
          llvm::formatv("  from {0} import *\n", exprInst->pythonModuleName)
              .str());
      escapedPythonExprOS.write_escaped("except:\n  pass\n");
    }
  }
  escapedPythonExprOS.write_escaped(pythonExpr);

  std::string moduleName = state.getNextPythonExpressionModuleName();
  std::string wrappedPythonExpr =
      llvm::formatv(pythonWrapperExpr, escapedPythonExpr, moduleName).str();
  impl->expressionLogger.debugLog("Wrapped python code:\n{0}",
                                  wrappedPythonExpr.data());

  // Build the Mojo expression we'll use to process the python. This consists of
  // the wrapped python expression, and implicit imports for any of the
  // top-level entities we want extract from the python expression.
  std::string mojoExpr;
  llvm::raw_string_ostream mojoExprOS(mojoExpr);

  // Evaluate the wrapped python expression.
  mojoExprOS << "var __lldb_repl_python__ = __mojo_repl_Python()\n\n";
  mojoExprOS << "if not __lldb_repl_python__.eval(\"";
  mojoExprOS.write_escaped(wrappedPythonExpr);
  mojoExprOS
      << "\"):\n  raise Error('The Python expression raised an exception')\n";

  // If persistent results are enabled, we also import top-level symbols from
  // the python module into the mojo context.
  if (!m_options.GetSuppressPersistentResult()) {
    mojoExprOS << llvm::formatv(
        "var {0} = __mojo_repl_Python.import_module(\"{0}\")\n\n", moduleName);

    // Import the interesting top-level symbols from the python module into the
    // mojo context.
    if (failed(importPythonSymbolsIntoMojo(exeCtx.GetTargetRef().GetDebugger(),
                                           pythonExpr, moduleName, mojoExprOS,
                                           diagnosticManager)))
      return failure();
  }

  // Now that we've got a Mojo expression, parse it the way we would any other
  // expression.
  m_expr_text = mojoExprOS.str();
  impl->pythonModuleName = std::move(moduleName);

  return wrapTextAndParseExpression(diagnosticManager, exeCtx, exeScope, state);
}

//===----------------------------------------------------------------------===//
// Execution

lldb::ExpressionResults MojoUserExpression::DoExecute(
    lldb_private::DiagnosticManager &diagnosticManager,
    lldb_private::ExecutionContext &exeCtx,
    const lldb_private::EvaluateExpressionOptions &options,
    lldb::UserExpressionSP &sharedPtrToMe, lldb::ExpressionVariableSP &result) {
  lldb::ExpressionResults results = JitUserExpression::DoExecute(
      diagnosticManager, exeCtx, options, sharedPtrToMe, result);
  auto lldbExprFailedVar = impl->persistentState.getVar(
      lldb_private::ConstString("__mojo_repl_expr_failed"));
  if (lldbExprFailedVar)
    impl->persistentState.RemovePersistentVariable(lldbExprFailedVar);
  if (results != lldb::eExpressionCompleted)
    return results;

  if (!lldbExprFailedVar)
    llvm::report_fatal_error("Expected to find variable "
                             "`__mojo_repl_expr_failed` in the persistent "
                             "state.");

  // Extract the value of __mojo_repl_expr_failed.
  DataExtractor extractor(lldbExprFailedVar->GetValueBytes(),
                          *lldbExprFailedVar->GetByteSize(),
                          exeCtx.GetProcessRef().GetByteOrder(),
                          exeCtx.GetProcessRef().GetAddressByteSize());
  lldb::offset_t offset = 0;
  lldb::offset_t addr = extractor.GetAddress(&offset);

  bool exprFailed;
  Status status = Status();
  exeCtx.GetProcessRef().ReadMemory((lldb::addr_t)addr, &exprFailed, 1, status);

  // Now that we have the value, we can check whether the expression failed or
  // not.
  auto expressionInstances = impl->persistentState.getExpressionInstances();
  if (exprFailed) {
    // The expression failed, so we won't persist any variables defined in the
    // expression.
    for (auto &var : expressionInstances.back()->persistentVariables)
      impl->persistentState.RemovePersistentVariable(var);

    // TODO: eventually we should put the exception into the persistent
    // state.
    impl->expressionLogger.errorLog(
        "Unhandled exception caught during execution");
    return lldb::eExpressionCompleted;
  }

  return lldb::eExpressionCompleted;
}
