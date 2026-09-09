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
// This file defines the main interface for the Mojo Jupyter kernel. It handles
// interacting with the Jupyter kernel protocol and the Mojo LLDB REPL.
//
//===----------------------------------------------------------------------===//

#include "../MojoLLDB/ExpressionParser/MojoExpressionVariable.h"
#include "../MojoLLDB/Logging/MojoExpressionLogger.h"
#include "../MojoLLDB/REPL/MojoREPL.h"
#include "../MojoLLDB/ScriptingBridge/SBClassUtils.h"
#include "Mojo/MojoJupyter/MatplotlibInitialization.h"

#include "Mojo/MojoJupyter/Kernel.h"
#include "Mojo/MojoLLDB/Plugin.h"
#include "Mojo/MojoTooling/CodeComplete.h"
#include "Mojo/Support/Configuration.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/STLExtras.h"
#include "Support/SymbolExport.h"
#include "lldb/API/LLDB.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Listener.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <filesystem>
#include <thread>

#define DEBUG_TYPE "mojo-jupyter"

using namespace lldb;
using namespace lldb_private;
using namespace M;
using namespace M::KGEN::Mojo;
using namespace M::Mojo::Jupyter;

/// An output function used to send output to the Jupyter kernel. The first
/// argument is the output type, and the second is the output string.
using RawOutputFn = void (*)(const char *, const char *);

/// A function used to send code completion results to the Jupyter kernel.
using RawCompletionFn = void (*)(const char *);

/// Return the persistent expression state for Mojo.
static MojoPersistentExpressionState *
getPersistentExpressionState(const TargetSP &target) {
  return static_cast<MojoPersistentExpressionState *>(
      target->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeMojo));
}

//===----------------------------------------------------------------------===//
// MojoExpressionEvaluationOptions
//===----------------------------------------------------------------------===//

namespace {
struct MojoExpressionEvaluationOptions : public SBExpressionOptions {
  MojoExpressionEvaluationOptions() {
    SetLanguage(lldb::eLanguageTypeMojo);
    SetUnwindOnError(false);
    SetGenerateDebugInfo(true);

    SetTimeoutInMicroSeconds(0);

    ref().SetREPLEnabled(true);
    ref().SetColorizeErrors(true);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// MojoKernel
//===----------------------------------------------------------------------===//

/// This class contains all of the various state needed to run the Mojo Jupyter
/// kernel.
struct MojoKernel::Impl {
  /// Information related to a specific kernel cell.
  struct KernelCellState {
    KernelCellState(StringRef id) : id(id) {}

    KernelCellState(const KernelCellState &) = delete;
    KernelCellState &operator=(const KernelCellState &) = delete;

    /// The string identifier of the cell.
    std::string id;
  };

  /// This struct represents a single expression evaluation request. It is used
  /// to pass the result of the evaluation back to the caller, which can query
  /// the status of the execution.
  struct ExpressionExecutionState {
    ExpressionExecutionState(KernelCellState &cellState)
        : finished(false), cellState(cellState) {}
    SBError error;
    SBValue result;
    std::thread executionThread;
    std::atomic<bool> finished;
    KernelCellState &cellState;
  };

  Impl(OutputFn outputFn, bool initializeMatPlotLib)
      : outputFn(std::move(outputFn)),
        mojoExpressionListener(
            Listener::MakeListener("mojo-type-system.listener")),
        // If we aren't initialized matplot, just mark it as finished now.
        matplotlibInitialized(!initializeMatPlotLib) {}

  ~Impl() {
    if (process->IsValid())
      process->Destroy(/*force_kill=*/true);
    SBDebugger sbdebugger = SBDebuggerUtils::create(debugger);
    SBDebugger::Destroy(sbdebugger);
    SBDebugger::Terminate();
  }

  /// Initialize the kernel.
  ///
  /// If `lldbInitFile` is not empty, LLDB will silently execute all the
  /// commands in this file upon initialization of the kernel.
  LogicalResult initialize(std::optional<ContextRef> ctx, StringRef mojoReplExe,
                           StringRef workingDirectory,
                           ArrayRef<std::string> additionalDirectories,
                           StringRef lldbInitFile);

  /// Start execution of the given cell identifier and expression string.
  /// `storeHistory` indicates if variables and state from this expression
  /// should be persisted. Returns the state of the expression execution.
  ExecutionFinishedState startExecution(StringRef cellId, StringRef expr,
                                        bool storeHistory);

  /// Check if the current expression has finished execution, also taking this
  /// time to flush any collected output.
  ExecutionFinishedState checkExecutionFinished();

  /// Interrupt the currently running execution.
  void interruptExecution();

  /// Perform code completion at the given position within the given code
  /// string. The completion function will be called with the completion
  /// results.
  void codeComplete(StringRef code, int completionPos,
                    CompletionFn completionFn);

private:
  /// Initialize the target.
  LogicalResult initializeTarget(StringRef mojoReplExe);

  /// Launch the mojo-repl-entry-point process.
  LogicalResult launchReplProcess(StringRef workingDirectory,
                                  ArrayRef<std::string> additionalDirectories);

  /// Initialize the inline matplotlib backend within the python interop.
  LogicalResult initializeMatplotlib();

  /// Report an error to the Jupyter kernel.
  LogicalResult reportKernelError(const Twine &message) {
    llvm::errs() << "error: " << message << "\n";
    sendOutput("error", message.str());
    return failure();
  }

  /// Send output to the Jupyter kernel.
  void sendOutput(StringRef type, StringRef output) {
    LLVM_DEBUG(llvm::dbgs()
               << "Sending output: " << type << ": " << output << "\n");
    outputFn(type.data(), output.data());
  }

  /// Flush the LLDB output streams associated within the current execution
  /// state.
  void flushLLDBStreams();

  /// Initialize the given cell for execution. Returns the associated cell
  /// state, or nullptr if the cell was invalid.
  KernelCellState &initializeCellForExecution(StringRef cellId) {
    auto [it, inserted] = cells.insert({cellId, nullptr});
    if (inserted)
      it->second = std::make_unique<KernelCellState>(it->first());
    return *it->second;
  }

  /// The output function used to send output to the Jupyter kernel.
  OutputFn outputFn;

  /// Various LLDB state used for tracking the repl process.
  DebuggerSP debugger;
  TargetSP target;
  ProcessSP process;
  MojoExpressionEvaluationOptions exprOpts;
  ThreadSP mainThread;
  ListenerSP mojoExpressionListener;

  /// The mojo persistent expression state.
  MojoPersistentExpressionState *exprState = nullptr;

  /// The current execution state, or nullopt if no execution is currently
  /// happening.
  std::optional<ExpressionExecutionState> executionState;

  /// Information about each of the cells that have been executed.
  llvm::StringMap<std::unique_ptr<KernelCellState>> cells;

  /// A bool indicating if matplotlib has been initialized.
  bool matplotlibInitialized = false;
};

MojoKernel::MojoKernel(OutputFn outputFn, bool initializeMatPlotLib)
    : impl(new Impl(std::move(outputFn), initializeMatPlotLib)) {}
MojoKernel::~MojoKernel() = default;

LogicalResult
MojoKernel::initialize(std::optional<ContextRef> ctx, StringRef mojoReplExe,
                       StringRef workingDirectory,
                       ArrayRef<std::string> additionalDirectories,
                       StringRef lldbInitFile) {
  return impl->initialize(ctx, mojoReplExe, workingDirectory,
                          additionalDirectories, lldbInitFile);
}

ExecutionFinishedState MojoKernel::startExecution(StringRef cellId,
                                                  StringRef expr,
                                                  bool storeHistory) {
  return impl->startExecution(cellId, expr, storeHistory);
}

ExecutionFinishedState MojoKernel::checkExecutionFinished() {
  return impl->checkExecutionFinished();
}

ExecutionFinishedState MojoKernel::executeAndWait(StringRef cellId,
                                                  StringRef expr,
                                                  bool storeHistory) {
  if (startExecution(cellId, expr, storeHistory) == kFinishedError)
    return kFinishedError;

  // Wait for the execution to finish.
  do {
    ExecutionFinishedState result = checkExecutionFinished();
    if (result != kNotFinished)
      return result;
  } while (true);
}

void MojoKernel::interruptExecution() { impl->interruptExecution(); }

void MojoKernel::codeComplete(StringRef code, int completionPos,
                              CompletionFn completionFn) {
  impl->codeComplete(code, completionPos, completionFn);
}

//===----------------------------------------------------------------------===//
// EntryPoint API
//===----------------------------------------------------------------------===//

MODULAR_EXPORT MojoKernel *initMojoKernel(RawOutputFn outputFn,
                                          const char *mojoReplExe,
                                          const char *workingDirectory,
                                          const char *lldbInitFile) {
  std::unique_ptr<MojoKernel> kernel =
      std::make_unique<MojoKernel>([=](StringRef type, StringRef output) {
        outputFn(type.data(), output.data());
      });
  if (failed(kernel->initialize(std::nullopt, mojoReplExe, workingDirectory,
                                /*additionalDirectories=*/{}, lldbInitFile)))
    return nullptr;
  return kernel.release();
}

MODULAR_EXPORT int startMojoExecution(MojoKernel *kernel, const char *cellId,
                                      const char *code, int storeHistory) {
  return kernel->startExecution(cellId, code, storeHistory);
}

MODULAR_EXPORT int checkMojoExecutionFinished(MojoKernel *kernel) {
  return kernel->checkExecutionFinished();
}

MODULAR_EXPORT void interruptMojoExecution(MojoKernel *kernel) {
  kernel->interruptExecution();
}

MODULAR_EXPORT void checkMojoCodeComplete(MojoKernel *kernel, const char *code,
                                          int completionPos,
                                          RawCompletionFn completionFn) {
  kernel->codeComplete(code, completionPos, [=](StringRef completion) {
    completionFn(completion.data());
  });
}

MODULAR_EXPORT void destroyMojoKernel(MojoKernel *kernel) { delete kernel; }

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

/// We want to restrict the set of LLDB commands that can be executed on the
/// notebook as a way to prevent external users from affecting the host
/// environment (e.g. killing or spawning processes, inspecting the entry-point,
/// etc.)
static void removeUnwantedCommands(Debugger &debugger) {
  auto isAllowed = [](const CommandObjectSP &obj) {
    /// The following list can grow as needed, but just be mindful of any
    /// possible vulnerability, as LLDB has more permissions than regular
    /// processes.
    static auto kAllowedCommands = {
        "apropos",
        "help",
        "mojo",
        // This is very useful for us to debug issues in the expression
        // evaluator.
        "log",
    };
    return llvm::any_of(kAllowedCommands, [&](StringRef allowed) {
      return obj->GetCommandName().starts_with(allowed);
    });
  };

  CommandInterpreter &interpreter = debugger.GetCommandInterpreter();

  CommandObject::CommandMap aliases = interpreter.GetAliases();
  for (auto &[name, obj] : aliases) {
    CommandObjectSP actualCommand =
        static_cast<CommandAlias *>(obj.get())->GetUnderlyingCommand();
    if (!isAllowed(actualCommand))
      interpreter.RemoveAlias(name);
  }

  CommandObject::CommandMap commands = interpreter.GetCommands();
  for (auto &[name, obj] : commands) {
    if (!isAllowed(obj))
      interpreter.RemoveCommand(name);
  }
}

static void runLLDBInitFile(Debugger &debugger, FileSpec lldbInitFile) {
  CommandInterpreterRunOptions options;
  CommandReturnObject result(/*colors=*/false);
  options.SetAddToHistory(false);
  options.SetEchoCommands(false);
  options.SetPrintErrors(true);
  options.SetSilent(true);
  debugger.GetCommandInterpreter().HandleCommandsFromFile(lldbInitFile, options,
                                                          result);
  if (!result.Succeeded()) {
    llvm::errs() << result.GetOutputString() << "\n";
    llvm::errs() << result.GetErrorString() << "\n";
    exit(EXIT_FAILURE);
  }
}

LogicalResult
MojoKernel::Impl::initialize(std::optional<ContextRef> ctx,
                             StringRef mojoReplExe, StringRef workingDirectory,
                             ArrayRef<std::string> additionalDirectories,
                             StringRef lldbInitFile) {
  // Initialize a new debugger instance.
  // We need to initialize with SBDebugger because that's the only way we can
  // support loading public plugins like MojoLLDB.
  if (SBError err = SBDebugger::InitializeWithErrorHandling(); err.Fail())
    return reportKernelError(err.GetCString());

  debugger = Debugger::CreateInstance();
  debugger->SetAsyncExecution(false);

  // If we got an LLDB init file, we execute it before anything else.
  if (!lldbInitFile.empty())
    runLLDBInitFile(*debugger, FileSpec(lldbInitFile));

  // For security reasons on public Jupyter notebooks, we want to remove some
  // commands that might give users ways to perform unwanted actions on the
  // host.
  removeUnwantedCommands(*debugger);

  // Initialize the Mojo LLDB plugin.
  ErrorOr<KGEN::MojoConfig> config = KGEN::MojoConfig::open();
  if (failed(config))
    return reportKernelError(config.getError());
  FileSpec mojoPlugin(config->getLLDBPluginPath().str());
  if (!FileSystem::Instance().Exists(mojoPlugin))
    return reportKernelError("unable to locate Mojo LLDB plugin");

  // LLDB needs access to an M::Context.  At initialization time, it will
  // create its own if one has not been provided to it, but this causes
  // problems if there is already an M::Context in the process, so hand off our
  // context before it initializes if we have one.
  if (ctx)
    KGEN::setLLDBPluginContext(ctx.value());

  CommandReturnObject result(/*colors=*/false);
  Status err;
  if (!debugger->LoadPlugin(mojoPlugin, err))
    return reportKernelError(err.AsCString());

  // Initialize the target.
  if (failed(initializeTarget(mojoReplExe)))
    return failure();

  // Launch the mojo-repl-entry-point process.
  if (failed(launchReplProcess(workingDirectory, additionalDirectories)))
    return failure();
  process = target->GetProcessSP();

  // Sets an infinite timeout so that users can run arbitrarily long
  // computations.
  mainThread = process->GetThreadList().GetThreadAtIndex(0);

  LLVM_DEBUG(llvm::dbgs() << "Successfully built Mojo Jupyter kernel\n");
  return success();
}

LogicalResult MojoKernel::Impl::initializeTarget(StringRef mojoReplExe) {
  // Compute a generic triple for the REPL target.
  llvm::Triple targetTriple = HostInfo::GetArchitecture().GetTriple();
  llvm::SmallString<16> osName;
  llvm::raw_svector_ostream os(osName);

  // Use the most generic sub-architecture.
  targetTriple.setArch(targetTriple.getArch());
  os << llvm::Triple::getOSTypeName(targetTriple.getOS());

  // Override the stub's minimum deployment target to the host os version.
  if (targetTriple.isOSDarwin())
    os << HostInfo::GetOSVersion().getAsString();
  targetTriple.setOSName(os.str());

  // Create a new target for the REPL executable.
  debugger->GetTargetList().CreateTarget(
      *debugger, mojoReplExe,
      /*target_triple=*/targetTriple.getTriple().c_str(),
      /*add_dependent_modules=*/eLoadDependentsYes,
      /*platform_options=*/nullptr, target);

  if (!target)
    return reportKernelError("failed to create target: invalid debugger");
  exprState = getPersistentExpressionState(target);

  return success();
}

LogicalResult MojoKernel::Impl::launchReplProcess(
    StringRef workingDirectory, ArrayRef<std::string> additionalDirectories) {
  if (llvm::Error err = MojoREPL::launchEntryPointProcess(
          *target, *debugger, workingDirectory, additionalDirectories)) {
    return reportKernelError(
        "Failed to launch `mojo-repl-entry-point` process: " +
        llvm::toString(std::move(err)));
  }
  MojoExpressionLogger::getLoggerForTarget(*target).AddListener(
      mojoExpressionListener, MojoExpressionLogger::eAllMessagesMask);

  return success();
}

//===----------------------------------------------------------------------===//
// Matplotlib
//===----------------------------------------------------------------------===//

LogicalResult MojoKernel::Impl::initializeMatplotlib() {
  // Initialize the matplotlib backend by running the initialization code with
  // the REPL.
  if (startExecution("matplotlib", kInitMatplotlibStr,
                     /*storeHistory=*/false) == kFinishedError)
    return failure();

  // Wait for the execution to finish.
  do {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (int result = checkExecutionFinished())
      return success(result == kFinishedSuccessfully);
  } while (true);
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

ExecutionFinishedState MojoKernel::Impl::startExecution(StringRef cellId,
                                                        StringRef expr,
                                                        bool storeHistory) {
  // Before we start executing, check to see if we need to initialize anything.
  if (!std::exchange(matplotlibInitialized, true)) {
    if (failed(initializeMatplotlib()))
      return kFinishedError;
  }
  executionState.emplace(initializeCellForExecution(cellId));

  // Start execution of the expression in a separate thread, so that way the
  // calling client can control waiting for the expression to complete.
  executionState->executionThread =
      std::thread([this, storeHistory, expr = std::string(expr)]() mutable {
        LLVM_DEBUG(llvm::dbgs() << "Executing expression: " << expr << "\n");

        // If the expression starts with `:`, then it is an LLDB command,
        // otherwise it is a Mojo expression.
        SBValue value;
        if (StringRef command(expr); command.consume_front(":")) {
          CommandReturnObject result(/*colors=*/false);
          target->GetDebugger().GetCommandInterpreter().HandleCommand(
              command.rtrim().str().c_str(),
              /*add_to_history=*/lldb_private::eLazyBoolNo, result);
          sendOutput("output", result.GetOutputString());
          if (!result.GetErrorString().empty())
            sendOutput("error", result.GetErrorString());
        } else {
          MojoExpressionEvaluationOptions options = exprOpts;
          if (!storeHistory)
            options.SetSuppressPersistentResult(true);

          value = SBTargetUtils::create(target).EvaluateExpression(expr.data(),
                                                                   options);
        }

        executionState->result = value;
        executionState->error = value.GetError();

        // Mark the execution as finished.
        executionState->finished = true;
      });
  return kNotFinished;
}

void MojoKernel::Impl::flushLLDBStreams() {
  // Reading the following streams from LLDB is thread safe because each reader
  // has its own mutex.

  // Flush type system messages.
  lldb::EventSP event;

  // The following gets the stream of events without timeout. All the messages
  // will be read eventually anyway.
  while (mojoExpressionListener->GetEvent(event, std::chrono::seconds(0))) {
    MojoExpressionLogger::handleEvent(
        event, [&](StringRef type, StringRef msg) {
          // If the message isn't an error, print it out for the user to see as
          // part of stderr.
          sendOutput(type == "error" ? type : "stderr", msg);
        });
    event->Clear();
  }

  char outputBuffer[1024];

  // Read stdout from the process.
  Status unused;
  while (int readLen = process->GetSTDOUT(outputBuffer, 1023, unused)) {
    outputBuffer[readLen] = '\0';
    StringRef data(outputBuffer, readLen);
    LLVM_DEBUG(llvm::dbgs() << "stdout: " << readLen << " : " << data << "\n");
    sendOutput("stdout", data);
  }
  // Read stderr from the process.
  while (int readLen = process->GetSTDERR(outputBuffer, 1024, unused)) {
    outputBuffer[readLen] = '\0';
    StringRef data(outputBuffer, readLen);
    LLVM_DEBUG(llvm::dbgs() << "stderr: " << readLen << " : " << data << "\n");
    sendOutput("stderr", data);
  }
}

ExecutionFinishedState MojoKernel::Impl::checkExecutionFinished() {
  if (!executionState)
    return kFinishedSuccessfully;

  // Check to see if the expression is still executing.
  if (!executionState->finished) {
    flushLLDBStreams();
    return kNotFinished;
  }
  flushLLDBStreams();

  // The expression has finished executing, process the results.
  LLVM_DEBUG(llvm::dbgs() << "Finished executing expression\n");

  ExecutionFinishedState finishState = kFinishedSuccessfully;
  // Process the result.
  auto errorType = executionState->error.GetType();
  if (errorType == eErrorTypeInvalid) {
    sendOutput("stdout", executionState->result.GetObjectDescription());
  } else if (errorType != eErrorTypeGeneric) {
    StringRef executionError(executionState->error.GetCString());

    // If the expression failed to parse, LLDB adds in an extra message, strip
    // that out.
    executionError.consume_front("expression failed to parse:\n");

    // If the output is simply "unknown error", this indicates that LLDB didn't
    // have a diagnostic for the specific problem. In these cases, the REPL
    // ensures that the user is alerted to the problem, so there isn't a need to
    // add the unhelpful error message.
    if (executionError != "unknown error")
      sendOutput("stderr", executionError);

    // Set the finish state to an error.
    finishState = kFinishedError;
  } else {
    // eErrorTypeGeneric can be thrown if the expression doesn't have a result,
    // even if it succeeded.
    // TODO: Revisit this once we have more TypeSystem functionality
    //   implemented.
    executionState->error.Clear();
  }

  // Clean up the state now that we're done with it.
  executionState->executionThread.join();
  executionState.reset();
  return finishState;
}

void MojoKernel::Impl::interruptExecution() { process->SendAsyncInterrupt(); }

//===----------------------------------------------------------------------===//
// Code Completion
//===----------------------------------------------------------------------===//

void MojoKernel::Impl::codeComplete(StringRef code, int completionPos,
                                    CompletionFn completionFn) {
  std::vector<CodeCompletionResult> results =
      MojoREPL::handleREPLCodeComplete(*target, code, completionPos);
  for (const CodeCompletionResult &result : results)
    completionFn(result.label.data());
}
