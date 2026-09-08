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

#include "MojoREPL.h"
#include "../ExpressionParser/MojoExpressionVariable.h"
#include "../Language/MojoLanguage.h"
#include "../Logging/MojoExpressionLogger.h"

#include "Mojo/MojoTooling/CodeComplete.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/Support/Configuration.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBProcess.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/DumpValueObjectOptions.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/StreamFile.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Host.h"

using namespace M;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

static llvm::Error createStringError(StringRef message) {
  return llvm::make_error<llvm::StringError>(message,
                                             llvm::inconvertibleErrorCode());
}
template <typename... Args>
static llvm::Error createStringError(const char *format, Args &&...args) {
  return createStringError(
      llvm::formatv(format, std::forward<Args>(args)...).str());
}

static MojoTypeSystem &getMojoTypeSystem(Target &target) {
  if (auto typeSystemOr =
          target.GetScratchTypeSystemForLanguage(lldb::eLanguageTypeMojo))
    return *static_cast<MojoTypeSystem *>(typeSystemOr.get().get());
  llvm::report_fatal_error(
      "The Mojo type system plug-in must have already been registered.");
}

//===----------------------------------------------------------------------===//
// Target event listening
//===----------------------------------------------------------------------===//

void MojoREPL::flushExpressionEventsAndProcessStreams() {
  std::scoped_lock<std::mutex> lock(flushStreamsMutex);

  lldb::TargetSP target = getTarget();

  // Report a message to the error stream.
  auto sendUserOutput = [&](StringRef type, StringRef message) {
    errorStream->AsRawOstream() << "[User] " << message << "\n";
    errorStream->Flush();
  };

  lldb::EventSP event;
  while (mojoExpressionListener->GetEvent(event, std::chrono::seconds(0))) {
    // Handle the mojo expression events by logging them to error stream.
    MojoExpressionLogger::getLoggerForTarget(*target).handleEvent(
        event, sendUserOutput);
  }

  if (lldb::ProcessSP process = target->GetProcessSP()) {
    target->GetDebugger().FlushProcessOutput(*process, /*flush_stdout=*/true,
                                             /*flush_stdout=*/true);
  }
}

static void eventThreadFunction(
    const std::atomic_bool &stopEventThread,
    std::function<void(void)> flushExpressionEventsAndProcessStreams) {
  while (!stopEventThread) {
    flushExpressionEventsAndProcessStreams();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  // We flush one last time in case the process emitted some messages after the
  // previous loop was told to stop by the REPL's destructor. Otherwise the
  // debugger might exit before the messages are displayed to the user.
  flushExpressionEventsAndProcessStreams();
}

llvm::Error MojoREPL::OnExpressionEvaluated(
    const ExecutionContext &exe_ctx, llvm::StringRef code,
    const EvaluateExpressionOptions &expr_options,
    lldb::ExpressionResults execution_results,
    const lldb::ValueObjectSP &result_valobj_sp, const Status &error) {
  // We flush right after an expression was evaluated but before the next one is
  // executed. Otherwise we might have a race condition when executing
  // expressions in batch mode, in which the events of an expression are merged
  // with the events of a subsequent expression. This makes this method a
  // synchronization point between event processing and the REPL.
  flushExpressionEventsAndProcessStreams();
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// MojoREPL
//===----------------------------------------------------------------------===//

char MojoREPL::ID;

MojoREPL::MojoREPL(Target &target)
    : llvm::RTTIExtends<MojoREPL, REPL>(target),
      mojoExpressionListener(
          Listener::MakeListener("mojo-repl.type-system-listener")),
      targetWP(target.shared_from_this()),
      errorStream(target.GetDebugger().GetAsyncErrorStream()) {
  // Get a pointer to the mojo type system. We need that to read the various
  // log messages.
  auto typeSystemOr =
      target.GetScratchTypeSystemForLanguage(lldb::eLanguageTypeMojo);
  if (!typeSystemOr)
    llvm::report_fatal_error("must be able to get the mojo type system");

  typeSystem = std::static_pointer_cast<MojoTypeSystem>(*typeSystemOr);

  if (!typeSystem)
    llvm::report_fatal_error("must be able to get the mojo type system");

  MojoExpressionLogger::getLoggerForTarget(target).AddListener(
      mojoExpressionListener, MojoExpressionLogger::eAllMessagesMask);

  eventThread = std::thread([this] {
    eventThreadFunction(stopEventThread,
                        [this]() { flushExpressionEventsAndProcessStreams(); });
  });

  // Here we set the default expr eval options for all REPL expressions.
  m_expr_options.SetTimeout(Timeout<std::micro>(std::nullopt));
  m_expr_options.SetOneThreadTimeout(Timeout<std::micro>(std::nullopt));
}

MojoREPL::~MojoREPL() {
  if (eventThread.joinable()) {
    stopEventThread = true;
    eventThread.join();
  }
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

/// Create a repl instance for a given target. `replOptions` contains a set of
/// options to be passed to the repl.
static llvm::Expected<lldb::REPLSP>
createInstanceFromTarget(Target &target, const char *replOptions) {
  // Sanity check the target to make sure a REPL would work here.
  if (!target.GetProcessSP() || !target.GetProcessSP()->IsAlive()) {
    return createStringError(
        "can't launch a Mojo REPL without a running process");
  }

  lldb::REPLSP repl = std::make_shared<MojoREPL>(target);
  repl->SetCompilerOptions(replOptions);
  return repl;
}

/// Create a target for use by the repl. The target is created by launching the
/// mojo-repl-entry-point utility executable. The executable is expected to be
/// adjacent to the location of plugin library.
static llvm::Expected<lldb::TargetSP> createMojoReplTarget(Debugger &debugger) {
  ErrorOr<KGEN::MojoConfig> config = KGEN::MojoConfig::open();
  if (failed(config))
    return createStringError("failed to open Mojo configuration: {0}",
                             config.takeError());

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

  // Create a target for the repl executable.
  lldb::TargetSP target;
  Status error = debugger.GetTargetList().CreateTarget(
      debugger, config->getREPLEntryPoint(), targetTriple.getTriple(),
      eLoadDependentsYes, /*platform_options=*/nullptr, target);
  if (!error.Success()) {
    return createStringError("failed to create REPL target: {0}",
                             error.AsCString());
  }
  return target;
}

/// Create a break point within the repl target to provide an anchor for the
/// repl to execute expressions.
static llvm::Error createReplBreakpoint(Target &target) {
  // Limit the breakpoint to the target's executable module.
  lldb::ModuleSP exeModule = target.GetExecutableModule();
  if (!exeModule) {
    target.Destroy();
    return createStringError("unable to resolve REPL executable module");
  }
  FileSpecList containingModules;
  containingModules.Append(exeModule->GetFileSpec());

  // Create the breakpoint.
  lldb::BreakpointSP breakpoint = target.CreateBreakpoint(
      &containingModules, /*containingSourceFiles=*/nullptr,
      /*func_name=*/"mojo_repl_main", lldb::eFunctionNameTypeAuto,
      lldb::eLanguageTypeUnknown, /*offset=*/0,
      /*offset_is_insn_count=*/false,
      /*skip_prologue=*/eLazyBoolCalculate,
      /*internal=*/true,
      /*request_hardware=*/false);
  if (breakpoint->GetNumLocations() == 0)
    return createStringError(
        "failed to resolve REPL breakpoint for 'mojo_repl_main'");

  breakpoint->SetBreakpointKind("REPL");
  return llvm::Error::success();
}

llvm::Error MojoREPL::launchEntryPointProcess(
    Target &target, Debugger &debugger, StringRef workingDirectory,
    ArrayRef<std::string> additionalImportDirectories) {
  // Create a breakpoint in the target to anchor the REPL.
  if (llvm::Error error = createReplBreakpoint(target))
    return error;
  MojoTypeSystem &typeSystem = getMojoTypeSystem(target);

  // The following disables a warning that is thrown when the entry-point is
  // built with optimizations. This warning pollutes the output and is not
  // helpful because the entry point is actually an empty program.
  ExecutionContext ctx;
  target.CalculateExecutionContext(ctx);
  target.SetPropertyValue(&ctx, eVarSetOperationAssign,
                          /*path=*/"process.optimization-warnings",
                          /*value=*/"false");

  ProcessLaunchInfo launchInfo;
  if (target.GetDisableSTDIO())
    launchInfo.GetFlags().Set(lldb::eLaunchFlagDisableSTDIO);
  if (!workingDirectory.empty()) {
    launchInfo.SetWorkingDirectory(FileSpec(workingDirectory));
    typeSystem.pushWorkingDirectory(workingDirectory);
  }
  typeSystem.addImportDirectories(additionalImportDirectories);

  lldb::ModuleSP exeModule = target.GetExecutableModule();

  // Configure the launch info to use the target's argv0.
  llvm::StringRef targetSettingsArgv0 = target.GetArg0();
  if (!targetSettingsArgv0.empty()) {
    launchInfo.GetArguments().AppendArgument(targetSettingsArgv0);
    launchInfo.SetExecutableFile(exeModule->GetPlatformFileSpec(), false);
  } else {
    launchInfo.SetExecutableFile(exeModule->GetPlatformFileSpec(), true);
  }

  // Configure the launch environment to use the target's environment. In
  // addition, we also ensure that the library path includes the directory
  // containing the REPL executable.
  launchInfo.GetEnvironment() = target.GetEnvironment();
  launchInfo.GetEnvironment()["LD_LIBRARY_PATH"] +=
      (":" + exeModule->GetFileSpec().GetDirectory()).str();

  // Launch the process synchronously, waiting for it to stop at the REPL
  // breakpoint.
  debugger.SetAsyncExecution(false);
  Status error = target.Launch(launchInfo, nullptr);
  debugger.SetAsyncExecution(true);
  if (!error.Success()) {
    return createStringError("failed to launch REPL process: {0}",
                             error.AsCString());
  }

  lldb::ProcessSP process = target.GetProcessSP();
  if (!process)
    return createStringError("failed to launch REPL process");

  // Functor used to report an error, and destroy the process.
  auto emitError = [&](StringRef errorMsg) {
    process->Destroy(/*force_kill=*/false);
    return createStringError(errorMsg);
  };

  lldb::StateType state = process->GetState();
  if (state != lldb::eStateStopped)
    return emitError("failed to stop process at REPL breakpoint");

  ThreadList &threadList = process->GetThreadList();
  if (threadList.GetSize() == 0)
    return emitError("process is not in a valid state (no threads)");

  lldb::ThreadSP thread = threadList.GetSelectedThread();
  if (!thread) {
    thread = threadList.GetThreadAtIndex(0);
    threadList.SetSelectedThreadByID(thread->GetID());
    assert(thread && "there should be at least one thread");
  }
  thread->SetSelectedFrameByIndex(0);

  return llvm::Error::success();
}

/// Create a repl instance for a given debugger. `replOptions` contains a set of
/// options to be passed to the repl.
static llvm::Expected<lldb::REPLSP>
createInstanceFromDebugger(Debugger &debugger, const char *replOptions) {
  llvm::Expected<lldb::TargetSP> target = createMojoReplTarget(debugger);
  if (!target)
    return target.takeError();

  // Launch the repl process and wait for it to trigger the breakpoint.
  if (llvm::Error error = MojoREPL::launchEntryPointProcess(**target, debugger))
    return error;

  // Start the debugger's default event handler thread.
  debugger.StartEventHandlerThread();

  // Destroy the process and the event handler thread after a fatal error.
  auto cleanupOnError = llvm::scope_exit([&]() {
    if (lldb::ProcessSP process = (**target).GetProcessSP())
      process->Destroy(/*force_kill=*/false);
    debugger.StopEventHandlerThread();
  });

  // The process is active and stopped, we can build the REPL now.
  lldb::REPLSP repl = std::make_shared<MojoREPL>(**target);
  repl->SetCompilerOptions(replOptions);
  (*target)->SetREPL(lldb::eLanguageTypeMojo, repl);

  // Disable the cleanup, since we have a valid repl session now.
  cleanupOnError.release();

  if (isatty(STDIN_FILENO)) {
    llvm::outs() << "Welcome to Mojo! 🔥\n\nExpressions are delimited by a "
                    "blank line.\nType `:quit` to exit the REPL and `:mojo "
                    "help` for further assistance.\n\n";
  }
  return repl;
}

static void runCommand(Debugger &debugger, StringRef command) {
  CommandReturnObject result(/*colors=*/false);
  debugger.GetCommandInterpreter().HandleCommand(
      command.data(), /*add_to_history=*/lldb_private::eLazyBoolNo, result);
}

/// Enable the LLDB symbol cache, which can effectively speed up the startup
/// time of subsequent invocations by caching the parsed debug info and symbol
/// tables.
/// The index is stored in ~/.lldb/repl-symbol-cache. In fact, ~/.lldb is a
/// folder used already to store many LLDB-related bits, like the editline
/// history. If this folder is not writable, the cache won't work.
/// This cache is set up to have a retention per file of 1 day. There's no need
/// to cache beyond that.
/// It's also possible to set a size limit to the cache, but as it's only used
/// for the REPL, we might not need to set this up.
[[maybe_unused]] static void enableREPLSymbolCache(Debugger &debugger) {
  llvm::SmallString<128> lldbCacheDir;
  FileSystem::Instance().GetHomeDirectory(lldbCacheDir);
  llvm::sys::path::append(lldbCacheDir, ".lldb");
  llvm::sys::path::append(lldbCacheDir, "repl-symbol-cache");

  runCommand(debugger, "settings set symbols.enable-lldb-index-cache true");
  runCommand(debugger,
             "settings set symbols.lldb-index-cache-expiration-days 1");
  runCommand(
      debugger,
      ("settings set symbols.lldb-index-cache-path " + lldbCacheDir).str());
}

/// The following optional features are "sometimes" expensive, but we rather
/// disable them just in case.
static void disableExpensiveFeatures(Debugger &debugger) {
  // We don't need JIT debugging.
  runCommand(debugger, "settings set plugin.jit-loader.gdb.enable off");
  // We don't need to process debug info upfront. This is more useful for remote
  // debugging, e.g. android.
  runCommand(debugger, "settings set target.preload-symbols false");
  // We don't need to enable custom scripts embedded in debug info.
  runCommand(debugger,
             "settings set target.load-script-from-symbol-file false");
}

static void setupLLDBForREPL(Debugger &debugger) {
  disableExpensiveFeatures(debugger);

// FIXME(MOTO-471): the symbol cache currently has an issue on Mac where
// re-exported symbols are parsed incorrectly, leading to non-deterministic
// crashes.
#if !defined(__APPLE__)
  enableREPLSymbolCache(debugger);
#endif
}

/// Create a repl instance for either the given target, or the given
/// debugger. `replOptions` contains a set of options to be passed to the
/// repl.
static lldb::REPLSP createInstance(Status &error, lldb::LanguageType language,
                                   Debugger *debugger, Target *target,
                                   const char *replOptions) {
  // Needed because the caller might have forgotten to clear this value.
  error.Clear();

  if (!target && !debugger) {
    error = Status::FromErrorString(
        "must have a debugger or a target to create a REPL");
    return nullptr;
  }

  setupLLDBForREPL(debugger ? *debugger : target->GetDebugger());

  if (target) {
    auto repl = createInstanceFromTarget(*target, replOptions);
    if (repl)
      return *repl;
    return error = Status::FromError(repl.takeError()), nullptr;
  }
  auto repl = createInstanceFromDebugger(*debugger, replOptions);
  if (repl)
    return *repl;
  return error = Status::FromError(repl.takeError()), nullptr;
}

void MojoREPL::Initialize() {
  LanguageSet languages;
  languages.Insert(lldb::eLanguageTypeMojo);
  PluginManager::RegisterPlugin(getPluginNameStatic(), "Mojo language REPL",
                                createInstance, languages);
}

void MojoREPL::Terminate() { PluginManager::UnregisterPlugin(createInstance); }

const char *MojoREPL::GetHelpPrologue() {
  // Let's try to keep this within 60 chars to make sure it fits nicely even in
  // small terminals.
  return R"(
The Mojo REPL (Read-Eval-Print-Loop) acts like an
interpreter of Mojo expressions, which are delimited by a
blank line. These expressions can define top-level
variables, functions, structs, and other declarations, which
are persisted across expressions. For example:

  1> var my_var = "Welcome to Mojo!"
  2.
  2> print(my_var)
  3.
  Welcome to Mojo!

Besides that, it is possible to execute Python expressions
using the %%python magic, which offers seamless interop with
Mojo code, including exposing top-level python declarations
in subsequent Mojo expressions. For example:

  1> %%python
  2> import sys
  3> print("Python version from python:", sys.version)
  4.
  Python version from python: 3.11.4

  5> print("Python version from Mojo:", sys.version)
  6.
  Python version from Mojo: 3.11.4

As the Mojo REPL is based on LLDB, the complete set of LLDB
debugging commands is also available as described below.

  Commands must be prefixed with a colon at the REPL prompt
  (:quit for example).
  Typing just a colon followed by return will switch to the
  LLDB prompt.

Type "< path" to read in code from a text file "path".

Finally, we encourage you to submit feature requests and error reports in
https://github.com/modular/modular/issues.
)";
}

const char *MojoREPL::IOHandlerGetHelpPrologue() {
  // FIXME(18991): CommandInterpreter distorts the output returned by this
  // function.
  return GetHelpPrologue();
}
//===----------------------------------------------------------------------===//
// Source Code Handling
//===----------------------------------------------------------------------===//

bool MojoREPL::SourceIsComplete(const std::string &source) {
  SmallVector<StringRef> lines;
  StringRef(source).split(lines, "\n");

  // If the last line is empty, then the source is complete. The only case in
  // which this fails is if the last expression is a multi-line string because
  // we'd be forbidding empty lines in them.
  return lines.empty() || lines.back().trim().empty();
}

lldb::offset_t MojoREPL::GetDesiredIndentation(const StringList &lines,
                                               int cursorPosition,
                                               int tabSize) {
  // We only indent if we are at the beginning of a line (cursorPosition == 0)
  // and it's not the first line (lines.getSize() >= 2).
  if (cursorPosition != 0 || lines.GetSize() < 2)
    return LLDB_INVALID_OFFSET;
  // We base our indent level on the previous line. If it creates a new scope or
  // starts a collection of elements, we increase the indent level, otherwise,
  // we keep the level from the previous line, unless it starts with keywords
  // that terminate the current scope, like `break` or `return`, in which case
  // we look for the most recent line that has a lower indent level, and we use
  // that one. This last heuristic fails in the presence of multiline strings,
  // but this is a case so rare that it's fine to fail occasionally. A proper
  // way to handle this would be to use the mojo parser, which is non-trivial
  // effort.
  auto getIndentPrefix = [](StringRef line) {
    return line.substr(0, line.find_first_not_of(" \t\n\v\f\r"));
  };

  StringRef prevLine(lines[lines.GetSize() - 2]);
  StringRef prevLineTrimmed = prevLine.trim();
  StringRef prevIndentPrefix = getIndentPrefix(prevLine);
  lldb::offset_t prevIndentLevel = prevIndentPrefix.size();

  // Prev is empty, so we deindent all the way to level 0 because we'll be
  // starting a brand new expression.
  if (prevLineTrimmed.empty())
    return 0;

  // Prev line is a comment.
  if (prevLineTrimmed[0] == '#')
    return prevIndentLevel;

  // Prev line creates a scope or starts a collection.
  if (llvm::is_contained({'{', '[', ':'}, prevLineTrimmed.back()))
    return prevIndentLevel + tabSize;

  static auto scopeModifiers = {"pass", "return", "continue", "break"};
  // Prev line breaks the control flow.
  if (llvm::any_of(scopeModifiers, [&](StringRef keyword) {
        return prevLineTrimmed.starts_with(keyword);
      })) {
    for (const std::string &line : llvm::reverse(llvm::drop_end(lines, 2))) {
      StringRef indentPrefix = getIndentPrefix(line);
      if (indentPrefix.size() < prevIndentLevel)
        return indentPrefix.size();
    }
    // This might only happen if the source code is incorrect, so returning 0
    // indent as fallback is fine.
    return 0;
  }
  return prevIndentLevel;
}

void MojoREPL::CompleteCode(const std::string &current_code,
                            CompletionRequest &request) {
  // If we're completing a partially written token, grab that and use it for
  // filtering results.
  StringRef completionPrefix;
  const Args::ArgEntry &completionArg = request.GetParsedArg();
  if (!completionArg.IsQuoted() &&
      llvm::all_of(completionArg.ref(), [](char c) {
        return llvm::isAlnum(c) || c == '_' || c == '$';
      })) {
    completionPrefix = completionArg.ref();
  }

  std::vector<CodeCompletionResult> result = handleREPLCodeComplete(
      *getTarget(), current_code + "\n", current_code.size());
  for (const CodeCompletionResult &completion : result) {
    if (completionPrefix.empty() ||
        StringRef(completion.label).starts_with(completionPrefix))
      request.AddCompletion(completion.label);
  }
}

std::vector<CodeCompletionResult>
MojoREPL::handleREPLCodeComplete(Target &target, StringRef code,
                                 uint64_t completionPos) {
  // Collect the current persistent variables.
  SmallVector<std::pair<StringRef, Type>> variables;
  MojoTypeSystem &typeSystem = getMojoTypeSystem(target);
  auto *persistentState = static_cast<MojoPersistentExpressionState *>(
      typeSystem.GetPersistentExpressionState());
  persistentState->collectPersistentVariables(variables);

  // Call into the parser context to perform the code completion.
  return typeSystem.getParserContext().codeCompleteREPLExpression(
      code, completionPos, variables);
}

//===----------------------------------------------------------------------===//
// Variable Printing
//===----------------------------------------------------------------------===//

bool MojoREPL::PrintOneVariable(Debugger &debugger,
                                lldb::LockableStreamFileSP &output,
                                lldb::ValueObjectSP &valobj,
                                ExpressionVariable *var) {
  // TODO: If a ExpressionVariable was passed, check first if that variable is
  // just an automatically created expression result. These variables are
  // already printed by the REPL so this is done to prevent printing the
  // variable twice.
  auto options = DumpValueObjectOptions::DefaultOptions();
  options.SetShowTypes(true);

  lldb_private::LockedStreamFile lockedStream = output->Lock();

  bool useColor = debugger.GetUseColor();
  if (useColor)
    fprintf(lockedStream.GetFile().GetStream(),
            ANSI_ESCAPE1(ANSI_FG_COLOR_CYAN));

  if (llvm::Error error = valobj->Dump(lockedStream, options))
    lockedStream << "error: " << llvm::toString(std::move(error));

  if (useColor)
    fprintf(lockedStream.GetFile().GetStream(), ANSI_ESCAPE1(ANSI_CTRL_NORMAL));

  return true;
}
