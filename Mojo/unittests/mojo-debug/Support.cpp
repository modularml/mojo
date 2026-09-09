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

#include "Support.h"
#include "Mojo/Support/Configuration.h"
#include "lldb/API/SBDebugger.h"
#include "llvm/Support/Program.h"

#include <cstdlib>

using namespace M;
using namespace lldb;

/// The leak sanitizer shows errors, probably because we load libpython via
/// LLDB.
extern "C" const char *__asan_default_options() { return "detect_leaks=0"; }

static TempDir createTempDir() {
  ErrorOr<TempDir> tempDirOr = TempDir::create("mojo-debug.%%%%%%");
  if (failed(tempDirOr))
    llvm::report_fatal_error(tempDirOr.takeError().get());
  return std::move(*tempDirOr);
}

MojoSource::MojoSource(StringRef fileName) {
  path = std::filesystem::absolute(
             std::filesystem::path(std::getenv("MODULAR_PATH")))
             .lexically_normal() /
         "Mojo" / "unittests" / "mojo-debug" / "inputs" / fileName.str();
  pathStr = path.string();

  auto bufferOr = toModularErrorOr(llvm::MemoryBuffer::getFile(pathStr));
  if (failed(bufferOr))
    llvm::report_fatal_error(Twine("Error reading the file ") + pathStr + ": " +
                             bufferOr.getError());
  llvm::MemoryBuffer &buffer = *bufferOr->get();
  contents = buffer.getBuffer();

  StringRef(contents).split(lines, '\n');
}

std::vector<int> MojoSource::findLinesWithText(StringRef text) const {
  std::vector<int> result;
  for (size_t i = 0, e = lines.size(); i < e; ++i)
    if (lines[i].contains(text))
      result.push_back(i + 1);
  return result;
}

int MojoSource::findFirstLineWithText(StringRef text) const {
  auto lines = findLinesWithText(text);
  return lines.empty() ? -1 : lines[0];
}

MojoBinary::MojoBinary(const std::shared_ptr<MojoSource> &source,
                       bool suppressBuildOutput)
    : source(source), outDir(createTempDir()),
      binPath(outDir.getPath() /
              (source->getFilesystemPath().filename().string() + ".exe")) {

  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr))
    llvm::report_fatal_error(configOr.getError());

  StringRef mojo = configOr->getDriverPath();
  std::vector<std::optional<StringRef>> redirects;
  if (suppressBuildOutput) {
    for (size_t i = 0; i < 3; ++i)
      redirects.emplace_back("");
  }
  int ec = llvm::sys::ExecuteAndWait(
      mojo, {mojo, "build", "-g", "-O0", source->getPath(), "-o", binPath},
      /*Env=*/std::nullopt, redirects);
  if (ec)
    llvm::report_fatal_error(
        llvm::formatv("`mojo build -g -O0 '{0}'` failed with exit code {1}. "
                      "This test won't run unless you fix this.",
                      source->getPath(), std::to_string(ec)));
}

static SBDebugger *g_debugger_ptr = nullptr;

/// Clean up LLDB on exit. Destroy the debugger first to release internal
/// resources, then terminate LLDB to unregister built-in plugins.
/// LLDBPluginTerminate runs afterward via its own atexit (LIFO ordering)
/// to unregister the Mojo plugin components.
static void cleanupLLDB() {
  if (g_debugger_ptr) {
    SBDebugger::Destroy(*g_debugger_ptr);
    g_debugger_ptr = nullptr;
  }
  SBDebugger::Terminate();
}

/// Acquire a singleton instance of a debugger.
static SBDebugger getOrCreateSBDebugger() {
  static std::once_flag flag;
  static SBDebugger debugger;
  std::call_once(flag, []() {
    // Initialize the singleton debugger.
    SBError err = SBDebugger::InitializeWithErrorHandling();
    if (err.Fail()) {
      llvm::report_fatal_error(llvm::formatv(
          "Couldn't create the debugger instance: {0}", err.GetCString()));
    }
    debugger = SBDebugger::Create(/*source_init_files=*/false);
    g_debugger_ptr = &debugger;
    debugger.SetAsync(false);

    // Launch the test lldbinit file
    SBFileSpec lldbInitPath(
        (std::filesystem::path(std::getenv("MODULAR_PATH")) / "Mojo" /
         "unittests" / "lit-lldb-init.in")
            .string()
            .c_str());

    if (!lldbInitPath.Exists())
      llvm::report_fatal_error("lldbinit doesn't exist");

    SBCommandReturnObject result;
    SBExecutionContext exeCtx;
    SBCommandInterpreterRunOptions options;
    options.SetPrintResults(false);
    options.SetEchoCommands(false);
    debugger.GetCommandInterpreter().HandleCommandsFromFile(
        lldbInitPath, exeCtx, options, result);

    if (std::string error = result.GetError(); !error.empty())
      llvm::outs() << std::string(result.GetOutput()) << "\n" << error << "\n";

    // Load the MojoLLDB plugin
    ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
    if (failed(configOr))
      llvm::report_fatal_error(Twine("failed to parse 'modular.cfg': ") +
                               configOr.getError());
    std::error_code ec;
    StringRef mojoLLDB = configOr->getLLDBPluginPath();
    if (!std::filesystem::exists(mojoLLDB.str(), ec) || ec)
      llvm::report_fatal_error("unable to resolve the MojoLLDB plugin path");
    debugger.HandleCommand(("plugin load " + mojoLLDB).str().c_str());

    // Ensure LLDB is properly terminated on exit. Registered after plugin
    // load so it runs before LLDBPluginTerminate's atexit (atexit is LIFO).
    // cleanupLLDB handles the full teardown sequence: destroy debugger,
    // unregister Mojo plugins, then terminate LLDB built-in plugins.
    std::atexit(cleanupLLDB);
  });
  return debugger;
}

/// Execute the provided command using the provided context (thread, process or
/// frame).
///
/// Note: it's better to use this instead of `debugger.HandleCommand()` because
/// it doesn't work nicely if multiple targets exist at once, which happens when
/// multiple test files are executed simultaneously.
template <typename Ctx>
static CommandResult runCommandForContext(StringRef command, Ctx context) {
  SBCommandReturnObject result;
  SBExecutionContext exeCtx(context);
  SBTarget target = exeCtx.GetTarget();
  SBDebugger dbg = getOrCreateSBDebugger();
  dbg.SetSelectedTarget(target);
  dbg.GetCommandInterpreter().HandleCommand(command.data(), exeCtx, result);

  std::string output = std::string(result.GetOutput());
  std::string error = std::string(result.GetError());
  return {result.Succeeded(), output, error};
}

/// Similar to runCommandForContext, but the output and error are printed
/// right away.
///
/// Returns true if and only if the command succeeded.
template <typename Ctx>
static bool dumpCommandForContext(StringRef command, Ctx context) {
  CommandResult result = runCommandForContext(command, context);
  if (!result.output.empty())
    llvm::outs() << result.output << "\n";
  if (!result.error.empty())
    llvm::outs() << result.error << "\n";
  return result.success;
}

/// Traverses the input file looking for the `# breakpoint` comment, and
/// places a breakpoint at the lines where it appears.
static void setBreakpointsForComments(const MojoSource &source,
                                      SBTarget &target) {

  for (int line : source.findLinesWithText("# breakpoint")) {
    SBBreakpoint bp =
        target.BreakpointCreateByLocation(source.getPath().data(), line);
    if (!bp.GetNumLocations())
      llvm::report_fatal_error(llvm::formatv(
          "Couldn't set a breakpoint at {0}:{1}", source.getPath(), line));
  }
}

static StopContext runTarget(SBTarget target, MojoBinary binary) {
  SBLaunchInfo launchInfo = target.GetLaunchInfo();

  SBError err;
  target.Launch(launchInfo, err);
  if (err.Fail()) {
    llvm::report_fatal_error(
        llvm::formatv("Couldn't launch the target: {0}", err.GetCString()));
  }

  SBProcess process = target.GetProcess();
  if (!process.IsValid())
    llvm::report_fatal_error("Invalid process");

  // This ensures the process didn't exit
  if (process.GetState() != lldb::eStateStopped)
    llvm::report_fatal_error("Process is not stopped");

  SBThread thread = process.GetSelectedThread();
  return StopContext{std::move(binary), target, process, thread,
                     thread.GetFrameAtIndex(0)};
}

void StopContext::updateAfterStateChange(StringRef action) {
  this->process = this->target.GetProcess();
  if (this->process.GetState() != lldb::eStateStopped)
    llvm::report_fatal_error("Process is not stopped after " + action);

  this->thread = process.GetSelectedThread();
  this->frame = this->thread.GetFrameAtIndex(0);
}

void StopContext::stepOver() {
  thread.StepOver();
  updateAfterStateChange("step over");
}

void StopContext::stepInto() {
  thread.StepInto();
  updateAfterStateChange("step into");
}

void StopContext::resume() {
  process.Continue();
  updateAfterStateChange("resume");
}

CommandResult StopContext::runCommand(StringRef command) {
  return runCommandForContext(command, frame);
}

StopContext M::buildAndLaunch(StringRef fileName) {
  auto source = std::make_shared<MojoSource>(fileName);

  // TODO(28608): support a test mode for JIT debugging besides AOT.
  MojoBinary binary(source, /*suppressBuildOutput=*/true);
  SBTarget target =
      getOrCreateSBDebugger().CreateTarget(binary.getPath().data());
  if (!target.IsValid())
    llvm::report_fatal_error("Invalid target");

  setBreakpointsForComments(*source, target);

  return runTarget(target, std::move(binary));
}
