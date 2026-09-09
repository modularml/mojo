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

#ifndef KGEN_LIB_MOJOLLDB_REPL_MOJOREPL_H
#define KGEN_LIB_MOJOLLDB_REPL_MOJOREPL_H

#include "../TypeSystem/MojoTypeSystem.h"
#include "lldb/Expression/REPL.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-public.h"
#include <string>
#include <thread>
#include <vector>

namespace M::KGEN::Mojo {
struct CodeCompletionResult;

/// This class implements a MOJO repl plugin for LLDB.
class MojoREPL : public llvm::RTTIExtends<MojoREPL, lldb_private::REPL> {
public:
  // LLVM RTTI support
  static char ID;

  MojoREPL(lldb_private::Target &target);
  ~MojoREPL() override;

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  static void Initialize();
  static void Terminate();

  static llvm::StringRef getPluginNameStatic() { return "Mojo REPL"; }

  /// Launch the entry point process that is used to JIT Mojo expressions.
  MODULAR_VISIBILITY_EXPORT static llvm::Error launchEntryPointProcess(
      lldb_private::Target &target, lldb_private::Debugger &debugger,
      StringRef workingDirectory = {},
      ArrayRef<std::string> additionalImportDirectories = {});

  /// Return the common help message to show in commands like `:help` or `:mojo
  /// help`.
  static const char *GetHelpPrologue();

  //===--------------------------------------------------------------------===//
  // Code Completion
  //===--------------------------------------------------------------------===//

  /// Perform a REPL code completion within the given type system.
  MODULAR_VISIBILITY_EXPORT static std::vector<CodeCompletionResult>
  handleREPLCodeComplete(lldb_private::Target &target, StringRef code,
                         uint64_t completionPos);

protected:
  lldb_private::Status DoInitialization() override { return {}; }

  //===--------------------------------------------------------------------===//
  // Utilities
  //===--------------------------------------------------------------------===//

  StringRef GetSourceFileBasename() override {
    return lldb_private::ConstString("repl.mojo");
  }

  lldb::LanguageType GetLanguage() override { return lldb::eLanguageTypeMojo; }

  std::shared_ptr<MojoTypeSystem> getTypeSystem() { return typeSystem; }

  const char *IOHandlerGetHelpPrologue() override;

  //===--------------------------------------------------------------------===//
  // Source Code Handling
  //===--------------------------------------------------------------------===//

  /// Return a string of characters which trigger auto-indentation.
  const char *GetAutoIndentCharacters() override { return ""; }

  /// Return if the given source string is a complete expression for the repl.
  bool SourceIsComplete(const std::string &source) override;

  /// Return the desired indentation for indenting the given source.
  lldb::offset_t GetDesiredIndentation(const lldb_private::StringList &lines,
                                       int cursorPosition,
                                       int tabSize) override;

  /// Process a code completion request on the given source.
  void CompleteCode(const std::string &current_code,
                    lldb_private::CompletionRequest &request) override;

  //===--------------------------------------------------------------------===//
  // Variable Printing
  //===--------------------------------------------------------------------===//

  /// Print the value of the given value object to the given output stream. An
  /// expression variable may optionally be provided.
  bool
  PrintOneVariable(lldb_private::Debugger &debugger,
                   lldb::LockableStreamFileSP &output,
                   lldb::ValueObjectSP &valobj,
                   lldb_private::ExpressionVariable *var = nullptr) override;

private:
  llvm::Error OnExpressionEvaluated(
      const lldb_private::ExecutionContext &exe_ctx, llvm::StringRef code,
      const lldb_private::EvaluateExpressionOptions &expr_options,
      lldb::ExpressionResults execution_results,
      const lldb::ValueObjectSP &result_valobj_sp,
      const lldb_private::Status &error) override;

  /// Flush expression events and the inferior's stdout/stderr streams.
  void flushExpressionEventsAndProcessStreams();

  lldb::TargetSP getTarget() { return targetWP.lock(); }

  /// This thread will listen to events in the underlying target and assumes
  /// there is only one target at a time.
  std::thread eventThread;
  std::atomic_bool stopEventThread = false;
  lldb::ListenerSP mojoExpressionListener;
  lldb::TargetWP targetWP;
  lldb::StreamSP errorStream;
  std::mutex flushStreamsMutex;
  std::shared_ptr<MojoTypeSystem> typeSystem;
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_REPL_MOJOREPL_H
