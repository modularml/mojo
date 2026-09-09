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
// This file defines the API for interacting with the Mojo Jupyter Kernel.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOJUPYTER_KERNEL_H
#define KGEN_MOJOJUPYTER_KERNEL_H

#include "Support/Context.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/FunctionExtras.h"

namespace M::Mojo::Jupyter {

/// These values were chosen because both `finished` and `error` are 'truthy',
/// and indicate that the execution has finished. Therefore, only clients that
/// care if there was an error have to do anything with it.
///
/// When updating this enum, please take care to also update the enums in
/// mojo-jupyter-executor/main.cpp and mojokernel.py.
enum ExecutionFinishedState : int {
  kNotFinished = 0,
  kFinishedSuccessfully = 1,
  kFinishedError = 2,
};

/// This class contains all of the various state needed to run the Mojo Jupyter
/// kernel.
class MODULAR_VISIBILITY_EXPORT MojoKernel {
public:
  /// An output function used to send output to the Jupyter kernel. The first
  /// argument is the output type, and the second is the output string.
  using OutputFn = llvm::unique_function<void(StringRef, StringRef)>;

  /// A function used to send code completion results to the Jupyter kernel.
  using CompletionFn = llvm::function_ref<void(StringRef)>;

  MojoKernel(OutputFn outputFn, bool initializeMatPlotLib = true);
  ~MojoKernel();

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
                                        bool storeHistory = true);

  /// Check if the current expression has finished execution, also taking this
  /// time to flush any collected output.
  ExecutionFinishedState checkExecutionFinished();

  /// Utility function to start and wait for the execution of the given cell
  /// identifier and expression string. `storeHistory` indicates if variables
  /// and state from this expression should be persisted. Returns the state of
  /// the expression execution.
  ExecutionFinishedState executeAndWait(StringRef cellId, StringRef expr,
                                        bool storeHistory = true);

  /// Interrupt the currently running execution.
  void interruptExecution();

  /// Perform code completion at the given position within the given code
  /// string. The completion function will be called with the completion
  /// results.
  void codeComplete(StringRef code, int completionPos,
                    CompletionFn completionFn);

private:
  /// The internal implementation of the MojoKernel.
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace M::Mojo::Jupyter

#endif // KGEN_MOJOJUPYTER_KERNEL_H
