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

#ifndef KGEN_OBJECTCOMPILER_LLVMIRUTILS_H
#define KGEN_OBJECTCOMPILER_LLVMIRUTILS_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Module.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// LLVMModuleAndContext
//===----------------------------------------------------------------------===//

/// A pair of an LLVM module and the LLVM context that holds ownership of the
/// objects. This is a useful class for parallelizing LLVM and managing
/// ownership of LLVM instances.
class LLVMModuleAndContext {
public:
  /// Expose the underlying LLVM context to create the module. This is the only
  /// way to access the LLVM context to prevent accidental sharing.
  ErrorOrSuccess create(
      function_ref<ErrorOr<std::unique_ptr<llvm::Module>>(llvm::LLVMContext &)>
          createModule);

  llvm::Module &operator*() { return *module; }
  llvm::Module *operator->() { return module.get(); }

  void reset();

  llvm::StringSet<> duplicatedFns;

private:
  /// LLVM context stored in a unique pointer so that we can move this type.
  std::unique_ptr<llvm::LLVMContext> ctx =
      std::make_unique<llvm::LLVMContext>();
  /// The paired LLVM module.
  std::unique_ptr<llvm::Module> module;
};

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

using LLVMSplitProcessFn =
    function_ref<void(llvm::unique_function<LLVMModuleAndContext()>,
                      std::optional<int64_t>, unsigned)>;

/// Helper to create a lambda that just forwards a preexisting modu.e.
inline llvm::unique_function<LLVMModuleAndContext()>
forwardModule(LLVMModuleAndContext &&module) {
  return [module = std::move(module)]() mutable { return std::move(module); };
}

/// support for splitting an LLVM module into multiple parts using exported
/// functions as anchors, and pull in all dependency on the call stack into one
/// module.
void splitPerExported(LLVMModuleAndContext module,
                      LLVMSplitProcessFn processFn);

/// support for splitting an LLVM module into multiple parts with each part
/// contains only one function (with exception for coroutine related functions.)
void splitPerFunction(
    LLVMModuleAndContext module, LLVMSplitProcessFn processFn,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
    int64_t inputModuleIdx = 0, unsigned numFunctionBase = 0);

} // namespace M::KGEN

#endif // KGEN_OBJECTCOMPILER_LLVMIRUTILS_H
