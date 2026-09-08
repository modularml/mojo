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

#include "Mojo/Support/CLOptionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace llvm::codegen;

static llvm::ManagedStatic<llvm::codegen::RegisterCodeGenFlags> codegenFlagsOpt;

void M::registerCommandFlags() {
  // Register llvm::codegen::RegisterCodegenFlags flags.
  // E.g. we want to use denormal-fp-math-f32
  *codegenFlagsOpt;

  // Remove duplicated flags that are conflicting with other mojo and kgen
  // options.
  llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options =
      llvm::cl::getRegisteredOptions();
  options["march"]->removeArgument();
  options["mcpu"]->removeArgument();
  options["mattr"]->removeArgument();
  options["large-data-threshold"]->removeArgument();
  options["relocation-model"]->removeArgument();
}
