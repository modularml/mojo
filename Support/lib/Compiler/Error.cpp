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

#include "Support/Compiler/Error.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace M;

void M::emitLimitedError(function_ref<InFlightDiagnostic()> emitError,
                         ErrorLimit &limit) {
  ++limit.errorCount;
  if (limit.errorLimit > 0 && limit.errorCount > limit.errorLimit) {
    return;
  }

  InFlightDiagnostic diag = emitError();

  // Emit message if hits error limit.
  if (limit.errorCount == limit.errorLimit)
    diag.attachNote() << "too many errors emitted, stopping now";
}

bool M::isLocationInPrelude(const Location &loc) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << loc;
  return str.find("_startup.mojo") != std::string::npos;
}
