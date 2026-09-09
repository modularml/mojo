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

#ifndef KGEN_SUPPORT_ERROR_H
#define KGEN_SUPPORT_ERROR_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"

namespace M {
struct ErrorLimit {
  int errorLimit = 0;
  int errorCount = 0;
};

/// Emit error with a limit check.
void emitLimitedError(function_ref<InFlightDiagnostic()> emitError,
                      ErrorLimit &limit);

/// Helper function to check if a location is from the mojo startup module or
/// not.
bool isLocationInPrelude(const Location &loc);

} // namespace M

#endif // KGEN_SUPPORT_ERROR_H
