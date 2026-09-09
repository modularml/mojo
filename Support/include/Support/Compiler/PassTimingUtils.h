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

#ifndef SUPPORT_COMPILER_PASSTIMINGUTILS_H
#define SUPPORT_COMPILER_PASSTIMINGUTILS_H

#include "Support/ErrorOr.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"

namespace M {
/// Enables pass timing on the `PassManger` object and dumps the tree structured
/// JSON to a temp file in `outDir`. The ostream object is initialized in the
/// function
ErrorOrSuccess
configureMLIRPassTimingJSONOutput(mlir::PassManager &pm, llvm::StringRef outDir,
                                  llvm::StringRef passPipelineName);
} // namespace M

#endif // SUPPORT_COMPILER_PASSTIMINGUTILS_H
