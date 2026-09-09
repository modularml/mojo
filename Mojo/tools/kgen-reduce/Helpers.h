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

#ifndef KGEN_REDUCE_HELPERS_H
#define KGEN_REDUCE_HELPERS_H

#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"

#include <cstdint>
#include <string>

namespace llvm {
class ToolOutputFile;
} // namespace llvm

namespace M {
/// Get the current time in milliseconds.
uint64_t getCurTimeMs();

/// Get a filename for a snapshot file.
std::string getTempFileName();

/// Write the module to a temporary file that will by default be deleted on
/// exit.
ErrorOr<std::unique_ptr<llvm::ToolOutputFile>>
getTempFile(ModuleOp module, const Twine &fileName, StringRef pipeline);

/// Store the module to a permanent file.
ErrorOrSuccess stashFile(ModuleOp module, const Twine &fileName,
                         StringRef pipeline);

/// Indicate that a file should be removed on exit from the process instead.
void unkeepToolOutputFile(llvm::ToolOutputFile &file);

/// Return true if a region is stubbed.
bool isStubbed(Region &region);

/// Stub a region and hold its old body in `owner`.
void stubRegion(Region &region, Region &owner);
} // namespace M

#endif // KGEN_REDUCE_HELPERS_H
