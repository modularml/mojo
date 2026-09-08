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

#ifndef KGEN_COMPILER_SAVEASMOUTPUT_H
#define KGEN_COMPILER_SAVEASMOUTPUT_H

/// Helpers for writing save-temps and kernel-offload output files.
///
/// Two families:
///   1. Save-temps helpers: writeBytesToTempWithHash, writeTempModule.
///   2. Kernel-offload output naming: reserveOffloadOutputBaseName,
///      offloadOutputPath, kOffloadWritesAttrName, flushOffloadWrites.

#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Save-temps helpers
//===----------------------------------------------------------------------===//

/// Write \p buf to a content-addressed file at
/// <saveTempsPrefix>.<xxh128><postfix>.  No-op when saveTempsPrefix is empty.
mlir::LogicalResult writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                                             const std::string &postfix,
                                             llvm::StringRef buf);

mlir::LogicalResult writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                                             const std::string &postfix,
                                             llvm::MemoryBufferRef buf);

/// Serialize \p module to a string and save it via writeBytesToTempWithHash.
/// The output path is <saveTempsPrefix><phase>.<hash><fileExt>.
template <typename ModuleT>
mlir::LogicalResult writeTempModule(const std::string &saveTempsPrefix,
                                    const std::string &phase, ModuleT &module,
                                    const std::string &fileExt = ".ll") {
  if (saveTempsPrefix.empty())
    return mlir::success();
  const std::string finalSavePrefix = saveTempsPrefix + phase;
  std::string str;
  llvm::raw_string_ostream ss(str);
  ss << module;
  return writeBytesToTempWithHash(finalSavePrefix, fileExt, str);
}

//===----------------------------------------------------------------------===//
// Offload output naming helpers
//===----------------------------------------------------------------------===//

/// Reserve a disambiguated base name for an offload output file.
/// \p ext is the target-specific file extension (e.g. ".s", ".ll").
/// The collision key is "<sanitized-name><ext>" so different extensions track
/// independently. \p nameCountMap is updated in place.
std::string reserveOffloadOutputBaseName(mlir::StringAttr rawName,
                                         llvm::StringRef ext,
                                         llvm::StringMap<int> &nameCountMap);

/// Return the output path for an offload kernel file.
/// Format: <prefix>_<fileName>, where fileName is "<baseName><ext>".
std::string offloadOutputPath(llvm::StringRef prefix, llvm::StringRef fileName);

//===----------------------------------------------------------------------===//
// Pending offload writes
//===----------------------------------------------------------------------===//

/// Module attribute that holds pending offload writes past cachedTransform.
/// compileOffloads() encodes the files it wants to save (.asm or .ll)
/// as a dictionary (file name → content) on the module;
/// cachedTransform serializes it into the cache alongside the IR.
/// flushOffloadWrites() reads this attribute after cachedTransform returns,
/// writing files on both the cache-hit path (deserialized module)
/// and the cache-miss path.
/// The keys are file names, not paths, so a cache entry stays valid when the
/// output directory changes.
inline constexpr llvm::StringLiteral kOffloadWritesAttrName =
    "kgen.offload_debug_files";

/// Flush all pending offload writes on \p module and remove the attribute.
/// Each file is written to \p outputPrefix joined with its recorded name.
/// Must be called after cachedTransform so files are written on both hit and
/// miss paths.
ErrorOrSuccess flushOffloadWrites(mlir::ModuleOp module,
                                  llvm::StringRef outputPrefix);

} // namespace M::KGEN

#endif // KGEN_COMPILER_SAVEASMOUTPUT_H
