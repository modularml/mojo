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

#ifndef KGEN_SUPPORT_FILEUTILS_H
#define KGEN_SUPPORT_FILEUTILS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include <string>

namespace M::KGEN {

/// Write the given `buf` to a file with the given prefix and postfix.
/// Appends a hash based on `buf` contents to the emitted file name.
/// If `saveTempsPrefix` is empty, does nothing and returns success.
mlir::LogicalResult writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                                             const std::string &postfix,
                                             llvm::StringRef buf);

/// Write the given `buf` to a file with the given prefix and postfix.
/// Appends a hash based on `buf` contents to the emitted file name.
/// If `saveTempsPrefix` is empty, does nothing and returns success.
mlir::LogicalResult writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                                             const std::string &postfix,
                                             llvm::MemoryBufferRef buf);

} // namespace M::KGEN

#endif // KGEN_SUPPORT_FILEUTILS_H
