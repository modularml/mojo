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

#include "Mojo/Support/FileUtils.h"

#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/xxhash.h"

using namespace M;
using namespace KGEN;

mlir::LogicalResult
M::KGEN::writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                                  const std::string &postfix,
                                  llvm::StringRef buf) {
  if (saveTempsPrefix.empty())
    return mlir::success();

  // Include unique hash as part of name.
  assert(sizeof(uint8_t) == sizeof(char) && "Assume char is 8 bits");
  llvm::XXH128_hash_t hash =
      llvm::xxh3_128bits(llvm::arrayRefFromStringRef(buf));
  std::string outPath =
      saveTempsPrefix + "." + llvm::utohexstr(hash.high64, /*LowerCase=*/true) +
      llvm::utohexstr(hash.low64, /*LowerCase=*/true) + postfix;

  auto outFile = mlir::openOutputFile(outPath);
  if (!outFile)
    return mlir::failure();
  outFile->os() << buf;
  outFile->keep();
  return mlir::success();
}

mlir::LogicalResult
M::KGEN::writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                                  const std::string &postfix,
                                  llvm::MemoryBufferRef buf) {
  return writeBytesToTempWithHash(saveTempsPrefix, postfix, buf.getBuffer());
}
