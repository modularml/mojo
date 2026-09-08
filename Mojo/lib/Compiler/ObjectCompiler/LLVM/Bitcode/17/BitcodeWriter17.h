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

#ifndef KGEN_COMPILER_LLVMIR_BITCODE_BITCODEWRITER17_H
#define KGEN_COMPILER_LLVMIR_BITCODE_BITCODEWRITER17_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/Allocator.h"
#include <array>

#include <memory>
#include <vector>

namespace llvm {

class BitstreamWriter;
class Module;
class raw_ostream;

} // namespace llvm

namespace M::KGEN::LLVM {

/// BitcodeWriter17 - Legacy bitcode writer for LLVM bitcode version 17
/// This class writes LLVM modules to bitcode format version 17, which is
/// required for Metal compatibility.
class BitcodeWriter17 {

  std::unique_ptr<llvm::BitstreamWriter> Stream;

  llvm::StringTableBuilder StrtabBuilder{llvm::StringTableBuilder::RAW};

  // Owns any strings created by the irsymtab writer until we create the
  // string table.
  llvm::BumpPtrAllocator Alloc;

  bool WroteStrtab = false, WroteSymtab = false;

  void writeBlob(unsigned Block, unsigned Record, llvm::StringRef Blob);

  std::vector<llvm::Module *> Mods;

public:
  /// Create a BitcodeWriter17 that writes to Buffer.
  BitcodeWriter17(llvm::SmallVectorImpl<char> &Buffer);
  BitcodeWriter17(llvm::raw_ostream &FS);

  ~BitcodeWriter17();

  /// Attempt to write a symbol table to the bitcode file. This must be called
  /// at most once after all modules have been written.
  ///
  /// A reader does not require a symbol table to interpret a bitcode file;
  /// the symbol table is needed only to improve link-time performance. So
  /// this function may decide not to write a symbol table. It may so decide
  /// if, for example, the target is unregistered or the IR is malformed.
  void writeSymtab();

  /// Write the bitcode file's string table. This must be called exactly once
  /// after all modules and the optional symbol table have been written.
  void writeStrtab();

  /// Copy the string table for another module into this bitcode file. This
  /// should be called after copying the module itself into the bitcode file.
  void copyStrtab(llvm::StringRef Strtab);

  /// Write the specified module to the buffer specified at construction time.
  ///
  /// If \c ShouldPreserveUseListOrder, encode the use-list order for each \a
  /// Value in \c M.  These will be reconstructed exactly when \a M is
  /// deserialized.
  ///
  /// If \c Index is supplied, the bitcode will contain the summary index
  /// (currently for use in ThinLTO optimization).
  ///
  /// \p GenerateHash enables hashing the Module and including the hash in the
  /// bitcode (currently for use in ThinLTO incremental build).
  ///
  /// If \p ModHash is non-null, when GenerateHash is true, the resulting
  /// hash is written into ModHash. When GenerateHash is false, that value
  /// is used as the hash instead of computing from the generated bitcode.
  /// Can be used to produce the same module hash for a minimized bitcode
  /// used just for the thin link as in the regular full bitcode that will
  /// be used in the backend.
  void writeModule(const llvm::Module &M,
                   bool ShouldPreserveUseListOrder = false,
                   const llvm::ModuleSummaryIndex *Index = nullptr,
                   bool GenerateHash = false,
                   llvm::ModuleHash *ModHash = nullptr);

  void writeIndex(const llvm::ModuleSummaryIndex *Index,
                  const void *ModuleToSummariesForIndex);
};

/// Write the specified module summary index to the given raw output stream
/// using bitcode format version 17.
void WriteIndexToFile17(const llvm::ModuleSummaryIndex &Index,
                        llvm::raw_ostream &Out,
                        const void *ModuleToSummariesForIndex = nullptr);

} // namespace M::KGEN::LLVM

#endif // KGEN_COMPILER_LLVMIR_BITCODE_BITCODEWRITER17_H
