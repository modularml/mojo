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

//===- llvm/Bitcode/BitcodeWriter.h - Bitcode writers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces to write LLVM bitcode files/streams.
//
//===----------------------------------------------------------------------===//

// Get this file fromllvm-project/llvm/include/llvm/Bitcode/BitcodeWriter.h
// with tag llvmorg-19.1.7

#ifndef KGEN_COMPILER_LLVMIR_BITCODE_BITCODEWRITER19_H
#define KGEN_COMPILER_LLVMIR_BITCODE_BITCODEWRITER19_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llvm {

class BitstreamWriter;
class Module;
class raw_ostream;
} // end namespace llvm

namespace M::KGEN::LLVM {
class BitcodeWriter19 {
  std::unique_ptr<llvm::BitstreamWriter> Stream;

  llvm::StringTableBuilder StrtabBuilder{llvm::StringTableBuilder::RAW};

  // Owns any strings created by the irsymtab writer until we create the
  // string table.
  llvm::BumpPtrAllocator Alloc;

  bool WroteStrtab = false, WroteSymtab = false;

  void writeBlob(unsigned Block, unsigned Record, llvm::StringRef Blob);

  std::vector<llvm::Module *> Mods;

public:
  /// Create a BitcodeWriter19 that writes to Buffer.
  BitcodeWriter19(llvm::SmallVectorImpl<char> &Buffer);
  BitcodeWriter19(llvm::raw_ostream &FS);

  ~BitcodeWriter19();

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

  /// Write the specified thin link bitcode file (i.e., the minimized bitcode
  /// file) to the buffer specified at construction time. The thin link
  /// bitcode file is used for thin link, and it only contains the necessary
  /// information for thin link.
  ///
  /// ModHash is for use in ThinLTO incremental build, generated while the
  /// IR bitcode file writing.
  void writeThinLinkBitcode(const llvm::Module &M,
                            const llvm::ModuleSummaryIndex &Index,
                            const llvm::ModuleHash &ModHash);

  void writeIndex(const llvm::ModuleSummaryIndex *Index,
                  const std::map<std::string, llvm::GVSummaryMapTy>
                      *ModuleToSummariesForIndex,
                  const llvm::GVSummaryPtrSet *DecSummaries);
};

/// Write the specified thin link bitcode file (i.e., the minimized bitcode
/// file) to the given raw output stream, where it will be written in a new
/// bitcode block. The thin link bitcode file is used for thin link, and it
/// only contains the necessary information for thin link.
///
/// ModHash is for use in ThinLTO incremental build, generated while the IR
/// bitcode file writing.
void writeThinLinkBitcodeToFile19(const llvm::Module &M, llvm::raw_ostream &Out,
                                  const llvm::ModuleSummaryIndex &Index,
                                  const llvm::ModuleHash &ModHash);

/// Write the specified module summary index to the given raw output stream,
/// where it will be written in a new bitcode block. This is used when
/// writing the combined index file for ThinLTO. When writing a subset of the
/// index for a distributed backend, provide the \p ModuleToSummariesForIndex
/// map. \p DecSummaries specifies the set of summaries for which the
/// corresponding value should be imported as a declaration (prototype).
void writeIndexToFile19(const llvm::ModuleSummaryIndex &Index,
                        llvm::raw_ostream &Out,
                        const std::map<std::string, llvm::GVSummaryMapTy>
                            *ModuleToSummariesForIndex = nullptr,
                        const llvm::GVSummaryPtrSet *DecSummaries = nullptr);

/// If EmbedBitcode is set, save a copy of the llvm IR as data in the
///  __LLVM,__bitcode section (.llvmbc on non-MacOS).
/// If available, pass the serialized module via the Buf parameter. If not,
/// pass an empty (default-initialized) MemoryBufferRef, and the serialization
/// will be handled by this API. The same behavior happens if the provided Buf
/// is not bitcode (i.e. if it's invalid data or even textual LLVM assembly).
/// If EmbedCmdline is set, the command line is also exported in
/// the corresponding section (__LLVM,_cmdline / .llvmcmd) - even if CmdArgs
/// were empty.
void embedBitcodeInModule19(llvm::Module &M, llvm::MemoryBufferRef Buf,
                            bool EmbedBitcode, bool EmbedCmdline,
                            const std::vector<uint8_t> &CmdArgs);

} // namespace M::KGEN::LLVM

#endif // KGEN_COMPILER_LLVMIR_BITCODE_BITCODEWRITER19_H
