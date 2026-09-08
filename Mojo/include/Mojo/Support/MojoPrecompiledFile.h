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
//
// This file provides utilities for reading/writing Mojo package files.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_MOJOPRECOMPILEDFILE_H
#define KGEN_SUPPORT_MOJOPRECOMPILEDFILE_H

#include "Config/Version.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/MemoryBuffer.h"

namespace M::KGEN {

struct MojoPrecompiledFileVersion final {
  int major = 0;
  int minor = 0;
  int patch = 0;
  std::string label = "";

  bool hasValidLabel = true;
  enum class ABRC {
    Alpha = 0,
    Beta,
    RC,
    None,
  } abrc = ABRC::None;
  std::optional<int> nABRC;
  std::optional<int> postN;
  std::optional<int> devN;

  MojoPrecompiledFileVersion() = default;
  MojoPrecompiledFileVersion(int maj, int min, int pat)
      : major(maj), minor(min), patch(pat) {}

  MojoPrecompiledFileVersion(const M::ProjectVersion &version)
      : major(version.major), minor(version.minor), patch(version.patch),
        label(version.label) {
    if (!(hasValidLabel = parseLabel())) {
      // Reset the parsed label fields on error
      abrc = ABRC::None;
      nABRC.reset();
      postN.reset();
      devN.reset();
    }
  }

  // Returns true on success, false on any consumeInteger failure; at exit,
  // labelRef must be empty for the parse to be considered valid.
  bool parseLabel() {
    StringRef labelRef = label;
    // Parse the label: [{a|b|rc}N][.postN][.devN]
    if (labelRef.consume_front("alpha") || labelRef.consume_front("a"))
      abrc = ABRC::Alpha;
    else if (labelRef.consume_front("beta") || labelRef.consume_front("b"))
      abrc = ABRC::Beta;
    else if (labelRef.consume_front("rc") || labelRef.consume_front("c"))
      abrc = ABRC::RC;
    if (abrc != ABRC::None) {
      int x;
      if (labelRef.consumeInteger(10, x) || x < 0)
        return false;
      nABRC = x;
    }
    if (labelRef.consume_front(".post")) {
      int x;
      if (labelRef.starts_with('.') || labelRef.empty())
        x = 0;
      else if (labelRef.consumeInteger(10, x) || x < 0)
        return false;
      postN = x;
    }
    if (labelRef.consume_front(".dev")) {
      int x;
      if (labelRef.empty())
        x = 0;
      else if (labelRef.consumeInteger(10, x) || x < 0)
        return false;
      devN = x;
    }
    return labelRef.empty();
  }

  static auto makeVersionCmpKey(const MojoPrecompiledFileVersion &ver) {
    // The "abrc" slot normally sorts by enum value. But a `.dev` on a no-pre
    // version is pre-pre-release, so it must sort *below* every explicit pre.
    int preClass = (ver.abrc == ABRC::None && !ver.postN && ver.devN)
                       ? -1
                       : static_cast<int>(ver.abrc);
    auto iABRCN = ver.nABRC.value_or(std::numeric_limits<int>::max());
    auto iDevN = ver.devN.value_or(std::numeric_limits<int>::max());
    auto iPostN = ver.postN.value_or(std::numeric_limits<int>::min());

    // 0.26.3.devN < 1.0.0b1.devN < 1.0.0b1.devN+1 < 1.0.0b1 < 1.0.0b2.devN
    // < 1.0.0b3 < 1.0.0rcN < 1.0.0 < 1.0.0.postN < 1.0.0.postN+1 < 1.1.0
    return std::tuple(ver.major, ver.minor, ver.patch, preClass, iABRCN, iPostN,
                      iDevN);
  }

  bool operator<(const MojoPrecompiledFileVersion &other) const {
    return makeVersionCmpKey(*this) < makeVersionCmpKey(other);
  }

  bool operator>(const MojoPrecompiledFileVersion &other) const {
    return makeVersionCmpKey(*this) > makeVersionCmpKey(other);
  }

  bool operator==(const MojoPrecompiledFileVersion &other) const {
    return std::tie(major, minor, patch, abrc, nABRC, postN, devN) ==
           std::tie(other.major, other.minor, other.patch, other.abrc,
                    other.nABRC, other.postN, other.devN);
  }

  // Format version for display
  std::string toString() const {
    std::string result = std::to_string(major) + "." + std::to_string(minor) +
                         "." + std::to_string(patch);
    if (!label.empty())
      result += label;
    return result;
  }

  operator bool() const {
    return major != 0 || minor != 0 || patch != 0 || !label.empty();
  }
};

/// Format version of the Mojo package binary encoding.
enum class MojoPrecompiledFileFormatVersion : uint8_t {
  /// Version 1: uncompressed MLIR bytecode.
  V1 = 1,
  /// Version 2: zstd-compressed MLIR bytecode with uncompressed size in header.
  V2 = 2,
};

/// Represents the header section of a Mojo package file, coming before the MLIR
/// section.
struct MojoPrecompiledFileHeader {
  MojoPrecompiledFileVersion mojoVersion;
  MojoPrecompiledFileVersion modularVersion;
  std::string mlirChecksum;
  MojoPrecompiledFileFormatVersion version =
      MojoPrecompiledFileFormatVersion::V2;
  uint64_t uncompressedSize = 0;
  size_t headerSize;

  size_t getSizeInBytes() const { return headerSize; }
  void dump() const;
};

/// Write the bytecode for the given operation to the provided output stream as
/// a Mojo package file. For streams where it matters, the given stream should
/// be in "binary" mode.
LogicalResult writePrecompiledFile(Operation *op, raw_ostream &os);

/// Write the bytecode for the given operation to the provided output stream as
/// a Mojo package file. For streams where it matters, the given stream should
/// be in "binary" mode.
/// Note: public visibility, intended only for round-trip unit testing
LogicalResult writePrecompiledFile(Operation *op,
                                   MojoPrecompiledFileVersion &mojoVer,
                                   MojoPrecompiledFileVersion &maxVer,
                                   StringRef mlirChecksum, raw_ostream &os);

/// Holds the MLIR buffer extracted from a Mojo package. For compressed (v2+)
/// packages, `ownedData` holds the decompressed data and `buffer` references
/// it. For uncompressed (v1) packages, `ownedData` is null and `buffer`
/// references the original package file.
struct MojoPrecompiledFileMLIRBuffer {
  llvm::MemoryBufferRef buffer;
  std::unique_ptr<llvm::MemoryBuffer> ownedData;
};

/// Returns whether the memory buffer points to a valid Mojo pre-compiled File
/// (.mojoc) file. Checks only the magic bytes at the beginning of the buffer.
bool isMojoPrecompiledFile(llvm::MemoryBufferRef buffer);

/// Returns whether the Mojo package (represented by its header) is compatible
/// with the current compiler.
bool isCompatiblePrecompiledFile(const MojoPrecompiledFileHeader &header);

/// Returns whether the Mojo package (represented by its header) is compatible
/// with the current compiler, and returns a message explaining any cause of
/// incompatibility.
ErrorOrSuccess
checkCompatiblePrecompiledFile(const MojoPrecompiledFileHeader &header,
                               StringRef packageName = "");

/// Compares two (package) versions and returns how 'other' compares to the
/// 'base' with a human-readable message on inequality. Optionally takes
/// human-readable names for each version and will add those to the message.
ErrorOrSuccess checkVersion(const MojoPrecompiledFileVersion &base,
                            const MojoPrecompiledFileVersion &other,
                            llvm::StringRef baseName = "",
                            llvm::StringRef otherName = "");

/// Reads and returns the Mojo package header section of a Mojo package file.
/// Returns an Error on failure. The buffer is read-only; the pointer to
/// the MLIR section can be computed by offsetting the buffer by the size of the
/// returned header (MojoPrecompiledFileHeader::getSizeInBytes).
ErrorOr<MojoPrecompiledFileHeader>
readPrecompiledFileHeader(llvm::MemoryBufferRef buffer);

// Read a Mojo package, returning both the header and the MLIR buffer.
// For compressed packages, the returned MojoPrecompiledFileMLIRBuffer owns
// the decompressed data.
ErrorOr<std::pair<MojoPrecompiledFileHeader, MojoPrecompiledFileMLIRBuffer>>
getMLIRBufferAndHeaderFromPrecompiledFile(llvm::MemoryBufferRef buffer);

// Read a Mojo package, returning the MLIR buffer if the header is compatible,
// or else an error if ignoreIncompatiblePrecompiledFiles is false. For
// compressed packages, the returned MojoPrecompiledFileMLIRBuffer owns the
// decompressed data.
ErrorOr<MojoPrecompiledFileMLIRBuffer>
getMLIRBufferFromPrecompiledFile(llvm::MemoryBufferRef buffer,
                                 bool ignoreIncompatiblePrecompiledFiles);

} // namespace M::KGEN

#endif // KGEN_SUPPORT_MOJOPRECOMPILEDFILE_H
