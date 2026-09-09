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
// This file provides utilities for reading/writing Mojo precompiled files.
//
//===----------------------------------------------------------------------===//

#include "Mojo/Support/MojoPrecompiledFile.h"
#include "Mojo/DialectChecksum/DialectChecksum.h"
#include "Support/Error.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/Process.h"

using namespace M;
using namespace KGEN;

static llvm::ArrayRef<uint8_t> asBytes(llvm::StringRef s) {
  return {reinterpret_cast<const uint8_t *>(s.data()), s.size()};
}

/// Returns whether the Mojo precompiled file (represented by its header) is
/// compatible with the current compiler.
ErrorOrSuccess
M::KGEN::checkCompatiblePrecompiledFile(const MojoPrecompiledFileHeader &header,
                                        StringRef packageName) {
  // Check whether the MLIR checksums match. If they do, assume the precompiled
  // file is okay.
  if (header.mlirChecksum == M::getMojoMlirDialectChecksum())
    return success();

  // We have a mismatch. Since MLIR checksums aren't meaningful to users, prefer
  // to return an error containing version information as a proxy.
  MojoPrecompiledFileVersion currentVer = M::getMojoVersion();

  std::string errMsg =
      "Mojo precompiled file is incompatible with the current version "
      "of the Mojo compiler";

  // The most common case (for end users) is that both the precompiled file
  // and compiler have version information. Report that.
  if (currentVer && header.mojoVersion) {
    std::string packageVerStr = ". Precompiled file";
    if (!packageName.empty())
      packageVerStr += (Twine(" '") + packageName + "'").str();
    packageVerStr += " version";

    if (auto err = checkVersion(header.mojoVersion, currentVer, packageVerStr,
                                "compiler version")) {
      return Error(errMsg + err.getError());
    }

    // Otherwise the versions are the same but the MLIR checksums don't match -
    // this is unlikely.
    return Error(
        errMsg + packageVerStr + " " + header.mojoVersion.toString() +
        " matches the compiler version, but has incompatible MLIR bytecode");
  }

  // Otherwise one or both of the precompiled file and compiler are missing
  // version information, so was either built using, or is, an internal compiler
  // build.

  if (currentVer) {
    errMsg += " (" + currentVer.toString() + "). Precompiled file";
    if (!packageName.empty())
      errMsg += (Twine(" '") + packageName + "'").str();
    return Error(errMsg + " is missing version information");
  }

  errMsg += ". Precompiled file";
  if (!packageName.empty())
    errMsg += (Twine(" '") + packageName + "'").str();
  return Error(errMsg + " was built with version " +
               header.mojoVersion.toString() +
               " but the compiler is missing version information");
}

ErrorOrSuccess M::KGEN::checkVersion(const MojoPrecompiledFileVersion &base,
                                     const MojoPrecompiledFileVersion &other,
                                     llvm::StringRef baseName,
                                     llvm::StringRef otherName) {
  // Note: these comparisons ignore labels
  if (base == other)
    return success();
  StringRef what = base > other ? "newer" : "older";
  return Error(baseName + (baseName.empty() ? "" : " ") + base.toString() +
               " is " + what + " than " + otherName +
               (otherName.empty() ? "" : " ") + other.toString());
}

bool M::KGEN::isCompatiblePrecompiledFile(
    const MojoPrecompiledFileHeader &header) {
  return !checkCompatiblePrecompiledFile(header).isError();
}

template <typename T>
static void writeInt(llvm::raw_ostream &os, uint64_t x) {
  SmallVector<char> buffer(sizeof(T));
  llvm::support::endian::write<T>(buffer.data(), static_cast<T>(x),
                                  llvm::endianness::little);
  os << buffer;
}

static void writeVersion(MojoPrecompiledFileVersion version,
                         llvm::raw_ostream &os) {
  writeInt<uint8_t>(os, version.major);
  writeInt<uint8_t>(os, version.minor);
  writeInt<uint16_t>(os, version.patch);
  // Write out the label, null terminated.
  os << version.label << '\0';
}

LogicalResult M::KGEN::writePrecompiledFile(Operation *op,
                                            MojoPrecompiledFileVersion &mojoVer,
                                            MojoPrecompiledFileVersion &maxVer,
                                            StringRef mlirChecksum,
                                            llvm::raw_ostream &os) {
  // Serialize the MLIR bytecode to a temporary buffer first so we can compress
  // it before writing.
  std::string mlirBuf;
  llvm::raw_string_ostream mlirStream(mlirBuf);
  if (failed(mlir::writeBytecodeToFile(op, mlirStream)))
    return failure();

  // Compress the MLIR bytecode with zstd.
  SmallVector<uint8_t> compressed;
  llvm::compression::zstd::compress(
      asBytes(mlirBuf), compressed,
      llvm::compression::zstd::BestSizeCompression);

  [[maybe_unused]] auto streamPos = os.tell();
  os << "MPKG";
  writeInt<uint8_t>(os,
                    static_cast<uint8_t>(MojoPrecompiledFileFormatVersion::V2));
  os << "..."; // plus 3 reserved bytes.

  writeVersion(mojoVer, os);

  writeVersion(maxVer, os);

  // Write the nul-terminated MLIR checksum
  os << mlirChecksum << '\0';

  // Write the uncompressed size of the MLIR section (v2).
  writeInt<uint64_t>(os, mlirBuf.size());

  // Align the header size to 8 bytes.
  auto bytesWritten = os.tell() - streamPos;
  auto paddingBytes = llvm::alignTo(bytesWritten, 8) - bytesWritten;
  while (paddingBytes--)
    writeInt<uint8_t>(os, 0);

  // Write the compressed MLIR data.
  os.write(reinterpret_cast<const char *>(compressed.data()),
           compressed.size());
  return success();
}

LogicalResult M::KGEN::writePrecompiledFile(Operation *op,
                                            llvm::raw_ostream &os) {
  MojoPrecompiledFileVersion mojoVersion = M::getMojoVersion();
  MojoPrecompiledFileVersion maxVersion = M::getMAXVersion();
  // The MLIR checksum
  StringRef mlirChecksum = M::getMojoMlirDialectChecksum();
  return writePrecompiledFile(op, mojoVersion, maxVersion, mlirChecksum, os);
}

bool M::KGEN::isMojoPrecompiledFile(llvm::MemoryBufferRef buffer) {
  return buffer.getBuffer().starts_with("MPKG");
}

template <typename T>
ErrorOr<std::pair<T, llvm::StringRef>> readInt(llvm::StringRef buffer) {
  if (buffer.size() < sizeof(T))
    return Error("read past end of buffer");
  return std::make_pair<T, llvm::StringRef>(
      llvm::support::endian::read<T, llvm::support::unaligned>(
          buffer.data(), llvm::endianness::little),
      buffer.drop_front(sizeof(T)));
}

static ErrorOr<std::pair<MojoPrecompiledFileVersion, llvm::StringRef>>
readVersion(llvm::StringRef buffer) {
  MojoPrecompiledFileVersion version;

  // Major
  if (auto err = readInt<uint8_t>(buffer))
    return err.takeError();
  else
    std::tie(version.major, buffer) = *err;
  // Minor
  if (auto err = readInt<uint8_t>(buffer))
    return err.takeError();
  else
    std::tie(version.minor, buffer) = *err;
  // Patch
  if (auto err = readInt<uint16_t>(buffer))
    return err.takeError();
  else
    std::tie(version.patch, buffer) = *err;

  // Parse the NUL terminated string matching the version label.
  version.label = buffer.take_until([](char c) { return c == '\0'; });

  // Check we've reached a nul terminator
  auto remaining = buffer.drop_front(version.label.size());
  if (remaining.empty() || remaining.front() != '\0')
    return Error("invalid version encoding");

  return std::make_pair(version, remaining.drop_front());
}

ErrorOr<MojoPrecompiledFileHeader>
M::KGEN::readPrecompiledFileHeader(llvm::MemoryBufferRef buffer) {
  llvm::StringRef bufferStr = buffer.getBuffer();
  if (!isMojoPrecompiledFile(buffer))
    return Error("invalid magic bytes");

  // A precompiled file header must be at least 8 bytes, to begin with. We'll
  // keep checking as we go.
  if (bufferStr.size() < 8)
    return Error("invalid header size");

  MojoPrecompiledFileHeader header;
  // Skip past the 4 magic bytes
  bufferStr = bufferStr.drop_front(4);

  // Read the single-byte encoding version information
  uint8_t rawVersion;
  if (auto err = readInt<uint8_t>(bufferStr))
    return err.takeError();
  else
    std::tie(rawVersion, bufferStr) = *err;
  header.version = static_cast<MojoPrecompiledFileFormatVersion>(rawVersion);

  // Skip past the 3 currently unused bytes.
  bufferStr = bufferStr.drop_front(3);

  auto mojoVersionOrErr = readVersion(bufferStr);
  if (mojoVersionOrErr.isError())
    return mojoVersionOrErr.takeError();
  std::tie(header.mojoVersion, bufferStr) = *mojoVersionOrErr;

  auto modularVersionOrErr = readVersion(bufferStr);
  if (modularVersionOrErr.isError())
    return modularVersionOrErr.takeError();
  std::tie(header.modularVersion, bufferStr) = *modularVersionOrErr;

  header.mlirChecksum = bufferStr.take_until([](char c) { return c == '\0'; });
  bufferStr = bufferStr.drop_front(header.mlirChecksum.size());

  // Skip past the NUL terminator (as readVersion does for version labels).
  if (bufferStr.empty() || bufferStr.front() != '\0')
    return Error("invalid checksum encoding");
  bufferStr = bufferStr.drop_front(1);

  // Version 2 adds zstd compression with an uncompressed size field.
  if (header.version == MojoPrecompiledFileFormatVersion::V2) {
    if (auto sizeOrErr = readInt<uint64_t>(bufferStr))
      return sizeOrErr.takeError();
    else
      std::tie(header.uncompressedSize, bufferStr) = *sizeOrErr;
  }

  header.headerSize =
      llvm::alignTo(buffer.getBufferSize() - bufferStr.size(), 8);
  if (buffer.getBufferSize() < header.headerSize)
    return Error("invalid header size");

  return header;
}

// Return a buffer from a Mojo precompiled file, skipping the header bytes and
// decompressing if necessary.
ErrorOr<std::pair<MojoPrecompiledFileHeader, MojoPrecompiledFileMLIRBuffer>>
M::KGEN::getMLIRBufferAndHeaderFromPrecompiledFile(
    llvm::MemoryBufferRef buffer) {
  auto header = readPrecompiledFileHeader(buffer);
  if (header.isError())
    return Error("invalid Mojo precompiled file '" +
                 buffer.getBufferIdentifier() + "': " + header.getError());

  auto mlirData = buffer.getBuffer().drop_front(header->getSizeInBytes());
  MojoPrecompiledFileMLIRBuffer result;

  if (header->version == MojoPrecompiledFileFormatVersion::V2) {
    // Decompress zstd-compressed MLIR section.
    auto writableBuf = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(
        header->uncompressedSize, buffer.getBufferIdentifier());
    size_t uncompSize = header->uncompressedSize;
    if (llvm::Error err = llvm::compression::zstd::decompress(
            asBytes(mlirData),
            reinterpret_cast<uint8_t *>(writableBuf->getBufferStart()),
            uncompSize)) {
      return Error("failed to decompress Mojo precompiled file '" +
                   buffer.getBufferIdentifier() +
                   "': " + llvm::toString(std::move(err)));
    }
    result.ownedData = std::move(writableBuf);
    result.buffer = *result.ownedData;
  } else {
    // Uncompressed: reference the original buffer directly.
    result.buffer =
        llvm::MemoryBufferRef(mlirData, buffer.getBufferIdentifier());
  }

  return std::make_pair(*header, std::move(result));
}

ErrorOr<MojoPrecompiledFileMLIRBuffer>
M::KGEN::getMLIRBufferFromPrecompiledFile(
    llvm::MemoryBufferRef buffer, bool ignoreIncompatiblePrecompiledFileErrs) {
  auto mlirBufferAndHeaderOrErr =
      getMLIRBufferAndHeaderFromPrecompiledFile(buffer);
  if (mlirBufferAndHeaderOrErr.isError())
    return mlirBufferAndHeaderOrErr.takeError();
  auto &[header, mlirResult] = *mlirBufferAndHeaderOrErr;
  if (!llvm::sys::Process::GetEnv("MOJO_NO_VALIDATE_MOJOPKG") &&
      !ignoreIncompatiblePrecompiledFileErrs) {
    if (auto err = KGEN::checkCompatiblePrecompiledFile(
            header, buffer.getBufferIdentifier())) {
      return Error(
          std::string(err.getError()) +
          ". To proceed, recreate the precompiled file with `mojo precompile`, "
          "or use the version of the compiler it was created with.");
    }
  }
  return std::move(mlirResult);
}

void MojoPrecompiledFileHeader::dump() const {
  llvm::dbgs() << "Encoding ver " << static_cast<int>(version) << "\n";
  llvm::dbgs() << "Mojo Version: " << mojoVersion.major << "."
               << mojoVersion.minor << "." << mojoVersion.patch
               << mojoVersion.label << "\n";
  llvm::dbgs() << "Modular Version: " << modularVersion.major << "."
               << modularVersion.minor << "." << modularVersion.patch
               << modularVersion.label << "\n";
  llvm::dbgs() << "MLIR Checksum: "
               << (mlirChecksum.empty() ? "<none>" : mlirChecksum) << "\n";
  if (version == MojoPrecompiledFileFormatVersion::V2)
    llvm::dbgs() << "Compression: zstd (uncompressed size: " << uncompressedSize
                 << ")\n";
}
