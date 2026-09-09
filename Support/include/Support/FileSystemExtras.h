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

#ifndef SUPPORT_FILESYSTEM_EXTRAS_H
#define SUPPORT_FILESYSTEM_EXTRAS_H

#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Alignment.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace llvm {
class MemoryBuffer;
} // namespace llvm

namespace M {

/// This function searches for an existing subdirectory in the list of
/// directories in a PATH like environment variable, and returns the first
/// subdirectory found according to the order of the entries in the PATH like
/// environment variable. The defaults hold for Unix-like systems but break for
/// Windows.
std::optional<std::string> findDirInEnvPath(llvm::StringRef subdirName,
                                            llvm::StringRef envName = "PATH",
                                            char separator = ':');

/// Write to a file (creating if necessary) serialized with any other
/// ...UnderLock operation, even in parallel across processes.  Writing will
/// also appear atomic to readers not aware of LLVM lock files.
ErrorOr<std::filesystem::path>
writeFileUnderLock(const std::filesystem::path &filePath,
                   llvm::function_ref<void(raw_ostream &)> writeContent);

/// Read a file exclusively, serializing with other ...UnderLock operations,
/// even in parallel across processes.  Other processes using readFileUnderLock
/// will wait for the operation to complete before initiating their reads.  The
/// operation is only atomic with respect to processes abiding by the LLVM lock
/// file convention -- no atomicity guarantees are provided with respect to
/// writers not aware of the LLVM lock file convention concurrently operating
/// on the file.
ErrorOrSuccess
readFileUnderLock(const std::filesystem::path &filePath,
                  llvm::function_ref<void(const std::filesystem::path &)> read);

/// Append to a file exclusively, serializing with other ...UnderLock
/// operations, even in parallel across processes.  Other processes appending
/// will block while the append is in progress.  If the process crashes in the
/// middle of appending, other processes may witness a partially-appended
/// state.  Processes not aware of the LLVM lock file convention may also
/// witness partially-appended states while the append is in progress.
ErrorOrSuccess
appendFileUnderLock(const std::filesystem::path &filePath,
                    llvm::function_ref<void(raw_ostream &)> appendContent);

/// This class provides a tempfile implementation. The llvm::sys version has
/// some really odd behavior that is tricky to manage, so we provide our own
/// implementation.
class TempFile {
public:
  /// Create a TempFile and return any errors during creation. The model is
  /// something like `myString-%%%%%.ext` - the `%` characters are filled in
  /// with random numbers/letters. Non absolute paths are created in the system
  /// temp directory.
  static ErrorOr<TempFile> create(StringRef model);
  /// TempFiles are move-able.
  TempFile(TempFile &&other);
  /// Destroy the temp file, and remove it from the filesystem if `keepFile` is
  /// not specified.
  ~TempFile();

  /// Keep the tempfile after the destructor runs - useful for debugging.
  void keep() { keepFile = true; }

  /// remove removes the file. The file will still be cleaned up (or at
  /// least attempted to remove) when the class is destroyed.
  void remove();

  /// close closes the file. The path will still be available.
  void close();

  /// Get the file descriptor as an integer. This file is open as of the
  /// completion of the `create` call. Note that if the file has been closed,
  /// an fd < 0 may be returned, which the caller should check for.
  int getFD() { return fd; }

  /// Return the path to the temp file. This path is absolute.
  const std::filesystem::path &getPath() const { return path; }

  /// Get the size of the temp file in bytes.
  ErrorOr<size_t> getSize();

private:
  TempFile(int fd, std::string path) : fd(fd), path(std::move(path)) {}
  /// These are not copy-able.
  TempFile(const TempFile &other) = delete;

  int fd = -1;
  std::filesystem::path path;
  bool keepFile = false;
};

/// This class provides a temporary directory, in the same fashion as TempFile.
class TempDir {
public:
  /// See TempFile::create.
  static ErrorOr<TempDir> create(StringRef model);
  TempDir(TempDir &&other);
  ~TempDir();

  /// Keep the directory after the destructor.
  void keep() { keepFile = true; }

  /// Remove the directory.
  void remove();

  /// Return the path to the directory.
  const std::filesystem::path &getPath() const { return path; }

private:
  TempDir(std::filesystem::path path) : path(std::move(path)) {}
  TempDir(const TempDir &other) = delete;

  std::filesystem::path path;
  bool keepFile = false;
};

/// Invokes the provided callback, writing the output to a temporary file whose
/// name is based on the provided model.
ErrorOr<TempFile> writeTempFile(const Twine &model,
                                function_ref<void(raw_ostream &)> writeFn);
ErrorOr<TempFile> writeTempFile(const Twine &model, StringRef buffer);

/// Open the filename specified as the argument and return a memory buffer, or
/// an error message on failure.
M::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
openInputFile(StringRef inputFilename,
              std::optional<llvm::Align> align = std::nullopt);

} // namespace M
#endif // SUPPORT_FILESYSTEM_EXTRAS_H
