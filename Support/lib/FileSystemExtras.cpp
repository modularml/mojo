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

#include "Support/FileSystemExtras.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/AdvisoryLock.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

using namespace M;

/// The maximum number of retries we will do when trying to take a lock file
/// on an error.
static constexpr int kLockMaxRetriesOnError = 1'000;

static std::string errorCodeToString(const llvm::Error &err) {
  SmallVector<std::string, 2> errors;
  llvm::visitErrors(err, [&errors](const llvm::ErrorInfoBase &ei) {
    std::error_code ec = ei.convertToErrorCode();
    errors.push_back("ErrorCode value: " + std::to_string(ec.value()) +
                     " category: " + ec.category().name() +
                     " message: " + ec.message());
  });

  return llvm::join(errors.begin(), errors.end(), "\n");
}

/// Do a file operation under an LLVM file lock - readFileUnderLock and
/// writeFileUnderLock do almost exactly the same thing, so this keeps the
/// common code common.
template <typename T>
static ErrorOr<T>
doLockedFileOperation(const std::filesystem::path &filePath,
                      llvm::function_ref<ErrorOr<T>()> callable) {
  int retries = 0;
  std::string filePathStr = filePath.string();

  // Lock or wait for the file to be able to operate on it.
  while (true) {
    llvm::LockFileManager lockManager(filePathStr);
    bool owned;
    if (llvm::Error err = lockManager.tryLock().moveInto(owned)) {
      // We don't need to do anything with the error here, since we will just
      // retry, but we also don't want to retry indefinitely so we have a limit.
      if (retries++ > kLockMaxRetriesOnError) {
        std::error_code parentExistsEC;
        bool parentExists =
            std::filesystem::exists(filePath.parent_path(), parentExistsEC);
        std::string ecMsg = errorCodeToString(err);
        std::string parentMsg =
            "Parent path: " + std::string(filePath.parent_path().c_str()) +
            " exists: " + std::to_string(parentExists);
        if (parentExistsEC) {
          parentMsg +=
              " ErrorCode value: " + std::to_string(parentExistsEC.value()) +
              " category: " + parentExistsEC.category().name() +
              " message: " + parentExistsEC.message();
        }

        return Error("unable to take lock file for '" + filePathStr + "': " +
                     toString(std::move(err)) + " " + ecMsg + parentMsg);
      }
      consumeError(std::move(err));
      continue;
    } else if (!owned) {
      // Wait for the other process to finish touching the file.
      switch (lockManager.waitForUnlockFor(std::chrono::seconds(90))) {
      case llvm::WaitForUnlockResult::Success:
        // We now have the lock file, and can proceed to operate on the file if
        // the other process didn't do it.
        break;
      case llvm::WaitForUnlockResult::OwnerDied:
        // The owner died, try again to take the file.
        continue;
      case llvm::WaitForUnlockResult::Timeout:
        // We timed out when trying to acquire the lock for the file.
        // TODO: We could try again, but the default timeout is 1.5 minutes.
        return Error("timed out waiting for lock file for '" + filePathStr +
                     "'");
      }
    }

    return callable();
  }
}

ErrorOr<std::filesystem::path> M::writeFileUnderLock(
    const std::filesystem::path &filePath,
    llvm::function_ref<void(llvm::raw_ostream &)> writeContent) {
  std::string filePathStr = filePath.string();

  // A helper function to write the content into the file.
  auto writeFile = [&]() -> ErrorOr<std::filesystem::path> {
    llvm::Error err = llvm::writeToOutput(
        filePathStr, [&](llvm::raw_ostream &os) -> llvm::Error {
          if (filePathStr == "/dev/null" || filePathStr == "-") {
            // For /dev/null we get a raw_null_ostream and for "-" and instance
            // of llvm::outs(). Both of them doesn't have any error checking
            // associated with it.
            writeContent(os);
            os.flush();
          } else {
            // The raw_ostream created by writeToOutput for a filePath will be a
            // raw_fd_ostream.
            auto *fdStream = static_cast<llvm::raw_fd_ostream *>(&os);
            writeContent(*fdStream);
            fdStream->flush();
            if (fdStream->has_error()) {
              llvm::Error err =
                  llvm::make_error<llvm::StringError>(fdStream->error());
              return err;
            }
          }

          return llvm::Error::success();
        });
    if (err)
      return toModularError(std::move(err));
    return filePath;
  };

  return doLockedFileOperation<std::filesystem::path>(filePath, writeFile);
}

ErrorOrSuccess M::readFileUnderLock(
    const std::filesystem::path &filePath,
    llvm::function_ref<void(const std::filesystem::path &)> read) {
  ErrorOr<Detail::Empty> err =
      doLockedFileOperation<Detail::Empty>(filePath, [&]() {
        read(filePath);
        return Detail::Empty();
      });
  // Apparently we can't convert from ErrorOr<Detail::Empty> to ErrorOrSuccess,
  // so we do it manually here.
  if (err)
    return err.takeError();
  return success();
}

ErrorOrSuccess
M::appendFileUnderLock(const std::filesystem::path &filePath,
                       llvm::function_ref<void(raw_ostream &)> appendContent) {
  std::string filePathStr = filePath.string();

  // A helper function to append to a file.
  auto appendFile = [&]() -> ErrorOr<Detail::Empty> {
    std::error_code ec;
    // We want to open an existing file, or create one if it doesn't already
    // exist. We only want to write to it, and we want to append to the end.
    llvm::raw_fd_ostream stream(filePathStr, ec, llvm::sys::fs::CD_OpenAlways,
                                llvm::sys::fs::FA_Write,
                                llvm::sys::fs::OF_Append);
    if (ec)
      return Error(ec.message());
    appendContent(stream);
    if (stream.has_error())
      return Error(stream.error().message());
    return Detail::Empty();
  };

  ErrorOr<Detail::Empty> err =
      doLockedFileOperation<Detail::Empty>(filePath, appendFile);
  if (err)
    return err.takeError();
  return success();
}

// llvm::sys::Process has a function called `llvm::sys::Process::FindInEnvPath`
// which looks for files (and files only) in PATH like environment variables.
// The version here is inspired by the original and has a similar contract but
// looks only for directories instead.
std::optional<std::string>
M::findDirInEnvPath(StringRef subdirName, StringRef envName, char separator) {
  assert(!llvm::sys::path::is_absolute(subdirName));
  std::optional<std::string> optPath = llvm::sys::Process::GetEnv(envName);
  if (!optPath)
    return {};

  const char envPathSeparatorStr[] = {separator, '\0'};
  SmallVector<StringRef, 8> dirs;
  StringRef(*optPath).split(dirs, envPathSeparatorStr);

  for (StringRef dir : dirs) {
    if (dir.empty())
      continue;

    SmallString<128> dirPath(dir);
    llvm::sys::path::append(dirPath, subdirName);
    if (llvm::sys::fs::exists(Twine(dirPath)) &&
        llvm::sys::fs::is_directory(Twine(dirPath))) {
      return std::string(dirPath);
    }
  }

  return std::nullopt;
}

ErrorOr<TempFile> M::writeTempFile(const Twine &model,
                                   function_ref<void(raw_ostream &)> writeFn) {
  std::error_code ec;
  std::filesystem::path path = std::filesystem::temp_directory_path(ec);
  if (ec)
    return Error(ec.message());
  path = path / model.str();

  auto tmpFileOr = TempFile::create(path.string());
  if (tmpFileOr.isError())
    return tmpFileOr.takeError();

  // Write the runtime to the temp file.
  llvm::raw_fd_ostream tmpOS(tmpFileOr->getFD(), /*shouldClose=*/false);
  writeFn(tmpOS);
  tmpOS.flush();
  tmpFileOr->close();

  return std::move(*tmpFileOr);
}

ErrorOr<TempFile> M::writeTempFile(const Twine &model, StringRef buffer) {
  return writeTempFile(model, [&](raw_ostream &os) { os << buffer; });
}

ErrorOr<TempFile> TempFile::create(StringRef model) {
  std::filesystem::path path = model.str();

  // If the path is relative, create it in the system temp directory.
  std::error_code ec;
  if (path.is_relative()) {
    path = std::filesystem::temp_directory_path(ec);
    if (ec)
      return Error(ec.message());
    path /= model.str();
  }

  // Create the parent directories if they don't exist.
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec)
    return Error(ec.message());

  int fd;
  SmallString<0> outFilePathVec;
  std::error_code err =
      llvm::sys::fs::createUniqueFile(path.string(), fd, outFilePathVec);
  if (err)
    return Error(err.message());

  return TempFile{fd, outFilePathVec.str().str()};
}

TempFile::TempFile(TempFile &&other)
    : fd(other.fd), path(std::move(other.path)), keepFile(other.keepFile) {
  other.fd = -1;
}

TempFile::~TempFile() {
  close();
  remove();
}

void TempFile::remove() {
  if (!keepFile) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
}

void TempFile::close() {
  if (fd != -1) {
    llvm::sys::fs::file_t nativeID = llvm::sys::fs::convertFDToNativeFile(fd);
    llvm::sys::fs::closeFile(nativeID);
    fd = -1;
  }
}

ErrorOr<size_t> TempFile::getSize() {
  std::error_code ec;
  uintmax_t size = std::filesystem::file_size(path, ec);
  if (size == (uintmax_t)-1)
    return Error(ec.message());

  return size;
}

ErrorOr<TempDir> TempDir::create(StringRef model) {
  std::filesystem::path dir;
  {
    // Create an delete a temporary file.
    auto file = TempFile::create(model);
    if (file.isError())
      return file.takeError();
    dir = file->getPath();
  }
  // Create the directory.
  std::error_code ec;
  std::filesystem::create_directory(dir, ec);
  if (ec)
    return Error(ec.message());
  return TempDir(dir);
}

TempDir::TempDir(TempDir &&other) : path(other.path) {
  other.keep(); // Don't remove on destruction.
}

TempDir::~TempDir() { remove(); }

void TempDir::remove() {
  if (!keepFile) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
}

/// Open the filename with a given alignment (if specified) as the argument and
/// return a memory buffer, or an error message on failure.
ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
M::openInputFile(StringRef inputFilename, std::optional<llvm::Align> align) {
  std::string errorMsg;
  auto result = align ? mlir::openInputFile(inputFilename, *align, &errorMsg)
                      : mlir::openInputFile(inputFilename, &errorMsg);
  if (result)
    return result;
  return Error(errorMsg);
}
