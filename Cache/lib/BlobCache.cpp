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

#include "Cache/BlobCache.h"
#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "Config/Version.h"
#include "Support/Base64.h"
#include "Support/Buffer.h"
#include "Support/CacheLog.h"
#include "Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "Support/Filesystem/DiskUsage.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/RCRef.h"
#include "Support/URI.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/xxhash.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

using namespace M;
using namespace Cache;
using namespace AsyncRT;

/// Provides a simple way to get an error given an optional encoded location and
/// a standard Error.
static EncodedDiagnostic getError(std::optional<EncodedLocation> loc,
                                  Error err) {
  if (loc)
    return {std::move(err), std::move(*loc)};

  return UnknownLocationDecoder::getDiagnostic(std::move(err));
}

/// Returns whether the given path is a directory that the current process can
/// write to. If the path does not exist, this attempts to create a writable
/// directory at that path.
static bool checkOrCreateWriteableDirectory(const std::filesystem::path &path) {
  [[maybe_unused]] std::error_code existsErr;
  if (std::filesystem::exists(path, existsErr)) {
    // If the path exists but is not a directory, return false.
    if (!std::filesystem::is_directory(path, existsErr)) {
      MODULAR_CACHE_LOG("fs")
          << path.string() << " exists but is not a directory\n";
      return false;
    }
    // Otherwise, check the write access permissions for the existing directory.
    bool writable =
        !llvm::sys::fs::access(path.string(), llvm::sys::fs::AccessMode::Write);
    MODULAR_CACHE_LOG("fs")
        << path.string()
        << " exists, writable=" << (writable ? "true" : "false") << "\n";
    return writable;
  }

  // If the path doesn't exist, create it. If creation was successful, we must
  // have write access.
  std::error_code createErr;
  std::filesystem::create_directories(path, createErr);
  if (createErr) {
    MODULAR_CACHE_LOG("fs")
        << path.string()
        << " does not exist, creation failed: " << createErr.message() << "\n";
  } else {
    MODULAR_CACHE_LOG("fs")
        << path.string() << " does not exist, created successfully\n";
  }
  return !createErr;
}

//===----------------------------------------------------------------------===//
// BlobCacheBackend
//===----------------------------------------------------------------------===//

AsyncValueRef<Chain>
BlobCacheBackend::insert(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash,
                         BufferRef obj, std::optional<EncodedLocation> loc) {
  EncodedLocation location = loc.has_value()
                                 ? std::move(*loc)
                                 : UnknownLocationDecoder::getEncodedLocation();

  auto chain =
      insertImpl(cpuDevice, keyHash.copy(), obj.copy(), location.copy());
  if (!delegate)
    return chain;

  // Arrange to wait and then insert into the delegate.
  auto result = AsyncValueRef<Chain>::allocate(cpuDevice);
  std::move(chain).andThenSync(
      [result = result.copy(), thisRef = copyRCRef(this),
       keyHash = std::move(keyHash), obj = std::move(obj),
       loc = std::move(location)](AsyncValueRef<Chain> &&chain) mutable {
        if (chain.isError())
          return std::move(result).setToError(chain.takeDiagnostic());
        auto insert =
            thisRef->delegate->insert(result.getCPUDevice(), std::move(keyHash),
                                      std::move(obj), std::move(loc));
        std::move(insert).andThenSync(
            [result =
                 std::move(result)](AsyncValueRef<Chain> &&insert) mutable {
              if (insert.isError())
                return std::move(result).setToError(insert.takeDiagnostic());
              return std::move(result).emplace();
            });
      });
  return result;
}

ErrorOrSuccess BlobCacheBackend::insertSync(StringRef keyHash, BufferRef obj) {
  auto err = insertSyncImpl(keyHash, obj.copy());
  if (err.isError())
    return err.takeError();
  if (!delegate)
    return success();

  // Insert synchronously into the delegate as well.
  return delegate->insertSync(keyHash, std::move(obj));
}

AsyncValueRef<Chain>
BlobCacheBackend::insertImpl(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash,
                             BufferRef obj,
                             std::optional<EncodedLocation> loc) {
  // Wrap the synchronous implementation by default.
  auto result = AsyncValueRef<Chain>::allocate(cpuDevice);
  addTask(cpuDevice, [thisRef = copyRCRef(this), keyHash = std::move(keyHash),
                      obj = std::move(obj), result = result.copy(),
                      loc = std::move(loc)]() mutable {
    if (auto err = thisRef->insertSync(keyHash->getBuffer(), std::move(obj))) {
      return std::move(result).setToError(
          getError(std::move(loc), err.takeError()));
    }
    std::move(result).emplace();
  });
  return result;
}

AsyncValueRef<bool>
BlobCacheBackend::contains(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash,
                           std::optional<EncodedLocation> loc) {
  EncodedLocation location = loc.has_value()
                                 ? std::move(*loc)
                                 : UnknownLocationDecoder::getEncodedLocation();

  auto chain = containsImpl(cpuDevice, keyHash.copy(), location.copy());
  if (!delegate)
    return chain;

  // Check the delegate, if this fails.
  auto result = AsyncValueRef<bool>::allocate(cpuDevice);
  std::move(chain).andThenSync(
      [result = result.copy(), thisRef = copyRCRef(this),
       keyHash = std::move(keyHash),
       loc = std::move(location)](AsyncValueRef<bool> &&chain) mutable {
        if (chain.isError())
          return std::move(result).setToError(chain.takeDiagnostic());
        if (*chain)
          return std::move(result).emplace(true); // Value is locally available.
        // Need to schedule a delegate contains call.
        auto contains = thisRef->delegate->contains(
            result.getCPUDevice(), std::move(keyHash), std::move(loc));
        std::move(contains).andThenSync(
            [result =
                 std::move(result)](AsyncValueRef<bool> &&contains) mutable {
              if (contains.isError())
                return std::move(result).setToError(contains.takeDiagnostic());
              return std::move(result).emplace(*contains);
            });
      });
  return result;
}

ErrorOr<bool> BlobCacheBackend::containsSync(StringRef keyHash) {
  auto errOr = containsSyncImpl(keyHash);
  if (errOr.isError())
    return errOr.takeError();
  if (*errOr)
    return true;
  if (!delegate)
    return false;

  // Check the delegate synchronously.
  return delegate->containsSync(keyHash);
}

AsyncValueRef<bool>
BlobCacheBackend::containsImpl(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash,
                               std::optional<EncodedLocation> loc) {
  // Wrap the synchronous implementation by default.
  auto result = AsyncValueRef<bool>::allocate(cpuDevice);
  addTask(cpuDevice, [thisRef = copyRCRef(this), keyHash = std::move(keyHash),
                      result = result.copy(), loc = std::move(loc)]() mutable {
    auto errOr = thisRef->containsSync(keyHash->getBuffer());
    if (errOr.isError()) {
      return std::move(result).setToError(
          getError(std::move(loc), errOr.takeError()));
    }
    std::move(result).emplace(*errOr);
  });
  return result;
}

AsyncValueRef<std::optional<BufferRef>>
BlobCacheBackend::find(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash,
                       std::optional<EncodedLocation> loc) {
  EncodedLocation location = loc.has_value()
                                 ? std::move(*loc)
                                 : UnknownLocationDecoder::getEncodedLocation();

  auto chain = findImpl(cpuDevice, keyHash.copy(), location.copy());
  if (!delegate)
    return chain;

  // Check the delegate, if this fails.
  auto result = AsyncValueRef<std::optional<BufferRef>>::allocate(cpuDevice);
  std::move(chain).andThenSync(
      [result = result.copy(), thisRef = copyRCRef(this),
       keyHash = std::move(keyHash), loc = std::move(location)](
          AsyncValueRef<std::optional<BufferRef>> &&chain) mutable {
        if (chain.isError())
          return std::move(result).setToError(chain.takeDiagnostic());
        if (chain->has_value())
          return std::move(result).emplace(
              std::move(**chain)); // Found locally.
        // Need to attempt to find within the delegate.
        auto found = thisRef->delegate->find(result.getCPUDevice(),
                                             keyHash.copy(), loc.copy());
        std::move(found).andThenSync(
            [thisRef = thisRef.copy(), result = std::move(result),
             keyHash = std::move(keyHash), loc = std::move(loc)](
                AsyncValueRef<std::optional<BufferRef>> &&found) mutable {
              if (found.isError())
                return std::move(result).setToError(found.takeDiagnostic());
              if (!found->has_value())
                return std::move(result).emplace(std::nullopt); // Not found.
              // We need to insert locally.
              auto inserted =
                  thisRef->insert(result.getCPUDevice(), std::move(keyHash),
                                  (*found)->copy(), std::move(loc));
              std::move(inserted).andThenSync(
                  [result = std::move(result), obj = std::move(**found)](
                      AsyncValueRef<Chain> &&inserted) mutable {
                    if (inserted.isError())
                      return std::move(result).setToError(
                          inserted.takeDiagnostic());
                    return std::move(result).emplace(
                        std::move(obj)); // Finally, put the buffer.
                  });
            });
      });
  return result;
}

ErrorOr<std::optional<BufferRef>>
BlobCacheBackend::findSync(StringRef keyHash) {
  auto errOr = findSyncImpl(keyHash);
  if (errOr.isError())
    return errOr.takeError();
  if (errOr->has_value())
    return std::move(**errOr);
  if (!delegate)
    return std::nullopt;

  // Check the delegate synchronously.
  auto delegateErrOr = delegate->findSync(keyHash);
  if (delegateErrOr.isError())
    return delegateErrOr.takeError();
  if (!delegateErrOr->has_value())
    return std::nullopt;
  BufferRef buf = std::move(**delegateErrOr);

  // Insert the value locally.
  auto insertOr = insertSync(keyHash, buf.copy());
  if (insertOr.isError())
    return insertOr.takeError();

  // Return the loaded buffer.
  return std::move(buf);
}

AsyncValueRef<std::optional<BufferRef>>
BlobCacheBackend::findImpl(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash,
                           std::optional<EncodedLocation> loc) {
  // Wrap the synchronous execution by default.
  auto result = AsyncValueRef<std::optional<BufferRef>>::allocate(cpuDevice);
  addTask(cpuDevice, [thisRef = copyRCRef(this), keyHash = std::move(keyHash),
                      result = result.copy(), loc = std::move(loc)]() mutable {
    auto errOr = thisRef->findSync(keyHash->getBuffer());
    if (errOr.isError())
      return std::move(result).setToError(
          getError(std::move(loc), errOr.takeError()));
    if (!errOr->has_value())
      return std::move(result).emplace(std::nullopt);
    BufferRef buf = std::move(**errOr);
    std::move(result).emplace(std::move(buf));
  });
  return result;
}

void BlobCacheBackend::appendDelegate(RCRef<BlobCacheBackend> d) {
  if (!delegate)
    delegate = std::move(d);
  else
    delegate->appendDelegate(std::move(d));
}

//===----------------------------------------------------------------------===//
// InMemoryBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides an in-memory backend that stores memory buffers in an
/// llvm::StringMap.
struct InMemoryBackend : public BlobCacheBackend {
  ErrorOrSuccess insertSyncImpl(StringRef keyHash, BufferRef obj) override {
    std::lock_guard<std::shared_mutex> lock(mutex);
    cache[keyHash] = std::move(obj);
    return success();
  }

  ErrorOr<bool> containsSyncImpl(StringRef keyHash) override {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return cache.count(keyHash);
  }

  ErrorOr<std::optional<BufferRef>> findSyncImpl(StringRef keyHash) override {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto found = cache.find(keyHash);
    if (found == cache.end())
      return std::nullopt;
    return found->second.copy();
  }

  llvm::StringMap<BufferRef> cache;
  mutable std::shared_mutex mutex;
};
} // namespace

RCRef<BlobCacheBackend> M::Cache::getInMemoryBackend() {
  return RCRef<InMemoryBackend>::create();
}

//===----------------------------------------------------------------------===//
// FilesystemBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides a filesystem-backed backend that primarily stores the buffers in
/// binary files on disk. If read-only, no writes are performed, only reads.
struct FilesystemBackend : public BlobCacheBackend {
  explicit FilesystemBackend(const std::filesystem::path &basePath,
                             bool readOnly)
      : basePath(basePath.string()), readOnly(readOnly) {}

  ErrorOrSuccess insertSyncImpl(StringRef keyHash, BufferRef obj) override {
    // Check if we already have the object in the filesystem cache - if we do,
    // then don't bother writing it again.
    ErrorOr<bool> containsOr = containsSync(keyHash);
    if (!containsOr.isError() && *containsOr) {
      MODULAR_CACHE_LOG("fs") << "insertSyncImpl: already cached, skipping: "
                              << encodeURLSafeBase64(keyHash) << "\n";
      return success();
    }

    // Otherwise, if the filesystem is read-only, we cannot write to it for
    // insertion.
    if (readOnly) {
      MODULAR_CACHE_LOG("fs") << "insertSyncImpl: read-only, skipping write\n";
      return success();
    }

    // Get the absolute path and create any directories we need to create.
    ErrorOr<std::filesystem::path> filePathOr = getAbsolutePathForKey(keyHash);
    if (filePathOr.isError())
      return filePathOr.takeError();

    std::error_code dirErr;
    std::filesystem::create_directories(filePathOr->parent_path(), dirErr);
    if (dirErr)
      return Error(dirErr.message());

    auto availableSizeOr = M::getAvailableDiskSpace(filePathOr->parent_path());
    if (availableSizeOr.isError())
      return availableSizeOr.takeError();

    MODULAR_CACHE_LOG("fs")
        << "insertSyncImpl: disk available=" << *availableSizeOr
        << " needed=" << obj->getBufferSize() << "\n";
    if (*availableSizeOr < obj->getBufferSize())
      return Error("cannot write to file to filesystem cache since available "
                   "space(" +
                   std::to_string(*availableSizeOr) +
                   ") is not enough to write " +
                   std::to_string(obj->getBufferSize()) + " bytes");

    // Functor used when we actually need to write out the file.
    auto writeContent = [&](raw_ostream &os) {
      // Copy the data into the file buffer.
      os.write(obj->getBufferStart(), obj->getBufferSize());

      // Compute and copy the hash as well.
      llvm::XXH128_hash_t hash =
          llvm::xxh3_128bits(llvm::arrayRefFromStringRef(obj->getBuffer()));
      os.write(llvm::bit_cast<char *>(&hash), sizeof(llvm::XXH128_hash_t));
    };

    // Safely process creating the file, taking into account that we may
    // have different processes trying to produce this file in parallel.
    if (auto err = writeFileUnderLock(*filePathOr, writeContent);
        err.isError()) {
      MODULAR_CACHE_LOG("fs")
          << "insertSyncImpl: write failed: " << filePathOr->string() << "\n";
      return err.takeError();
    }

    MODULAR_CACHE_LOG("fs")
        << "insertSyncImpl: write success: " << filePathOr->string() << "\n";
    return success();
  }

  ErrorOr<bool> containsSyncImpl(StringRef keyHash) override {
    std::error_code ec;
    ErrorOr<std::filesystem::path> absOr = getAbsolutePathForKey(keyHash);
    if (absOr.isError())
      return absOr.takeError();

    bool exists = std::filesystem::exists(*absOr, ec);
    if (ec)
      return Error(ec.message());

    if (!exists)
      return false;

    bool isDir = std::filesystem::is_directory(*absOr, ec);
    if (ec)
      return Error(ec.message());
    return !isDir;
  }

  ErrorOr<std::optional<BufferRef>> findSyncImpl(StringRef keyHash) override {
    // Get the file path and open it.
    ErrorOr<std::filesystem::path> filePath = getAbsolutePathForKey(keyHash);

    // No such file, return nullopt (not error).
    std::error_code ec;
    if (!std::filesystem::exists(*filePath, ec) || ec) {
      MODULAR_CACHE_LOG("fs")
          << "findSyncImpl: miss, file not found: " << filePath->string()
          << "\n";
      return std::nullopt;
    }

    // If the cache file for this key exists, then it will never be written
    // again. We can safely read it without a lock.
    auto bufOr = Buffer::getFile(*filePath);
    // If the file doesn't exist, or it's empty, return an error.
    if (bufOr.isError())
      return bufOr.takeError();

    BufferRef buffer = std::move(*bufOr);
    if (buffer->getBufferSize() == 0) {
      MODULAR_CACHE_LOG("fs")
          << "findSyncImpl: CORRUPTED (empty file): " << filePath->string()
          << "\n";
      return Error("file '" + Twine(filePath->string()) +
                   "' exists, but is empty");
    }

    StringRef contentsAndHash = buffer->getBuffer();

    // Get a StringRef of the contents without the hash.
    StringRef contents = contentsAndHash.drop_back(sizeof(llvm::XXH128_hash_t));
    llvm::XXH128_hash_t computedHash =
        xxh3_128bits(llvm::arrayRefFromStringRef(contents));
    StringRef storedHash =
        contentsAndHash.take_back(sizeof(llvm::XXH128_hash_t));

    // Check the computed hash against the hash in the file.
    if (memcmp(llvm::bit_cast<char *>(&computedHash), storedHash.data(),
               sizeof(llvm::XXH128_hash_t)) != 0) {
      MODULAR_CACHE_LOG("fs")
          << "findSyncImpl: CORRUPTED (hash mismatch): " << filePath->string()
          << "\n";
      return Error("corrupted file: stored hash and computed hash did not "
                   "match for file '" +
                   Twine(filePath->string()) + "'");
    }

    // Now that we've verified the integrity of the file, return a memory buffer
    // that holds just the contents.
    bufOr = Buffer::getFile(*filePath, contents.size(), /*offset=*/0);
    if (failed(bufOr))
      return bufOr.takeError();
    MODULAR_CACHE_LOG("fs") << "findSyncImpl: cache hit: " << contents.size()
                            << " bytes from " << filePath->string() << "\n";
    return BufferRef::take(bufOr->release());
  }

  ErrorOr<std::filesystem::path>
  getAbsolutePathForKey(StringRef keyHash) const {
    std::error_code ec;
    std::filesystem::path filepath(basePath);
    std::string encodedHash = encodeURLSafeBase64(keyHash);
    filepath /= encodedHash;

    std::filesystem::path absolute = std::filesystem::absolute(filepath, ec);
    if (ec)
      return Error(ec.message());

    return absolute;
  }

  /// The base path for the filesystem cache.
  std::string basePath;
  /// Whether the filesystem cache is read-only. If `true`, reads are performed
  /// as normal, whereas writes are silently ignored.
  bool readOnly;
};
} // namespace

RCRef<BlobCacheBackend>
M::Cache::getFilesystemBackend(const std::filesystem::path &basePath,
                               bool readOnly) {
  return RCRef<FilesystemBackend>::create(basePath, readOnly);
}

/// Returns a filesystem-based implementation of the BlobCacheBackend. The
/// `cacheDir` is used to derive a path for use by the filesystem backend. The
/// `version` specifies the version string of the cache, defaults to
/// MODULAR_VERSION_STRING if the provided version is empty.
static ErrorOr<RCRef<FilesystemBackend>>
getVersionedFilesystemBackend(const std::filesystem::path &cacheDir,
                              std::string_view version) {
  // If no version is specified, use the default version.
  if (version.empty())
    version = getModularVersionString();

  MODULAR_CACHE_LOG("fs") << "version=" << std::string_view(version) << "\n";

  std::error_code ec;
  std::filesystem::path base = cacheDir;
  if (!base.is_absolute()) {
    auto config = Config::open();
    if (config.isError())
      return config.takeError();
    auto cacheFolderPath = config->getModularCacheFolderPath();
    if (cacheFolderPath.isError())
      return Error(cacheFolderPath.getError());
    base = std::filesystem::absolute(*cacheFolderPath) / cacheDir;
    MODULAR_CACHE_LOG("fs")
        << "resolved relative path to: " << base.string() << "\n";
  }

  assert(base.is_absolute() && "must default to non-empty absolute path");
  bool readOnly = !checkOrCreateWriteableDirectory(base);
  MODULAR_CACHE_LOG("fs") << "basePath=" << base.string()
                          << " readOnly=" << (readOnly ? "true" : "false")
                          << "\n";

  // Always warn when a cache directory is not writable, regardless of
  // MODULAR_ENABLE_CACHE_LOGGING. Silent read-only degradation makes
  // performance problems very hard to diagnose.
  if (readOnly) {
    llvm::errs()
        << "Warning: cache directory '" << base.string()
        << "' is not writable; compilation caching will be disabled for "
           "this path. Set MODULAR_CACHE_DIR to a writable directory, or "
           "set MODULAR_ENABLE_CACHE_LOGGING=1 for full diagnostics.\n";
  }

  // If we have write access, do a little cache pruning on the host system in
  // order to keep disk usage down: iterate the base path and remove
  // directories that match the current suffix (after '-').
  // Keeping dirs that match the suffix and not prefix allows us to
  // keep cached parallel debug and release versions.
  if (!readOnly) {
    // File lock the directory to avoid multiple processes remove the
    // content in the directory at the same time.
    llvm::LockFileManager locker(base.string());
    if (locker.tryLock()) {
      // Extract version suffix (everything after the first '-')
      std::string_view suffix;
      size_t idx = version.find('-');
      if (idx != std::string_view::npos)
        suffix = version.substr(idx + 1, std::string_view::npos);

      for (const auto &dirEntry : std::filesystem::directory_iterator{base}) {
        // The directory entry must exist, be a directory, the parent must be
        // `base` and the directory 'filename' suffix must match the current
        // version suffix in order for it to be deleted.

        [[maybe_unused]] std::error_code ec0, ec1;
        if (std::filesystem::is_directory(dirEntry.path(), ec) &&
            (std::filesystem::canonical(dirEntry.path().parent_path(), ec0) ==
             std::filesystem::canonical(base, ec1))) {
          // Extract suffix from directory name
          std::string dirName = dirEntry.path().filename().string();
          // Remove if dirName is not the same as current version (outdated
          // cache) and has the same build type as the current one.
          if (dirName != version && dirName.ends_with(suffix)) {
            MODULAR_CACHE_LOG("fs")
                << "pruning old cache dir: " << dirEntry.path().string()
                << "\n";
            std::filesystem::remove_all(dirEntry, ec);
          }
        }
      }
    }
  }

  base = base / version;
  MODULAR_CACHE_LOG("fs") << "final cache path=" << base.string() << "\n";
  return RCRef<FilesystemBackend>::create(base, readOnly);
}

ErrorOr<RCRef<BlobCacheBackend>>
M::Cache::getFilesystemBackend(const std::filesystem::path &cacheDir,
                               std::string_view version) {
  return getVersionedFilesystemBackend(cacheDir, version);
}

ErrorOr<RCRef<BlobCacheBackend>>
M::Cache::getLocalDefaultBackendChain(const std::filesystem::path &cacheDir,
                                      std::string_view version) {
  auto filesystemOr = getFilesystemBackend(cacheDir, version);
  if (filesystemOr.isError())
    return filesystemOr.takeError();
  auto memory = getInMemoryBackend();
  memory->appendDelegate(std::move(*filesystemOr));
  MODULAR_CACHE_LOG("fs") << "backend chain created (memory + filesystem)\n";
  return std::move(memory);
}

ErrorOr<RCRef<BlobCacheBackend>>
M::Cache::getDefaultBackendChain(const URI &uri, std::string_view version) {
  StringRef scheme = uri.getScheme();
  if (scheme == "file") {
    std::string path(uri.getPath());
    return getLocalDefaultBackendChain(path, version);
  }

  return Error("Can't build BlobCache backend chain with unknown URI scheme: " +
               scheme);
}
