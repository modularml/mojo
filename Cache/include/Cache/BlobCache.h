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

#ifndef CACHE_BLOBCACHE_H
#define CACHE_BLOBCACHE_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Support/Chain.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "Support/URI.h"
#include "llvm/ADT/StringRef.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace M::Cache {
/// This class is the backend interface for a BlobCache. The backend contains a
/// pointer to its delegate, which is meant to be used as an option if this
/// backend has a cache miss. This means that the backends should be ordered on
/// priority - i.e. have an in-memory backend delegate to a remote backend, not
/// the other way around!
///
/// Conceptually, the backends form a linked-list that's sorted in priority
/// order that the BlobCache below will use to find an item.
class BlobCacheBackend : public ReferenceCounted<BlobCacheBackend> {
public:
  virtual ~BlobCacheBackend() {}

  /// Store the object `obj` with hash `keyHash`. This is expected to take
  /// ownership of the data in `obj` on success. Subclasses are expected to
  /// overwrite the current contents on a collision.
  AsyncRT::AsyncValueRef<AsyncRT::Chain>
  insert(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash, BufferRef obj,
         std::optional<EncodedLocation> loc = std::nullopt);

  /// May be overwritten to provide an asynchronous insert.
  virtual AsyncRT::AsyncValueRef<AsyncRT::Chain>
  insertImpl(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash, BufferRef obj,
             std::optional<EncodedLocation> loc = std::nullopt);

  /// Check if an item with key hash `keyHash` exists in this backend or in any
  /// of the delegates.
  AsyncRT::AsyncValueRef<bool>
  contains(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash,
           std::optional<EncodedLocation> loc = std::nullopt);

  /// May be overwritten to provide an asynchronous contains.
  virtual AsyncRT::AsyncValueRef<bool>
  containsImpl(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash,
               std::optional<EncodedLocation> loc = std::nullopt);

  /// Get the item with key hash `keyHash` from this backend or any of its
  /// delegates.
  AsyncRT::AsyncValueRef<std::optional<BufferRef>>
  find(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash,
       std::optional<EncodedLocation> loc = std::nullopt);

  /// May be overwritten to provide an asynchronous find.
  virtual AsyncRT::AsyncValueRef<std::optional<BufferRef>>
  findImpl(AsyncRT::CPUDevice &cpuDevice, BufferRef keyHash,
           std::optional<EncodedLocation> loc = std::nullopt);

  /// Subclasses that don't override insert should use this to provide the
  /// implementation of actually storing an item.
  ErrorOrSuccess insertSync(StringRef keyHash, BufferRef obj);

  /// Must be overwritten to provide synchronous insert.
  virtual ErrorOrSuccess insertSyncImpl(StringRef keyHash, BufferRef obj) = 0;

  /// Subclasses that don't override contains should use this to provide the
  /// implementation of checking if an item exists.
  ErrorOr<bool> containsSync(StringRef keyHash);

  /// Must be overwritten to provide synchronous contains.
  virtual ErrorOr<bool> containsSyncImpl(StringRef keyHash) = 0;

  /// Subclasses that don't override find should use this to provide the
  /// implementation of getting an item from storage.
  ErrorOr<std::optional<BufferRef>> findSync(StringRef keyHash);

  /// Must be overwritten to provide a synchronous find.
  virtual ErrorOr<std::optional<BufferRef>> findSyncImpl(StringRef keyHash) = 0;

  /// Add delegate to the end of the backend chain.
  virtual void appendDelegate(RCRef<BlobCacheBackend> d);

private:
  /// The next backend in the list. The public APIs handle nullptr here
  /// correctly, and the protected APIs (for the subclasses) should ignore the
  /// presence of this delegate entirely.
  RCRef<BlobCacheBackend> delegate;
};

/// This is the thing that users will interact with. It holds onto the list of
/// backends and calls into them, but its primary responsibility is to hash the
/// keys passed in to normalize the way we try to access the storage backends.
/// This cache supports any key type that can be hashed as long as the hash
/// method is provided through the `KeyInfo` type.
///
/// \tparam KeyInfo A struct that describes how to handle a key. It should be of
/// the form:
///
///   struct KeyInfo {
///     using KeyTy = SomeT;
///     static std::string hashKey(KeyTy key);
///   }
///
template <typename KeyInfo>
class BlobCache : public ReferenceCounted<BlobCache<KeyInfo>> {
public:
  explicit BlobCache(RCRef<BlobCacheBackend> backendList)
      : backendList(std::move(backendList)) {}

  using KeyTy = typename KeyInfo::KeyTy;

  /// Simple method to get the hash of a key via the KeyInfo struct. This is
  /// useful if (for example) we already have the object in the cache.
  std::string getHash(KeyTy key) const {
    return KeyInfo::hashKey(std::forward<KeyTy>(key));
  }

  /// Store an item in the provided backends. On a collision, the backends are
  /// expected to overwrite the existing contents, so it is incumbent on the
  /// user to use a strong hash function! Returns the cache key on success -
  /// this can be used for speeding up future hash computations or simply
  /// discarded.
  AsyncRT::AsyncValueRef<Chain>
  insertKeyed(AsyncRT::CPUDevice &cpuDevice, llvm::StringRef key, BufferRef obj,
              std::optional<EncodedLocation> loc = std::nullopt) {
    return backendList->insert(cpuDevice, Buffer::get(key), std::move(obj),
                               std::move(loc));
  }
  ErrorOrSuccess insertKeyedSync(llvm::StringRef key, BufferRef obj) {
    return backendList->insertSync(key, std::move(obj));
  }

  AsyncRT::AsyncValueRef<std::string>
  insert(AsyncRT::CPUDevice &cpuDevice, KeyTy key, BufferRef obj,
         std::optional<EncodedLocation> loc = std::nullopt) {
    std::string keyHash = KeyInfo::hashKey(std::forward<KeyTy>(key));
    AsyncRT::AsyncValueRef<AsyncRT::Chain> insertAsync =
        insertKeyed(cpuDevice, keyHash, std::move(obj), std::move(loc));

    // Allocate a space for the output.
    auto out = AsyncRT::AsyncValueRef<std::string>::allocate(cpuDevice);
    std::move(insertAsync)
        .andThenSync([keyHash = std::move(keyHash), out = out.copy()](
                         AsyncValueRef<AsyncRT::Chain> &&insertAsync) mutable {
          // If insertion failed, propagate the error. Otherwise, hand over the
          // key hash.
          if (insertAsync.isError())
            return std::move(out).setToError(insertAsync.takeDiagnostic());

          return std::move(out).emplace(keyHash);
        });

    return out;
  }
  ErrorOr<std::string> insertSync(KeyTy key, BufferRef obj) {
    std::string keyHash = KeyInfo::hashKey(std::forward<KeyTy>(key));
    auto errOr = insertKeyedSync(keyHash, std::move(obj));
    if (errOr.isError())
      return errOr.takeError();
    return keyHash;
  }

  /// Check if any of the provided backends have the item. If `outKeyHash` is
  /// provided, it will be set to the key hash, regardless of whether the item
  /// exists or not.
  AsyncRT::AsyncValueRef<bool>
  containsKeyed(AsyncRT::CPUDevice &cpuDevice, llvm::StringRef key,
                std::optional<EncodedLocation> loc = std::nullopt) const {
    return backendList->contains(cpuDevice, Buffer::get(key), std::move(loc));
  }
  ErrorOr<bool> containsKeyedSync(llvm::StringRef key) const {
    return backendList->containsSync(key);
  }
  AsyncRT::AsyncValueRef<bool>
  contains(AsyncRT::CPUDevice &cpuDevice, KeyTy key,
           std::optional<EncodedLocation> loc = std::nullopt,
           std::string *outKeyHash = nullptr) const {
    std::string keyHash = KeyInfo::hashKey(std::forward<KeyTy>(key));
    if (outKeyHash)
      *outKeyHash = keyHash;
    return containsKeyed(cpuDevice, keyHash, std::move(loc));
  }
  ErrorOr<bool> containsSync(KeyTy key,
                             std::string *outKeyHash = nullptr) const {
    std::string hash = KeyInfo::hashKey(std::forward<KeyTy>(key));
    if (outKeyHash)
      *outKeyHash = hash;
    return containsKeyedSync(hash);
  }

  /// Get the item from any of the provided backends. If `outKeyHash` is
  /// provided, it will be set to the key hash, regardless of whether the item
  /// exists or not.
  AsyncRT::AsyncValueRef<std::optional<BufferRef>>
  findKeyed(AsyncRT::CPUDevice &cpuDevice, llvm::StringRef key,
            std::optional<EncodedLocation> loc = std::nullopt) const {
    return backendList->find(cpuDevice, Buffer::get(key), std::move(loc));
  }
  ErrorOr<std::optional<BufferRef>> findKeyedSync(llvm::StringRef key) const {
    return backendList->findSync(key);
  }
  AsyncRT::AsyncValueRef<std::optional<BufferRef>>
  find(AsyncRT::CPUDevice &cpuDevice, KeyTy key,
       std::optional<EncodedLocation> loc = std::nullopt,
       std::string *outKeyHash = nullptr) const {
    std::string hash = KeyInfo::hashKey(std::forward<KeyTy>(key));
    if (outKeyHash)
      *outKeyHash = hash;
    return findKeyed(cpuDevice, hash, std::move(loc));
  }
  ErrorOr<std::optional<BufferRef>>
  findSync(KeyTy key, std::string *outKeyHash = nullptr) const {
    std::string hash = KeyInfo::hashKey(std::forward<KeyTy>(key));
    if (outKeyHash)
      *outKeyHash = hash;
    return findKeyedSync(hash);
  }

private:
  RCRef<BlobCacheBackend> backendList;
};

/// Returns an in-memory implementation of the BlobCacheBackend.
RCRef<BlobCacheBackend> getInMemoryBackend();

/// Returns a filesystem-based implementation of the BlobCacheBackend. If the
/// base path is not specified, then the backend will use the CWD. The cache
/// reads and writes to the filesystem by default, but if `readOnly` is
/// specified, only reads are performed.
RCRef<BlobCacheBackend>
getFilesystemBackend(const std::filesystem::path &basePath = "",
                     bool readOnly = false);

/// Returns a filesystem-based implementation of the BlobCacheBackend. The
/// `cacheDir` is used to derive a path for use by the filesystem backend. The
/// `version` specifies the version string of the cache, defaults to
/// MODULAR_VERSION_STRING if the provided version is empty.
ErrorOr<RCRef<BlobCacheBackend>>
getFilesystemBackend(const std::filesystem::path &cacheDir,
                     std::string_view version);

/// Returns a chain of pre-setup backends that represent the default chain,
/// inMemory->filesystem. The `cacheDir` is used to derive a path for use
/// by the filesystem backend. The `version` specifies the version string of the
/// cache, defaults to MODULAR_VERSION_STRING if the provided version is empty.
ErrorOr<RCRef<BlobCacheBackend>>
getLocalDefaultBackendChain(const std::filesystem::path &cacheDir = "",
                            std::string_view version = "");

ErrorOr<RCRef<BlobCacheBackend>>
getDefaultBackendChain(const URI &cacheUri, std::string_view version = "");

} // namespace M::Cache

#endif // CACHE_BLOBCACHE_H
