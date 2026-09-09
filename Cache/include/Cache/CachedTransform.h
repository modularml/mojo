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

#ifndef CACHE_CACHED_TRANSFORM_H
#define CACHE_CACHED_TRANSFORM_H

#include "AsyncRT/CompilerSupport/MLIRLocationDecoder.h"
#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "Cache/BlobCache.h"
#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/RCRef.h"
#include "mlir/IR/OpDefinition.h" // IWYU pragma: keep (incomplete type)
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/ErrorHandling.h"

#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace mlir {
class PassManager;
}

namespace M::Cache {

//===----------------------------------------------------------------------===//
// Generic Transformations
//===----------------------------------------------------------------------===//

/// The Cache dialect provides a method to cache generic transformations. This
/// struct defines the cache key as a BufferRef; the contents of which are
/// hashed; this is to disambiguate it from StringRef keys, which generally
/// have no hashing applied.
struct TransformCacheKey {
  using KeyTy = BufferRef;
  static std::string hashKey(KeyTy key);
};

/// Convenience typedef to reduce typing.
using TransformCache = BlobCache<TransformCacheKey>;

/// The most basic function that performs a transformation, writing the
/// cacheable results to the provided buffer. The transform should chain itself
/// on the provided AsyncValueRef.
///
/// For example:
///   auto runTransform = [](WriteableBufferRef buf, AnyAsyncValueRef chain)
///     -> AsyncValueRef<Chain> {
///    auto xform = doAsyncTransform(op, buf, std::move(chain));
///
///    // Allocate a space to put the result of the transformation. We'll chain
///    // off that.
///    auto result = AsyncValueRef<Chain>::allocate(chain.getCPUDevice());
///    xform.andThenSync([&]() mutable {
///      result.emplace(doSyncTransform(op, buf));
///    });
///    return result;
///  };
using WritingToBufferTransformFn =
    llvm::unique_function<AsyncRT::AnyAsyncValueRef(WriteableBufferRef,
                                                    AsyncRT::AnyAsyncValueRef)>;

// A transform fn which perform some transformation and returns the buffer to be
// cached.
using ReturnBufferTransformFn =
    llvm::unique_function<AsyncRT::AsyncValueRef<BufferRef>(
        AsyncRT::AnyAsyncValueRef)>;

/// This is the function that's called on a cache hit. It provides the
/// buffer that was in the cache for the requested lookup.
using CacheHitFn = llvm::unique_function<AsyncRT::AnyAsyncValueRef(BufferRef)>;

/// Profiler entry for run-time cache transforms.
using RuntimeCacheProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kOther, 1), Trace::kOther>;

namespace Detail {
/// These three detectors check for the ErrorOr-style APIs we care about for the
/// templated version of `cachedTransform` below.
template <typename T>
using HasIsError = decltype(std::declval<T>().isError());
template <typename T>
using HasTakeError = decltype(std::declval<T>().takeError());
template <typename T>
using HasTakeValue = decltype(std::declval<T>().takeValue());

/// Given a CacheHitFn-like callable, get the result type.
template <typename CacheHitFnT>
using ResultT = std::invoke_result_t<CacheHitFnT, BufferRef>;

template <typename CacheHitFnT>
using AsyncValueRefResultT =
    AsyncRT::AsyncValueRef<Detail::ResultT<CacheHitFnT>>;

/// Package up detection of member functions of ErrorOr.
template <typename CacheHitFnT>
constexpr bool is_result_error_or_v =
    llvm::is_detected<Detail::HasIsError,
                      Detail::ResultT<CacheHitFnT>>::value &&
    llvm::is_detected<Detail::HasTakeError,
                      Detail::ResultT<CacheHitFnT>>::value &&
    llvm::is_detected<Detail::HasTakeValue,
                      Detail::ResultT<CacheHitFnT>>::value;

template <typename FnT>
constexpr bool return_buffer_v =
    std::is_constructible_v<ReturnBufferTransformFn, FnT>;

template <typename FnT>
constexpr bool accepts_buffer_v =
    std::is_constructible_v<WritingToBufferTransformFn, FnT>;

} // namespace Detail

/// Run the specified transform, using the associated key for caching. When the
/// transform is run, the result AnyAsyncValueRef is resolved to the result of
/// the transform. If the transform is *not* run, then the result
/// AnyAsyncValueRef simply contains a Chain.
template <typename TransformFnT>
AsyncRT::AnyAsyncValueRef cachedTransform(
    EncodedLocation loc, const RCRef<TransformCache> &transformCache,
    AsyncRT::AnyAsyncValueRef chain, WriteableBufferRef transformKey,
    TransformFnT transformFn, CacheHitFn cacheHitFn,
    bool errorOnCacheInsertFailure = true, std::string *outKeyHash = nullptr) {
  TimeTraceScope traceScope(
      RuntimeCacheProfilerEntry::create("Cache::cachedTransform"));
  BufferRef keyBuffer = std::move(transformKey);

  // Try to find the key in the cache. The cache hit function should chain off
  // that and do the right for the cache state.
  auto foundOr =
      AsyncValueRef<std::optional<BufferRef>>::allocate(chain.getCPUDevice());
  chain.andThenSync([foundOr = foundOr.copy(), keyBuffer = keyBuffer.copy(),
                     transformCache = transformCache.copy(),
                     loc = std::move(loc), outKeyHash]() mutable {
    // Find the thing in the cache with the target op's location. This copy of
    // `keyBuffer` is local, so it's safe to move.
    auto f = transformCache->find(foundOr.getCPUDevice(), std::move(keyBuffer),
                                  std::move(loc), outKeyHash);
    std::move(f).andThenSync(
        [foundOr = foundOr.copy()](
            AsyncValueRef<std::optional<BufferRef>> &&f) mutable {
          if (f.isError())
            return std::move(foundOr).setToError(f.takeDiagnostic());

          std::move(foundOr).emplace(std::move(*f));
        });
  });

  // Allocate space for the output.
  AnyAsyncValueRef out = AnyAsyncValueRef::createIndirect(chain.getCPUDevice());
  std::move(foundOr).andThenSync(
      [out = out.copy(), transformCache = transformCache.copy(),
       transformFn = std::move(transformFn), keyBuffer = std::move(keyBuffer),
       cacheHitFn = std::move(cacheHitFn),
       errorOnCacheInsertFailure = errorOnCacheInsertFailure,
       outKeyHash](AsyncValueRef<std::optional<BufferRef>> &&foundOr) mutable {
        if (foundOr.isError())
          return std::move(out).setToError(
              foundOr.getPointer()->takeDiagnostic());

        if (foundOr->has_value())
          return std::move(out).resolveIndirect(
              cacheHitFn(std::move(**foundOr)));

        // No error but no cache hit.

        // If the caller provided a pointer to a string, clear it.
        if (outKeyHash)
          outKeyHash->clear();

        AnyAsyncValueRef xform;
        WriteableBufferRef writeableTransformResult;
        if constexpr (Detail::accepts_buffer_v<TransformFnT>) {
          // Run the transform. Use a 1 MB in-memory buffer.
          WriteableBufferRef transformResult = WriteableBuffer::get(
              /*size=*/0, /*alignment=*/{}, /*capacity=*/1024 * 1024);
          auto fn = WritingToBufferTransformFn(std::move(transformFn));
          xform = fn(transformResult.copy(), std::move(foundOr));
          writeableTransformResult = transformResult.copy();
        } else if constexpr (Detail::return_buffer_v<TransformFnT>) {
          auto fn = ReturnBufferTransformFn(std::move(transformFn));
          xform = fn(std::move(foundOr));
        } else {
          llvm_unreachable("unknown_fn_type");
        }

        // Insert the transform result into the cache.
        std::move(xform).andThenSync(
            [transformCache = transformCache.copy(), out = out.copy(),
             keyBuffer = std::move(keyBuffer),
             bufferWrittenInTransform = std::move(writeableTransformResult),
             errorOnCacheInsertFailure = errorOnCacheInsertFailure,
             outKeyHash](AnyAsyncValueRef &&xform) mutable {
              if (xform.isError())
                return std::move(out).setToError(xform.takeDiagnostic());

              // Only at this point (so the transform has finished successfully)
              // should we change the transform result ref to be read-only.
              BufferRef bufferToCache;
              if constexpr (Detail::return_buffer_v<TransformFnT>)
                bufferToCache = xform.get<BufferRef>().copy();
              else if constexpr (Detail::accepts_buffer_v<TransformFnT>)
                bufferToCache = std::move(bufferWrittenInTransform);
              else
                llvm_unreachable("unknown_fn_type");

              // Again, this keyBuffer is local, so it's safe to move.
              AsyncValueRef<std::string> hashOr = transformCache->insert(
                  out.getCPUDevice(), std::move(keyBuffer),
                  std::move(bufferToCache));
              std::move(hashOr).andThenSync(
                  [out = out.copy(), xform = xform.copy(),
                   errorOnCacheInsertFailure = errorOnCacheInsertFailure,
                   outKeyHash](AsyncValueRef<std::string> &&hashOr) mutable {
                    if (hashOr.isError() && errorOnCacheInsertFailure)
                      return std::move(out).setToError(hashOr.takeDiagnostic());

                    if (outKeyHash)
                      *outKeyHash = hashOr.get();
                    return std::move(out).resolveIndirect(xform.copy());
                  });
            });
      });

  return out;
}

/// This provides a templated version of `cachedTransform` that provides a sync
/// API for the cache hit function.
template <typename TransformFnT, typename CacheHitFnT>
AsyncRT::AnyAsyncValueRef
cachedTransform(EncodedLocation loc, RCRef<TransformCache> transformCache,
                AsyncRT::AnyAsyncValueRef chain,
                WriteableBufferRef transformKey, TransformFnT transformFn,
                CacheHitFnT cacheHitFn, bool errorOnCacheInsertFailure = true,
                std::string *outKeyHash = nullptr) {
  CacheHitFn onCacheHit;

  // If the cache hit function return something like an ErrorOr<T> propagate
  // failures properly.
  if constexpr (Detail::is_result_error_or_v<CacheHitFnT>) {
    onCacheHit = [chain = chain.copy(), loc = loc.copy(),
                  cacheHitFn = std::move(cacheHitFn)](BufferRef buf) mutable {
      auto resultOr = cacheHitFn(std::move(buf));
      if (resultOr.isError())
        return Detail::AsyncValueRefResultT<CacheHitFnT>::createError(
            chain.getCPUDevice(),
            EncodedDiagnostic(resultOr.takeError(), std::move(loc)));

      return Detail::AsyncValueRefResultT<CacheHitFnT>::createReady(
          chain.getCPUDevice(), resultOr.takeValue());
    };
  } else {
    onCacheHit = [chain = chain.copy(),
                  cacheHitFn = std::move(cacheHitFn)](BufferRef buf) {
      auto result = Detail::AsyncValueRefResultT<CacheHitFnT>::allocate(
          chain.getCPUDevice());
      result.copy().emplace(cacheHitFn(std::move(buf)));
      return result;
    };
  }
  return cachedTransform(std::move(loc), std::move(transformCache),
                         std::move(chain), std::move(transformKey),
                         std::move(transformFn), std::move(onCacheHit),
                         errorOnCacheInsertFailure, outKeyHash);
}

/// Profiler entry for compile-time cache transforms.
using CacheProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kCompiler, 2), Trace::kCompiler>;

//===----------------------------------------------------------------------===//
// Operation Transformations
//===----------------------------------------------------------------------===//

/// Transformation and cache functions that operate on a given operation.
using OpTransformFn = llvm::unique_function<AsyncRT::AnyAsyncValueRef(
    Operation *, WriteableBufferRef, AsyncRT::AnyAsyncValueRef)>;
using OpCacheHitFn =
    llvm::unique_function<AsyncRT::AnyAsyncValueRef(Operation *, BufferRef)>;

/// Helper method to write the given operation to the provided cache key.
LogicalResult writeOperationToCacheKey(Operation *op,
                                       const WriteableBufferRef &key);

/// Run the specified transform on the target operation. The transform must have
/// a key of some kind that can be associated with the operation. The semantics
/// of `cachedTransform` are that it will combine the input IR with the name of
/// the transform to map to a cached result.
///
/// When the transform is run, the result AnyAsyncValueRef is resolved to the
/// result of the transform. If the transform is *not* run, then the result
/// AnyAsyncValueRef simply contains a Chain.
template <typename TransformationFnT, typename CacheHitFnT>
AsyncRT::AnyAsyncValueRef
cachedTransform(Operation *target, RCRef<TransformCache> transformCache,
                AsyncRT::AnyAsyncValueRef chain,
                WriteableBufferRef transformKey,
                TransformationFnT &&transformFn, CacheHitFnT &&cacheHitFn,
                std::string *outKeyHash = nullptr) {
  if (failed(writeOperationToCacheKey(target, transformKey.copy()))) {
    chain.copy().setToError(AsyncRT::getMLIRDiagnostic(
        "failed to write bytecode file", target->getLoc()));
    return chain;
  }

  return cachedTransform(
      AsyncRT::MLIRLocationDecoder::getEncodedLocation(target->getLoc()),
      std::move(transformCache), std::move(chain), std::move(transformKey),
      [target, transformFn = std::forward<TransformationFnT>(transformFn)](
          WriteableBufferRef buf, AsyncRT::AnyAsyncValueRef chain) mutable {
        return transformFn(target, std::move(buf), std::move(chain));
      },
      [target, cacheHitFn = std::forward<CacheHitFnT>(cacheHitFn)](
          BufferRef buf) { return cacheHitFn(target, std::move(buf)); },
      /*errorOnCacheInsertFailure=*/true, outKeyHash);
}

/// Run the specified passes over the target operation (i.e. ModulePasses over a
/// ModuleOp). If the target operation and pass pipeline result in a cache hit,
/// that cache hit will simply replace the operation's region hash attribute
/// with the updated region hash attribute. The granularity of the result is a
/// region on the operation `target`. This function manifests its result as an
/// update to the RegionHashArrayAttr on `target` - it will update the region
/// hashes from the old versions (pre-transform) to the new versions (transform
/// applied).
/// moreOnMiss and moreOnHit are two callbacks for extra functionality to
/// perform for cache miss and hit accordingly if needed.
AsyncRT::AnyAsyncValueRef cachedTransform(
    Operation *target, RCRef<TransformCache> transformCache,
    AsyncRT::AnyAsyncValueRef chain, mlir::PassManager &pm,
    const std::function<void(Operation *)> &moreOnMiss = [](Operation *) {},
    const std::function<void(Operation *)> &moreOnHit = [](Operation *) {});

} // namespace M::Cache

#endif // CACHE_CACHED_TRANSFORM_H
