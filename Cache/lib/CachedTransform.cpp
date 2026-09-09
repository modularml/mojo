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

#include "Cache/CachedTransform.h"
#include "AsyncRT/CompilerSupport/MLIRLocationDecoder.h"
#include "AsyncRT/ForwardDecls.h"
#include "Support/Buffer.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/HashUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/RCRef.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace M;
using namespace Cache;
using namespace AsyncRT;

//===----------------------------------------------------------------------===//
// Generic Transformations
//===----------------------------------------------------------------------===//

std::string TransformCacheKey::hashKey(TransformCacheKey::KeyTy key) {
  TimeTraceScope scope(
      CacheProfilerEntry::create("TransformCacheKey::hashKey"));

  raw_xxhash128_stream hasher;

  // Take the 128-bit xxhash of the input.
  hasher << key->getBuffer();

  // Write the hash to the result buffer and return it.
  return hasher.hashString();
}

//===----------------------------------------------------------------------===//
// Operation Transformations
//===----------------------------------------------------------------------===//

LogicalResult Cache::writeOperationToCacheKey(Operation *op,
                                              const WriteableBufferRef &key) {
  TimeTraceScope scope(CacheProfilerEntry::create("writeOperationToCacheKey"));

  // Strip source locations before hashing so the key is relocatable: kernel
  // `.mojo` files are parsed with their on-disk absolute paths baked into
  // `FileLineColLoc`, so without this a cache warmed under one install path
  // (e.g. one venv's site-packages) misses under a different install path even
  // for byte-identical IR. Locations carry no semantic meaning for the
  // transform result, so dropping them is safe and matches what the MOGG
  // semantic hash already does (see ModuleHasher::setGeneratorHash). Clone
  // first: the live `op` is the pre-transform IR and still needs its locations
  // for diagnostics.
  OwningOpRef<Operation *> clone(op->clone());
  mlir::AttrTypeReplacer replacer;
  auto unknownLoc = mlir::UnknownLoc::get(op->getContext());
  replacer.addReplacement(
      [unknownLoc](mlir::LocationAttr) -> mlir::LocationAttr {
        return unknownLoc;
      });
  replacer.recursivelyReplaceElementsIn(clone.get(), /*replaceAttrs=*/false,
                                        /*replaceLocs=*/true);

  // Use bytecode when writing cache keys to ensure determinism across different
  // builds.
  return mlir::writeBytecodeToFile(clone.get(), *key);
}

/// Encode the given set of diagnostics in the provided buffer.
template <typename T>
static void encodeDiagnostics(T &&diagnostics, WriteableBufferRef buf) {
  llvm::support::endian::Writer writer(*buf, llvm::endianness::little);

  // Functor used to encode a string.
  auto encodeString = [&](StringRef str) {
    writer.write((uint64_t)str.size());
    *buf << str;
    buf->write((char)0);
  };

  // Write out the diagnostics.
  writer.write((uint64_t)llvm::size(diagnostics));
  for (Diagnostic &diag : diagnostics) {
    buf->write((char)diag.getSeverity());
    encodeString(mlir::debugString(diag.getLocation()));
    encodeString(diag.str());
    encodeDiagnostics(diag.getNotes(), buf.copy());
  }
}

/// Decode a set of diagnostics from the provided data. The provided data
/// pointer is updated to point to the next byte after the diagnostics.
static ErrorOrSuccess decodeDiagnostics(const char *&dataIt,
                                        const char *dataEnd, MLIRContext *ctx,
                                        std::vector<Diagnostic> &diagnostics) {
  // Functor for reading a uint64_t from the cache buffer.
  auto readInt = [&](uint64_t &value) -> ErrorOrSuccess {
    if ((dataIt + sizeof(uint64_t)) > dataEnd)
      return Error("failed to read int from cache buffer");
    value = llvm::support::endian::readNext<uint64_t, llvm::endianness::little,
                                            llvm::support::unaligned>(dataIt);
    return success();
  };

  // Functor for reading a string from the cache buffer.
  auto readString = [&](StringRef &str) -> ErrorOrSuccess {
    uint64_t size = 0;
    if (auto err = readInt(size))
      return err.takeError();
    if ((dataIt + size + 1) > dataEnd)
      return Error("failed to read string from cache buffer");
    str = StringRef(dataIt, size);
    dataIt += (size + 1);
    return success();
  };

  // Write out the number of diagnostics.
  uint64_t numDiagnostics = 0;
  if (auto err = readInt(numDiagnostics))
    return err;
  for (uint64_t i = 0; i < numDiagnostics; ++i) {
    if (dataIt == dataEnd)
      return Error("failed to read diagnostic from cache buffer");
    char severity = *dataIt++;

    // Read in the location.
    StringRef locationStr;
    if (auto err = readString(locationStr))
      return err;
    LocationAttr loc = dyn_cast_if_present<LocationAttr>(
        mlir::parseAttribute(locationStr, ctx, Type(), /*numRead=*/nullptr,
                             /*isKnownNullTerminated=*/true));
    if (!loc)
      return Error("failed to parse location in cached diagnostic");
    mlir::Diagnostic diag(loc, static_cast<mlir::DiagnosticSeverity>(severity));

    // Read in the message.
    StringRef message;
    if (auto err = readString(message))
      return err;
    diag << message;

    // Read in the notes.
    std::vector<Diagnostic> notes;
    if (auto err = decodeDiagnostics(dataIt, dataEnd, ctx, notes))
      return err;
    for (Diagnostic &note : notes)
      diag.attachNote() = std::move(note);

    diagnostics.push_back(std::move(diag));
  }
  return success();
}
static ErrorOrSuccess decodeDiagnostics(StringRef &data, MLIRContext *ctx,
                                        std::vector<Diagnostic> &diagnostics) {
  const char *dataIt = data.data();
  if (auto err = decodeDiagnostics(dataIt, data.end(), ctx, diagnostics))
    return err;

  // Update the data string.
  data = StringRef(dataIt, data.end() - dataIt);
  return success();
}

/// Create a deep copy of the given diagnostic.
static Diagnostic copyDiag(const Diagnostic &diag) {
  Diagnostic newDiag(diag.getLocation(), diag.getSeverity());
  newDiag << diag.str();
  for (auto &note : diag.getNotes())
    newDiag.attachNote() = copyDiag(note);
  return newDiag;
}

/// Run a pass manager's passes as a cached transform.
AnyAsyncValueRef
Cache::cachedTransform(Operation *target, RCRef<TransformCache> transformCache,
                       AnyAsyncValueRef chain, mlir::PassManager &pm,
                       const std::function<void(Operation *)> &moreOnMiss,
                       const std::function<void(Operation *)> &moreOnHit) {
  auto keyBuf = WriteableBuffer::get();
  pm.printAsTextualPipeline(*keyBuf);

  // Callback that runs the pass manager and puts the correct region hash attr
  // on the op.
  auto runTransform =
      [&pm, moreOnMiss](Operation *op, WriteableBufferRef buf,
                        AnyAsyncValueRef chain) -> AsyncValueRef<Chain> {
    TimeTraceScope traceScope(CacheProfilerEntry::create(
        "Cache::cachedTransform(Operation *)::runTransform"));
    // Allocate a space to put the result of the pass manager (the emitted
    // diagnostics). We'll chain off that for the deflation.
    auto pmResult =
        AsyncValueRef<std::vector<Diagnostic>>::allocate(chain.getCPUDevice());
    std::move(chain).andThenSync([op, &pm, moreOnMiss,
                                  pmResult = pmResult.copy()](
                                     AnyAsyncValueRef &&chain) mutable {
      moreOnMiss(op);

      if (chain.isError())
        return std::move(pmResult).setToError(chain.takeDiagnostic());

      // Collect the diagnostics emitted while running the pass manager. These
      // will get cached with the bytecode.
      std::vector<Diagnostic> diagnostics;
      auto handlerFn = [&](const Diagnostic &diag) {
        diagnostics.push_back(copyDiag(diag));

        // Return failure to allow the main handler to process the diagnostic.
        return failure();
      };
      mlir::ScopedDiagnosticHandler diagHandler(op->getContext(), handlerFn);

      if (failed(pm.run(op))) {
        return std::move(pmResult).setToError(getMLIRDiagnostic(
            Error("failed to run the pass manager"), op->getLoc()));
      }

      std::move(pmResult).emplace(std::move(diagnostics));
    });

    auto out = AsyncValueRef<Chain>::allocate(pmResult.getCPUDevice());
    // Just write the bytecode and return.
    std::move(pmResult).andThenSync(
        [op, buf = std::move(buf), out = out.copy()](
            AsyncValueRef<std::vector<Diagnostic>> &&pmResult) mutable {
          if (pmResult.isError())
            return std::move(out).setToError(pmResult.takeDiagnostic());

          // Workaround for https://linear.app/modularml/issue/MOCO-3656
          // The bytecode reader for FusedLoc (no-metadata, code 12) calls
          // FusedLoc::get(ctx, locs) which applies a deduplication
          // optimization that returns a non-FusedLoc when locs.size()==1,
          // causing cast<FusedLoc> to crash.  Prevent writing such degenerate
          // FusedLocs (1 location, null metadata) by collapsing them to their
          // single inner location before serialization.
          // TODO(MOCO-3656): remove once the upstream MLIR fix lands in
          // BuiltinDialectBytecode.td.
          {
            mlir::AttrTypeReplacer replacer;
            replacer.addReplacement(
                [](mlir::FusedLoc loc) -> mlir::LocationAttr {
                  if (loc.getLocations().size() == 1 && !loc.getMetadata())
                    return mlir::cast<mlir::LocationAttr>(
                        loc.getLocations()[0]);
                  return loc;
                });
            replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/false,
                                                  /*replaceLocs=*/true);
          }

          // Write out the bytecode.
          TimeTraceScope traceScope(
              CacheProfilerEntry::create("writeBytecodeToFile"));
          if (failed(mlir::writeBytecodeToFile(op, *buf))) {
            return std::move(out).setToError(getMLIRDiagnostic(
                "failed to write bytecode file", op->getLoc()));
          }

          // Write out the diagnostics.
          uint64_t bufSize = buf->getBuffer().size();
          encodeDiagnostics(*pmResult, buf.copy());

          // Write out the bytecode size as a footer so that we can step
          // over it when reading the diagnostics. We encode this at the end
          // to make sure that the bytecode is still aligned.
          llvm::support::endian::Writer(*buf, llvm::endianness::little)
              .write(bufSize);

          std::move(out).emplace();
        });
    return out;
  };

  // Callback that on a cache hit reads the region hashes out of the cache and
  // places them on the operation.
  auto onCacheHit = [moreOnHit](Operation *op,
                                const BufferRef &buf) -> ErrorOrSuccess {
    moreOnHit(op);
    TimeTraceScope traceScope(CacheProfilerEntry::create(
        "Cache::cachedTransform(Operation *)::onCacheHit"));
    StringRef buffer = buf->getBuffer();
    MLIRContext *ctx = op->getContext();

    // Decode the cached diagnostics and re-emit them.
    auto readDiagnostics = [ctx](StringRef &diagBuffer) -> ErrorOrSuccess {
      std::vector<Diagnostic> diagnostics;
      if (auto err = decodeDiagnostics(diagBuffer, ctx, diagnostics))
        return err;
      for (Diagnostic &diag : diagnostics)
        ctx->getDiagEngine().emit(std::move(diag));
      return success();
    };

    // Read in the cached diagnostics encoded after the bytecode. The footer
    // of the buffer contains the size of the bytecode section.
    uint64_t bytecodeSize =
        llvm::support::endian::read<uint64_t, llvm::support::unaligned>(
            buffer.end() - sizeof(uint64_t), llvm::endianness::little);
    StringRef bytecodeBuffer = buffer.take_front(bytecodeSize);
    buffer = buffer.drop_front(bytecodeSize);

    // Read in the encoded diagnostics.
    if (auto err = readDiagnostics(buffer))
      return err;

    // Parse the bytecode.
    std::unique_ptr<llvm::MemoryBuffer> bytecode =
        llvm::MemoryBuffer::getMemBuffer(bytecodeBuffer, /*BufferName=*/"",
                                         /*RequiresNullTerminator=*/false);
    OwningOpRef<Operation *> cachedOp = readOpFromBytecodeFile(
        *bytecode, mlir::ParserConfig(op->getContext(),
                                      /*verifyAfterParse=*/false));

    // Get the body from the parsed op and onto the op we're using.
    for (auto [cached, opRegion] :
         llvm::zip(cachedOp->getRegions(), op->getRegions()))
      opRegion.takeBody(cached);
    op->setAttrs(cachedOp->getAttrDictionary());

    return success();
  };

  return cachedTransform(target, std::move(transformCache), std::move(chain),
                         std::move(keyBuf), std::move(runTransform),
                         std::move(onCacheHit));
}
