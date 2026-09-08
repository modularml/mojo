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

#include "Mojo/ExecutionEngine/JIT/StaticArchiveLayer.h"

#include "Cache/Support/Keys.h"
#include "Mojo/Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "llvm/ExecutionEngine/Orc/COFFPlatform.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Debugging/DebugInfoSupport.h"
#include "llvm/ExecutionEngine/Orc/ObjectFileInterface.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/Support/Base64.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

using namespace M;
using namespace KGEN;
using namespace Cache;

//===----------------------------------------------------------------------===//
// StaticArchiveObjectMaterializationUnit
//===----------------------------------------------------------------------===//

namespace {
class StaticArchiveObjectMaterializationUnit
    : public llvm::orc::MaterializationUnit {
public:
  StaticArchiveObjectMaterializationUnit(llvm::orc::ObjectLayer &objLayer,
                                         llvm::MemoryBufferRef objectBuffer,
                                         Interface &interface)
      : MaterializationUnit(interface), objectBuffer(objectBuffer),
        genLayer(objLayer) {}

  /// Provide a name for this MU that will show up in ORC debug logs.
  StringRef getName() const override {
    return "KGEN::StaticArchiveObjectMaterializationUnit";
  }

  /// Given a MaterializationResponsibility, push the object file buffer onto
  /// the base layer.
  void materialize(
      std::unique_ptr<llvm::orc::MaterializationResponsibility> mr) override {
    genLayer.emit(std::move(mr),
                  llvm::MemoryBuffer::getMemBuffer(
                      objectBuffer, /*RequiresNullTerminator=*/false));
  }

  /// Notify that the symbol `name` has been overridden.
  void discard(const llvm::orc::JITDylib &jd,
               const llvm::orc::SymbolStringPtr &name) override {}

  llvm::MemoryBufferRef objectBuffer;
  llvm::orc::ObjectLayer &genLayer;
};
} // namespace

//===----------------------------------------------------------------------===//
// StaticArchiveMaterializationLayer
//===----------------------------------------------------------------------===//

StaticArchiveLayer::StaticArchiveLayer(llvm::orc::ObjectLayer &objLayer,
                                       llvm::orc::ExecutionSession &sess,
                                       const llvm::DataLayout &dl,
                                       AddToSearchOrderFn add)
    : MaterializationLayer(LayerKind::kStaticArchiveLayer, sess, dl,
                           std::move(add)),
      objectLayer(objLayer) {}

ErrorOrSuccess StaticArchiveLayer::add(StringRef libName, BufferRef object) {
  auto dylibOr = getOrCreateDylib(libName);
  if (dylibOr.isError())
    return dylibOr.takeError();
  llvm::orc::JITDylib *dylib = *dylibOr;

  // If the archive creation succeeds we store a ref to this buffer so the
  // data won't be deallocated until the JIT is destroyed. This version of
  // MemoryBuffer::getMemBuffer produces a non-owning buffer.
  std::unique_ptr<llvm::MemoryBuffer> objectMemBuf =
      llvm::MemoryBuffer::getMemBuffer(object->getBuffer(),
                                       /*BufferName=*/"",
                                       /*RequiresNullTerminator=*/false);
  // Store a ref to the buffer data.
  objectBuffers.push_back(object.copy());

  // Generate a materialization unit for binary object.
  // TODO: We really shouldn't have to do this, we should be able to use a
  // static library generator instead. This unfortunately doesn't work well with
  // the current generator model in orc, where some platforms (like MSVC) define
  // "terminal" generators as part of platform setup.
  llvm::orc::ResourceTrackerSP resourceTracker =
      dylib->getDefaultResourceTracker();
  llvm::Error err = llvm::Error::success();

  auto objectBuffer = object->getMemBufferRef();
  auto objectInterface = toModularErrorOr(
      llvm::orc::getObjectFileInterface(session, objectBuffer));
  if (objectInterface.isError())
    return objectInterface.takeError();
  if (auto defineErr = toModularErrorOr(dylib->define(
          std::make_unique<StaticArchiveObjectMaterializationUnit>(
              objectLayer, objectBuffer, *objectInterface),
          resourceTracker)))
    return defineErr;
  if (err)
    return toModularError(std::move(err));

  return success();
}
