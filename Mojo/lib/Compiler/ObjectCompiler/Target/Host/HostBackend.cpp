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

#include "HostBackend.h"

#include "Mojo/Compiler/SaveAsmOutput.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Target/Host/HostTraits.h"
#include "Target/TargetTraits.h"

#include "mlir/IR/Location.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

const TargetTraits *HostBackend::traits() const { return &HostTraits::get(); }

ErrorOr<BufferRef> HostBackend::emitAssembly(llvm::Module &module,
                                             EmitContext &ctx) const {
  WriteableBufferRef buf = WriteableBuffer::get();
  if (ErrorOrSuccess error =
          ctx.runLlc(module, *buf, /*createObjectFile=*/false)) {
    return Error(Twine(error.getError()) +
                 ", llc failed to codegen LLVM IR to object code");
  }

  if (!ctx.options.saveTempsPrefix.empty()) {
    ErrorOr<const TargetTraits *> traitsOr = TargetTraitsRegistry::get().lookup(
        llvm::Triple(ctx.options.targetTriple));
    if (traitsOr.isError())
      return Error(traitsOr.getError());
    const TargetTraits *traits = *traitsOr;
    StringRef toEmit(buf->getBufferStart(), buf->getBufferSize());
    if (mlir::failed(writeBytesToTempWithHash(ctx.options.saveTempsPrefix,
                                              traits->getAsmExtension().str(),
                                              toEmit)))
      return Error("failed to save asm to saveTempsPrefix");
  }
  return buf;
}

ErrorOr<BufferRef> HostBackend::emitObject(llvm::Module &module,
                                           EmitContext &ctx) const {
  WriteableBufferRef codeBuf = WriteableBuffer::get();
  if (ErrorOrSuccess error =
          ctx.runLlc(module, *codeBuf, /*createObjectFile=*/true)) {
    return Error(Twine(error.getError()) +
                 ", llc failed to codegen LLVM IR to object code");
  }

  StringRef name = "mojo-object";
  if (auto fileLoc = ctx.loc->findInstanceOf<mlir::FileLineColLoc>())
    name = llvm::sys::path::filename(fileLoc.getFilename());
  std::string moduleName = (name + llvm::Twine(ctx.moduleIdx)).str();

  return ctx.linkObject(BufferRef::take(codeBuf.release()), moduleName);
}

ErrorOr<BufferRef>
HostBackend::createArchive(llvm::MutableArrayRef<BufferRef> objects,
                           llvm::StringRef moduleName, EmitContext &ctx) const {
  // The host archive flow still runs through ObjectCompiler::emitArchive /
  // MCLinker directly and is not routed through the backend yet.
  return Error("HostBackend::createArchive is not wired");
}

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetBackend<HostBackend> registerHostBackend;
#pragma GCC diagnostic pop
} // namespace

} // namespace M::KGEN
