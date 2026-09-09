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

#include "Support/Context.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/CompilerSupport/LLVMThreadPool.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Support/MDialect/MDialect.h"

using namespace M;

//===---------------------------------------------------------------------===//
// MContextExtension
//===---------------------------------------------------------------------===//

namespace {

/// Dialect extension to inject an RCRef<M::Context> into MDialect.
class MContextExtension : public mlir::DialectExtensionBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MContextExtension)

  /// Apply this extension once MDialect is loaded.
  explicit MContextExtension(ContextRef ref, bool enableThreadPool)
      : DialectExtensionBase(MDialect::getDialectNamespace()),
        ctxRef(std::move(ref)), enableThreadPool(enableThreadPool) {}

  /// Apply this extension to the given context and the required dialects.
  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    auto *dialect = cast<MDialect>(dialects.front());

    dialect->setInternal(ctxRef.copy());
    if (context->isMultithreadingEnabled() || !enableThreadPool)
      return;
    AsyncRT::LLVMThreadPool *tp = ctxRef->get<AsyncRT::LLVMThreadPool>();
    if (!tp) {
      if (AsyncRT::CPUDevice *cpuDevice = ctxRef->get<AsyncRT::CPUDevice>())
        tp = &ctxRef->emplace<AsyncRT::LLVMThreadPool>(*cpuDevice);
    }

    // If the cpuDevice is available, enable threading in MLIR with it.
    if (tp)
      context->setThreadPool(*tp);
  }

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<MContextExtension>(ctxRef.copy(), enableThreadPool);
  }

private:
  ContextRef ctxRef;
  bool enableThreadPool;
};

} // namespace

void M::registerContext(mlir::DialectRegistry &registry, ContextRef &ref,
                        bool enableThreadPool) {
  std::unique_ptr<mlir::DialectExtensionBase> ctxExtension =
      std::make_unique<MContextExtension>(ref.copy(), enableThreadPool);
  registry.addExtension(mlir::TypeID::get<MContextExtension>(),
                        std::move(ctxExtension));
}

void M::registerContext(mlir::MLIRContext &ctx, ContextRef &ref,
                        bool enableThreadPool) {
  DialectRegistry registry;
  registerContext(registry, ref, enableThreadPool);
  ctx.appendDialectRegistry(registry);
}

ContextRef M::loadContext(mlir::MLIRContext *ctx) {
  StringRef name = MDialect::getDialectNamespace();
  return static_cast<MDialect *>(ctx->getOrLoadDialect(name))->getInternal();
}
