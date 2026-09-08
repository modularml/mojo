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
#ifndef KGEN_EXECUTIONENGINE_JIT_STATICARCHIVELAYER_H
#define KGEN_EXECUTIONENGINE_JIT_STATICARCHIVELAYER_H

#include "MaterializationLayer.h"

namespace M::KGEN {
//===----------------------------------------------------------------------===//
// StaticArchiveLayer
//===----------------------------------------------------------------------===//

/// This layer provides a way to add a static archive to the ExecutionEngine.
/// All symbols in the archive are made available for use and lookup.
class StaticArchiveLayer : public MaterializationLayer {
public:
  /// The StaticArchiveLayer needs a reference to the base object linking layer
  /// so it can feed the archive bytes into the linker.
  StaticArchiveLayer(llvm::orc::ObjectLayer &objLayer,
                     llvm::orc::ExecutionSession &sess,
                     const llvm::DataLayout &dl, AddToSearchOrderFn add);

  /// Add the archive in `archive` to the library `libName`. Stores a reference
  /// to `archive` inside the class to ensure its lifetime matches the lifetime
  /// of the ExecutionEngine.
  ErrorOrSuccess add(StringRef libName, BufferRef object);

  static bool classof(const MaterializationLayer *layer) {
    return layer->getKind() == LayerKind::kStaticArchiveLayer;
  }

private:
  llvm::orc::ObjectLayer &objectLayer;
  SmallVector<BufferRef> objectBuffers;
};
} // namespace M::KGEN

#endif // KGEN_EXECUTIONENGINE_JIT_STATICARCHIVELAYER_H
