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

#ifndef KGEN_EXECUTIONENGINE_JIT_MATERIALIZATIONLAYER_H
#define KGEN_EXECUTIONENGINE_JIT_MATERIALIZATIONLAYER_H

#include "Support/Buffer.h"
#include "Support/Compiler/Sanitizers.h"
#include "Support/ErrorOr.h"
#include "Support/FunctionExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/TypeName.h"

namespace M::KGEN {
//===----------------------------------------------------------------------===//
// MaterializationLayer
//===----------------------------------------------------------------------===//

/// Provides a base class we can use to store pointers to Layers in the
/// ExecutionEngine.
///
/// Layers must implement an `add` function that the ExecutionEngine can use:
///
///  ErrorOrSuccess add(StringRef libName, Args args...);
///
/// The base class doesn't have a virtual method to override largely because
/// each class will have its own requirements for what needs to be passed into
/// `add`.
class MaterializationLayer {
public:
  enum LayerKind {
    kStaticArchiveLayer,
  };
  virtual ~MaterializationLayer() = default;

  /// Nothing in this class hierarchy is copyable.
  MaterializationLayer(const MaterializationLayer &other) = delete;

  /// Check if this layer has an error.
  bool hasError() const { return error.has_value(); }

  LayerKind getKind() const { return kind; }

  /// Take the error from this layer.
  Error takeError() {
    assert(hasError());
    return std::move(*error);
  }

protected:
  using AddToSearchOrderFn =
      llvm::unique_function<ErrorOrSuccess(StringRef, llvm::orc::JITDylib *)>;

  MaterializationLayer(LayerKind kind, llvm::orc::ExecutionSession &sess,
                       const llvm::DataLayout &dl, AddToSearchOrderFn add);

  /// Get or create a dylib with name `libName`. Subclasses should always use
  /// this method rather than manipulating the ExecutionSession directly.
  ErrorOr<llvm::orc::JITDylib *> getOrCreateDylib(StringRef libName);

  /// Mangle and intern `name` in the ExecutionSession.
  llvm::orc::SymbolStringPtr mangleAndIntern(StringRef name);

  /// Layers should override this function if they need to filter the symbols
  /// coming from the current process. The MaterializationLayer automatically
  /// adds visibility to current process symbols when creating a new dylib, so
  /// this allows layers to customize that behavior.
  virtual llvm::unique_function<bool(const llvm::orc::SymbolStringPtr &)>
  getTargetProcessSymbolFilter() {
    return {};
  }

protected:
  llvm::orc::ExecutionSession &session;
  const llvm::DataLayout &dataLayout;
  AddToSearchOrderFn addToSearchOrder;

  /// Stores an optional Error that an individual layer can set to be checked
  /// later. This is necessary because the MaterializationUnit may call into a
  /// function in the layer that has no other way to report that error.
  std::optional<Error> error = std::nullopt;

private:
  LayerKind kind;
};
} // namespace M::KGEN

#endif // KGEN_EXECUTIONENGINE_JIT_MATERIALIZATIONLAYER_H
