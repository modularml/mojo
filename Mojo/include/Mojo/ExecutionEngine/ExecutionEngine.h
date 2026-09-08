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

#ifndef KGEN_COMPILER_EXECUTIONENGINE_H
#define KGEN_COMPILER_EXECUTIONENGINE_H

#include "Support/Buffer.h"
#include "Support/Compiler/Sanitizers.h"
#include "Support/ErrorOr.h"
#include "Support/FunctionExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/DylibManager.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/TypeName.h"

namespace llvm {
class TargetMachine;
} // namespace llvm

namespace M::KGEN {
class MaterializationLayer;
class ObjectCompiler;

//===----------------------------------------------------------------------===//
// ExecutionEngineOptions
//===----------------------------------------------------------------------===//

/// This is a struct of options that the ExecutionEngine wants to have on
/// construction. These are like the KGEN compilation options, but we want to
/// avoid depending on them directly.
struct ExecutionEngineOptions {
  /// Whether or not to register the GDB plugins.
  bool registerDebugPlugins = false;
  bool registerPerfPlugins = false;

  /// Set to true if the executing engine is being used to cross-compile. This
  /// will forgo any JIT setup and capabilities.
  bool crossCompiling = false;

  /// An ORC ExecutorProcessControl that the user can specify.
  std::unique_ptr<llvm::orc::ExecutorProcessControl> epc = nullptr;

  /// Libraries to load.
  SmallVector<StringRef> libraryPaths;
};

//===----------------------------------------------------------------------===//
// CompiledFunc
//===----------------------------------------------------------------------===//

/// This class provides an interface to interact with a compiled func. You
/// can either invoke the func, or get it as an object. The lifetime of one of
/// these objects is tied to the ExecutionEngine through the `cache` member.
/// This could be relaxed by using a pointer instead, but that would require
/// getObject to fail if the cache is unavailable, and there's currently no use
/// case for such a feature so we will leave it to the future.
class CompiledFunc {
public:
  /// Invoke this func. This has exactly the signature the compiled func
  /// does. Intended to have perfect forwarding of arguments into the
  /// function, and of return values from the function.
  ///
  /// Disable ubsan function type check here because the function pointer might
  /// be jitted without ubsan, thus does not contain RTTI for ubsan to look up.
  template <typename ReturnT, typename... Args>
  __attribute__((no_sanitize("function"))) ReturnT invoke(Args... args) {
    // Cast the function pointer and invoke it directly.
    return ((ReturnT (*)(Args...))fn)(std::forward<Args>(args)...);
  }

  /// Return the pointer to the compiled function.
  void *getFunctionPointer() const { return fn; }

private:
  /// Construct a CompiledFunc object. This constructor is private because it
  /// needs a reference to the cache that the ExecutionEngine holds, so it
  /// should really only be constructed from the ExecutionEngine or something
  /// like it.
  CompiledFunc(void *ptr) : fn(ptr) {}
  friend class ExecutionEngine;

  /// Pointer to the function to invoke.
  void *fn;
};

//===----------------------------------------------------------------------===//
// ExecutionEngine
//===----------------------------------------------------------------------===//

/// This class provides an interface to the LLVM ORCJIT. It can compile
/// individual funcs (already lowered to the LLVM dialect) to object code. It
/// caches the objects themselves so we can retrieve them later and write them
/// to a file. The fundamental unit this class deals with is a single llvm
/// function because that's the minimum granularity we would want to use for
/// caching and search.
class ExecutionEngine {
public:
  ~ExecutionEngine();

  //===--------------------------------------------------------------------===//
  // Constructors
  //===--------------------------------------------------------------------===//
  /// Create an ExecutionEngine with StaticArchiveLayer.
  /// This enables the user to load static archives, which is
  /// pretty much the base requirement.
  static ErrorOr<std::unique_ptr<ExecutionEngine>>
  createWithStandardLayers(ExecutionEngineOptions options,
                           const llvm::TargetMachine &tm);

  //===--------------------------------------------------------------------===//
  // Adding/getting layers
  //===--------------------------------------------------------------------===//
  /// Add a layer into the ExecutionEngine. This passes the variables that are
  /// private to the ExecutionEngine into the layer constructor and constructs
  /// the layer in-place.
  template <typename T, typename... Args>
  T &addLayer(Args &&...args) {
    assert(findLayer<T>() == nullptr && "duplicate layer found");
    return cast<T>(*layers.emplace_back(std::make_unique<T>(
        std::forward<Args>(args)..., *executionSession, dataLayout,
        [&](StringRef name, llvm::orc::JITDylib *dylib) {
          return addToSearchOrder(name, dylib);
        })));
  }

  /// Get a layer of type T. Asserts that the layer is found in the
  /// ExecutionEngine and returns a reference to it.
  template <typename T>
  T &getLayer() const {
    auto found =
        llvm::find_if(layers, [](const auto &layer) { return isa<T>(*layer); });
    assert(found != layers.end() && "can't find this layer...");
    return cast<T>(**found);
  }

  /// Constructs and adds an object with libName to the layer of LayerT.
  /// However, if libName already exists then is a no-op. Thread safe.
  template <typename LayerT, typename... Args>
  ErrorOrSuccess addIfAbsent(StringRef libName, Args &&...args) {
    LayerT *found = findLayer<LayerT>();
    if (!found)
      return Error("could not find layer of type " +
                   llvm::getTypeName<LayerT>());

    std::lock_guard<std::mutex> guard(mu);
    if (executionSession->getJITDylibByName(libName))
      return success();

    return found->add(libName, std::forward<Args>(args)...);
  }

  //===--------------------------------------------------------------------===//
  // Adding symbols/objects/etc.
  //===--------------------------------------------------------------------===//
  /// Add *something* to the ExecutionEngine. Uses `LayerT` to find the layer to
  /// add *to*, and then calls the layer's `add` function.
  template <typename LayerT, typename... Args>
  ErrorOrSuccess add(StringRef libName, Args &&...args) {
    LayerT *found = findLayer<LayerT>();
    if (!found)
      return Error("could not find layer of type " +
                   llvm::getTypeName<LayerT>());

    return found->add(libName, std::forward<Args>(args)...);
  }

  llvm::orc::ExecutionSession &getExecutionSession() {
    return *executionSession;
  }

  //===--------------------------------------------------------------------===//
  // Compiled symbol lookup
  //===--------------------------------------------------------------------===//

  /// Look up a func and return it as a CompiledFunc object if we can find it.
  ErrorOr<CompiledFunc> lookup(StringRef symbol);

  /// Look up the provided symbol only in the provided dylib and any others
  /// added to its link order. Note that this bypasses the default search order,
  /// and must therefore must be used with caution.
  ErrorOr<CompiledFunc> lookup(StringRef libName, StringRef symbol);

  //===--------------------------------------------------------------------===//
  // JIT Execution
  //===--------------------------------------------------------------------===//

  /// Run the entry point in the specified library as the main function of a
  /// program. This will invoke the entry point through the ORC RT if available.
  ErrorOrSuccess runProgram(StringRef libName, StringRef entryPoint,
                            function_ref<ErrorOrSuccess(void *)> runFn);

  /// Get the name of the global constructor function to call in JIT mode.
  static constexpr const char *getGlobalCtorFnName() {
    return "KGEN_EE_JIT_GlobalConstructor";
  }
  /// Get the name of the global destructor function to call in JIT mode.
  static constexpr const char *getGlobalDtorFnName() {
    return "KGEN_EE_JIT_GlobalDestructor";
  }

  //===--------------------------------------------------------------------===//
  // Misc
  //===--------------------------------------------------------------------===//

  /// Get the base object linking layer.
  llvm::orc::ObjectLinkingLayer &getLinkingLayer() { return *objectLayer; }
  const llvm::DataLayout &getDataLayout() const { return dataLayout; }

  /// Add a JITDylib to the search order for symbol resolution. Asserts if the
  /// dylib already exists - users should generally be cautious about adding
  /// dylibs to the search order.
  ErrorOrSuccess addToSearchOrder(StringRef name, llvm::orc::JITDylib *dylib);

private:
  //===--------------------------------------------------------------------===//
  // Constructors
  //===--------------------------------------------------------------------===//
  explicit ExecutionEngine(std::unique_ptr<llvm::orc::ExecutionSession> session,
                           const llvm::DataLayout &dl);

  /// This class is not copy-constructible.
  ExecutionEngine(const ExecutionEngine &other) = delete;

  /// Create an ExecutionEngine with no layers. This is generally not very
  /// useful unless the user wants to customize exactly which layers go into the
  /// JIT.
  static ErrorOr<std::unique_ptr<ExecutionEngine>>
  create(ExecutionEngineOptions options, const llvm::TargetMachine &tm);

  /// Mangle and intern a string name.
  llvm::orc::SymbolStringPtr mangleAndIntern(StringRef name);

  /// Look up the provided symbol with the given search order. This is a
  /// generalization of the two lookup methods above, we just don't want to
  /// expose the notion of a 'search order' to users cause it's easy to mis-use.
  ErrorOr<CompiledFunc>
  lookupWithSearchOrder(const llvm::orc::JITDylibSearchOrder &order,
                        StringRef symbol);

  //===--------------------------------------------------------------------===//
  // finding layers
  //===--------------------------------------------------------------------===//
  /// Find a layer of type T. Because we can only have one layer of each kind,
  /// this simply iterates the layer list and returns a stable pointer to the
  /// layer of type T. If that layer cannot be found, returns nullptr.
  template <typename T>
  T *findLayer() const {
    auto found =
        llvm::find_if(layers, [](const auto &layer) { return isa<T>(*layer); });
    if (found == layers.end())
      return nullptr;
    return &cast<T>(**found);
  }

  /// The ORC requires an ExecutionSession - this is how it coordinates
  /// execution across processes/machines.
  std::unique_ptr<llvm::orc::ExecutionSession> executionSession = nullptr;

  /// DylibManager for managing dynamic libraries in the target process.
  std::unique_ptr<llvm::orc::DylibManager> dylibMgr = nullptr;

  /// JITLink linker. This is what drives all the linking underneath our JIT.
  std::unique_ptr<llvm::orc::ObjectLinkingLayer> objectLayer = nullptr;

  /// Protects the addition to all the layers in 'layers' when called via
  /// the thread-safe addIfAbsent.
  std::mutex mu;

  /// List of materialization layers the JIT has. The base is *always* the
  /// object linking layer. We are having more than 2 layers total:
  /// (1) StaticArchiveLayer, (2) KGEN object generation,
  SmallVector<std::unique_ptr<MaterializationLayer>, 2> layers;

  /// Keep a set of known dylibs and a dylib search order - this will make it
  /// easy to (a) make sure we only have unique dylibs and (b) cache the
  /// search order so we don't recreate it on every lookup.
  llvm::StringSet<> knownDylibs;
  llvm::orc::JITDylibSearchOrder searchOrder;

  /// We need to hold onto a pointer to the data layout because it holds onto
  /// some state.
  llvm::DataLayout dataLayout;

  /// List of buffers that contain archive files added to the JIT. This holds
  /// references to them so they aren't deallocated underneath our feet.
  SmallVector<BufferRef> archiveBuffers;
};
} // namespace M::KGEN

#endif // KGEN_COMPILER_EXECUTIONENGINE_H
