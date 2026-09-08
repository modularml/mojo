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

#ifndef KGEN_ELABORATOR_INTERPRETERCACHE_H
#define KGEN_ELABORATOR_INTERPRETERCACHE_H

#include "Cache/CachedTransform.h"
#include "IREvaluator.h"
#include "Mojo/Interpreter/BytecodeInterpreter.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/Threading/Shared.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "mlir/Support/ThreadLocalCache.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// InterpreterCache
//===----------------------------------------------------------------------===//

/// A concrete function is a region with a lazily generated interpreter
/// bytecode.
class ConcreteFunction {
public:
  ConcreteFunction() : region(nullptr) {}
  ConcreteFunction(Region &region)
      : region(&region), compiled(std::make_shared<CompiledRegion>()) {}

  /// Get the bytecode or generate it if necessary.
  ErrorTreeOr<const FunctionIRBytecode *> getOrCompile(TargetInfoAttr target,
                                                       bool optimize) {
    return compiled->compileIfNecessary(*region, target, optimize);
  }

  operator bool() const { return region; }

private:
  struct CompiledRegion {
    ErrorTreeOr<const FunctionIRBytecode *>
    compileIfNecessary(Region &region, TargetInfoAttr target, bool optimize);

    /// A clone of the original function if we choose to optimize it before
    /// generating bytecode.
    OwningOpRef<FuncOp> clone;
    /// Shared bytecode generation.
    Shared<std::optional<FunctionIRBytecode>> bytecode;
    /// A flag to indicate whether the bytecode has already been generated.
    std::atomic<bool> compiled = false;
  };

  /// The region to compile.
  Region *region;
  /// A shared pointer to a compiled bytecode that can be shared across
  /// threads.
  std::shared_ptr<CompiledRegion> compiled;
};

class InterpreterCache : public BytecodeCompiler {
public:
  InterpreterCache(TargetInfoAttr target, bool optimize)
      : target(target), optimize(optimize) {}

  void addRegion(Region &region) {
    cache.modify(
        [&](auto &map) { map.try_emplace(&region, ConcreteFunction(region)); });
  }

  ErrorTreeOr<const FunctionIRBytecode *>
  compileBytecode(Region &region) override {
    ConcreteFunction &value = (*tlc)[&region];
    if (!value)
      value = cache.read([&](auto &map) { return map.at(&region); });
    return value.getOrCompile(target, optimize);
  }

private:
  TargetInfoAttr target;
  bool optimize;

  // Shared top-level cache.
  Shared<DenseMap<Region *, ConcreteFunction>> cache;
  // Per-thread cache to reduce contention.
  mlir::ThreadLocalCache<DenseMap<Region *, ConcreteFunction>> tlc;
};

//===----------------------------------------------------------------------===//
// ElaborationState
//===----------------------------------------------------------------------===//

class ElaborationState {
  enum State { NEW_IMPL_SCOPE, NEW_PARAM_SCOPE, ADVANCE, ERROR };

public:
  /// Return the elaboration state that indicates an operation was successfully
  /// processed.
  static ElaborationState advance() { return ADVANCE; }
  /// Return the elaboration state that indicates elaboration should be
  /// pre-empted by a new frame within a function.
  static ElaborationState skipFrame() { return NEW_IMPL_SCOPE; }
  /// Return the elaboration state that indicates elaboration should be
  /// pre-empted by a new parameter node.
  static ElaborationState skipNode() { return NEW_PARAM_SCOPE; }
  /// Return the elaboration state that indicates a fatal error has occurred
  /// during elaboration.
  static ElaborationState error() { return ERROR; }

  /// Return true if the frame should be skipped.
  bool shouldSkipFrame() const { return state == NEW_IMPL_SCOPE; }
  /// Return true if the node should be skipped.
  bool shouldSkipNode() const { return state == NEW_PARAM_SCOPE; }
  /// Return true if an error occurred.
  bool isError() const { return state == ERROR; }

  /// Allow implicit conversion from `LogicalResult`.
  ElaborationState(LogicalResult result)
      : state(succeeded(result) ? ADVANCE : ERROR) {}

private:
  ElaborationState(State state) : state(state) {}

  State state;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_INTERPRETERCACHE_H
