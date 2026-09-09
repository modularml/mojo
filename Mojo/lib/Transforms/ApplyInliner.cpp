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

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Compiler/Threading.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/Matchers.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ApplyInlinerPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_APPLYINLINER
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct ApplyInlinerPass : impl::ApplyInlinerBase<ApplyInlinerPass> {
  using ApplyInlinerBase::ApplyInlinerBase;
  void runOnOperation() override;
};

struct FunctionTrait {
  static std::optional<FunctionTrait> identify(GeneratorOp func);

  /// This trait represents a function that trivially forwards a single
  /// register-passable argument.
  ///
  /// ```mlir
  /// kgen.generator @anything(%arg0: !SomeType) -> !SomeType {
  ///   kgen.return %arg0 : !SomeType
  /// }
  /// ```
  ///
  /// This means we can peephole `apply(@anything, x) -> x` regardless of
  /// parameter bindings.
  struct RegForward {};

  /// This trait represents a function that trivially forwards a single
  /// register-passable argument through a byref_result result slot.
  ///
  /// ```mlir
  /// kgen.generator @anything(%value: !SomeType,
  ///         %result: !kgen.pointer<!SomeType> byref_result) -> !kgen.none {
  ///   pop.store %value, %result
  ///   kgen.return %none
  /// }
  /// ```
  ///
  /// This means we can peephole `apply_result_slot(@anything, x) -> x`
  /// regardless of parameter bindings.
  struct RegSlotForward {};

  /// This trait represents a function that simply returns a constant value with
  /// no side effects.
  ///
  /// ```mlir
  /// kgen.generator @constant() -> !SomeType {
  ///   %0 = kgen.param.constant: !SomeType = <...>
  ///   kgen.return %0 : !SomeType
  /// }
  /// ```
  ///
  /// This means we can peephole `apply(@constant) -> value`, while substituting
  /// any parameter values.
  struct RegConstant {
    Attribute value;
  };

  SmartVariant<RegForward, RegSlotForward, RegConstant> impl;
  GeneratorOp generatorOp;
};
} // namespace

/// Use this function instead of `getNextNode()` on an operation to skip over
/// debug operations.
static Operation *getNextNonDebugNode(Operation *op) {
  do {
    op = op->getNextNode();
  } while (op &&
           isa_and_nonnull<DebugInfo::DebugInfoDialect>(op->getDialect()));
  return op;
}

/// Get the first non-debug operation in a block.
static Operation *getFirstNonDebugNode(Block *block) {
  Operation *op = &block->front();
  if (isa_and_nonnull<DebugInfo::DebugInfoDialect>(op->getDialect()))
    return getNextNonDebugNode(op);
  return op;
}

std::optional<FunctionTrait> FunctionTrait::identify(GeneratorOp func) {
  // In all patterns, the function is terminated by a return.
  auto ret = dyn_cast<ReturnOp>(func.getBody()->getTerminator());
  if (!ret)
    return {};
  FuncType sig = func.getFuncTypeGenerator().getBody();
  Operation *first = getFirstNonDebugNode(func.getBody());

  // Check one argument, one result, return is only operation in the body. It
  // follows that the return operand must be the function argument, since there
  // is nothing else it can be.
  if (sig.getNumArguments() == 1 && ret.getNumOperands() == 1 && ret == first)
    return FunctionTrait{RegForward{}, func};

  // Check zero arguments, one result, the first operation is a constant, return
  // is the only other operation. It follows that the return operand must be the
  // constant.
  auto cst = dyn_cast<ParamConstantOp>(first);
  if (cst && sig.getNumArguments() == 0 && ret.getNumOperands() == 1 &&
      getNextNonDebugNode(cst) == ret)
    return FunctionTrait{RegConstant{cst.getValue()}, func};

  // Check two arguments, one result, the return value is a none constant, and
  // one of the first two ops is a store, and the third op is a return.
  NoneAttr noneAttr;
  POP::StoreOp store;
  if (sig.getNumArguments() == 2 && ret.getNumOperands() == 1 &&
      mlir::matchPattern(ret.getOperand(0), mlir::m_Constant(&noneAttr)) &&
      ((store = dyn_cast<POP::StoreOp>(first)) ||
       (store = dyn_cast<POP::StoreOp>(getNextNonDebugNode(first)))) &&
      (getNextNonDebugNode(store) == ret ||
       getNextNonDebugNode(ret.getOperand(0).getDefiningOp()) == ret)) {
    int valueIdx = -1;
    if (sig.getArgConvention(1) == ArgConvention::ByRefResult)
      valueIdx = 0;
    // Check that one of the arguments is a result slot, the result slot
    // argument is the dest of the store, and the other argument is the value.
    if (valueIdx != -1 && store.getPtr() == func.getArgument(!valueIdx) &&
        store.getArg() == func.getArgument(valueIdx))
      return FunctionTrait{RegSlotForward{}, func};
  }

  return {};
}

void ApplyInlinerPass::runOnOperation() {
  DenseMap<StringAttr, FunctionTrait> funcTraits;
  for (auto func : getOperation().getOps<GeneratorOp>())
    if (std::optional<FunctionTrait> trait = FunctionTrait::identify(func))
      funcTraits.try_emplace(func.getSymNameAttr(), std::move(*trait));

  mlir::SymbolTableAnalysis symTabAnalysis(getOperation());
  mlir::LockedSymbolTableCollection symtabs(symTabAnalysis.getSymbolTables());
  KGEN::SymTabEvaluationContext context(getOperation(), symtabs);
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](SugarAttr sugar) -> TypedAttr {
    llvm_unreachable("sugar should be replaced by now");
  });
  replacer.addReplacement([&funcTraits,
                           &context](ParamOperatorAttr apply) -> TypedAttr {
    if (apply.getOpcode() != POC::Apply &&
        apply.getOpcode() != POC::ApplyResultSlot)
      return apply;
    auto cst = dyn_cast<SymbolConstantAttr>(apply.getOperand(0));
    if (!cst)
      return apply;
    auto it =
        funcTraits.find(cast<FlatSymbolRefAttr>(cst.getSymbol()).getAttr());
    if (it == funcTraits.end())
      return apply;

    FunctionTrait trait = it->second;
    GeneratorOp func = trait.generatorOp;

    if (isa<FunctionTrait::RegForward, FunctionTrait::RegSlotForward>(
            trait.impl)) {
      // Handle RegForward/RegSlotForward.
      //
      // In both cases, create an identity GeneratorAttr in the form of
      // `alias inlinedForm[input0, input1, ...] = input0`
      //
      // NOTE: Currently, it is unnecessarily general, but the general form
      // could be reused to handle general "always_inline("builtin")" cases.
      auto inputs = func.getFunctionType().getInputs();
      auto genAttr = GeneratorAttr::get(
          {inputs}, ParamIndexRefAttr::get(0, 0, inputs.front()));
      func.setInlinedFormAttr(genAttr);
      return apply.getOperand(1);
    }

    // Handle RegConstant.
    assert(func.getFunctionType().getNumInputs() == 0);
    auto regCst = cast<FunctionTrait::RegConstant>(trait.impl);
    func.setInlinedFormAttr(cast<TypedAttr>(regCst.value));
    ParameterEvaluator evaluator(func.getInputParams(), cst.getParamValues());
    evaluator.setEvaluationContext(&context);

    return cast<TypedAttr>(evaluator.getReboundAttribute(regCst.value));
  });

  // The replacers have an internal cache, so make sure to share them correctly.
  auto substTrivialFuncs = [](mlir::AttrTypeReplacer &replacer, Operation *op) {
    replacer.recursivelyReplaceElementsIn(
        op, /*replaceAttrs=*/true, /*replaceLocs=*/true, /*replaceTypes=*/true);
  };
  std::vector<Operation *> ops;
  // Substitution cannot be run in parallel due to data race to `inlinedForm`
  // attribute.
  // TODO: If this becomes a bottleneck, need to refactor replacer and
  // SymTabEvaluationContext:
  for (Operation &op : getOperation().getOps())
    substTrivialFuncs(replacer, &op);
}
