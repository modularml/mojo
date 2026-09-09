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
//
// This file implements the PointerRewriter pass for Metal backend that
// rewriters opaque pointers to typed pointers as it's expected by The Metal.
//
// The pass is taken from Julia's LLVM downgrader
// https://github.com/JuliaLLVM/llvm-downgrade
//
//===----------------------------------------------------------------------===//

#include "PointerRewriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace M::KGEN;

// Demote ConstantAggregate values that contain pointer elements to sequences
// of insertvalue instructions. This is needed because Metal's LLVM 5.0 bitcode
// reader checks that aggregate element types match the values' types, but the
// opaque-to-typed pointer downgrade creates mismatches (e.g., struct element
// becomes {}* while the global value is [2 x i8]*). By converting to
// insertvalue chains, we avoid ConstantAggregates with pointers in the bitcode
// constants block entirely.
static bool demotePointerConstantAggregates(Module &module) {
  SmallVector<std::pair<Instruction *, int>, 8> worklist;
  for (Function &func : module) {
    for (BasicBlock &bb : func) {
      for (Instruction &inst : bb) {
        for (const Use &op : inst.operands()) {
          auto *constAggregate = dyn_cast<ConstantAggregate>(op);
          if (!constAggregate)
            continue;
          // Check if any element is a pointer
          bool hasPtr = false;
          for (unsigned i = 0, e = constAggregate->getNumOperands(); i != e;
               ++i) {
            if (constAggregate->getOperand(i)->getType()->isPointerTy()) {
              hasPtr = true;
              break;
            }
          }
          if (hasPtr)
            worklist.push_back({&inst, op.getOperandNo()});
        }
      }
    }
  }
  if (worklist.empty())
    return false;

  for (auto [inst, opIdx] : worklist) {
    auto *constAggregate = cast<ConstantAggregate>(inst->getOperand(opIdx));

    // Build an insertvalue chain: start with undef, insert each element
    Value *agg = UndefValue::get(constAggregate->getType());
    for (unsigned i = 0, e = constAggregate->getNumOperands(); i != e; ++i) {
      Value *elem = constAggregate->getOperand(i);
      auto *iv = InsertValueInst::Create(agg, elem, {i});
      iv->insertBefore(inst->getIterator());
      agg = iv;
    }
    inst->setOperand(opIdx, agg);
  }
  return true;
}

// Demote all constant expressions that produce pointers, to their
// corresponding instructions so that we can more easily rewrite them.
// getAsInstruction() only unwraps one level, so nested ConstantExprs get
// pushed back onto the worklist. Defensive backstop.
static constexpr unsigned kMaxPointerConstexprNestingDepth = 64;

static bool demotePointerConstexprs(Module &module) {
  // (instruction, operand index, nesting depth within its ConstantExpr chain)
  SmallVector<std::tuple<Instruction *, int, unsigned>, 8> worklist;
  for (Function &func : module)
    for (BasicBlock &bb : func)
      for (Instruction &inst : bb)
        for (const Use &op : inst.operands())
          if (isa<ConstantExpr>(op))
            worklist.push_back({&inst, op.getOperandNo(), 0});
  if (worklist.empty())
    return false;

  while (!worklist.empty()) {
    auto [inst, opIdx, depth] = worklist.pop_back_val();
    if (depth > kMaxPointerConstexprNestingDepth)
      report_fatal_error(
          "PointerRewriter: compiler pointer constant-expression nesting "
          "exceeded the expected depth. Please file a bug report.");
    ConstantExpr *ce = cast<ConstantExpr>(inst->getOperand(opIdx));
    Instruction *newInst = ce->getAsInstruction();
    newInst->insertBefore(inst->getIterator());
    inst->setOperand(opIdx, newInst);
    for (const Use &innerOp : newInst->operands())
      if (isa<ConstantExpr>(innerOp))
        worklist.push_back({newInst, innerOp.getOperandNo(), depth + 1});
  }
  return true;
}

// determine the typed function type based on !arg_eltypes metadata
static FunctionType *getTypedFunctionType(const Function *func) {
  auto &ctx = func->getContext();
  auto *fTy = func->getFunctionType();

  // Apple's air backend lowers emask only on a typed `i8*`; the opaque `{}*`
  // MLIR emits fails the macOS-27 PSO bitcode upgrade. BitcodeWriter17's Call
  // path types the call site to match.
  if (func->getName().starts_with("llvm.agx3.") &&
      func->getName().contains(".with.emask")) {
    auto args = fTy->params().vec();
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      if (auto *opaquePtrTy = dyn_cast<PointerType>(args[i]))
        args[i] = TypedPointerType::get(Type::getInt8Ty(ctx),
                                        opaquePtrTy->getAddressSpace());
    }
    return FunctionType::get(fTy->getReturnType(), args, fTy->isVarArg());
  }

  // handle known intrinsics
  if (func->isIntrinsic()) {
    switch (func->getIntrinsicID()) {
    case Intrinsic::vastart:
      // void @llvm.va_start(i8* <arglist>)
      return FunctionType::get(
          Type::getVoidTy(ctx),
          {TypedPointerType::get(
              Type::getInt8Ty(ctx),
              fTy->getParamType(0)->getPointerAddressSpace())},
          false);
    case Intrinsic::vaend:
      // void @llvm.va_end(i8* <arglist>)
      return FunctionType::get(
          Type::getVoidTy(ctx),
          {TypedPointerType::get(
              Type::getInt8Ty(ctx),
              fTy->getParamType(0)->getPointerAddressSpace())},
          false);
    case Intrinsic::vacopy:
      // void @llvm.va_copy(i8* <destarglist>, i8* <srcarglist>)
      return FunctionType::get(
          Type::getVoidTy(ctx),
          {TypedPointerType::get(
               Type::getInt8Ty(ctx),
               fTy->getParamType(0)->getPointerAddressSpace()),
           TypedPointerType::get(
               Type::getInt8Ty(ctx),
               fTy->getParamType(1)->getPointerAddressSpace())},
          false);
    case Intrinsic::gcroot:
      // void @llvm.gcroot(i8** %ptrloc, i8* %metadata)
      return FunctionType::get(
          Type::getVoidTy(ctx),
          {TypedPointerType::get(
               TypedPointerType::get(
                   Type::getInt8Ty(ctx),
                   fTy->getParamType(0)->getPointerAddressSpace()),
               fTy->getParamType(0)->getPointerAddressSpace()),
           TypedPointerType::get(
               Type::getInt8Ty(ctx),
               fTy->getParamType(1)->getPointerAddressSpace())},
          false);
    case Intrinsic::gcread:
      // i8* @llvm.gcread(i8* %ObjPtr, i8** %Ptr)
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(ctx),
                                fTy->getReturnType()->getPointerAddressSpace()),
          {TypedPointerType::get(
               Type::getInt8Ty(ctx),
               fTy->getParamType(0)->getPointerAddressSpace()),
           TypedPointerType::get(
               TypedPointerType::get(
                   Type::getInt8Ty(ctx),
                   fTy->getParamType(1)->getPointerAddressSpace()),
               fTy->getParamType(1)->getPointerAddressSpace())},
          false);
    case Intrinsic::gcwrite:
      // void @llvm.gcwrite(i8* %P1, i8* %Obj, i8** %P2)
      return FunctionType::get(
          Type::getVoidTy(ctx),
          {TypedPointerType::get(
               Type::getInt8Ty(ctx),
               fTy->getParamType(0)->getPointerAddressSpace()),
           TypedPointerType::get(
               Type::getInt8Ty(ctx),
               fTy->getParamType(1)->getPointerAddressSpace()),
           TypedPointerType::get(
               TypedPointerType::get(
                   Type::getInt8Ty(ctx),
                   fTy->getParamType(2)->getPointerAddressSpace()),
               fTy->getParamType(2)->getPointerAddressSpace())},
          false);
    case Intrinsic::returnaddress:
      // i8* @llvm.returnaddress(i32 <level>)
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(ctx),
                                fTy->getReturnType()->getPointerAddressSpace()),
          {Type::getInt32Ty(ctx)}, false);
    case Intrinsic::addressofreturnaddress:
      // i8* @llvm.addressofreturnaddress()
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(ctx),
                                fTy->getReturnType()->getPointerAddressSpace()),
          {}, false);
    case Intrinsic::sponentry:
      // i8* @llvm.sponentry()
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(ctx),
                                fTy->getReturnType()->getPointerAddressSpace()),
          {}, false);
    case Intrinsic::frameaddress:
      // i8* @llvm.frameaddress(i32 <level>)
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(ctx),
                                fTy->getReturnType()->getPointerAddressSpace()),
          {Type::getInt32Ty(ctx)}, false);
    case Intrinsic::stacksave:
      // i8* @llvm.stacksave()
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(ctx),
                                fTy->getReturnType()->getPointerAddressSpace()),
          {}, false);
    case Intrinsic::stackrestore:
      // void @llvm.stackrestore(i8* %ptr)
      return FunctionType::get(
          Type::getVoidTy(ctx),
          {TypedPointerType::get(
              Type::getInt8Ty(ctx),
              fTy->getParamType(0)->getPointerAddressSpace())},
          false);
    case Intrinsic::prefetch:
      // void @llvm.prefetch(i8* <address>, i32 <rw>, i32 <locality>, i32 <cache
      // type>)
      return FunctionType::get(
          Type::getVoidTy(ctx),
          {TypedPointerType::get(
               Type::getInt8Ty(ctx),
               fTy->getParamType(0)->getPointerAddressSpace()),
           Type::getInt32Ty(ctx), Type::getInt32Ty(ctx), Type::getInt32Ty(ctx)},
          false);
    case Intrinsic::clear_cache:
      // void @llvm.clear_cache(i8*, i8*)
      return FunctionType::get(
          Type::getVoidTy(ctx),
          {TypedPointerType::get(
               Type::getInt8Ty(ctx),
               fTy->getParamType(0)->getPointerAddressSpace()),
           TypedPointerType::get(
               Type::getInt8Ty(ctx),
               fTy->getParamType(1)->getPointerAddressSpace())},
          false);
    case Intrinsic::thread_pointer:
      // i8* @llvm.thread.pointer()
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(ctx),
                                fTy->getReturnType()->getPointerAddressSpace()),
          {}, false);
    }
  }

  // look at the !arg_eltypes metadata
  MDNode *metadata = func->getMetadata("arg_eltypes");
  if (!metadata)
    return fTy;

  auto args = fTy->params().vec();
  for (unsigned i = 0; i < metadata->getNumOperands(); i += 2) {
    auto idxConstant =
        cast<ConstantAsMetadata>(metadata->getOperand(i))->getValue();
    int idx = cast<ConstantInt>(idxConstant)->getZExtValue();
    Type *eltTy = cast<ValueAsMetadata>(metadata->getOperand(i + 1))
                      ->getValue()
                      ->getType();

    auto opaquePtrTy = cast<PointerType>(args[idx]);
    auto typedPtrTy =
        TypedPointerType::get(eltTy, opaquePtrTy->getAddressSpace());
    args[idx] = typedPtrTy;
  }
  return FunctionType::get(fTy->getReturnType(), args, fTy->isVarArg());
}

// prepend an instruction's pointer operand with a no-op bitcast
static void prependBitcast(Module &module, Instruction *inst, int idx) {
  Value *value = inst->getOperand(idx);
  assert(value->getType()->isPtrOrPtrVectorTy() &&
         "Expected a pointer operand");

  // Check if the bitcast would actually be a no-op
  // Don't create bitcasts that would be invalid (e.g., between address spaces)
  Type *dstType = value->getType(); // This is a true no-op bitcast

  // For Metal, we should only create bitcasts if they're truly no-op
  // The original issue was that these "no-op" bitcasts weren't actually no-op
  // when dealing with different address spaces in the typed pointer system

  // Create the bitcast - this is a true no-op since src and dst are the same
  auto *cst = CastInst::Create(Instruction::BitCast, value, dstType);

  if (auto *phi = dyn_cast<PHINode>(inst)) {
    // we can't insert before phis, so rewrite in the incoming block instead
    auto *bb = phi->getIncomingBlock(idx);
    cst->insertBefore(bb->getTerminator()->getIterator());
  } else {
    cst->insertBefore(inst->getIterator());
  }

  inst->setOperand(idx, cst);
}

// replace all uses of a value with no-op bitcasts
static void replaceWithBitcast(Module &module, Value *value) {
  assert(value->getType()->isPtrOrPtrVectorTy() && "Expected a pointer value");

  // Find all uses
  SmallVector<std::pair<Instruction *, unsigned>, 8> worklist;
  for (Use &use : value->uses()) {
    auto user = use.getUser();
    if (auto *inst = dyn_cast<Instruction>(user))
      worklist.push_back({inst, use.getOperandNo()});
  }

  // Insert no-op bitcasts
  for (auto item : worklist) {
    Instruction *inst = item.first;
    int idx = item.second;
    prependBitcast(module, inst, idx);
  }
}

/// append a single instruction's pointer return value with a no-op bitcast
/// FIXME: Revisit the need of this function
static void appendBitcast(Module &module, Instruction *inst) {
  assert(inst->getType()->isPtrOrPtrVectorTy() &&
         "Expected a pointer-returning instruction");
  // Create a true no-op bitcast (same type to same type)
  Instruction *cst =
      CastInst::Create(Instruction::BitCast, inst, inst->getType());
  cst->insertBefore(inst->getNextNode()->getIterator());

  inst->replaceAllUsesWith(cst);
  // HACK: undo the part of the RAUW which messed with our input argument
  cst->setOperand(0, inst);
}

// bitcast uses of globals, for which we can infer the element type based on the
// global's type
static bool bitcastGlobals(Module &module) {
  // Find all globals
  SmallVector<GlobalVariable *, 8> worklist;
  for (GlobalVariable &gv : module.globals())
    worklist.push_back(&gv);
  if (worklist.empty())
    return false;

  // Insert bitcasts
  for (GlobalVariable *gv : worklist) {
    replaceWithBitcast(module, gv);
  }

  return true;
}

// bitcast operands to instructions, by inferring the element type by inspecting
// the instruction
static bool bitcastInstructionOperands(Module &module) {
  // Find all instructions with pointer inputs or outputs
  SmallVector<Instruction *, 8> worklist;
  for (Function &func : module) {
    for (BasicBlock &bb : func) {
      for (Instruction &inst : bb) {
        if (auto *load = dyn_cast<LoadInst>(&inst))
          worklist.push_back(load);
        else if (auto *store = dyn_cast<StoreInst>(&inst))
          worklist.push_back(store);
        else if (auto *atom = dyn_cast<AtomicRMWInst>(&inst))
          worklist.push_back(atom);
        else if (auto *atom = dyn_cast<AtomicCmpXchgInst>(&inst))
          worklist.push_back(atom);
        else if (auto *gep = dyn_cast<GetElementPtrInst>(&inst))
          worklist.push_back(gep);
        else if (auto *alloca = dyn_cast<AllocaInst>(&inst))
          worklist.push_back(alloca);
      }
    }
  }
  if (worklist.empty())
    return false;

  // Add no-op bitcasts
  for (Instruction *inst : worklist) {
    if (auto *load = dyn_cast<LoadInst>(inst)) {
      prependBitcast(module, load, load->getPointerOperandIndex());
    } else if (auto *store = dyn_cast<StoreInst>(inst)) {
      prependBitcast(module, store, store->getPointerOperandIndex());
    } else if (auto *atom = dyn_cast<AtomicRMWInst>(inst)) {
      prependBitcast(module, atom, atom->getPointerOperandIndex());
    } else if (auto *atom = dyn_cast<AtomicCmpXchgInst>(inst)) {
      prependBitcast(module, atom, atom->getPointerOperandIndex());
    } else if (auto *gep = dyn_cast<GetElementPtrInst>(inst)) {
      prependBitcast(module, gep, gep->getPointerOperandIndex());
      appendBitcast(module, gep);
    } else if (auto *alloca = dyn_cast<AllocaInst>(inst)) {
      if (llvm::any_of(alloca->users(), [](User *user) {
            auto *intr = dyn_cast<IntrinsicInst>(user);
            return intr &&
                   (intr->getIntrinsicID() == Intrinsic::lifetime_start ||
                    intr->getIntrinsicID() == Intrinsic::lifetime_end);
          })) {
        // LLVM verifies that pointer of lifetime.start/lifetime.end comes
        // directly from alloca
        continue;
      }
      appendBitcast(module, alloca);
    } else
      llvm_unreachable("Unhandled instruction");
  }

  return true;
}

// bitcast operands to calls, whose type can be altered by metadata attached to
// the function
static bool bitcastFunctionOperands(Module &module) {
  for (Function &func : module) {
    auto *fTy = func.getFunctionType();
    auto *newFTy = getTypedFunctionType(&func);
    if (fTy == newFTy)
      continue;

    // convert calls to this function
    for (User *u : func.users()) {
      if (auto *ci = dyn_cast<CallInst>(u)) {
        for (unsigned idx = 0, e = ci->arg_size(); idx < e; ++idx) {
          auto oldTy = fTy->getParamType(idx);
          auto newTy = newFTy->getParamType(idx);
          if (oldTy == newTy)
            continue;

          prependBitcast(module, ci, idx);
        }
      }
    }
  }

  return false;
}

// Check if function is a Metal kernel that needs device address space mapping
static bool isMetalKernelFunction(const Function &func) {
  if (func.isDeclaration())
    return false;

  // Check for explicit kernel attributes
  if (func.hasFnAttribute("kernel") || func.hasFnAttribute("metal.kernel") ||
      func.hasFnAttribute("gpu.kernel")) {
    return true;
  }

  // Check for GPU calling conventions
  if (func.getCallingConv() == CallingConv::PTX_Kernel ||
      func.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
      func.getCallingConv() == CallingConv::SPIR_KERNEL) {
    return true;
  }

  // Heuristic: External functions with pointer arguments are likely kernels
  if (func.hasExternalLinkage()) {
    for (const Argument &arg : func.args()) {
      if (arg.getType()->isPointerTy()) {
        return true;
      }
    }
  }

  return false;
}

// build a map of values to typed pointer types
PointerRewriter::PointerTypeMap
PointerRewriter::buildPointerMap(const Module &module) {
  PointerRewriter::PointerTypeMap pointerMap;

  // globals
  for (const GlobalVariable &gv : module.globals()) {
    Type *eltTy = gv.getValueType();
    unsigned addrSpace = gv.getAddressSpace();
    auto typedPtrTy = TypedPointerType::get(eltTy, addrSpace);
    pointerMap[&gv] = typedPtrTy;
  }

  // instructions
  for (const Function &func : module) {
    for (const BasicBlock &bb : func) {
      for (const Instruction &inst : bb) {
        if (auto *load = dyn_cast<LoadInst>(&inst)) {
          pointerMap[load->getPointerOperand()] = TypedPointerType::get(
              load->getType(), load->getPointerAddressSpace());
        } else if (auto *store = dyn_cast<StoreInst>(&inst)) {
          pointerMap[store->getPointerOperand()] =
              TypedPointerType::get(store->getValueOperand()->getType(),
                                    store->getPointerAddressSpace());
        } else if (auto *atom = dyn_cast<AtomicRMWInst>(&inst)) {
          pointerMap[atom->getPointerOperand()] = TypedPointerType::get(
              atom->getValOperand()->getType(), atom->getPointerAddressSpace());
        } else if (auto *atom = dyn_cast<AtomicCmpXchgInst>(&inst)) {
          pointerMap[atom->getPointerOperand()] =
              TypedPointerType::get(atom->getNewValOperand()->getType(),
                                    atom->getPointerAddressSpace());
        } else if (auto *gep = dyn_cast<GetElementPtrInst>(&inst)) {
          // Only map the gep's pointer operand if it's not already mapped
          if (pointerMap.find(gep->getPointerOperand()) == pointerMap.end()) {
            pointerMap[gep->getPointerOperand()] = TypedPointerType::get(
                gep->getSourceElementType(), gep->getPointerAddressSpace());
          }

          // Map the gep instruction itself to its result element type
          pointerMap[gep] = TypedPointerType::get(gep->getResultElementType(),
                                                  gep->getAddressSpace());

        } else if (auto *alloca = dyn_cast<AllocaInst>(&inst)) {
          pointerMap[alloca] = TypedPointerType::get(alloca->getAllocatedType(),
                                                     alloca->getAddressSpace());
        }
      }
    }
  }

  // functions
  for (const Function &func : module) {
    auto *fTy = func.getFunctionType();
    auto *newFTy = getTypedFunctionType(&func);

    // Map function arguments to their typed pointer types and build a new
    // function type
    SmallVector<Type *, 8> paramTypes;
    bool needsNewFunctionType = false;

    for (unsigned i = 0; i < fTy->getNumParams(); ++i) {
      Type *paramType = fTy->getParamType(i);

      if (paramType->isPointerTy()) {
        const Argument *arg = func.getArg(i);

        // Check if the new function type has a typed pointer for this parameter
        if (i < newFTy->getNumParams() &&
            newFTy->getParamType(i) != paramType) {
          Type *newParamType = newFTy->getParamType(i);
          if (auto *typedPtrType = dyn_cast<TypedPointerType>(newParamType)) {
            pointerMap[arg] = typedPtrType;
            paramTypes.push_back(typedPtrType);
            needsNewFunctionType = true;
            continue;
          }
        }

        // Try to infer the element type from how the argument is used
        Type *inferredElemType = nullptr;
        for (const User *u : arg->users()) {
          if (auto *gep = dyn_cast<GetElementPtrInst>(u)) {
            inferredElemType = gep->getSourceElementType();
            break;
          } else if (auto *load = dyn_cast<LoadInst>(u)) {
            inferredElemType = load->getType();
            break;
          } else if (auto *store = dyn_cast<StoreInst>(u)) {
            if (store->getPointerOperand() == arg) {
              inferredElemType = store->getValueOperand()->getType();
              break;
            }
          }
        }

        if (inferredElemType) {
          // For Metal kernel functions, use device address space (1) for
          // pointer arguments
          unsigned addressSpace = paramType->getPointerAddressSpace();
          if (isMetalKernelFunction(func) && addressSpace == 0) {
            addressSpace = 1; // Metal device address space
          }

          auto *inferredTypedPtr =
              TypedPointerType::get(inferredElemType, addressSpace);
          pointerMap[arg] = inferredTypedPtr;
          paramTypes.push_back(inferredTypedPtr);
          needsNewFunctionType = true;
        } else if (isMetalKernelFunction(func) &&
                   paramType->getPointerAddressSpace() == 0) {
          // For Metal kernel functions, default to float* in device address
          // space
          auto *floatType = Type::getFloatTy(func.getContext());
          auto *deviceFloatPtr =
              TypedPointerType::get(floatType, 1); // Device address space
          pointerMap[arg] = deviceFloatPtr;
          paramTypes.push_back(deviceFloatPtr);
          needsNewFunctionType = true;
        } else {
          paramTypes.push_back(paramType);
        }
      } else if (false && isMetalKernelFunction(func) &&
                 paramType->isIntegerTy(64)) {
        // DISABLED: For Metal kernel functions, convert i64 index parameters to
        // i32
        Type *i32Type = Type::getInt32Ty(func.getContext());
        paramTypes.push_back(i32Type);
        needsNewFunctionType = true;
      } else {
        paramTypes.push_back(paramType);
      }
    }

    // Create a new function type with typed pointers if needed
    if (needsNewFunctionType) {
      newFTy =
          FunctionType::get(fTy->getReturnType(), paramTypes, fTy->isVarArg());
    }

    pointerMap[&func] = TypedPointerType::get(newFTy, func.getAddressSpace());

    if (fTy == newFTy)
      continue;

    for (unsigned int i = 0; i < fTy->getNumParams(); ++i) {
      auto oldTy = fTy->getParamType(i);
      auto newTy = newFTy->getParamType(i);
      if (oldTy == newTy)
        continue;

      for (const User *u : func.users()) {
        if (auto *ci = dyn_cast<CallInst>(u)) {
          pointerMap[ci->getArgOperand(i)] = cast<TypedPointerType>(newTy);
        }
      }
    }
  }

  return pointerMap;
}

// Remove metadata that contains references to typed pointers
bool PointerRewriter::cleanupTypedPointerMetadata(Module &module) {
  bool changed = false;

  for (Function &func : module) {
    // Remove arg_eltypes metadata since it references typed pointers
    if (func.hasMetadata("arg_eltypes")) {
      func.setMetadata("arg_eltypes", nullptr);
      changed = true;
    }
  }

  return changed;
}

bool PointerRewriter::runImpl(Module &module) {
  // Demote ConstantAggregates with pointer elements to insertvalue chains
  // BEFORE demoting constant expressions, since the insertvalue chain uses
  // the original pointer values directly.
  bool changed = demotePointerConstantAggregates(module);

  // get rid of constant expressions so that we can more easily rewrite them
  changed |= demotePointerConstexprs(module);

  // insert no-op bitcasts surrounding pointer values
  changed |= bitcastGlobals(module);
  changed |= bitcastInstructionOperands(module);
  changed |= bitcastFunctionOperands(module);

  // Remove metadata that references typed pointers
  changed |= cleanupTypedPointerMetadata(module);

  return changed;
}

PreservedAnalyses PointerRewriter::run(Module &module,
                                       ModuleAnalysisManager &MAM) {
  if (runImpl(module))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
