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
// This file implements utilities for binding concrete (non-parametric) values
// to generator parameters.
//
//===----------------------------------------------------------------------===//

#include "ConcreteBindings.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// ConcreteBindingReplacer
//===----------------------------------------------------------------------===//

ConcreteBindingReplacer::ConcreteBindingReplacer(
    ArrayRef<TypedAttr> paramBindings) {
  bindings.reserve(paramBindings.size());
  numUnbound = 0;
  for (TypedAttr binding : paramBindings) {
    ParamBinding pb;
    if (isa<UnboundAttr>(binding)) {
      pb.isBound = false;
      pb.newIndex = numUnbound++;
      pb.boundValue = nullptr;
    } else {
      pb.isBound = true;
      pb.boundValue = binding;
    }
    bindings.push_back(pb);
  }
}

SmallVector<Type> ConcreteBindingReplacer::getRemappedUnboundParamTypes(
    ArrayRef<Type> originalTypes) {
  assert(originalTypes.size() == bindings.size() &&
         "mismatched param types and bindings");

  SmallVector<Type> result;
  result.reserve(numUnbound);

  for (auto [idx, type] : llvm::enumerate(originalTypes)) {
    if (!bindings[idx].isBound)
      result.push_back(getReboundType(type));
  }
  return result;
}

Attribute ConcreteBindingReplacer::tryReplace(Attribute attr, size_t depth) {
  if (auto ref = dyn_cast<ParamIndexRefAttr>(attr)) {
    // Only substitute references at the target depth (depth 0 = root scope).
    if (ref.getDepth() != depth)
      return nullptr; // Let the base class recurse.

    unsigned index = ref.getIndex();
    assert(index < bindings.size() && "parameter index out of range");

    const ParamBinding &binding = bindings[index];
    Type remappedType = replaceImpl(ref.getType(), depth);

    if (binding.isBound) {
      TypedAttr result = binding.boundValue;

      // If the types don't match but are canonically equal, rebind.
      if (result.getType() != remappedType &&
          isEqualCanon(result.getType(), remappedType))
        result = ParamOperatorAttr::getRebind(result, remappedType);

      return result;
    }

    // Unbound: create a new index ref with the updated index.
    // The depth stays the same since we're not changing scope structure.
    if (binding.newIndex == ref.getIndex() && remappedType == ref.getType())
      return ref;

    return ParamIndexRefAttr::get(ref.getDepth(), binding.newIndex,
                                  remappedType);
  }

  return nullptr;
}

PogListAttr ConcreteBindingReplacer::specializeMetadata(
    PogListAttr genMetadata, ArrayRef<TypedAttr> paramBindings,
    ArrayRef<Type> paramTypes, function_ref<InFlightDiagnostic()> emitErrorFn) {
  if (!genMetadata)
    return {};

  // Precompute bound/unbound info and index mapping in a single O(N) pass.
  llvm::BitVector boundParams(paramBindings.size());
  SmallVector<unsigned> newIndexMap(paramBindings.size());
  unsigned unboundCount = 0;
  for (auto [idx, binding] : llvm::enumerate(paramBindings)) {
    if (!isa<UnboundAttr>(binding)) {
      boundParams.set(idx);
    } else {
      newIndexMap[idx] = unboundCount++;
    }
  }

  ParameterEvaluator metadataEvaluator;
  for (auto [idx, binding] : llvm::enumerate(paramBindings)) {
    if (isa<UnboundAttr>(binding)) {
      Type remappedType = getReboundType(paramTypes[idx]);
      metadataEvaluator.appendIndexBinding(
          ParamIndexRefAttr::get(/*depth=*/0, newIndexMap[idx], remappedType));
    } else {
      metadataEvaluator.appendIndexBinding(binding);
    }
  }

  return genMetadata.getSpecializedMetadata(metadataEvaluator, boundParams,
                                            emitErrorFn);
}

//===----------------------------------------------------------------------===//
// Free functions for specializing generators with concrete bindings
//===----------------------------------------------------------------------===//

GeneratorType M::KGEN::getSpecializedWithConcreteBindings(
    GeneratorType gen, ArrayRef<TypedAttr> paramBindings,
    function_ref<InFlightDiagnostic()> emitErrorFn) {
  if (paramBindings.empty())
    return gen;

  ArrayRef<Type> paramTypes = gen.getInputParamTypes();
  if (paramBindings.size() != paramTypes.size()) {
    if (emitErrorFn) {
      emitErrorFn() << "generator type expects " << paramTypes.size()
                    << " parameters but got bindings for "
                    << paramBindings.size();
    }
    return {};
  }

  ConcreteBindingReplacer replacer(paramBindings);
  SmallVector<Type> newParamTypes =
      replacer.getRemappedUnboundParamTypes(paramTypes);
  Type newBody = replacer.getReboundType(gen.getBody());
  PogListAttr rawMetadata = gen.getParamListAttrs();
  PogListAttr genMetadata = replacer.specializeMetadata(
      rawMetadata, paramBindings, paramTypes, emitErrorFn);
  if (rawMetadata && !genMetadata)
    return {};

  return GeneratorType::get(newParamTypes, newBody, genMetadata);
}

GeneratorAttr M::KGEN::getSpecializedWithConcreteBindings(
    GeneratorAttr gen, ArrayRef<TypedAttr> paramBindings,
    function_ref<InFlightDiagnostic()> emitErrorFn) {
  if (paramBindings.empty())
    return gen;

  ArrayRef<Type> paramTypes = gen.getInputParamTypes();
  if (paramBindings.size() != paramTypes.size()) {
    if (emitErrorFn) {
      emitErrorFn() << "generator attr expects " << paramTypes.size()
                    << " parameters but got bindings for "
                    << paramBindings.size();
    }
    return {};
  }
  ConcreteBindingReplacer replacer(paramBindings);
  SmallVector<Type> newParamTypes =
      replacer.getRemappedUnboundParamTypes(paramTypes);
  TypedAttr newBody = replacer.getReboundAttribute(gen.getBody());
  PogListAttr rawMetadata = gen.getMetadata();
  PogListAttr genMetadata = replacer.specializeMetadata(
      rawMetadata, paramBindings, paramTypes, emitErrorFn);
  if (rawMetadata && !genMetadata)
    return {};

  return GeneratorAttr::get(newBody.getContext(), newBody, newParamTypes,
                            genMetadata);
}
