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

#include "Support/Compiler/DomainAwareReplacer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <type_traits>
#include <utility>

using namespace M;

DomainAwareReplacer::DomainAwareReplacer()
    : cache([&](CacheKey attr) { return breakCycleImpl(attr); }) {}

//===----------------------------------------------------------------------===//
// Application
//===----------------------------------------------------------------------===//

LogicalResult DomainAwareReplacer::replaceElementsIn(Operation *op,
                                                     DomainId domain,
                                                     bool replaceAttrs,
                                                     bool replaceLocs,
                                                     bool replaceTypes) {
  // Functor that replaces the given element if the new value is different,
  // otherwise returns nullptr.
  // If the replacement can not be performed, return failure.
  auto replaceIfDifferent =
      [&](auto element) -> FailureOr<std::conditional_t<
                            std::is_convertible_v<decltype(element), Attribute>,
                            Attribute, Type>> {
    auto replacement = replace(element, domain);
    if (succeeded(replacement)) {
      return *replacement != element ? *replacement : nullptr;
    }
    return failure();
  };

  // Update the attribute dictionary.
  if (replaceAttrs) {
    auto newAttrs = replaceIfDifferent(op->getAttrDictionary());
    if (failed(newAttrs))
      return failure();
    if (*newAttrs)
      op->setAttrs(cast<DictionaryAttr>(*newAttrs));
  }

  // If we aren't updating locations or types, we're done.
  if (!replaceTypes && !replaceLocs)
    return success();

  // Update the location.
  if (replaceLocs) {
    FailureOr<Attribute> newLoc = replaceIfDifferent(op->getLoc());
    if (failed(newLoc))
      return failure();
    if (*newLoc)
      op->setLoc(cast<LocationAttr>(*newLoc));
  }

  // Update the result types.
  if (replaceTypes) {
    for (OpResult result : op->getResults()) {
      FailureOr<Type> newType = replaceIfDifferent(result.getType());
      if (failed(newType))
        return failure();
      if (*newType)
        result.setType(*newType);
    }
  }

  // Update any nested block arguments.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument &arg : block.getArguments()) {
        if (replaceLocs) {
          FailureOr<Attribute> newLoc = replaceIfDifferent(arg.getLoc());
          if (failed(newLoc))
            return failure();
          if (*newLoc)
            arg.setLoc(cast<LocationAttr>(*newLoc));
        }

        if (replaceTypes) {
          FailureOr<Type> newType = replaceIfDifferent(arg.getType());
          if (failed(newType))
            return failure();
          if (*newType)
            arg.setType(*newType);
        }
      }
    }
  }

  return success();
}

template <typename T>
static void updateSubElementImpl(T element,
                                 DomainAwareReplacer::DomainId domain,
                                 DomainAwareReplacer &replacer,
                                 SmallVectorImpl<T> &newElements,
                                 FailureOr<bool> &changed) {
  // Bail early if we failed at any point.
  if (failed(changed))
    return;

  // Guard against potentially null inputs. We always map null to null.
  if (!element) {
    newElements.push_back(nullptr);
    return;
  }

  // Replace the element.
  FailureOr<T> result = replacer.replace(element, domain);
  if (succeeded(result)) {
    newElements.push_back(*result);
    if (*result != element)
      changed = true;
  } else {
    changed = failure();
  }
}

template <typename T>
static T replaceSubElements(T interface, DomainAwareReplacer::DomainId domain,
                            DomainAwareReplacer &replacer) {
  // Walk the current sub-elements, replacing them as necessary.
  SmallVector<Attribute, 16> newAttrs;
  SmallVector<Type, 16> newTypes;
  FailureOr<bool> changed = false;
  interface.walkImmediateSubElements(
      [&](Attribute element) {
        updateSubElementImpl(element, domain, replacer, newAttrs, changed);
      },
      [&](Type element) {
        updateSubElementImpl(element, domain, replacer, newTypes, changed);
      });
  if (failed(changed))
    return nullptr;

  // If any sub-elements changed, use the new elements during the replacement.
  T result = interface;
  if (*changed)
    result = interface.replaceImmediateSubElements(newAttrs, newTypes);
  return result;
}

/// Shared implementation of replacing a given attribute or type element.
template <typename T, typename ReplaceFns>
static T replaceElementImpl(T element, DomainAwareReplacer::DomainId domain,
                            ReplaceFns &replaceFns,
                            DomainAwareReplacer &replacer) {
  T result = element;
  WalkResult walkResult = WalkResult::advance();
  for (auto &replaceFn : llvm::reverse(replaceFns)) {
    if (std::optional<std::pair<T, WalkResult>> newRes = replaceFn(element)) {
      std::tie(result, walkResult) = *newRes;
      break;
    }
  }

  // If an error occurred, return nullptr to indicate failure.
  if (walkResult.wasInterrupted() || !result) {
    return nullptr;
  }

  // Handle replacing sub-elements if this element is also a container.
  if (!walkResult.wasSkipped()) {
    // Replace the sub elements of this element, bailing if we fail.
    if (!(result = replaceSubElements(result, domain, replacer))) {
      return nullptr;
    }
  }

  return result;
}

template <typename T>
T DomainAwareReplacer::cachedReplaceImpl(T element, DomainId domain) {
  AttrOrType taggedElement(element);
  decltype(cache)::CacheEntry cacheEntry =
      cache.lookupOrInit(CacheKey{taggedElement, domain});
  if (auto resultOpt = cacheEntry.get())
    return T::getFromOpaquePointer(*resultOpt);

  T result;
  if constexpr (std::is_same_v<T, Attribute>)
    result =
        replaceElementImpl(element, domain, attrReplacementFns[domain], *this);
  else
    result =
        replaceElementImpl(element, domain, typeReplacementFns[domain], *this);

  cacheEntry.resolve(result.getAsOpaquePointer());
  return result;
}

std::optional<const void *>
DomainAwareReplacer::breakCycleImpl(CacheKey element) {
  AttrOrType taggedElement = element.first;
  DomainId domain = element.second;
  if (auto attr = dyn_cast<Attribute>(taggedElement)) {
    for (auto &cyclicReplaceFn : llvm::reverse(attrCycleBreakerFns[domain])) {
      if (std::optional<Attribute> newRes = cyclicReplaceFn(attr)) {
        return newRes->getAsOpaquePointer();
      }
    }
  } else {
    auto type = dyn_cast<Type>(taggedElement);
    for (auto &cyclicReplaceFn : llvm::reverse(typeCycleBreakerFns[domain])) {
      if (std::optional<Type> newRes = cyclicReplaceFn(type)) {
        return newRes->getAsOpaquePointer();
      }
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void DomainAwareReplacer::addReplacement(ReplaceFn<Attribute> fn,
                                         DomainId domain) {
  attrReplacementFns[domain].emplace_back(std::move(fn));
}

void DomainAwareReplacer::addReplacement(ReplaceFn<Type> fn, DomainId domain) {
  typeReplacementFns[domain].push_back(std::move(fn));
}

void DomainAwareReplacer::addCycleBreaker(CycleBreakerFn<Attribute> fn,
                                          DomainId domain) {
  attrCycleBreakerFns[domain].emplace_back(std::move(fn));
}

void DomainAwareReplacer::addCycleBreaker(CycleBreakerFn<Type> fn,
                                          DomainId domain) {
  typeCycleBreakerFns[domain].emplace_back(std::move(fn));
}
