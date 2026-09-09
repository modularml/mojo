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

#ifndef SUPPORT_STL_EXTRAS_H
#define SUPPORT_STL_EXTRAS_H

#include "Support/AlignedAlloc.h"
#include "Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include <cstddef>
#include <type_traits>

namespace M {

/// Converts an enumeration to its underlying type. Note this function is
/// available as part of the STL in C++23.
template <typename Enum>
constexpr std::underlying_type_t<Enum> to_underlying(Enum e) {
  return static_cast<std::underlying_type_t<Enum>>(e);
}

//===----------------------------------------------------------------------===//
// failableInterleave
//===----------------------------------------------------------------------===//

/// Call a function for each element in the range and a second function in
/// between every pair of elements. Either function can fail, in which case
/// iteration aborts and the function as a whole fails.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
auto failableInterleave(ForwardIterator begin, ForwardIterator end,
                        UnaryFunctor eachFn, NullaryFunctor betweenFn)
    -> decltype(betweenFn()) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    if (failed(betweenFn()) || failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
auto failableInterleave(const Container &c, UnaryFunctor eachFn,
                        NullaryFunctor betweenFn) {
  return failableInterleave(c.begin(), c.end(), eachFn, betweenFn);
}

//===----------------------------------------------------------------------===//
// contains_if
//===----------------------------------------------------------------------===//

/// Returns true if there is at least one element in the range that satisfies
/// the unary predicate.
template <typename Range, typename UnaryPredicate>
bool contains_if(Range &&range, UnaryPredicate pred) {
  auto it = llvm::find_if(range, pred);
  return it != llvm::adl_end(range);
}

//===----------------------------------------------------------------------===//
// AlignedAllocator
//===----------------------------------------------------------------------===//

/// An allocator that can be used in STL data structures with a dynamic
/// alignment value.
template <typename T>
class AlignedAllocator {
public:
  using value_type = T;
  using pointer = T *;

  AlignedAllocator(size_t align) : align(align) {}

  pointer allocate(size_t n) { return (pointer)alignedAlloc(align, n); }

  void deallocate(pointer p, size_t n) { alignedFree(p); }

private:
  size_t align;
};

} // namespace M

#endif // SUPPORT_STL_EXTRAS_H
