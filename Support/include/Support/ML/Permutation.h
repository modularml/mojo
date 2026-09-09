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
// Utilities for working with permutations.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_PERMUTATION_H
#define SUPPORT_ML_PERMUTATION_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace M {

/// Returns a vector with the contents of the data vector rearranged according
/// to the permutation vector and stride.
///
/// Example:
///   data = [aa, ab, ba, bb]
///   permutation = [1, 0]
///   stride = 2
///
///   returns: [ba, bb, aa, ab]
///
/// TODO move out of ML as a more general-purpose array transformation utility.
template <typename T>
static SmallVector<T> permute(ArrayRef<T> data, ArrayRef<int64_t> permutation,
                              int64_t stride = 1) {
  SmallVector<T> output;

  for (int64_t permIdx : permutation)
    for (int64_t j = 0; j < stride; ++j)
      output.emplace_back(data[permIdx * stride + j]);

  return output;
}

template <typename T>
static SmallVector<T> permute(const SmallVector<T> &data,
                              ArrayRef<int64_t> permutation,
                              int64_t stride = 1) {
  ArrayRef<T> dataRef(data);
  return permute(dataRef, permutation, stride);
}

/// Returns a vector where applying `permutation` to permute dimensions achieves
/// output. Only supports stride 1 for now.
///
/// Example:
///   data        : [a, b, c, d] <-- returns `data` given the below
///   permutation : [3, 1, 2, 0]
///   output      : [d, b, c, a]
template <typename T>
static SmallVector<T> permuteReverse(ArrayRef<T> output,
                                     ArrayRef<int64_t> permutation) {
  SmallVector<T> input(output.begin(), output.end());
  for (auto [permIdx, outputValue] : llvm::zip(permutation, output)) {
    input[permIdx] = outputValue;
  }
  return input;
}

template <typename T>
static SmallVector<T> permuteReverse(const SmallVector<T> &data,
                                     ArrayRef<int64_t> permutation) {
  ArrayRef<T> dataRef(data);
  return permuteReverse(dataRef, permutation);
}

/// Returns a permutation where applying to `input` returns `output`.
/// `input` and `output` must be composed of the same unique elements.
///
/// Example:
///   input       : [a, b, c, d]
///   permutation : [3, 1, 2, 0] <-- returns `permutation`
///   output      : [d, b, c, a]
template <typename T>
static SmallVector<int64_t> solvePermutation(ArrayRef<T> output,
                                             ArrayRef<T> input) {
  SmallVector<int64_t> answer;
  answer.reserve(output.size());

  DenseMap<T, int64_t> inputsToIndex;
  for (auto [i, ele] : llvm::enumerate(input)) {
    inputsToIndex[ele] = i;
  }
  for (auto ele : output) {
    answer.push_back(inputsToIndex[ele]);
  }
  return answer;
}

template <typename T>
static SmallVector<int64_t> solvePermutation(const SmallVector<T> &output,
                                             const SmallVector<T> &input) {
  ArrayRef<int64_t> outputRef(output);
  ArrayRef<int64_t> inputRef(input);
  return solvePermutation(outputRef, inputRef);
}

} // namespace M

#endif // SUPPORT_ML_PERMUTATION_H
