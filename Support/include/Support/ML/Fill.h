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

#ifndef SUPPORT_ML_FILL_H
#define SUPPORT_ML_FILL_H

#include "Support/ErrorOr.h"
#include "Support/ML/DType.h"
#include <cstddef>

namespace M {
//===----------------------------------------------------------------------===//
// Scalar value generation
//===----------------------------------------------------------------------===//

/// This kernel fills the specified generic buffer with a single "1" or "1.0"
/// real value.  Complex numbers have their imaginary component set to zero.
ErrorOrSuccess getScalarOne(void *destPtr, DType eltType);

/// This kernel fills the specified generic buffer with a single "-1" or "-1.0"
/// real value.  Complex numbers have their imaginary component set to zero.
ErrorOrSuccess getScalarNegativeOne(void *destPtr, DType eltType);

//===----------------------------------------------------------------------===//
// Memory Fills
//===----------------------------------------------------------------------===//

/// This kernel fills the specified generic buffer with a constant value
/// specified by "element".  This returns a non-empty error on failure.
ErrorOrSuccess fillHomogeneous(void *destPtr, size_t numElements, DType eltType,
                               const void *elementPtr);

/// This kernel fills the specified generic buffer with a random values.
/// This returns a non-empty error on failure.
///
/// TODO: This is not implemented in a very general way, the bounds should be
/// passed it or something.
ErrorOrSuccess fillRandom(void *destPtr, size_t numElements, DType eltType);

} // namespace M

#endif // SUPPORT_ML_FILL_H
