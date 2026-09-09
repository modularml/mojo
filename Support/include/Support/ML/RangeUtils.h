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
// Utilities for working with integer ranges, both at compile time and runtime.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_RANGEUTILS_H
#define SUPPORT_ML_RANGEUTILS_H

#include "Support/ErrorOr.h"
#include <cstdint>

namespace M {

/// Return the number of results given the start/limit/step values. Returns an
/// error if the given values are invalid.
ErrorOr<int64_t> getRangeNumElements(int64_t start, int64_t limit,
                                     int64_t step);

} // namespace M

#endif // SUPPORT_ML_RANGEUTILS_H
