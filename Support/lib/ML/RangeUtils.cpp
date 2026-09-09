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

#include "Support/ML/RangeUtils.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <cstdlib>

using namespace M;

ErrorOr<int64_t> M::getRangeNumElements(int64_t start, int64_t end,
                                        int64_t step) {
  if (step == 0)
    return Error("step must not be zero");

  bool stepSign = step > 0;
  int64_t intervalLen = (stepSign ? 1 : -1) * (end - start);
  if (intervalLen < 0) {
    return Error(Twine("limit must be ") + (stepSign ? "greater" : "less") +
                 " than or equal to start when step is " +
                 (stepSign ? "positive" : "negative"));
  }

  return llvm::divideCeil(intervalLen, std::abs(step));
}
