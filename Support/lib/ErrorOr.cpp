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

#include "Support/ErrorOr.h"
#include "Support/Error.h"
#include "Support/LogicalResult.h"
#include "llvm/Support/Error.h"
#include <utility>

using namespace M;

ErrorOrSuccess M::toModularErrorOr(llvm::Error error) {
  if (error)
    return Error(llvm::toString(std::move(error)));
  return success();
}
