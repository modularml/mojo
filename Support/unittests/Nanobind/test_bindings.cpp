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

#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "nanobind/nanobind.h"
#include "llvm/Support/LogicalResult.h"

namespace nb = nanobind;

NB_MODULE(bindings, m) {
  m.def("return_logical_result_success",
        [] { return llvm::LogicalResult::success(); });
  m.def("return_logical_result_failure",
        [] { return llvm::LogicalResult::failure(); });

  m.def("return_error_or_success_success",
        []() -> M::ErrorOrSuccess { return M::ErrorOrSuccess(); });
  m.def("return_error_or_success_failure",
        []() -> M::ErrorOrSuccess { return M::Error("failed"); });

  m.def("return_error_or_success", []() -> M::ErrorOr<int> { return 42; });
  m.def("return_error_or_failure",
        []() -> M::ErrorOr<int> { return M::Error("failed"); });
}
