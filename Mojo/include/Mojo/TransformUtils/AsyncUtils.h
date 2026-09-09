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

#ifndef KGEN_TRANSFORMUTILS_ASYNCUTILS_H
#define KGEN_TRANSFORMUTILS_ASYNCUTILS_H

namespace M::KGEN {

/// An AsyncContinuationField represents a slot in the continuation data
/// structure of a coroutine. Multiple passes depend on the layout of this data
/// structure so we store its definition in a shared location.
enum AsyncContinuationField {
  State = 0,
  ResumeFunction = 1,
  CallbackFn = 2,
  ClosureState = 3,
  ErrorSlot = 4,
  ResultSlot = 5,
  Promise = 6,
  Frame = 7
};

} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_ASYNCUTILS_H
