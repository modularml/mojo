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

#ifndef ASYNCRT_SUPPORT_CHAIN_H
#define ASYNCRT_SUPPORT_CHAIN_H

namespace M::AsyncRT {

/// This type is used to model dependencies between side-effecting operations,
/// by turning these side effects into explicitly modeled values.  Its runtime
/// representation is a zero sized value.
///
class Chain {
public:
  static void swap(Chain &lhs, Chain &rhs) {}
};

} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_CHAIN_H
