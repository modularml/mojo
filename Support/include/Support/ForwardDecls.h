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
// This file forward declares Support types in a canonical place.  This avoids
// scattering forward declarations throughout the codebase.
//
// This only covers the widely used types, not esoteric things like
// AlignedAlloc and ConcatenationTree.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_FORWARD_DECLS_H
#define SUPPORT_FORWARD_DECLS_H

namespace M {
class DType;
class ErrorOrSuccess;
template <typename T>
class ErrorOr;

template <typename T>
class RCRef;
template <typename T>
RCRef<T> copyRCRef(T *ptr);
template <typename T>
RCRef<T> takeRCRef(T *ptr);

template <typename SubClass>
class ReferenceCounted;
} // namespace M

#endif // SUPPORT_FORWARD_DECLS_H
