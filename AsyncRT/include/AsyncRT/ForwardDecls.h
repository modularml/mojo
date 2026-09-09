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
// This file forward declares AsyncRT types in a canonical place and imports
// them into the Modular M namespace.  This avoids scattering forward
// declarations throughout the codebase.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_FORWARD_DECLS_H
#define ASYNCRT_FORWARD_DECLS_H

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::AsyncRT {
// AsyncRT/Support Declarations
class Chain;
class EncodedLocation;
class EncodedDiagnostic;
class LocationDecoder;

// AsyncRT/Runtime Declarations
class Allocator;
class AsyncValue;
class AnyAsyncValueRef;
template <typename T>
class AsyncValueRef;
class CPUDevice;

} // namespace M::AsyncRT

//===----------------------------------------------------------------------===//
// Using Declarations
//===----------------------------------------------------------------------===//

namespace M {
// AsyncRT/Support Declarations
using AsyncRT::Chain;
using AsyncRT::EncodedDiagnostic;
using AsyncRT::EncodedLocation;
using AsyncRT::LocationDecoder;

// AsyncRT/Runtime Declarations
using AsyncRT::Allocator;
using AsyncRT::AnyAsyncValueRef;
using AsyncRT::AsyncValue;
using AsyncRT::AsyncValueRef;
using AsyncRT::CPUDevice;
} // namespace M

#endif // ASYNCRT_FORWARD_DECLS_H
