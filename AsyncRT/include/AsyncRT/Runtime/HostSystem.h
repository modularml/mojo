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

#ifndef ASYNCRT_RUNTIME_HOST_SYSTEM_H
#define ASYNCRT_RUNTIME_HOST_SYSTEM_H

#include "AsyncRT/Runtime/CPUDevice.h"
#include "Support/SymbolExport.h"

namespace M::AsyncRT {

/// Returns a reference to the process-wide global AsyncRT CPUDevice, creating
/// it on first use with \p source and \p options. If a global CPUDevice already
/// exists, triggers a fatal error if \p options do not match those used at
/// creation, and returns a copy of the existing reference.
/// \p allowUsingExistingOptions may be set to true to disable the check that
/// the CPUDevice options match and discard the provided options, but the caller
/// should ensure that it is safe to do so.
MODULAR_CXX_EXPORT CPUDeviceRef
getOrCreateCPUDevice(CPUDeviceSource source,
                     const CPUDeviceOptions &options = CPUDeviceOptions(),
                     bool allowUsingExistingOptions = false);

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_HOST_SYSTEM_H
