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

#ifndef KGEN_TOOLS_MOJO_COMMON_LLDB_H
#define KGEN_TOOLS_MOJO_COMMON_LLDB_H

#include "Support/Driver/DriverSupport.h"
#include "Support/ErrorOr.h"

namespace M {

/// Invokes an LLDB process with the provided arguments.
int invokeLLDB(const State &state, ArrayRef<std::string> lldbArgs,
               ArrayRef<std::string> runArgs = {}, bool dryRun = false);

} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_LLDB_H
