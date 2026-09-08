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
// This file defines the API for interacting with the Mojo LLDB plugin.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOLLDB_PLUGIN_H
#define KGEN_MOJOLLDB_PLUGIN_H

#include "Support/Context.h"
#include "Support/SymbolExport.h"

namespace M::KGEN {

/// Set the context to use inside the LLDB plugin.  This should be set before
/// the LLDB plugin initializes.
MODULAR_VISIBILITY_EXPORT void setLLDBPluginContext(ContextRef ctx);

} // namespace M::KGEN

#endif // KGEN_MOJOLLDB_PLUGIN_H
