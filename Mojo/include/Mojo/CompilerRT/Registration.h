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

#ifndef KGEN_COMPILERRT_REGISTRATION_H
#define KGEN_COMPILERRT_REGISTRATION_H

#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"

//===----------------------------------------------------------------------===//
// Initialize.cpp
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT LLVM_ATTRIBUTE_USED bool KGEN_CompilerRT_Initialize();

//===----------------------------------------------------------------------===//
// Globals.cpp
//===----------------------------------------------------------------------===//

/// Allow parts of the execution engine to inject globals.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_InsertGlobal(llvm::StringRef name, void *value);

//===----------------------------------------------------------------------===//
// Python.cpp
//===----------------------------------------------------------------------===//

/// If not already set, this sets the `PYTHONPATH` environment variable to point
/// to the typical directories that contain Python modules. These directories
/// are discovered by invoking `python` and querying it for the paths it has
/// been configured to use.
///
/// If an error prevents `PYTHONPATH` from being set, this returns a pointer to
/// a non-empty string literal with an error message. Otherwise, this returns a
/// pointer to an empty string literal.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT const char *
KGEN_CompilerRT_Python_SetPythonPath();

#endif // KGEN_COMPILERRT_REGISTRATION_H
