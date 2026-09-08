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

#ifndef KGEN_UNITTESTS_MOJO_LSP_SERVER_SUPPORT_H
#define KGEN_UNITTESTS_MOJO_LSP_SERVER_SUPPORT_H

#include "../tools/mojo-lsp-test-client/LSPBatchClient.h"

namespace lsp = llvm::lsp;

namespace M {

/// Create a test client that asserts the execution doesn't fail and also dumps
/// the contents of server IO files upon errors.
LSPBatchClient createTestClient(bool attachDebugger = false);

/// Create a document from a file located in the `/inputs` folder.
Document createDocumentFromInputFile(StringRef fileName);

/// Create a document from a file located in the `/inputs-with-package` folder.
Document createDocumentFromInputFileWithinPackage(StringRef fileName);

} // namespace M

#endif // KGEN_UNITTESTS_MOJO_LSP_SERVER_SUPPORT_H
