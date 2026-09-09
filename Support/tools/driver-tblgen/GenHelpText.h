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

#ifndef SUPPORT_TOOLS_DRIVERTBLGEN_GENHELPTEXT_H
#define SUPPORT_TOOLS_DRIVERTBLGEN_GENHELPTEXT_H

namespace M {

class BackendRegistry;

/// Registers the "gen-help-text" backend.
void registerGenHelpTextBackend(BackendRegistry &registry);
} // namespace M

#endif // SUPPORT_TOOLS_DRIVERTBLGEN_GENHELPTEXT_H
