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

#ifndef MOJO_DOC_H
#define MOJO_DOC_H

namespace M {

class SubcommandRegistry;

/// Initializes the `doc` subcommand and its various options, and registers
/// its callback function with the registry.
void registerDocSubcommand(SubcommandRegistry &registry);

} // namespace M

#endif // MOJO_DOC_H
