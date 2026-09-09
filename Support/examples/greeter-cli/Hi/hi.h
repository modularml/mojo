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

#ifndef SUPPORT_EXAMPLES_GREETER_CLI_HI_H
#define SUPPORT_EXAMPLES_GREETER_CLI_HI_H

namespace M {

class SubcommandRegistry;

// This adds a `hi` subcommand function to the registry, under the name "hi".
void registerHiSubcommand(SubcommandRegistry &registry);
} // namespace M

#endif // SUPPORT_EXAMPLES_GREETER_CLI_HI_H
