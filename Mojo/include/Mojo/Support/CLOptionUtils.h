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

#ifndef SUPPORT_COMPILER_CLOPTIONUTILS_H
#define SUPPORT_COMPILER_CLOPTIONUTILS_H

namespace M {

// Register llvm::codegen::RegisterCodegenFlags flags.
// E.g. we want to use denormal-fp-math-f32
void registerCommandFlags();

} // namespace M

#endif // SUPPORT_COMPILER_CLOPTIONUTILS_H
