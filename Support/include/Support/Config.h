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

#ifndef SUPPORT_CONFIG_H
#define SUPPORT_CONFIG_H

namespace mlir {
class MLIRContext;
class PassManager;
} // namespace mlir

namespace M {
/// This function configures the MLIR context according to the current build
/// configuration. In modular production builds, it disables IR dumps in
/// diagnostics.
void configureMLIRContext(mlir::MLIRContext &ctx);

/// This function configures the MLIR pass manager according to the current
/// build configuration. In modular production builds, it disables verification
/// after all passes.
void configurePassManager(mlir::PassManager &mgr);
} // namespace M

#endif // SUPPORT_CONFIG_H
