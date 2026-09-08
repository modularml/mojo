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
// This file registers all the dialects in the KGEN library.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_INITALLDIALECTS_H
#define KGEN_TOOLCOMMON_INITALLDIALECTS_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace M {
/// Add all the KGEN dialects and extensions to the provided registry.
void registerAllKGENDialects(mlir::DialectRegistry &registry);
/// Register all required LLVMIR translation interfaces.
void registerKGENToLLVMTranslation(mlir::DialectRegistry &registry);
/// Preload all KGEN dialects in the MLIR context.
void preloadAllKGENDialects(mlir::MLIRContext *ctx);
} // namespace M

#endif // KGEN_TOOLCOMMON_INITALLDIALECTS_H
