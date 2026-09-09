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
// This file forward declares and imports various common LLVM and MLIR datatypes
// that we want to use unqualified in the modular M namespace.
//
// Note that most of these are forward declared and then imported into the
// M (Modular) namespace with using decls, rather than being #included.  This is
// because we want clients to explicitly #include the files they need.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_LLVM_COMPILER_FORWARD_DECLS_H
#define SUPPORT_LLVM_COMPILER_FORWARD_DECLS_H

// Include everything in LLVMForwardDecls.
#include "Support/LLVMForwardDecls.h"
#include "mlir/Support/LLVM.h"
#include <cstdint>
#include <type_traits>

// Forward declarations of classes to be imported in to the M namespace.
namespace mlir {
namespace detail { // NOLINT, upstream namespace
template <typename T>
class DenseArrayAttrImpl;
} // namespace detail
class ArrayAttr;
class AsmParser;
class AsmPrinter;
class Attribute;
class BFloat16Type;
class Block;
class IRMapping;
class IRRewriter;
class BlockArgument;
class BoolAttr;
class Builder;
class BuiltinDialect;
class NamedAttrList;
class ConversionPattern;
class ConversionPatternRewriter;
class ConversionTarget;
class DenseElementsAttr;
class DenseIntElementsAttr;
using DenseBoolArrayAttr = detail::DenseArrayAttrImpl<bool>;
using DenseI8ArrayAttr = detail::DenseArrayAttrImpl<int8_t>;
class DenseResourceElementsAttr;
namespace detail {
template <typename Ty>
struct TypedValue;
} // namespace detail
template <typename T>
struct DialectResourceBlobHandle;
class Diagnostic;
class Dialect;
class DialectAsmParser;
class DialectAsmPrinter;
class DialectRegistry;
class DictionaryAttr;
class ElementsAttr;
class FileLineColLoc;
class FlatSymbolRefAttr;
class Float128Type;
class Float16Type;
class Float32Type;
class Float64Type;
class Float80Type;
class Float8E3M4Type;
class Float8E4M3Type;
class Float8E4M3FNType;
class Float8E4M3FNUZType;
class Float8E5M2Type;
class Float8E5M2FNUZType;
class Float8E8M0FNUType;
class FloatAttr;
class FloatTF32Type;
class FloatType;
class FunctionType;
class FusedLoc;
class ImplicitLocOpBuilder;
class IndexType;
class InFlightDiagnostic;
class IntegerAttr;
class IntegerType;
class Location;
class LocationAttr;
class MemRefType;
class MLIRContext;
class ModuleOp;
class MutableOperandRange;
class NamedAttribute;
class NamedAttrList;
class NoneType;
class OpAsmDialectInterface;
class OpAsmParser;
class OpAsmPrinter;
class OpBuilder;
class OperandRange;
class Operation;
class OpFoldResult;
class OpOperand;
class OptionalParseResult;
class OpResult;
class Pass;
class PatternRewriter;
class RankedTensorType;
class Region;
class RegionRange;
template <typename HandleT>
class ResourceBlobManagerDialectInterfaceBase;
class RewritePattern;
class RewritePatternSet;
class ShapedType;
class SplatElementsAttr;
class StringAttr;
class SymbolRefAttr;
class SymbolTable;
class SymbolTableCollection;
class TupleType;
class Type;
class TypeAttr;
class TypeConverter;
class TypeID;
class TypeRange;
class TypeStorage;
class TypedAttr;
class UnitAttr;
class UnknownLoc;
class Value;
class ValueRange;
class VectorType;
class WalkResult;
enum class RegionKind;
struct CallInterfaceCallable;
struct MemRefAccess;
struct OperationState;

template <typename SourceOp>
class OpConversionPattern;
template <typename SourceOp>
class OpInterfaceConversionPattern;
template <typename SourceOp>
struct OpInterfaceRewritePattern;
template <typename T>
class OperationPass;
template <typename SourceOp>
struct OpRewritePattern;
template <typename OpTy>
class OwningOpRef;

using DefaultTypeStorage = TypeStorage;
using OpAsmSetValueNameFn = function_ref<void(Value, StringRef)>;
namespace OpTrait {}
namespace quant { // NOLINT, upstream namespace
class CalibratedQuantizedType;
class QuantizedType;
class UniformQuantizedType;
class UniformQuantizedPerAxisType;
} // namespace quant
} // namespace mlir

// Import things we want into our namespace.
namespace M {
using mlir::ArrayAttr;
using mlir::AsmParser;
using mlir::AsmPrinter;
using mlir::Attribute;
using mlir::BFloat16Type;
using mlir::Block;
using mlir::BlockArgument;
using mlir::BoolAttr;
using mlir::Builder;
using mlir::CallInterfaceCallable;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::DefaultTypeStorage;
using mlir::DenseBoolArrayAttr;
using mlir::DenseElementsAttr;
using mlir::DenseI8ArrayAttr;
using mlir::DenseIntElementsAttr;
using mlir::DenseResourceElementsAttr;
using mlir::IRMapping;
using mlir::IRRewriter;
using DenseResourceElementsHandle =
    mlir::DialectResourceBlobHandle<mlir::BuiltinDialect>;
using DenseResourceElementsHandleManager =
    mlir::ResourceBlobManagerDialectInterfaceBase<DenseResourceElementsHandle>;
using mlir::Diagnostic;
using mlir::Dialect;
using mlir::DialectAsmParser;
using mlir::DialectAsmPrinter;
using mlir::DialectRegistry;
using mlir::DictionaryAttr;
using mlir::ElementsAttr;
using mlir::FileLineColLoc;
using mlir::FlatSymbolRefAttr;
using mlir::Float128Type;
using mlir::Float16Type;
using mlir::Float32Type;
using mlir::Float64Type;
using mlir::Float80Type;
using mlir::Float8E3M4Type;
using mlir::Float8E4M3FNType;
using mlir::Float8E4M3FNUZType;
using mlir::Float8E4M3Type;
using mlir::Float8E5M2FNUZType;
using mlir::Float8E5M2Type;
using mlir::Float8E8M0FNUType;
using mlir::FloatAttr;
using mlir::FloatTF32Type;
using mlir::FloatType;
using mlir::FunctionType;
using mlir::FusedLoc;
using mlir::ImplicitLocOpBuilder;
using mlir::IndexType;
using mlir::InFlightDiagnostic;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::Location;
using mlir::LocationAttr;
using mlir::MemRefAccess;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::MutableOperandRange;
using mlir::NamedAttribute;
using mlir::NamedAttrList;
using mlir::NoneType;
using mlir::OpAsmDialectInterface;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpAsmSetValueNameFn;
using mlir::OpBuilder;
using mlir::OpConversionPattern;
using mlir::OperandRange;
using mlir::Operation;
using mlir::OperationPass;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::OpInterfaceConversionPattern;
using mlir::OpInterfaceRewritePattern;
using mlir::OpOperand;
using mlir::OpResult;
using mlir::OpRewritePattern;
using mlir::OptionalParseResult;
using mlir::OwningOpRef;
using mlir::Pass;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::Region;
using mlir::RegionKind;
using mlir::RegionRange;
using mlir::RewritePattern;
using mlir::RewritePatternSet;
using mlir::ShapedType;
using mlir::SplatElementsAttr;
using mlir::StringAttr;
using mlir::SymbolRefAttr;
using mlir::SymbolTable;
using mlir::SymbolTableCollection;
using mlir::TupleType;
using mlir::Type;
using mlir::TypeAttr;
using mlir::TypeConverter;
using mlir::TypedAttr;
template <typename Ty, typename Value = mlir::Value>
/// If Ty is mlir::Type this will select `Value` instead of having a wrapper
/// around it. This helps resolve ambiguous conversion issues.
using TypedValue =
    std::conditional_t<std::is_same_v<Ty, mlir::Type>, mlir::Value,
                       mlir::detail::TypedValue<Ty>>;
using mlir::TypeRange;
using mlir::TypeStorage;
using mlir::UnitAttr;
using mlir::UnknownLoc;
using mlir::Value;
using mlir::ValueRange;
using mlir::VectorType;
using mlir::WalkResult;
using mlir::quant::CalibratedQuantizedType;
using mlir::quant::QuantizedType;
using mlir::quant::UniformQuantizedPerAxisType;
using mlir::quant::UniformQuantizedType;

namespace OpTrait = mlir::OpTrait;
} // namespace M

#endif // SUPPORT_LLVM_FORWARD_DECLS_H
