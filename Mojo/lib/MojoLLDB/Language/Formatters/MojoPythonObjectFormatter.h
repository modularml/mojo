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

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_FORMATTERS_MOJOPYTHONOBJECTFORMATTER_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_FORMATTERS_MOJOPYTHONOBJECTFORMATTER_H

// NOLINTNEXTLINE(readability-identifier-naming)
namespace lldb_private {
class Stream;
class TypeSummaryOptions;
class ValueObject;
} // namespace lldb_private

namespace M::KGEN::Mojo {

/// Summary provider for Mojo's PythonObject type.
///
/// PythonObject wraps a reference-counted PyObject* pointer. The formatter
/// reads CPython's PyObject / PyTypeObject structures directly to extract
/// the Python type name, and decodes the underlying value for a handful of
/// common built-ins. No Python code runs under the debugger.
///
/// Targeted at the REPL, where each top-level `var` is held behind a
/// pointer slot; the formatter does one extra dereference accordingly. In
/// non-REPL contexts the slot indirection may not apply, in which case the
/// PyType_Type self-reference probe fails and the formatter prints
/// "<unreadable 0x...>" rather than a bogus decoded value. See the
/// implementation for details.
///
/// Auto-detects release vs. debug-build CPython by probing the PyType_Type
/// self-reference invariant at each candidate header size (release: 2 *
/// pointerSize; debug, with _ob_next/_ob_prev: 4 * pointerSize).
///
/// Display format:
///   - "None" when the object's Python type is NoneType
///   - "<class 'bool'>  = True | False"
///   - "<class 'int'>   = <decoded small int>" (values that fit in a single
///     30-bit digit; larger ints fall back to the address form)
///   - "<class 'float'> = <double>"
///   - "<class 'list' | 'tuple' | 'dict'> = <N> items"
///   - "<class '<typename>'> = 0x<addr>" for everything else
///   - "<unreadable 0x<addr>>" when the pointer is null/invalid or type
///     info can't be read
bool mojoPythonObjectSummaryProvider(
    lldb_private::ValueObject &valobj, lldb_private::Stream &stream,
    const lldb_private::TypeSummaryOptions &options);

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_FORMATTERS_MOJOPYTHONOBJECTFORMATTER_H
