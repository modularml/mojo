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

#include "MojoPythonObjectFormatter.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/DataExtractor.h"

using namespace lldb_private;
using namespace M::KGEN::Mojo;

namespace {
// Upstream LLDB changed Process::ReadPointerFromMemory to return
// llvm::Expected<addr_t> and drop the Status& out-param. Preserve the previous
// behavior of yielding LLDB_INVALID_ADDRESS on a failed read.
lldb::addr_t readPointer(Process &process, lldb::addr_t addr) {
  llvm::Expected<lldb::addr_t> valOrErr = process.ReadPointerFromMemory(addr);
  if (!valOrErr) {
    llvm::consumeError(valOrErr.takeError());
    return LLDB_INVALID_ADDRESS;
  }
  return *valOrErr;
}
} // namespace

bool M::KGEN::Mojo::mojoPythonObjectSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  // PythonObject is a register-passable wrapper around a PyObject*. In the
  // REPL, each top-level `var` is stored behind a pointer slot, so the 8
  // bytes we read out of the ValueObject are the *address* of that slot,
  // and the slot's contents are the actual PyObject*.
  //
  // ASSUMPTION: this formatter is built for the REPL, where the
  // double-indirection above is the norm. When a PythonObject is materialized
  // in another context — e.g. an inlined field of a stack-allocated struct
  // in a compiled binary — the bytes returned by GetData() may already be the
  // PyObject* itself, in which case the extra ReadPointerFromMemory below
  // dereferences a pointer-to-pointer and yields garbage. The self-reference
  // probe in layoutHolds() will then fail to fix-point and we fall through to
  // the "<unreadable 0x...>" fallback rather than printing a bogus value.
  //
  // We avoid navigating via GetChildMemberWithName on PythonObject's
  // _obj_ptr chain because the _mlir_value-elision synthetic provider
  // registered on `^!lit\.struct<.*>` can hit a cantFail inside LLDB when
  // probed for fields a given struct doesn't have. Going through process
  // memory directly sidesteps that.
  lldb::ValueObjectSP nonSynth = valobj.GetNonSyntheticValue();
  if (!nonSynth || !nonSynth->GetError().Success())
    return false;

  ExecutionContext execCtx(valobj.GetExecutionContextRef());
  lldb::ProcessSP process = execCtx.GetProcessSP();
  if (!process)
    return false;

  const uint64_t pointerSize = process->GetAddressByteSize();

  Status err;
  DataExtractor data;
  const size_t dataBytes = nonSynth->GetData(data, err);
  if (err.Fail() || dataBytes < pointerSize)
    return false;

  lldb::offset_t dataOffset = 0;
  const lldb::addr_t slotAddr = data.GetAddress(&dataOffset);

  // Address printed by the unreadable fallback: the slot address until we
  // successfully dereference it, then the PyObject* itself. Lets the user
  // see the most-derived address available when type-info reads fail.
  lldb::addr_t fallbackAddr = slotAddr;
  auto fallback = [&] {
    stream.Printf("<unreadable 0x%llx>", (unsigned long long)fallbackAddr);
    return true;
  };

  if (slotAddr == 0 || slotAddr == LLDB_INVALID_ADDRESS)
    return fallback();

  // Dereference the slot to get the actual PyObject*.
  const lldb::addr_t objAddr = readPointer(*process, slotAddr);
  if (objAddr == 0 || objAddr == LLDB_INVALID_ADDRESS)
    return fallback();
  fallbackAddr = objAddr;

  // PyObject header layout depends on whether CPython was built with
  // Py_TRACE_REFS (debug builds). The header is either:
  //   release: { Py_ssize_t ob_refcnt; PyTypeObject *ob_type; }
  //            → 2 * pointerSize, ob_type at offset pointerSize
  //   debug:   { PyObject *_ob_next; PyObject *_ob_prev;
  //              Py_ssize_t ob_refcnt; PyTypeObject *ob_type; }
  //            → 4 * pointerSize, ob_type at offset 3 * pointerSize
  //
  // Detect which by walking the ob_type chain up to PyType_Type's
  // self-reference fix-point (PyType_Type.ob_type == PyType_Type). Every
  // Python object eventually reaches PyType_Type via a finite chain:
  //   obj → T → type → type                       (default metaclass)
  //   instance → MyClass → ABCMeta → type → type  (custom metaclass)
  // At the wrong header layout we'd be following garbage pointers, so the
  // loop either bails on an unreadable read or fails to fix-point within
  // the hop budget. The budget of 8 is a defensive upper bound — typical
  // chains reach the fix-point in 2–4 hops.
  //
  // ob_type sits one pointer before the end of the header in both layouts
  // (release: ob_refcnt, ob_type; debug: _ob_next, _ob_prev, ob_refcnt,
  // ob_type), so its offset is `headerSize - pointerSize`.
  auto layoutHolds = [&](uint64_t headerSize) {
    const uint64_t tOff = headerSize - pointerSize;
    lldb::addr_t typePtr = readPointer(*process, objAddr + tOff);
    if (typePtr == 0 || typePtr == LLDB_INVALID_ADDRESS)
      return false;
    for (int i = 0; i < 8; ++i) {
      lldb::addr_t next = readPointer(*process, typePtr + tOff);
      if (next == 0 || next == LLDB_INVALID_ADDRESS)
        return false;
      if (next == typePtr)
        return true;
      typePtr = next;
    }
    return false;
  };

  uint64_t headerSize = 2 * pointerSize;
  if (!layoutHolds(headerSize)) {
    headerSize = 4 * pointerSize;
    if (!layoutHolds(headerSize))
      return fallback();
  }

  const lldb::addr_t typePtr =
      readPointer(*process, objAddr + headerSize - pointerSize);
  if (typePtr == 0 || typePtr == LLDB_INVALID_ADDRESS)
    return fallback();

  // PyTypeObject has the same header as PyObject plus PyVarObject's ob_size,
  // then tp_name. So tp_name lives at `headerSize + pointerSize` from the
  // type object base (skipping header + ob_size).
  const lldb::addr_t namePtr =
      readPointer(*process, typePtr + headerSize + pointerSize);
  if (namePtr == 0 || namePtr == LLDB_INVALID_ADDRESS)
    return fallback();

  static const size_t maxTypeName = 256;
  char typeNameBuf[maxTypeName];
  const size_t nameBytes =
      process->ReadCStringFromMemory(namePtr, typeNameBuf, maxTypeName, err);
  if (err.Fail() || nameBytes == 0)
    return fallback();

  llvm::StringRef typeName(typeNameBuf);

  // Python None is a non-null singleton, so it lands here rather than in the
  // invalid-pointer fallback above.
  if (typeName == "NoneType") {
    stream << "None";
    return true;
  }

  // Value decoders for common built-ins. These follow the CPython 3.12+
  // layout:
  //
  //   PyLongObject { PyObject base; _PyLongValue long_value; }
  //   _PyLongValue { uintptr_t lv_tag; digit ob_digit[N]; }
  //
  // lv_tag encoding (from Python's Include/cpython/longintrepr.h — CPython
  // names these SIGN_MASK and NON_SIZE_BITS):
  //   low 2 bits     — sign (0 = positive, 1 = zero, 2 = negative)
  //   bits >> 3      — digit count
  //
  // Older (≤ 3.11) CPython used a signed Py_ssize_t ob_size in the same
  // slot; the two encodings are disjoint, and this decoder will likely
  // produce wrong results against those builds. Also assumes the default
  // PYLONG_BITS_IN_DIGIT=30 build where sizeof(digit) == 4 — builds
  // configured with PYLONG_BITS_IN_DIGIT=15 use 2-byte digits and would
  // need a 2-byte read here. Mojo currently bundles a new-enough default
  // Python.
  auto decodeSmallInt = [&](int64_t &out) -> bool {
    static constexpr uint64_t signMask = 3;
    static constexpr uint64_t nonSizeBits = 3;
    static constexpr uint64_t signZero = 1;
    static constexpr uint64_t signNegative = 2;

    Status e;
    uint64_t lvTag = process->ReadUnsignedIntegerFromMemory(
        objAddr + headerSize, pointerSize, 0, e);
    if (e.Fail())
      return false;
    const uint64_t sign = lvTag & signMask;
    const uint64_t digitCount = lvTag >> nonSizeBits;
    if (sign == signZero) {
      out = 0;
      return true;
    }
    if (digitCount != 1) // large ints fall through to the caller's fallback
      return false;
    uint64_t digit = process->ReadUnsignedIntegerFromMemory(
        objAddr + headerSize + pointerSize, 4, 0, e);
    if (e.Fail())
      return false;
    out = (sign == signNegative) ? -(int64_t)digit : (int64_t)digit;
    return true;
  };

  if (typeName == "bool") {
    // bool is a PyLong subclass — non-zero digit means True.
    int64_t value;
    if (decodeSmallInt(value)) {
      stream.Printf("<class 'bool'> = %s", value != 0 ? "True" : "False");
      return true;
    }
  } else if (typeName == "int") {
    int64_t value;
    if (decodeSmallInt(value)) {
      stream.Printf("<class 'int'> = %lld", (long long)value);
      return true;
    }
  } else if (typeName == "float") {
    // PyFloatObject: PyObject header + double ob_fval.
    Status e;
    double value = 0.0;
    if (process->ReadMemory(objAddr + headerSize, &value, sizeof(value), e) ==
            sizeof(value) &&
        !e.Fail()) {
      // %.17g is the minimum precision needed to round-trip an IEEE-754
      // double through a decimal text form. Less than that loses bits.
      stream.Printf("<class 'float'> = %.17g", value);
      return true;
    }
  } else if (typeName == "list" || typeName == "tuple" || typeName == "dict") {
    // For list/tuple this is PyVarObject's ob_size. dict is not a
    // PyVarObject — PyDictObject is PyObject_HEAD + Py_ssize_t ma_used at
    // the same offset, and ma_used is exactly len(dict). Either way, a
    // pointer-sized signed int at `headerSize` gives the correct count.
    Status e;
    int64_t obSize = (int64_t)process->ReadSignedIntegerFromMemory(
        objAddr + headerSize, pointerSize, 0, e);
    if (!e.Fail()) {
      stream.Printf("<class '%s'> = %lld items", typeNameBuf,
                    (long long)obSize);
      return true;
    }
  }

  stream.Printf("<class '%s'> = 0x%llx", typeNameBuf,
                (unsigned long long)objAddr);
  return true;
}
