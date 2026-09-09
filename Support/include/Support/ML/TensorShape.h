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
// This file declares the TensorShape class.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_TENSORSHAPE_H
#define SUPPORT_ML_TENSORSHAPE_H

#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LLVMYAMLForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace M {

/// The maximum tensor rank for any tensor shape.
/// This value must match max_rank in Kernels/mojo/Stdlib/Buffer.mojo
constexpr size_t kMaxRank = 8;

/// The value representing dynamic dimension in TensorShape and TensorSpec.
constexpr int64_t kDynamicDimensionValue = -1;

/// The value representing dynamic rank in TensorShape and TensorSpec.
constexpr int64_t kDynamicRankValue = -1;

/// Flag to signal construction of dynamic ranked shapes.
enum TensorRankStyle { kStaticallyRanked = 0, kDynamicallyRanked = 1 };

//===----------------------------------------------------------------------===//
// Detail::TensorShapeStorage
//===----------------------------------------------------------------------===//

namespace Detail {
/// The distinguished uint8_t rank representing 'dynamic' or 'unknown'.
constexpr uint8_t kDynamicRank = std::numeric_limits<uint8_t>::max();

static_assert(kMaxRank < kDynamicRank);

/// This class implements a storage class to hold tensor shapes in a compact
/// 16-byte format that is suitable for long term storage on the heap.  It is
/// carefully laid out to hold common tensor sizes inline without losing
/// support for the full generality of tensor shapes.
class TensorShapeStorage {
  /// This supports two inline representations and an out-of-line one:
  ///  1) k16 can hold up to 6 dimensions when they fit into 16-bits.
  ///  2) k32 can hold up to 4 dimension where the first three fits in
  ///     32-bits and the last fits in 8 bits (typically channels or batch
  ///     size).
  ///  3) kOutOfLine is used for the general case.
  ///
  /// Important: Identical shapes have the same representation kind to allow
  /// efficient shape comparison with memcmp for k16 and k32.
  ///
  /// Each representation has an additional 8 bits of unused "auxiliary"
  /// storage.  This is used to hold a DType for TensorSpec.  We keep
  /// this at the end of the storage so we can efficiently omit it from
  /// memset/memcpy operations.
  enum class RepKind : uint8_t { k16 = 0, k32 = 1, kOutOfLine = 2 };

  struct Rep16 {
    int16_t dims[6]; // < 0 if dynamic
    uint8_t unused;
    RepKind kind;
    uint8_t rank; // <= kMaxRank, == kDynamicRank if dynamic
    uint8_t auxiliary;
  };
  struct Rep32 {
    int32_t dims[3]; // < 0 if dynamic
    int8_t dim3;     // < 0 if dynamic
    RepKind kind;
    uint8_t rank; // <= kMaxRank, == kDynamicRank if dynamic
    uint8_t auxiliary;
  };

  struct RepOutOfLine {
    ssize_t *dims; // < 0 if dynamic
    // FIXME: This isn't correct for big endian systems, but we check with
    // static_assert below.
    uint8_t padding[13 - sizeof(void *)];
    RepKind kind;
    uint8_t rank; // <= kMaxRank, == kDynamicRank if dynamic
    uint8_t auxiliary;
  };

  union {
    Rep16 rep16;
    Rep32 rep32;
    RepOutOfLine repOutOfLine;
  } representation;

public:
  /// Default construct to rank-0 shape with zero dimensions.
  TensorShapeStorage() {
    memset(&representation, 0, sizeof(representation));
    representation.repOutOfLine.kind = RepKind::k32;
  }

  ~TensorShapeStorage() {
    if (isOutOfLine())
      delete[] representation.repOutOfLine.dims;
  }

  TensorShapeStorage(const TensorShapeStorage &other) {
    representation.rep32.kind = RepKind::k32;
    operator=(other);
  }
  TensorShapeStorage(TensorShapeStorage &&other) {
    representation.rep32.kind = RepKind::k32;
    operator=(other);
  }
  void operator=(const TensorShapeStorage &other) {
    if (&other == this)
      return;
    if (isOutOfLine())
      delete[] representation.repOutOfLine.dims;
    memcpy(&representation, &other.representation, sizeof(representation));
    if (isOutOfLine()) {
      representation.repOutOfLine.dims = new ssize_t[getRank()];
      memcpy(representation.repOutOfLine.dims,
             other.representation.repOutOfLine.dims,
             getRank() * sizeof(ssize_t));
    }
  }
  void operator=(TensorShapeStorage &&other) {
    if (isOutOfLine())
      delete[] representation.repOutOfLine.dims;
    memcpy(&representation, &other.representation, sizeof(representation));
    // Take ownership of an out-of-line pointer if present.
    other.representation.repOutOfLine.kind = RepKind::k16;
  }

  /// Returns true if shape is statically ranked.
  bool hasRank() const {
    static_assert(offsetof(Rep16, rank) == offsetof(Rep32, rank) &&
                      offsetof(Rep16, rank) == offsetof(RepOutOfLine, rank),
                  "Layout mismatch inside of TensorShape");
    // Because all of the representations store their rank in the same place,
    // we can just access an arbitrary one.
    return representation.rep16.rank != kDynamicRank;
  }

  /// Returns true if shape is statically ranked and all dimensions are
  /// known (ie non-negative).
  bool isStatic() const;

  /// Returns the rank.
  /// Requires hasRank().
  size_t getRank() const {
    assert(hasRank() && "tensor shape is dynamically ranked");
    return representation.rep16.rank;
  }

  /// Returns dimension at idx. The result may be negative to indicate
  /// 'dynamic' or 'unknown'. Requires hasRank().
  ssize_t operator[](size_t idx) const {
    assert(idx < getRank() && "idx is out of range for tensor shape rank");
    auto rep = getRepKind();
    if (rep == RepKind::k32) {
      assert(idx < 4 && "you can only fit 4 dimensions in k32");
      return idx != 3 ? representation.rep32.dims[idx]
                      : representation.rep32.dim3;
    }
    if (rep == RepKind::k16) {
      assert(idx < 6 && "you can only fit 6 dimensions in k16");
      return representation.rep16.dims[idx];
    }
    return representation.repOutOfLine.dims[idx];
  }

  // Provide access to the auxiliary storage.
  uint8_t getAuxiliary() const {
    static_assert(offsetof(Rep16, auxiliary) == offsetof(Rep32, auxiliary) &&
                      offsetof(Rep16, auxiliary) ==
                          offsetof(RepOutOfLine, auxiliary),
                  "Layout mismatch inside of TensorShape");
    // Because all of the representations store their auxiliary in the same
    // place, we can just access an arbitrary one.
    return representation.rep16.auxiliary;
  }
  void setAuxiliary(uint8_t value) { representation.rep16.auxiliary = value; }

  /// Provides random access iteration, but only a read-only version.
  /// Negative values denote 'dynamic' or 'unknown' dimensions.
  /// Requires hasRank().
  class iterator : public llvm::iterator_facade_base<
                       iterator, std::random_access_iterator_tag, ssize_t> {
  public:
    using Base =
        llvm::iterator_facade_base<iterator, std::random_access_iterator_tag,
                                   ssize_t>;

    iterator(const TensorShapeStorage *shape, size_t dimIdx)
        : shape(shape), dimIdx(dimIdx) {
      assert(shape->hasRank() && "tensor shape is dynamically ranked");
    }

    iterator &operator+=(Base::difference_type n) {
      dimIdx += n;
      return *this;
    }
    iterator &operator-=(Base::difference_type n) {
      dimIdx -= n;
      return *this;
    }
    Base::difference_type operator-(iterator rhs) const {
      assert(shape == rhs.shape && "iterators from different shapes!");
      return Base::difference_type(dimIdx - rhs.dimIdx);
    }
    bool operator==(const iterator &rhs) const {
      assert(shape == rhs.shape && "iterators from different shapes!");
      return dimIdx == rhs.dimIdx;
    }
    ssize_t operator*() const { return (*shape)[dimIdx]; }

  private:
    const TensorShapeStorage *shape;
    size_t dimIdx;
  };

  // We cannot support mutation through the iterator.
  using const_iterator = iterator;
  iterator begin() const { return iterator(this, 0); }
  iterator end() const { return iterator(this, getRank()); }

  /// Set shape to be dynamic ranked.
  void assignDynamic();

  /// Bulk reassignment of elements.
  /// TODO: Forcing dimensions to 64-bit is suboptimal on 32-bit hosts.
  void assign(ArrayRef<ssize_t> elements);

  bool equalsIncludingAux(const TensorShapeStorage &rhs) const {
    if (isOutOfLine())
      return equalsIncludingAuxOOL(rhs);
    return memcmp(&representation, &rhs.representation,
                  sizeof(representation)) == 0;
  }

  bool equalsExcludingAux(const TensorShapeStorage &rhs) const {
    if (isOutOfLine())
      return equalsExcludingAuxOOL(rhs);
    // The aux field is the last byte of the representation.
    return memcmp(&representation, &rhs.representation,
                  sizeof(representation) - 1) == 0;
  }

  RepKind getKind() const { return representation.rep16.kind; }

private:
  bool equalsIncludingAuxOOL(const TensorShapeStorage &rhs) const;
  bool equalsExcludingAuxOOL(const TensorShapeStorage &rhs) const;

  // Return the storage representation for this TensorShape.
  RepKind getRepKind() const {
    // Check the representations line up.
    static_assert(offsetof(Rep16, kind) == offsetof(Rep32, kind) &&
                      offsetof(Rep16, kind) == offsetof(RepOutOfLine, kind),
                  "Layout mismatch inside of TensorShape");
    // Because all of the representations store their kind in the same place,
    // we can just access an arbitrary one.
    return representation.rep16.kind;
  }

  bool isOutOfLine() const { return getRepKind() == RepKind::kOutOfLine; }
};

//===----------------------------------------------------------------------===//
// assign(...)
//===----------------------------------------------------------------------===//

// NOTE: These assignment helpers could live in TensorShapeStorage or
// TensorShape however older gcc versions complain.
template <typename IteratorType>
void assign(TensorShapeStorage &storage, IteratorType begin, IteratorType end,
            TensorRankStyle style) {
  switch (style) {
  case kStaticallyRanked:
    storage.assign(SmallVector<ssize_t, kMaxRank>(begin, end));
    break;
  case kDynamicallyRanked:
    assert(end == begin &&
           "dynamic ranked tensor shapes cannot have dimensions");
    storage.assignDynamic();
    break;
  }
}

template <typename ElementType>
void assign(TensorShapeStorage &storage, ArrayRef<ElementType> dims,
            TensorRankStyle style) {
  assign(storage, dims.begin(), dims.end(), style);
}

template <>
inline void assign<ssize_t>(TensorShapeStorage &storage, ArrayRef<ssize_t> dims,
                            TensorRankStyle style) {
  switch (style) {
  case kStaticallyRanked:
    storage.assign(dims);
    break;
  case kDynamicallyRanked:
    assert(dims.empty() &&
           "dynamic ranked tensor shapes cannot have dimensions");
    storage.assignDynamic();
    break;
  }
}

template <>
inline void assign<size_t>(TensorShapeStorage &storage, ArrayRef<size_t> dims,
                           TensorRankStyle style) {
  switch (style) {
  case kStaticallyRanked: {
    // Pointer cast to avoid copying the elements.
    ArrayRef<ssize_t> castedDims((const ssize_t *)dims.data(), dims.size());
    storage.assign(castedDims);
    break;
  }
  case kDynamicallyRanked:
    assert(dims.empty() &&
           "dynamic ranked tensor shapes cannot have dimensions");
    storage.assignDynamic();
    break;
  }
}

} // namespace Detail

//===----------------------------------------------------------------------===//
// TensorShape
//===----------------------------------------------------------------------===//

/// Represents the dimensions of a tensor.
///
/// TensorShapes may be used to represent both 'expected' and 'actual' shapes.
/// Expected shapes may have 'dynamic' or 'unknown' rank and dimensions. Actual
/// shapes should be fully known (ie actual.isStatic() is true), and should
/// refine their corresponding expected shape (ie expected.isRefinedBy(actual)
/// is true).
class TensorShape {
public:
  /// Constructs a tensor shape which is either dynamically ranked or has a
  /// static rank of zero.
  TensorShape(TensorRankStyle style = kStaticallyRanked) {
    if (style == kDynamicallyRanked)
      storage.assignDynamic();
  }

  /// Constructs a tensor shape from the given dimensions.
  ///
  /// If a dimension msb is set then it is taken to denote 'dynamic' or
  /// 'unknown'. Since unsigned 64-bit values are implicitly cast to signed,
  /// this is equivalent to any negative value denoting dynamic. However we do
  /// not support construction from uint32_ts since the meaning of the msb is
  /// ambiguous.
  ///
  /// If style is kDynamicRankStyle then the dimensions must be empty and the
  /// shape is taken to have a 'dynamic' or 'unknown' rank. Be aware many
  /// accessors assert that the shape hasRank().
  ///
  /// These constructors are defined explicitly (instead of as a template) so
  /// that implicit conversions from things like SmallVector will work.
  /*implicit*/ TensorShape(ArrayRef<int32_t> dims,
                           TensorRankStyle style = kStaticallyRanked) {
    assign(storage, dims, style);
  }
  /*implicit*/ TensorShape(ArrayRef<int64_t> dims,
                           TensorRankStyle style = kStaticallyRanked) {
    assign(storage, dims, style);
  }
  /*implicit*/ TensorShape(ArrayRef<uint64_t> dims,
                           TensorRankStyle style = kStaticallyRanked) {
    assign(storage, dims, style);
  }
#ifdef __APPLE__
  /*implicit*/ TensorShape(ArrayRef<size_t> dims,
                           TensorRankStyle style = kStaticallyRanked) {
    assign(storage, dims, style);
  }
  /*implicit*/ TensorShape(ArrayRef<ssize_t> dims,
                           TensorRankStyle style = kStaticallyRanked) {
    assign(storage, dims, style);
  }
#endif // __APPLE__

  template <typename ElementType,
            typename = std::enable_if_t<std::is_integral_v<ElementType>>>
  /*implicit*/ TensorShape(const std::initializer_list<ElementType> &dims,
                           TensorRankStyle style = kStaticallyRanked) {
    assign(storage, dims.begin(), dims.end(), style);
  }

  // Allow converting from a range of integer type, with elements that can be
  // converted to ssize_t.
  template <typename IteratorType>
  TensorShape(IteratorType begin, IteratorType end,
              TensorRankStyle style = kStaticallyRanked) {
    assign(storage, begin, end, style);
  }

  // This class has value semantics, implementing standard constructors,
  // assignment, copy construction etc.
  TensorShape(const TensorShape &) = default;
  TensorShape(TensorShape &&) = default;
  TensorShape &operator=(const TensorShape &) = default;
  TensorShape &operator=(TensorShape &&) = default;

  uint8_t getAuxiliaryStorage() const { return storage.getAuxiliary(); }
  void setAuxiliaryStorage(uint8_t value) { storage.setAuxiliary(value); }

  /// Returns true if shape is statically ranked.
  bool hasRank() const { return storage.hasRank(); }

  /// Returns true if shape is statically ranked and all dimensions are
  /// known (ie non-negative).
  bool isStatic() const { return storage.isStatic(); }

  /// Return the number of dimensions in this shape.
  /// Requires hasRank().
  size_t getRank() const { return storage.getRank(); }

  /// Return the underlying kind of the spec.
  /// Visible for testing only.
  uint8_t getKind() const { return (uint8_t)storage.getKind(); }

  /// Return the total number of elements in this tensor, which is the product
  /// of all the dimension sizes.
  /// Requires isStatic().
  size_t getNumElements() const {
    size_t result = 1;
    for (auto dim : *this) {
      assert(dim >= 0 && "attempting to get the number of elements of a "
                         "TensorSpec with unknown dimensions");
      result *= static_cast<size_t>(dim);
    }
    return result;
  }

  /// Returns success if argument tensor shape is a refinement of this tensor
  /// shape. That is, the two shapes agree on all static parts, and this tensor
  /// may have dynamic parts. Returns description of mismatch as Error
  /// otherwise.
  /// Requires staticShape.isStatic().
  ErrorOrSuccess isRefinedBy(const TensorShape &staticShape) const;

  // Support the typical iteration and subscripting operations.
  // Requires hasRank().
  using iterator = typename Detail::TensorShapeStorage::iterator;
  iterator begin() { return storage.begin(); }
  iterator end() { return storage.end(); }
  using const_iterator = typename Detail::TensorShapeStorage::const_iterator;
  const_iterator begin() const { return storage.begin(); }
  const_iterator end() const { return storage.end(); }

  ssize_t operator[](size_t i) const { return storage[i]; }

  /// Return the dimensions as a temporary, unpacked SmallVector.
  /// Requires hasRank().
  ///
  /// CAUTION: You probably don't need to call this! TensorShape supports
  /// indexing, iteration and getRank directly.
  SmallVector<ssize_t, kMaxRank> getDimsCopy() const {
    return SmallVector<ssize_t, kMaxRank>(begin(), end());
  }

  bool operator==(const TensorShape &rhs) const {
    return storage.equalsExcludingAux(rhs.storage);
  }
  bool operator!=(const TensorShape &rhs) const { return !(*this == rhs); }

  std::string getAsString() const;
  void print(raw_ostream &os) const;

  /// Parses a string of the form "dim0xdim1x...xdimN" or "*" into a
  /// TensorShape.
  static ErrorOr<TensorShape> parseFromString(StringRef);

protected:
  Detail::TensorShapeStorage storage;
};

static_assert(sizeof(TensorShape) == 16, "TensorShape should not grow");

inline raw_ostream &operator<<(raw_ostream &os, const TensorShape &value) {
  value.print(os);
  return os;
}

} // namespace M

LLVM_FWD_YAML_DECLARE_SCALAR_TRAITS(M::TensorShape)

#endif // SUPPORT_ML_TENSORSHAPE_H
