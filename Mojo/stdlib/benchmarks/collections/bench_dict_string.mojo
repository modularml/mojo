# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.collections.string.string_span import _to_string_list
from std.hashlib import default_comp_time_hasher, default_hasher
from std.os import abort
from std.pathlib import _dir_of_current_file
from std.sys import stderr

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    Format,
    Unit,
    keep,
    run,
)
from std.memory import (
    Allocation,
    dealloc,
    unsafe_memcpy,
    unsafe_memset_zero,
)
from std.testing import assert_equal


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
def make_small_keys(filename: String = "UN_charter_EN.txt") -> List[String]:
    """Make a `String` made of items in the `./data` directory.

    Args:
        filename: The name of the file inside the `./data` directory.
    """

    try:
        var directory = _dir_of_current_file() / "data"
        var f = open(directory / filename, "r")
        var content = f.read()
        return _to_string_list(content.split())
    except e:
        print(e, file=stderr)
    abort()


# ===-----------------------------------------------------------------------===#
# Long Key Data
# ===-----------------------------------------------------------------------===#
def make_long_keys(filename: String = "UN_charter_EN.txt") -> List[String]:
    """Make a `String` made of items in the `./data` directory.

    Args:
        filename: The name of the file inside the `./data` directory.
    """

    try:
        var directory = _dir_of_current_file() / "data"
        var f = open(directory / filename, "r")
        var content = f.read()
        return _to_string_list(content.split("\n"))
    except e:
        print(e, file=stderr)
    abort()


# ===-----------------------------------------------------------------------===#
# String Dict implementation for benchmarking baseline against Dict
# ===-----------------------------------------------------------------------===#

from std.bit import bit_width, pop_count


struct KeysContainer[KeyEndType: DType = .uint32](ImplicitlyCopyable, Sized):
    var keys: Pointer[UInt8, MutUntrackedOrigin]
    var allocated_bytes: Int
    var keys_end: Pointer[Scalar[Self.KeyEndType], MutUntrackedOrigin]
    var count: Int
    var capacity: Int

    def __init__(out self, capacity: Int):
        comptime assert (
            Self.KeyEndType == .uint8
            or Self.KeyEndType == .uint16
            or Self.KeyEndType == .uint32
            or Self.KeyEndType == .uint64
        ), "KeyEndType needs to be an unsigned integer"
        self.allocated_bytes = capacity << 3
        self.keys = alloc[UInt8]({count = self.allocated_bytes}).unsafe_leak()
        self.keys_end = alloc[Scalar[Self.KeyEndType]](
            {count = capacity}
        ).unsafe_leak()
        self.count = 0
        self.capacity = capacity

    def __init__(out self, *, copy: Self):
        self.allocated_bytes = copy.allocated_bytes
        self.count = copy.count
        self.capacity = copy.capacity
        self.keys = alloc[UInt8]({count = self.allocated_bytes}).unsafe_leak()
        unsafe_memcpy(dest=self.keys, src=copy.keys, count=self.allocated_bytes)
        self.keys_end = alloc[Scalar[Self.KeyEndType]](
            {count = self.capacity}
        ).unsafe_leak()
        unsafe_memcpy(
            dest=self.keys_end, src=copy.keys_end, count=self.capacity
        )

    def __deinit__(deinit self):
        dealloc(
            Allocation(
                unsafe_owned_ptr=self.keys,
                layout={count = self.allocated_bytes},
            )
        )
        dealloc(
            Allocation(
                unsafe_owned_ptr=self.keys_end, layout={count = self.capacity}
            )
        )

    @always_inline
    def add(mut self, key: StringSlice):
        var prev_end = (
            0 if self.count
            == 0 else self.keys_end[unsafe_offset=self.count - 1]
        )
        var key_length = key.byte_length()
        var new_end = prev_end + Scalar[Self.KeyEndType](key_length)

        var old_allocated_bytes = self.allocated_bytes
        var needs_realocation = False
        while new_end > Scalar[Self.KeyEndType](self.allocated_bytes):
            self.allocated_bytes += self.allocated_bytes >> 1
            needs_realocation = True

        if needs_realocation:
            var keys = alloc[UInt8](
                {count = self.allocated_bytes}
            ).unsafe_leak()
            unsafe_memcpy(dest=keys, src=self.keys, count=Int(prev_end))
            dealloc(
                Allocation(
                    unsafe_owned_ptr=self.keys,
                    layout={count = old_allocated_bytes},
                )
            )
            self.keys = keys

        unsafe_memcpy(
            dest=self.keys.unsafe_offset(prev_end),
            src=Pointer(key.unsafe_ptr()),
            count=key_length,
        )
        var count = self.count + 1
        if count >= self.capacity:
            var new_capacity = self.capacity + (self.capacity >> 1)
            var keys_end = alloc[Scalar[Self.KeyEndType]](
                {count = new_capacity}
            ).unsafe_leak()
            unsafe_memcpy(dest=keys_end, src=self.keys_end, count=self.capacity)
            dealloc(
                Allocation(
                    unsafe_owned_ptr=self.keys_end,
                    layout={count = self.capacity},
                )
            )
            self.keys_end = keys_end
            self.capacity = new_capacity

        self.keys_end.unsafe_store(self.count, new_end)
        self.count = count

    @always_inline
    def get(self, index: Int) -> StringSlice[ImmOrigin(origin_of(self))]:
        var keys_ptr = self.keys.as_imm().unsafe_origin_cast[origin_of(self)]()
        if index < 0 or index >= self.count:
            return StringSlice(
                unsafe_from_utf8=Span(unsafe_ptr=keys_ptr, length=0)
            )
        var start = 0 if index == 0 else Int(
            self.keys_end[unsafe_offset=index - 1]
        )
        var length = Int(self.keys_end[unsafe_offset=index]) - start
        return StringSlice(
            unsafe_from_utf8=Span(
                unsafe_ptr=keys_ptr.unsafe_offset(start), length=length
            )
        )

    @always_inline
    def clear(mut self):
        self.count = 0

    @always_inline
    def __getitem__(
        self, index: Int
    ) -> StringSlice[ImmOrigin(origin_of(self))]:
        return self.get(index)

    @always_inline
    def __len__(self) -> Int:
        return self.count

    def keys_vec(self, out result: List[StringSlice[origin_of(self)]]):
        var keys = type_of(result)(capacity=self.count)
        for i in range(self.count):
            keys.append(self[i])
        return keys^

    def print_keys(self):
        print("(" + String(self.count) + ")[", end="")
        for i in range(self.count):
            var end = ", " if i < self.count - 1 else ""
            print(self[i], end=end)
        print("]")


struct StringDict[
    V: Copyable & Deinitable,
    KeyCountType: DType = .uint32,
    KeyOffsetType: DType = .uint32,
    destructive: Bool = True,
    caching_hashes: Bool = True,
](Sized):
    var keys: KeysContainer[Self.KeyOffsetType]
    var key_hashes: Pointer[Scalar[Self.KeyCountType], MutUntrackedOrigin]
    var values: List[Self.V]
    var slot_to_index: Pointer[Scalar[Self.KeyCountType], MutUntrackedOrigin]
    var deleted_mask: Pointer[UInt8, MutUntrackedOrigin]
    var count: Int
    var capacity: Int

    def __init__(out self, capacity: Int = 16):
        comptime assert (
            Self.KeyCountType == .uint8
            or Self.KeyCountType == .uint16
            or Self.KeyCountType == .uint32
            or Self.KeyCountType == .uint64
        ), "KeyCountType needs to be an unsigned integer"
        self.count = 0
        if capacity <= 8:
            self.capacity = 8
        else:
            var icapacity = Int64(capacity)
            self.capacity = capacity if pop_count(icapacity) == 1 else 1 << Int(
                bit_width(icapacity)
            )
        self.keys = KeysContainer[Self.KeyOffsetType](capacity)

        comptime if Self.caching_hashes:
            self.key_hashes = alloc[Scalar[Self.KeyCountType]](
                {count = self.capacity}
            ).unsafe_leak()
        else:
            self.key_hashes = alloc[Scalar[Self.KeyCountType]](
                {count = 0}
            ).unsafe_leak()
        self.values = List[Self.V](capacity=capacity)
        self.slot_to_index = alloc[Scalar[Self.KeyCountType]](
            {count = self.capacity}
        ).unsafe_leak()
        unsafe_memset_zero(self.slot_to_index, self.capacity)

        comptime if Self.destructive:
            self.deleted_mask = alloc[UInt8](
                {count = self.capacity >> 3}
            ).unsafe_leak()
            unsafe_memset_zero(self.deleted_mask, self.capacity >> 3)
        else:
            self.deleted_mask = alloc[UInt8]({count = 0}).unsafe_leak()

    def __init__(out self, *, copy: Self):
        self.count = copy.count
        self.capacity = copy.capacity
        self.keys = copy.keys

        comptime if Self.caching_hashes:
            self.key_hashes = alloc[Scalar[Self.KeyCountType]](
                {count = self.capacity}
            ).unsafe_leak()
            unsafe_memcpy(
                dest=self.key_hashes,
                src=copy.key_hashes,
                count=self.capacity,
            )
        else:
            self.key_hashes = type_of(self.key_hashes).unsafe_dangling()
        self.values = copy.values.copy()
        self.slot_to_index = alloc[Scalar[Self.KeyCountType]](
            {count = self.capacity}
        ).unsafe_leak()
        unsafe_memcpy(
            dest=self.slot_to_index,
            src=copy.slot_to_index,
            count=self.capacity,
        )

        comptime if Self.destructive:
            self.deleted_mask = alloc[UInt8](
                {count = self.capacity >> 3}
            ).unsafe_leak()
            unsafe_memcpy(
                dest=self.deleted_mask,
                src=copy.deleted_mask,
                count=self.capacity >> 3,
            )
        else:
            self.deleted_mask = type_of(self.deleted_mask).unsafe_dangling()

    def __deinit__(deinit self):
        dealloc(
            Allocation(
                unsafe_owned_ptr=self.slot_to_index,
                layout={count = self.capacity},
            )
        )
        comptime if Self.destructive:
            dealloc(
                Allocation(
                    unsafe_owned_ptr=self.deleted_mask,
                    layout={count = self.capacity >> 3},
                )
            )
        comptime if Self.caching_hashes:
            dealloc(
                Allocation(
                    unsafe_owned_ptr=self.key_hashes,
                    layout={count = self.capacity},
                )
            )

    def __len__(self) -> Int:
        return self.count

    @always_inline
    def __contains__(self, key: StringSlice) -> Bool:
        return self._find_key_index(key) != 0

    def put(mut self, key: StringSlice, value: Self.V):
        if self.count >= self.capacity - (self.capacity >> 3):
            self._rehash()

        var key_hash = hash(key).cast[Self.KeyCountType]()
        var modulo_mask = self.capacity - 1
        var slot = Int(key_hash & Scalar[Self.KeyCountType](modulo_mask))
        while True:
            var key_index = Int(self.slot_to_index.unsafe_load(slot))
            if key_index == 0:
                self.keys.add(key)

                comptime if Self.caching_hashes:
                    self.key_hashes.unsafe_store(slot, key_hash)
                self.values.append(value.copy())
                self.count += 1
                self.slot_to_index.unsafe_store(
                    slot, Scalar[Self.KeyCountType](self.keys.count)
                )
                return

            comptime if Self.caching_hashes:
                var other_key_hash = self.key_hashes[unsafe_offset=slot]
                if other_key_hash == key_hash:
                    var other_key = self.keys[key_index - 1]
                    if other_key == key:
                        # replace value
                        self.values[key_index - 1] = value.copy()

                        comptime if Self.destructive:
                            if self._is_deleted(key_index - 1):
                                self.count += 1
                                self._not_deleted(key_index - 1)
                        return
            else:
                var other_key = self.keys[key_index - 1]
                if other_key == key:
                    # replace value
                    self.values[key_index - 1] = value.copy()

                    comptime if Self.destructive:
                        if self._is_deleted(key_index - 1):
                            self.count += 1
                            self._not_deleted(key_index - 1)
                    return

            slot = (slot + 1) & modulo_mask

    @always_inline
    def _is_deleted(self, index: Int) -> Bool:
        var offset = index >> 3
        var bit_index = index & 7
        return (
            self.deleted_mask.unsafe_offset(offset).unsafe_load()
            & UInt8(1 << bit_index)
            != 0
        )

    @always_inline
    def _deleted(self, index: Int):
        var offset = index >> 3
        var bit_index = index & 7
        var p = self.deleted_mask.unsafe_offset(offset)
        var mask = p.unsafe_load()
        p.unsafe_store(mask | UInt8((1 << bit_index)))

    @always_inline
    def _not_deleted(self, index: Int):
        var offset = index >> 3
        var bit_index = index & 7
        var p = self.deleted_mask.unsafe_offset(offset)
        var mask = p.unsafe_load()
        p.unsafe_store(mask & UInt8(~(1 << bit_index)))

    @always_inline
    def _rehash(mut self):
        var old_slot_to_index = self.slot_to_index
        var old_capacity = self.capacity
        self.capacity <<= 1
        var mask_capacity = self.capacity >> 3
        self.slot_to_index = alloc[Scalar[Self.KeyCountType]](
            {count = self.capacity}
        ).unsafe_leak()
        unsafe_memset_zero(self.slot_to_index, self.capacity)

        var key_hashes = self.key_hashes

        comptime if Self.caching_hashes:
            key_hashes = alloc[Scalar[Self.KeyCountType]](
                {count = self.capacity}
            ).unsafe_leak()

        comptime if Self.destructive:
            var deleted_mask = alloc[UInt8](
                {count = mask_capacity}
            ).unsafe_leak()
            unsafe_memset_zero(deleted_mask, mask_capacity)
            unsafe_memcpy(
                dest=deleted_mask,
                src=self.deleted_mask,
                count=old_capacity >> 3,
            )
            dealloc(
                Allocation(
                    unsafe_owned_ptr=self.deleted_mask,
                    layout={count = old_capacity >> 3},
                )
            )
            self.deleted_mask = deleted_mask

        var modulo_mask = self.capacity - 1
        for i in range(old_capacity):
            if old_slot_to_index[unsafe_offset=i] == 0:
                continue
            var key_hash: Scalar[Self.KeyCountType]

            comptime if Self.caching_hashes:
                key_hash = self.key_hashes[unsafe_offset=i]
            else:
                key_hash = hash(
                    self.keys[Int(old_slot_to_index[unsafe_offset=i] - 1)]
                ).cast[Self.KeyCountType]()

            var slot = Int(key_hash & Scalar[Self.KeyCountType](modulo_mask))

            # var searching = True
            while True:
                var key_index = Int(self.slot_to_index.unsafe_load(slot))

                if key_index == 0:
                    self.slot_to_index.unsafe_store(
                        slot, old_slot_to_index[unsafe_offset=i]
                    )
                    break
                    # searching = False

                else:
                    slot = (slot + 1) & modulo_mask

            comptime if Self.caching_hashes:
                key_hashes[unsafe_offset=slot] = key_hash

        comptime if Self.caching_hashes:
            dealloc(
                Allocation(
                    unsafe_owned_ptr=self.key_hashes,
                    layout={count = old_capacity},
                )
            )
            self.key_hashes = key_hashes
        dealloc(
            Allocation(
                unsafe_owned_ptr=old_slot_to_index,
                layout={count = old_capacity},
            )
        )

    def get(self, key: StringSlice, default: Self.V) -> Self.V:
        var key_index = self._find_key_index(key)
        if key_index == 0:
            return default.copy()

        comptime if Self.destructive:
            if self._is_deleted(key_index - 1):
                return default.copy()
        return self.values[key_index - 1].copy()

    def delete(mut self, key: StringSlice):
        comptime if not Self.destructive:
            return

        var key_index = self._find_key_index(key)
        if key_index == 0:
            return
        if not self._is_deleted(key_index - 1):
            self.count -= 1
        self._deleted(key_index - 1)

    def upsert(
        mut self,
        key: StringSlice,
        update: def(value: Optional[Self.V]) thin -> Self.V,
    ):
        var key_index = self._find_key_index(key)
        if key_index == 0:
            var value = update(None)
            self.put(key, value)
        else:
            key_index -= 1

            comptime if Self.destructive:
                if self._is_deleted(key_index):
                    self.count += 1
                    self._not_deleted(key_index)
                    self.values[key_index] = update(None)
                    return

            self.values[key_index] = update(self.values[key_index].copy())

    def clear(mut self):
        self.values.clear()
        self.keys.clear()
        unsafe_memset_zero(self.slot_to_index, self.capacity)

        comptime if Self.destructive:
            unsafe_memset_zero(self.deleted_mask, self.capacity >> 3)
        self.count = 0

    @always_inline
    def _find_key_index(self, key: StringSlice) -> Int:
        var key_hash = hash(key).cast[Self.KeyCountType]()
        var modulo_mask = self.capacity - 1

        var slot = Int(key_hash & Scalar[Self.KeyCountType](modulo_mask))
        while True:
            var key_index = Int(self.slot_to_index.unsafe_load(slot))
            if key_index == 0:
                return key_index

            comptime if Self.caching_hashes:
                var other_key_hash = self.key_hashes[unsafe_offset=slot]
                if key_hash == other_key_hash:
                    var other_key = self.keys[key_index - 1]
                    if other_key == key:
                        return key_index
            else:
                var other_key = self.keys[key_index - 1]
                if other_key == key:
                    return key_index

            slot = (slot + 1) & modulo_mask


# ===-----------------------------------------------------------------------===#
# Benchmark Dict init
# ===-----------------------------------------------------------------------===#
def bench_dict_init_with_short_keys[file_name: String](mut b: Bencher) raises:
    var keys = make_small_keys(file_name)

    @always_inline
    def call_fn() {imm}:
        var d = Dict[String, Int]()
        for i, key in enumerate(keys):
            d[key] = i
        keep(d._table._ctrl)

    b.iter(call_fn)


def bench_dict_init_with_long_keys[file_name: String](mut b: Bencher) raises:
    var keys = make_long_keys(file_name)

    @always_inline
    def call_fn() {imm}:
        var d = Dict[String, Int, default_hasher]()
        for i, key in enumerate(keys):
            d[key] = i
        keep(d._table._ctrl)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark StringDict init
# ===-----------------------------------------------------------------------===#
def bench_string_dict_init_with_short_keys[
    file_name: String
](mut b: Bencher) raises:
    var keys = make_small_keys(file_name)

    @always_inline
    def call_fn() {imm}:
        var d = StringDict[Int]()
        for i, key in enumerate(keys):
            d.put(key, i)
        keep(d.keys.keys)

    b.iter(call_fn)


def bench_string_dict_init_with_long_keys[
    file_name: String
](mut b: Bencher) raises:
    var keys = make_long_keys(file_name)

    @always_inline
    def call_fn() {imm}:
        var d = StringDict[Int]()
        for i, key in enumerate(keys):
            d.put(key, i)
        keep(d.keys.keys)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Validate StringDict against Dict
# ===-----------------------------------------------------------------------===#


def validate_dicts(
    file_name: String = "UN_charter_EN.txt", small_keys: Bool = True
) raises:
    var keys = make_small_keys(file_name) if small_keys else make_long_keys(
        file_name
    )
    print(
        "Number of keys:",
        len(keys),
        "small" if small_keys else "long",
        file_name,
    )
    var d = Dict[String, Int]()
    for i, key in enumerate(keys):
        d[key] = i

    var sd = StringDict[Int]()
    for i, key in enumerate(keys):
        sd.put(key, i)

    assert_equal(len(d), len(sd), "Length mismatch between Dict and StringDict")
    print("Length match between Dict and StringDict", len(d))


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    validate_dicts("UN_charter_EN.txt", small_keys=True)
    validate_dicts("UN_charter_EN.txt", small_keys=False)
    validate_dicts("UN_charter_AR.txt", small_keys=True)
    validate_dicts("UN_charter_AR.txt", small_keys=False)
    validate_dicts("UN_charter_ES.txt", small_keys=True)
    validate_dicts("UN_charter_ES.txt", small_keys=False)
    validate_dicts("UN_charter_RU.txt", small_keys=True)
    validate_dicts("UN_charter_RU.txt", small_keys=False)
    validate_dicts("UN_charter_zh-CN.txt", small_keys=True)
    validate_dicts("UN_charter_zh-CN.txt", small_keys=False)

    var m = Bench(
        BenchConfig(
            # out_file=_dir_of_current_file() / "bench_dict_string.csv",
            num_repetitions=5,
        )
    )
    m.bench_function(
        bench_dict_init_with_short_keys["UN_charter_EN.txt"],
        BenchId("bench_dict_init_with_short_keys EN"),
    )
    m.bench_function(
        bench_dict_init_with_short_keys["UN_charter_AR.txt"],
        BenchId("bench_dict_init_with_short_keys AR"),
    )
    m.bench_function(
        bench_dict_init_with_short_keys["UN_charter_ES.txt"],
        BenchId("bench_dict_init_with_short_keys ES"),
    )
    m.bench_function(
        bench_dict_init_with_short_keys["UN_charter_RU.txt"],
        BenchId("bench_dict_init_with_short_keys RU"),
    )
    m.bench_function(
        bench_dict_init_with_short_keys["UN_charter_zh-CN.txt"],
        BenchId("bench_dict_init_with_short_keys zh-CN"),
    )
    m.bench_function(
        bench_dict_init_with_long_keys["UN_charter_EN.txt"],
        BenchId("bench_dict_init_with_long_keys EN"),
    )
    m.bench_function(
        bench_dict_init_with_long_keys["UN_charter_AR.txt"],
        BenchId("bench_dict_init_with_long_keys AR"),
    )
    m.bench_function(
        bench_dict_init_with_long_keys["UN_charter_ES.txt"],
        BenchId("bench_dict_init_with_long_keys ES"),
    )
    m.bench_function(
        bench_dict_init_with_long_keys["UN_charter_RU.txt"],
        BenchId("bench_dict_init_with_long_keys RU"),
    )
    m.bench_function(
        bench_dict_init_with_long_keys["UN_charter_zh-CN.txt"],
        BenchId("bench_dict_init_with_long_keys zh-CN"),
    )

    m.bench_function(
        bench_string_dict_init_with_short_keys["UN_charter_EN.txt"],
        BenchId("bench_string_dict_init_with_short_keys EN"),
    )
    m.bench_function(
        bench_string_dict_init_with_short_keys["UN_charter_AR.txt"],
        BenchId("bench_string_dict_init_with_short_keys AR"),
    )
    m.bench_function(
        bench_string_dict_init_with_short_keys["UN_charter_ES.txt"],
        BenchId("bench_string_dict_init_with_short_keys ES"),
    )
    m.bench_function(
        bench_string_dict_init_with_short_keys["UN_charter_RU.txt"],
        BenchId("bench_string_dict_init_with_short_keys RU"),
    )
    m.bench_function(
        bench_string_dict_init_with_short_keys["UN_charter_zh-CN.txt"],
        BenchId("bench_string_dict_init_with_short_keys zh-CN"),
    )
    m.bench_function(
        bench_string_dict_init_with_long_keys["UN_charter_EN.txt"],
        BenchId("bench_string_dict_init_with_long_keys EN"),
    )
    m.bench_function(
        bench_string_dict_init_with_long_keys["UN_charter_AR.txt"],
        BenchId("bench_string_dict_init_with_long_keys AR"),
    )
    m.bench_function(
        bench_string_dict_init_with_long_keys["UN_charter_ES.txt"],
        BenchId("bench_string_dict_init_with_long_keys ES"),
    )
    m.bench_function(
        bench_string_dict_init_with_long_keys["UN_charter_RU.txt"],
        BenchId("bench_string_dict_init_with_long_keys RU"),
    )
    m.bench_function(
        bench_string_dict_init_with_long_keys["UN_charter_zh-CN.txt"],
        BenchId("bench_string_dict_init_with_long_keys zh-CN"),
    )

    m.dump_report()
