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

from std.os import abort

from std.builtin.sort import _quicksort


# DO NOT CHANGE
trait TuningConfig(TrivialRegisterPassable, Writable):
    ...


# DO NOT CHANGE
struct Table[type: TuningConfig](Writable):
    var configs: List[Self.type]
    var name: String
    var num_configs: Int

    def __init__(out self, configs: List[Self.type], name: String):
        self.configs = configs.copy()
        self.name = name
        self.num_configs = len(configs)

        if not self.check():
            abort(t"Failed to Compile Table: [{self.name}]")

    # Method to check there are no redundancies in table (based on __str__).
    def check(self) -> Bool:
        var keys = List[String]()
        var is_valid = True

        for i in range(len(self.configs)):
            var cfg = self.configs[i]
            var res = String(cfg)
            if res in keys:
                print(
                    "ERROR: Redundant Entry [",
                    self.name,
                    "][",
                    i,
                    "] ",
                    cfg,
                    sep="",
                )
                is_valid = False
                continue
            keys.append(res)
        return is_valid

    def write_to(self, mut writer: Some[Writer]):
        """Writes the table as a string.

        Args:
            writer: The writer to write to.
        """
        writer.write_string(self.name)
        for i in range(len(self.configs)):
            var cfg = self.configs[i]
            t"\n[{i}] {cfg}".write_to(writer)

    # Method `query_index` queries a unique list of values for each parameter.
    # Find the indices of all matching values in the list.
    # Notes:
    #   - `domain` is a list of indices to narrow down the search.
    #     These indices are marked valid in the flag and may not represent the entire domain.
    #   - Returns a list of matching indices, not the entire domain.
    def query_index[
        rule_fn: ImplicitlyCopyable & def(Self.type) -> Bool,
        domain: List[Int] = List[Int](),
    ](self, *, rule: rule_fn) -> List[Int]:
        var flag: List[Bool]

        comptime if len(domain):
            flag = List[Bool](length=self.num_configs, fill=False)
            for idx in materialize[domain]():
                flag[idx] = True
        else:
            flag = List[Bool](length=self.num_configs, fill=True)

        for i in range(self.num_configs):
            flag[i] &= rule(self.configs[i])
        var result_idx_list = List[Int]()

        for i in range(self.num_configs):
            if flag[i]:
                result_idx_list.append(i)
        return result_idx_list^

    # Apply rule on all configs in the table and return list of all the unique results.
    def query_values[
        ret_type: Comparable & ImplicitlyCopyable & Deinitable,
        rule_fn: ImplicitlyCopyable & def(Self.type) -> ret_type,
        domain: List[Int] = List[Int](),
    ](self, *, rule: rule_fn) -> List[ret_type]:
        var result = List[ret_type]()

        @always_inline
        def _get_search_domain() {imm} -> List[Int]:
            if len(materialize[domain]()):
                return materialize[domain]()
            else:
                return [idx for idx in range(self.num_configs)]

        var search_domain = _get_search_domain()

        for idx in search_domain:
            var value = rule(self.configs[idx])
            if value not in result:
                result.append(value)

        def _cmp(lsh: ret_type, rhs: ret_type) -> Bool:
            return lsh < rhs

        _quicksort(result, _cmp)
        return result^

    def find[
        rule_fn: ImplicitlyCopyable & def(Self.type) -> Bool,
    ](self, *, rule: rule_fn) -> List[Self.type]:
        var result = List[Self.type]()

        for config in self.configs:
            if rule(config):
                result.append(config)

        return result^
