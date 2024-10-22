# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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


class Simple:
    def __init__(self):
        pass


class WithGetItem:
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return "Keys: {0}".format(", ".join(map(str, key)))
        else:
            return "Key: {0}".format(key)


class WithGetItemException:
    def __getitem__(self, key):
        raise ValueError("Custom error")


class With2DGetItem:
    def __init__(self):
        self.data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def __getitem__(self, key):
        if isinstance(key, tuple) and all(isinstance(k, slice) for k in key):
            return [row[key[1]] for row in self.data[key[0]]]
        elif isinstance(key, tuple):
            return self.data[key[0]][key[1]]
        else:
            return self.data[key]


class Sliceable:
    def __getitem__(self, key):
        return key
