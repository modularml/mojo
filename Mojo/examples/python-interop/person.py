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

# DOC: Mojo/docs/site/manual/python/mojo-from-python.mdx

# The Mojo importer module will handle compilation of the Mojo files.
import mojo.importer  # noqa: F401, I001

import person_module  # type: ignore

person = person_module.Person("Sarah", 32)

print(person)
