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

from max.runtime.tracing import Trace, TraceLevel


def test_trace_context_with_dynamic_name() raises:
    # Exercise the Trace context manager to ensure no crashes and proper begin/end
    # against the Tracy bridge. We cannot assert on UI-visible names here, but
    # this covers the dynamic path used by __enter__/__exit__.
    with Trace[level=TraceLevel.OP](name="trace_context_dynamic"):
        pass

    # Also exercise start()/end() explicitly.
    var tr = Trace[level=TraceLevel.OP](name="trace_start_end_dynamic")
    tr.start()
    tr.end()


def main() raises:
    test_trace_context_with_dynamic_name()
