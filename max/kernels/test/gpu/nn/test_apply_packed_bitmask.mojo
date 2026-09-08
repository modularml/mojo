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

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from nn.gather_scatter import apply_packed_bitmask
from std.utils.numerics import neg_inf


def test_apply_packed_bitmask(ctx: DeviceContext) raises:
    # batch=2, vocab=40 -> packed_vocab = ceil(40 / 32) = 2 words/row.
    comptime batch = 2
    comptime vocab = 40
    comptime packed_vocab = 2
    comptime fill_value: Float32 = -10000.0

    # logits[b, v] = b * 100 + v, so every kept value is unique and checkable.
    var logits_stack = Array[Float32, batch * vocab](fill={})
    var logits = TileTensor(logits_stack, row_major[batch, vocab]())
    for b in range(batch):
        for v in range(vocab):
            logits[b, v] = Float32(b * 100 + v)

    # Build a packed bitmask by setting an explicit set of valid tokens, then
    # derive the expected masked logits from the same source of truth.
    var valid = Array[Bool, batch * vocab](fill=False)
    var packed_stack = Array[Int32, batch * packed_vocab](fill=0)
    var packed = TileTensor(packed_stack, row_major[batch, packed_vocab]())

    # Row 0: a spread of tokens incl. ones that cross the 32-bit word boundary.
    for v in [0, 1, 31, 32, 33, 39]:
        valid[0 * vocab + v] = True
    # Row 1: a different set, including the very last token.
    for v in [5, 7, 30, 38, 39]:
        valid[1 * vocab + v] = True

    for b in range(batch):
        for v in range(vocab):
            if valid[b * vocab + v]:
                packed[b, v >> 5] |= Int32(1) << Int32(v & 31)

    # Copy inputs to device.
    var logits_gpu_buf = ctx.enqueue_create_buffer[.float32](batch * vocab)
    ctx.enqueue_copy(logits_gpu_buf, logits._storage)
    var logits_gpu = TileTensor(logits_gpu_buf, row_major[batch, vocab]())

    var packed_gpu_buf = ctx.enqueue_create_buffer[.int32](batch * packed_vocab)
    ctx.enqueue_copy(packed_gpu_buf, packed._storage)
    var packed_gpu = TileTensor(
        packed_gpu_buf, row_major[batch, packed_vocab]()
    )

    var out_gpu_buf = ctx.enqueue_create_buffer[.float32](batch * vocab)
    var out_gpu = TileTensor(out_gpu_buf, row_major[batch, vocab]())

    apply_packed_bitmask[target="gpu"](
        out_gpu, logits_gpu, packed_gpu, fill_value, ctx
    )

    var out_stack = Array[Float32, batch * vocab](fill={})
    ctx.enqueue_copy(Span(out_stack), out_gpu_buf)
    ctx.synchronize()
    var out = TileTensor(out_stack, row_major[batch, vocab]())

    for b in range(batch):
        for v in range(vocab):
            var expected = Float32(b * 100 + v) if valid[
                b * vocab + v
            ] else fill_value
            if out[b, v] != expected:
                raise Error(
                    "out[",
                    b,
                    ", ",
                    v,
                    "] = ",
                    out[b, v],
                    " != ",
                    expected,
                )


def test_neg_inf_survives_the_mask(ctx: DeviceContext) raises:
    """A logit the model set to `-inf` must never take the finite fill.

    An architecture whose `lm_head` is wider than its tokenizer masks the
    untrained tail to `-inf` so those ids can never be sampled. A grammar
    knows nothing about that padding, so it marks the tail invalid and the
    finite fill would otherwise write `-10000` over the `-inf` -- making the
    ids reachable again on any row whose surviving logits sit near the fill.
    """
    comptime batch = 2
    comptime vocab = 40
    comptime packed_vocab = 2
    comptime fill_value: Float32 = -10000.0
    # Mirrors a padded vocabulary: ids at or above this are untrained.
    comptime unpadded = 37

    var logits_stack = Array[Float32, batch * vocab](fill={})
    var logits = TileTensor(logits_stack, row_major[batch, vocab]())
    for b in range(batch):
        for v in range(vocab):
            logits[b, v] = neg_inf[.float32]() if v >= unpadded else Float32(
                b * 100 + v
            )

    # Row 0: a grammar that allows a few real tokens and, as any grammar
    # would, none of the padded tail. Row 1: fully masked -- the row that
    # turns the finite fill into a uniform draw over the whole vocabulary.
    var valid = Array[Bool, batch * vocab](fill=False)
    for v in [0, 5, 31, 36]:
        valid[0 * vocab + v] = True

    var packed_stack = Array[Int32, batch * packed_vocab](fill=0)
    var packed = TileTensor(packed_stack, row_major[batch, packed_vocab]())
    for b in range(batch):
        for v in range(vocab):
            if valid[b * vocab + v]:
                packed[b, v >> 5] |= Int32(1) << Int32(v & 31)

    var logits_gpu_buf = ctx.enqueue_create_buffer[.float32](batch * vocab)
    ctx.enqueue_copy(logits_gpu_buf, logits._storage)
    var logits_gpu = TileTensor(logits_gpu_buf, row_major[batch, vocab]())

    var packed_gpu_buf = ctx.enqueue_create_buffer[.int32](batch * packed_vocab)
    ctx.enqueue_copy(packed_gpu_buf, packed._storage)
    var packed_gpu = TileTensor(
        packed_gpu_buf, row_major[batch, packed_vocab]()
    )

    var out_gpu_buf = ctx.enqueue_create_buffer[.float32](batch * vocab)
    var out_gpu = TileTensor(out_gpu_buf, row_major[batch, vocab]())

    apply_packed_bitmask[target="gpu"](
        out_gpu, logits_gpu, packed_gpu, fill_value, ctx
    )

    var out_stack = Array[Float32, batch * vocab](fill={})
    ctx.enqueue_copy(Span(out_stack), out_gpu_buf)
    ctx.synchronize()
    var out = TileTensor(out_stack, row_major[batch, vocab]())

    for b in range(batch):
        for v in range(vocab):
            var expected: Float32
            if v >= unpadded:
                # Masked by the model: stays excluded whatever the grammar says.
                expected = neg_inf[.float32]()
            elif valid[b * vocab + v]:
                expected = Float32(b * 100 + v)
            else:
                expected = fill_value
            if out[b, v] != expected:
                raise Error(
                    "out[",
                    b,
                    ", ",
                    v,
                    "] = ",
                    out[b, v],
                    " != ",
                    expected,
                )


def main() raises:
    with DeviceContext() as ctx:
        test_apply_packed_bitmask(ctx)
        test_neg_inf_survives_the_mask(ctx)
