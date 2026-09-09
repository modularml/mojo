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
"""Op implementation for non-maximum suppression."""

from max._core.dialects import kgen, rmo
from max.dtype import DType

from ..dim import Dim
from ..graph import Graph
from ..type import TensorType
from ..value import TensorValue, TensorValueLike


def non_maximum_suppression(
    boxes: TensorValueLike,
    scores: TensorValueLike,
    max_output_boxes_per_class: TensorValueLike,
    iou_threshold: TensorValueLike,
    score_threshold: TensorValueLike,
    out_dim: str = "num_selected",
) -> TensorValue:
    """Filters boxes with high intersection-over-union (IoU).

    Applies greedy non-maximum suppression independently per (batch, class)
    pair. For each pair, the algorithm:

    1. Discards boxes whose score is at or below ``score_threshold``.
    2. Sorts the remaining boxes by score in descending order.
    3. Greedily selects boxes, suppressing any later candidate whose IoU with
       an already-selected box exceeds ``iou_threshold``.
    4. Stops after ``max_output_boxes_per_class`` selections per pair.

    Boxes use ``(y1, x1, y2, x2)`` corner format. Coordinates may be normalized
    or absolute, since the op handles both.

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        box_type = TensorType(DType.float32, [1, 3, 4], device=device)
        score_type = TensorType(DType.float32, [1, 1, 3], device=device)
        with Graph(
            "nms", input_types=[box_type, score_type]
        ) as graph:
            boxes, scores = graph.inputs
            # Each output row is (batch_index, class_index, box_index), with a
            # data-dependent number of rows.
            selected = ops.non_maximum_suppression(
                boxes.tensor,
                scores.tensor,
                max_output_boxes_per_class=ops.constant(
                    2, DType.int64, device=device
                ),
                iou_threshold=ops.constant(0.5, DType.float32, device=device),
                score_threshold=ops.constant(
                    0.0, DType.float32, device=device
                ),
            )
            graph.output(selected)

    Args:
        boxes: The input boxes tensor of shape
            ``(batch_size, num_boxes, 4)``, with a float dtype.
        scores: The per-class scores of shape
            ``(batch_size, num_classes, num_boxes)``, with the same dtype as
            ``boxes``.
        max_output_boxes_per_class: A scalar ``int64`` tensor giving the
            maximum number of boxes to select per (batch, class) pair.
        iou_threshold: A scalar float tensor giving the IoU suppression
            threshold.
        score_threshold: A scalar float tensor giving the minimum score to
            consider.
        out_dim: The name for the dynamic output dimension, which is the number
            of selected boxes. Defaults to ``"num_selected"``.

    Returns:
        A ``TensorValue`` representing the selected boxes, with shape
        ``(out_dim, 3)`` and ``int64`` dtype. Each row is
        ``(batch_index, class_index, box_index)``.
    """
    boxes = TensorValue(boxes)
    scores = TensorValue(scores)
    max_output_boxes_per_class = TensorValue(max_output_boxes_per_class)
    iou_threshold = TensorValue(iou_threshold)
    score_threshold = TensorValue(score_threshold)

    result_type = TensorType(
        dtype=DType.int64,
        shape=[Dim(out_dim), 3],
        device=boxes.device,
    )

    return Graph.current._add_op_generated(
        rmo.MoNonMaximumSuppressionOp,
        result_type.to_mlir(),
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
