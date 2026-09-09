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


from layout import Coord, TileTensor, row_major
from nn.nms import non_max_suppression, non_max_suppression_shape_func

from std.utils import IndexList


struct BoxCoords[dtype: DType](TrivialRegisterPassable):
    var y1: Scalar[Self.dtype]
    var x1: Scalar[Self.dtype]
    var y2: Scalar[Self.dtype]
    var x2: Scalar[Self.dtype]

    def __init__(
        out self,
        y1: Scalar[Self.dtype],
        x1: Scalar[Self.dtype],
        y2: Scalar[Self.dtype],
        x2: Scalar[Self.dtype],
    ):
        self.y1 = y1
        self.x1 = x1
        self.y2 = y2
        self.x2 = x2


def fill_boxes[
    dtype: DType
](
    batch_size: Int,
    box_list: Span[BoxCoords[dtype], ...],
    boxes: TileTensor[mut=True, dtype, ...],
):
    comptime assert boxes.flat_rank == 3
    var num_boxes = len(box_list) // batch_size
    for i in range(len(box_list)):
        var coords = linear_offset_to_coords(
            i, IndexList[2](batch_size, num_boxes)
        )
        boxes[coords[0], coords[1], 0] = box_list[i].y1
        boxes[coords[0], coords[1], 1] = box_list[i].x1
        boxes[coords[0], coords[1], 2] = box_list[i].y2
        boxes[coords[0], coords[1], 3] = box_list[i].x2


def linear_offset_to_coords[
    rank: Int
](idx: Int, shape: IndexList[rank]) -> IndexList[rank]:
    var output = IndexList[rank](0)
    var curr_idx = idx
    for i in reversed(range(rank)):
        output[i] = curr_idx % shape[i]
        curr_idx //= shape[i]

    return output


def fill_scores[
    dtype: DType
](
    batch_size: Int,
    num_classes: Int,
    scores_list: Span[Scalar[dtype], ...],
    scores: TileTensor[mut=True, dtype, ...],
):
    comptime assert scores.flat_rank == 3
    var num_boxes = len(scores_list) // batch_size // num_classes
    var shape = IndexList[3](batch_size, num_classes, num_boxes)
    for i in range(len(scores_list)):
        var coords = linear_offset_to_coords(i, shape)
        scores[coords[0], coords[1], coords[2]] = scores_list[i]


def test_case[
    dtype: DType
](
    batch_size: Int,
    num_classes: Int,
    num_boxes: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
    max_output_boxes_per_class: Int,
    box_list: Span[BoxCoords[dtype], ...],
    scores_list: Span[Scalar[dtype], ...],
):
    # Create boxes tensor
    var boxes_shape = IndexList[3](batch_size, num_boxes, 4)
    var boxes_storage = List(
        length=boxes_shape.flattened_length(), fill=Scalar[dtype](0)
    )
    var boxes = TileTensor(
        boxes_storage,
        row_major(Coord(boxes_shape)),
    )
    fill_boxes[dtype](batch_size, box_list, boxes)

    # Create scores tensor
    var scores_shape = IndexList[3](batch_size, num_classes, num_boxes)
    var scores_storage = List(
        length=scores_shape.flattened_length(), fill=Scalar[dtype](0)
    )
    var scores = TileTensor(
        scores_storage,
        row_major(Coord(scores_shape)),
    )
    fill_scores[dtype](batch_size, num_classes, scores_list, scores)

    var shape = non_max_suppression_shape_func(
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    )
    var idxs_shape = IndexList[2](shape[0], shape[1])
    var idxs_storage = List(length=idxs_shape.flattened_length(), fill=Int64(0))
    var selected_idxs = TileTensor(
        idxs_storage,
        row_major(Coord(idxs_shape)),
    )
    non_max_suppression(
        boxes,
        scores,
        selected_idxs,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    )

    for i in range(Int(selected_idxs.layout.shape[0]().value())):
        print(selected_idxs[i, 0], end="")
        print(",", end="")
        print(selected_idxs[i, 1], end="")
        print(",", end="")
        print(selected_idxs[i, 2], end="")
        print(",", end="")
        print("")


def main():
    def test_no_score_threshold():
        print("== test_no_score_threshold")
        var box_list = [
            BoxCoords[.float32](0.0, 0.0, 1.0, 1.0),
            BoxCoords[.float32](0.0, 0.1, 1.0, 1.1),
            BoxCoords[.float32](0.0, -0.1, 1.0, 0.9),
            BoxCoords[.float32](0.0, 10.0, 1.0, 11.0),
            BoxCoords[.float32](0.0, 10.1, 1.0, 11.1),
            BoxCoords[.float32](0.0, 100.0, 1.0, 101.0),
        ]
        var scores_list: List[Float32] = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]

        test_case[.float32](
            1, 1, 6, Float32(0.5), Float32(0.0), 3, box_list, scores_list
        )

    def test_flipped_coords():
        print("== test_flipped_coords")
        var box_list = [
            BoxCoords[.float32](1.0, 1.0, 0.0, 0.0),
            BoxCoords[.float32](1.0, 1.1, 0.0, 0.1),
            BoxCoords[.float32](1.0, 0.9, 0.0, -0.1),
            BoxCoords[.float32](1.0, 11.0, 0.0, 10.0),
            BoxCoords[.float32](1.0, 11.1, 0.0, 10.1),
            BoxCoords[.float32](1.0, 101.0, 0.0, 100.0),
        ]
        var scores_list: List[Float32] = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]

        test_case[.float32](
            1, 1, 6, Float32(0.5), Float32(0.0), 3, box_list, scores_list
        )

    def test_reflect_over_yx():
        print("== test_reflect_over_yx")
        var box_list = [
            BoxCoords[.float32](-1.0, -1.0, 0.0, 0.0),
            BoxCoords[.float32](-1.0, -1.1, 0.0, -0.1),
            BoxCoords[.float32](-1.0, -0.9, 0.0, 0.1),
            BoxCoords[.float32](-1.0, -11.0, 0.0, -10.0),
            BoxCoords[.float32](-1.0, -11.1, 0.0, -10.1),
            BoxCoords[.float32](-1.0, -101.0, 0.0, -100.0),
        ]
        var scores_list: List[Float32] = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]

        test_case[.float32](
            1, 1, 6, Float32(0.5), Float32(0.0), 3, box_list, scores_list
        )

    def test_score_threshold():
        print("== test_score_threshold")
        var box_list = [
            BoxCoords[.float32](0.0, 0.0, 1.0, 1.0),
            BoxCoords[.float32](0.0, 0.1, 1.0, 1.1),
            BoxCoords[.float32](0.0, -0.1, 1.0, 0.9),
            BoxCoords[.float32](0.0, 10.0, 1.0, 11.0),
            BoxCoords[.float32](0.0, 10.1, 1.0, 11.1),
            BoxCoords[.float32](0.0, 100.0, 1.0, 101.0),
        ]
        var scores_list: List[Float32] = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]

        test_case[.float32](
            1, 1, 6, Float32(0.5), Float32(0.4), 3, box_list, scores_list
        )

    def test_limit_outputs():
        print("== test_limit_outputs")
        var box_list = [
            BoxCoords[.float32](0.0, 0.0, 1.0, 1.0),
            BoxCoords[.float32](0.0, 0.1, 1.0, 1.1),
            BoxCoords[.float32](0.0, -0.1, 1.0, 0.9),
            BoxCoords[.float32](0.0, 10.0, 1.0, 11.0),
            BoxCoords[.float32](0.0, 10.1, 1.0, 11.1),
            BoxCoords[.float32](0.0, 100.0, 1.0, 101.0),
        ]
        var scores_list: List[Float32] = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]

        test_case[.float32](
            1, 1, 6, Float32(0.5), Float32(0.0), 2, box_list, scores_list
        )

    def test_single_box():
        print("== test_single_box")
        var box_list = [
            BoxCoords[.float32](0.0, 0.0, 1.0, 1.0),
        ]
        var scores_list: List[Float32] = [0.9]

        test_case[.float32](
            1, 1, 1, Float32(0.5), Float32(0.0), 2, box_list, scores_list
        )

    def test_two_classes():
        print("== test_two_classes")
        var box_list = [
            BoxCoords[.float32](0.0, 0.0, 1.0, 1.0),
            BoxCoords[.float32](0.0, 0.1, 1.0, 1.1),
            BoxCoords[.float32](0.0, -0.1, 1.0, 0.9),
            BoxCoords[.float32](0.0, 10.0, 1.0, 11.0),
            BoxCoords[.float32](0.0, 10.1, 1.0, 11.1),
            BoxCoords[.float32](0.0, 100.0, 1.0, 101.0),
        ]
        var scores_list: List[Float32] = [
            0.9,
            0.75,
            0.6,
            0.95,
            0.5,
            0.3,
            0.9,
            0.75,
            0.6,
            0.95,
            0.5,
            0.3,
        ]

        test_case[.float32](
            1, 2, 6, Float32(0.5), Float32(0.0), 2, box_list, scores_list
        )

    def test_two_batches():
        print("== test_two_batches")
        var box_list = [
            BoxCoords[.float32](0.0, 0.0, 1.0, 1.0),
            BoxCoords[.float32](0.0, 0.1, 1.0, 1.1),
            BoxCoords[.float32](0.0, -0.1, 1.0, 0.9),
            BoxCoords[.float32](0.0, 10.0, 1.0, 11.0),
            BoxCoords[.float32](0.0, 10.1, 1.0, 11.1),
            BoxCoords[.float32](0.0, 100.0, 1.0, 101.0),
            BoxCoords[.float32](0.0, 0.0, 1.0, 1.0),
            BoxCoords[.float32](0.0, 0.1, 1.0, 1.1),
            BoxCoords[.float32](0.0, -0.1, 1.0, 0.9),
            BoxCoords[.float32](0.0, 10.0, 1.0, 11.0),
            BoxCoords[.float32](0.0, 10.1, 1.0, 11.1),
            BoxCoords[.float32](0.0, 100.0, 1.0, 101.0),
        ]
        var scores_list: List[Float32] = [
            0.9,
            0.75,
            0.6,
            0.95,
            0.5,
            0.3,
            0.9,
            0.75,
            0.6,
            0.95,
            0.5,
            0.3,
        ]

        test_case[.float32](
            2, 1, 6, Float32(0.5), Float32(0.0), 2, box_list, scores_list
        )

    # CHECK-LABEL: == test_no_score_threshold
    # CHECK: 0,0,3,
    # CHECK-NEXT: 0,0,0,
    # CHECK-NEXT: 0,0,5,
    test_no_score_threshold()

    # CHECK-LABEL: == test_flipped_coords
    # CHECK: 0,0,3,
    # CHECK-NEXT: 0,0,0,
    # CHECK-NEXT: 0,0,5,
    test_flipped_coords()

    # CHECK-LABEL: == test_reflect_over_yx
    # CHECK: 0,0,3,
    # CHECK-NEXT: 0,0,0,
    # CHECK-NEXT: 0,0,5,
    test_reflect_over_yx()

    # CHECK-LABEL: == test_score_threshold
    # CHECK: 0,0,3,
    # CHECK-NEXT: 0,0,0,
    test_score_threshold()

    # CHECK-LABEL: == test_limit_outputs
    # CHECK: 0,0,3,
    # CHECK-NEXT: 0,0,0,
    test_limit_outputs()

    # CHECK-LABEL: == test_single_box
    # CHECK: 0,0,0,
    test_single_box()

    # CHECK-LABEL: == test_two_classes
    # CHECK: 0,0,3,
    # CHECK-NEXT: 0,0,0,
    # CHECK-NEXT: 0,1,3,
    # CHECK-NEXT: 0,1,0,
    test_two_classes()

    # CHECK-LABEL: == test_two_batches
    # CHECK: 0,0,3,
    # CHECK-NEXT: 0,0,0,
    # CHECK-NEXT: 1,0,3,
    # CHECK-NEXT: 1,0,0,
    test_two_batches()
