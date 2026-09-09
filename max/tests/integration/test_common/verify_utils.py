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

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing
import torch
from scipy.spatial import distance
from scipy.special import rel_entr, softmax
from test_common.custom_args import CommaSeparatedList
from test_common.table import CONSOLE, PrettyTable
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import (
    LearnedPerceptualImagePatchSimilarity,
)

# --- Shared image metric computation (single source of truth for report + validators) ---


def _prepare_images_for_torchmetrics(img: np.ndarray) -> torch.Tensor:
    """Convert numpy image(s) to tensor (B, C, H, W) in [0, 1] for TorchMetrics."""
    if img.ndim == 3:
        img = img[None, ...]
    img = np.transpose(img, (0, 3, 1, 2))
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img.astype(np.float32))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images (single source of truth for report and SSIMValidator).

    Args:
        img1, img2: (H, W, C) or (B, H, W, C) in [0, 1] or [0, 255].
    Returns:
        SSIM score in [0, 1] (1 = identical). For batch, returns mean.
    """
    t1 = _prepare_images_for_torchmetrics(img1)
    t2 = _prepare_images_for_torchmetrics(img2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    with torch.no_grad():
        val = metric(t1.to(device), t2.to(device))
    out = val.cpu().numpy()
    return float(np.mean(out))


def compute_lpips(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Compute LPIPS distance between images (single source of truth for report and LPIPSValidator).

    Args:
        img1, img2: (H, W, C) or (B, H, W, C) in [0, 1] or [0, 255].
    Returns:
        Array of LPIPS distances, shape (B,) or (1,) for single image. Lower = more similar.
    """
    t1 = _prepare_images_for_torchmetrics(img1)
    t2 = _prepare_images_for_torchmetrics(img2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="squeeze", normalize=True
    ).to(device)
    with torch.no_grad():
        val = metric(t1.to(device), t2.to(device))
    return val.cpu().numpy().flatten()


@dataclass
class ValidationResult:
    """short_name of the Validator returning this result"""

    name: str
    """True if we're within tolerances, False otherwise"""
    success: bool
    """The message to print on failure. Not set if success == True"""
    message: str | None = None
    """The np.array of target values. Not set if success == True"""
    target: numpy.typing.NDArray | None = None  # type: ignore[type-arg]
    """The np.array of reference values. Not set if success == True"""
    reference: numpy.typing.NDArray | None = None  # type: ignore[type-arg]
    """The indices of the failing tolerances. Not set if success == True"""
    element_indices: numpy.typing.NDArray | None = None  # type: ignore[type-arg]
    """The metrics to display on failure. Not set if success == True"""
    data: list[numpy.typing.NDArray] = field(default_factory=list)  # type: ignore[type-arg]


class ValidationResultCollection:
    """Container to store multiple ValidationResults for a given model output

    If your model has 3 outputs, then each output will have its own
    ValidationResultCollection.
    """

    def __init__(self, *results: ValidationResult) -> None:
        self._results = {r.name: r for r in results}

    def any_failed(
        self,
    ) -> bool:
        """Returns true if any entry failed in the collection, False otherwise"""
        return any(not result.success for result in self._results.values())

    def get_failure_messages(
        self,
    ) -> str:
        """Returns a combined failure message for all entries in the collection."""
        message = ""
        num_failures = 0
        for result in self._results.values():
            if result.success:
                continue

            if message:
                message += "\n"

            assert result.message is not None
            message += result.message
            num_failures += 1

        if num_failures == 0 or len(self._results) == 1:
            # keep output simple if we didn't fail or if we only have one metric
            return message

        # decorate our text with more context if we have more than one metric
        return f"Failed {num_failures} validation metrics\n" + message + "\n"

    def get_result(self, result_name: str) -> ValidationResult:
        """Returns the ValidationResult by its name"""
        return self._results[result_name]

    def add_result(self, result: ValidationResult) -> None:
        """Add a new ValidationResult to the collection."""
        if result.name in self._results:
            raise ValueError(
                "Encountered repeat validation result for eval of type"
                f" {result.name}"
            )

        self._results[result.name] = result

    def merge_with(self, other_result: "ValidationResultCollection") -> None:
        """Merge self with another ValidationResultCollection."""
        for result in other_result._results.values():
            self.add_result(result)


class ValidatorBase(ABC):
    """Base class defining the interface for custom validation metrics"""

    def __init__(self, *args, **kwargs) -> None:  # noqa: B027
        pass

    @staticmethod
    @abstractmethod
    def short_name() -> str:
        """The name to reference this validator in the CLI args.
        Must be unique per Validator.
        """

    @staticmethod
    @abstractmethod
    def _indices_to_sort_by() -> list[int]:
        """The indices to sort by when printing result tables.
        If this returns [2,3], then two tables will be printed with the same
        data, the first sorted by the metric returned in index 2 and the second
        sorted by the metric in index 3."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _column_names() -> list[str]:
        """The column names of the metrics.
        The number of arrays returned from `validate` in
        `ValidationResult.metrics` must have the same length as this array.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _pretty_names() -> list[str]:
        """Names used for table titles.

        The length of the return value must match the len of `_indices_to_sort_by`.
        """
        raise NotImplementedError()

    @abstractmethod
    def threshold_str(self) -> str:
        """Returns a string describing the threshold for this metric"""
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """Aliases to validate"""
        return self.validate(*args, **kwargs)

    @abstractmethod
    def validate(
        self,
        target: numpy.typing.NDArray,  # type: ignore[type-arg]
        reference: numpy.typing.NDArray,  # type: ignore[type-arg]
        **kwargs,
    ) -> ValidationResultCollection:
        """Performs the validation with the given metric.

        Validate is called for each tensor in the model output.
        """
        raise NotImplementedError()

    @abstractmethod
    def _print_suggested_tolerances(
        self,
        targets: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        references: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        metrics: list[list[numpy.typing.NDArray]],  # type: ignore[type-arg]
    ) -> None:
        raise NotImplementedError()

    def print_error_table(
        self,
        ref_framework: str,
        target_framework: str,
        results: list[ValidationResultCollection],
        diff_count: int | None = None,
        print_suggested_tolerances: bool = False,
        **kwargs,
    ) -> None:
        """Prints the table with errors sorted by severity

        The `results` passed to this method should be the results accumulated
        after calling `validate` on each tensor in the model output.
        """
        targets = []
        references = []
        tensor_indices = []
        element_indices = []
        data: list[list[numpy.typing.NDArray]] = []  # type: ignore[type-arg]
        metric_names = self._column_names()
        for tensor_idx, result_collection in enumerate(results):
            result = result_collection.get_result(self.short_name())

            if result.success:
                continue

            if not data:
                # initialize metrics list
                data = [[] for _ in result.data]
            else:
                assert len(result.data) == len(data), (
                    "Expected number of metrics for each result to be consistent"
                )

            assert len(metric_names) == len(result.data), (
                "Expected number of metric_names to match number of returned"
                f" metrics for {self.short_name()} Validator"
            )

            for i in range(len(result.data)):
                data[i].append(result.data[i].flatten())

            # flatten element indices into individual numpy arrays
            # the rank of each output tensor may not be consistent,
            # so we can't concat this list
            assert result.element_indices is not None
            element_indices.extend(list(result.element_indices))
            tensor_indices.append(
                np.full_like(data[0][-1], tensor_idx, dtype=np.int64)
            )
            assert result.target is not None
            assert result.reference is not None
            targets.append(result.target)
            references.append(result.reference)

        if not any(m for m in data):
            # no failures to report, return
            return

        num_diff_outputs = len(data[0])
        concatted_data = []
        for i in range(len(data)):
            concatted_data.append(np.concatenate(data[i]))

        catted_tensor_indices = np.concatenate(tensor_indices)

        # iterate over each `sort_by_idx` and print a table
        max_val_str = ""
        for sort_by_idx, pretty_name in zip(
            self._indices_to_sort_by(), self._pretty_names(), strict=True
        ):
            CONSOLE.print(
                f"Failures by {pretty_name} ({ref_framework} vs"
                f" {target_framework}):"
            )
            max_val_str += (
                f"Max {pretty_name}: {concatted_data[sort_by_idx].max():.5e}\n"
            )
            assert diff_count is not None
            _print_diff_table(
                tensor_indices=catted_tensor_indices,
                element_indices=np.array(element_indices),
                data=concatted_data,
                column_names=metric_names,
                max_shown=diff_count,
                sort_by_idx=sort_by_idx,
            )

        num_diffs = concatted_data[0].size
        CONSOLE.print(
            f"{max_val_str}"
            f"🟡 {num_diff_outputs} outputs differ at {num_diffs} locations"
            f" ({self.threshold_str()})\n"
        )

        if print_suggested_tolerances:
            self._print_suggested_tolerances(
                targets, references, concatted_data
            )


class MultiValidator(ValidatorBase):
    """Wrapper class to execute multiple validators"""

    def __init__(self, *wrapped_validators: ValidatorBase, **kwargs) -> None:
        super().__init__(**kwargs)
        self._validators = wrapped_validators

    @staticmethod
    def short_name() -> str:
        raise NotImplementedError("should not call name in multi-validator")

    @staticmethod
    def _indices_to_sort_by() -> list[int]:
        raise NotImplementedError(
            "_indices_to_sort_by should not be called in MultiValidator"
        )

    @staticmethod
    def _pretty_names() -> list[str]:
        raise NotImplementedError(
            "_pretty_names should not be called in MultiValidator"
        )

    @staticmethod
    def _column_names() -> list[str]:
        raise NotImplementedError(
            "_column_names should not be called in MultiValidator"
        )

    def _print_suggested_tolerances(
        self,
        targets: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        references: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        metrics: list[list[numpy.typing.NDArray]],  # type: ignore[type-arg]
    ) -> None:
        raise NotImplementedError(
            "_print_suggested_tolerances should not be called in MultiValidator"
        )

    def threshold_str(self) -> str:
        return ", ".join(v.threshold_str() for v in self._validators)

    def validate(
        self,
        target: numpy.typing.NDArray,  # type: ignore[type-arg]
        reference: numpy.typing.NDArray,  # type: ignore[type-arg]
        **kwargs,
    ) -> ValidationResultCollection:
        overall_result = ValidationResultCollection()
        for validator in self._validators:
            result = validator(target, reference, **kwargs)
            overall_result.merge_with(result)

        return overall_result

    def print_error_table(
        self,
        ref_framework: str,
        target_framework: str,
        results: list[ValidationResultCollection],
        diff_count: int | None = None,
        print_suggested_tolerances: bool = False,
        **kwargs,
    ) -> None:
        for validator in self._validators:
            validator.print_error_table(
                ref_framework,
                target_framework,
                results,
                diff_count=diff_count,
                print_suggested_tolerances=print_suggested_tolerances,
                **kwargs,
            )


class ToleranceValidator(ValidatorBase):
    """Validator which checks absolute and relative tolerance."""

    _ATOL_IDX = 2
    _RTOL_IDX = 3

    def __init__(self, atol: float, rtol: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._atol = atol
        self._rtol = rtol

    @staticmethod
    def short_name() -> str:
        return "tol"

    @staticmethod
    def _indices_to_sort_by() -> list[int]:
        # print tables sorted by abs_diff and rel_diff
        return [2, 3]

    @staticmethod
    def _pretty_names() -> list[str]:
        return ["Absolute Diff", "Relative Diff"]

    @staticmethod
    def _column_names() -> list[str]:
        return ["reference", "target", "abs_diff", "rel_diff"]

    def threshold_str(self) -> str:
        return f"atol={self._atol} rtol={self._rtol}"

    def _print_suggested_tolerances(
        self,
        targets: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        references: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        metrics: list[list[numpy.typing.NDArray]],  # type: ignore[type-arg]
    ) -> None:
        target = np.concatenate([t.flatten() for t in targets])
        reference = np.concatenate([r.flatten() for r in references])

        max_atol = self._atol
        for abs_diff in metrics[self._ATOL_IDX]:
            temp_max_atol = abs_diff.max()
            max_atol = max(max_atol, temp_max_atol)

        max_rtol = self._rtol
        for rel_diff in metrics[self._RTOL_IDX]:
            temp_max_rtol = rel_diff.max()
            max_rtol = max(max_rtol, temp_max_rtol)

        _print_pareto_tolerances(
            target,
            reference,
            min_atol=self._atol,
            max_atol=max_atol,
            min_rtol=self._rtol,
            max_rtol=max_rtol,
        )

    def validate(
        self,
        target: numpy.typing.NDArray,  # type: ignore[type-arg]
        reference: numpy.typing.NDArray,  # type: ignore[type-arg]
        **kwargs,
    ) -> ValidationResultCollection:
        isoff = np.logical_not(
            _is_close(
                target,
                reference,
                absolute_tolerance=self._atol,
                relative_tolerance=self._rtol,
                equal_nan=True,
            )
        )

        if not isoff.any():
            return ValidationResultCollection(
                ValidationResult(self.short_name(), True)
            )

        reference = reference[isoff]
        target = target[isoff]

        abs_diff = np.abs(reference - target)

        # This may divide by zero but we don't care to print the warning.
        with np.errstate(divide="ignore"):
            rel_diff = abs_diff / np.maximum(np.abs(reference), np.abs(target))

        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        element_indices = np.transpose(np.where(isoff))

        # not within tolerance, report the error
        err_msg = (
            f"Output has {len(abs_diff)} differences that exceed"
            f" tolerance (Max Absolute Diff: {max_abs_diff:.5e},"
            f" Max Relative Diff: {max_rel_diff:.5e})"
        )

        result = ValidationResult(
            self.short_name(),
            False,
            err_msg,
            target,
            reference,
            element_indices,
            [reference, target, abs_diff, rel_diff],
        )

        return ValidationResultCollection(result)


class DistanceValidatorBase(ValidatorBase, ABC):
    """Base class to implement simple distance metrics.

    Child classes must implement `_compute_distance` and `threshold_str`.
    """

    def __init__(self, threshold: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._threshold = threshold

    def _print_suggested_tolerances(
        self,
        targets: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        references: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        metrics: list[list[numpy.typing.NDArray]],  # type: ignore[type-arg]
    ) -> None:
        assert len(metrics) == 1
        max_dist = np.array(metrics[0]).max()

        if math.isfinite(max_dist):
            base = math.pow(10, math.floor(math.log10(max_dist)) - 1)
            max_dist = math.ceil(max_dist / base) * base
        CONSOLE.print(
            f"Suggested {self._pretty_names()[0]} threshold: {max_dist:.1e}\n"
        )

    @abstractmethod
    def _compute_distance(
        self,
        target: numpy.typing.NDArray,  # type: ignore[type-arg]
        reference: numpy.typing.NDArray,  # type: ignore[type-arg]
    ) -> numpy.typing.NDArray:  # type: ignore[type-arg]
        raise NotImplementedError()

    @staticmethod
    def _column_names() -> list[str]:
        return ["distance"]

    @staticmethod
    def _indices_to_sort_by() -> list[int]:
        # print tables sorted by singular distance metric
        return [0]

    def validate(
        self,
        target: numpy.typing.NDArray,  # type: ignore[type-arg]
        reference: numpy.typing.NDArray,  # type: ignore[type-arg]
        **kwargs,
    ) -> ValidationResultCollection:
        distance = self._compute_distance(target, reference)
        non_finite = np.logical_not(np.isfinite(distance))
        isoff = np.logical_or(distance > self._threshold, non_finite)
        if not isoff.any():
            return ValidationResultCollection(
                ValidationResult(self.short_name(), True)
            )
        # NOTE: reference and target shapes differ from distance's shape
        # _compute_distance will reduces the last axis into single statistic
        reference = reference[isoff]
        target = target[isoff]
        if isoff.size <= 1:
            distance = distance.reshape(-1)
            isoff = isoff.reshape(-1)

        element_indices = np.transpose(np.where(isoff))
        max_distance = (
            np.nanmax(distance) if np.isfinite(distance).any() else float("nan")
        )
        num_diffs = np.count_nonzero(isoff)
        distance = distance[isoff]
        num_non_finite = np.count_nonzero(non_finite)

        # not within tolerance, report the error
        err_msg = (
            f"Output has {num_diffs} differences that exceed"
            f" {self._pretty_names()[0]} threshold (Max Distance:"
            f" {max_distance:.6e})"
        )
        if num_non_finite:
            err_msg += f", including {num_non_finite} non-finite values"

        result = ValidationResult(
            self.short_name(),
            False,
            err_msg,
            target,
            reference,
            element_indices,
            [distance],
        )

        return ValidationResultCollection(result)


class CosineSimilarityValidator(DistanceValidatorBase):
    """Validator to check Cosine similarity"""

    def __init__(self, cos_threshold: float, **kwargs) -> None:
        super().__init__(cos_threshold, **kwargs)

    @staticmethod
    def short_name() -> str:
        return "cos"

    @staticmethod
    def _pretty_names() -> list[str]:
        return ["Cosine Similarity"]

    def threshold_str(self) -> str:
        return f"cos_similarity={self._threshold}"

    def _compute_distance(
        self,
        target: numpy.typing.NDArray,  # type: ignore[type-arg]
        reference: numpy.typing.NDArray,  # type: ignore[type-arg]
    ) -> numpy.typing.NDArray:  # type: ignore[type-arg]
        flat_target = target.reshape((-1, target.shape[-1]))
        flat_ref = reference.reshape((-1, reference.shape[-1]))
        flat_distance = np.zeros(flat_ref.shape[:-1], dtype=reference.dtype)
        for i in range(flat_target.shape[0]):  # type: ignore
            flat_distance[i] = distance.cosine(flat_target[i], flat_ref[i])

        result = flat_distance.reshape(reference.shape[:-1])

        return result


class KLDivergenceValidator(DistanceValidatorBase):
    """Validator to check KLDivergence of output distributions"""

    def __init__(self, kl_div_threshold: float, **kwargs) -> None:
        super().__init__(kl_div_threshold, **kwargs)

    @staticmethod
    def short_name() -> str:
        return "kl"

    @staticmethod
    def _pretty_names():  # noqa: ANN205
        return ["KL Divergence"]

    def threshold_str(self) -> str:
        return f"kl_div={self._threshold}"

    @staticmethod
    def _smooth_probs(
        probs: numpy.typing.NDArray[np.floating],
        eps: float = 1e-10,
    ) -> numpy.typing.NDArray[np.floating]:
        """Smooths probabilities so no entry is below eps.

        Applies (1 - N*eps) * probs + eps, which preserves the simplex
        (probabilities still sum to 1) and guarantees a minimum value of eps.

        The choice of eps is a tradeoff between numerical stability and
        vocabulary size N. The (1 - N*eps) factor redistributes probability
        away from softmax values to fund the eps floor. For typical vocab
        sizes (~2e5), N*eps is ~2e-5, which is negligible for any
        significant probability.
        """
        n = probs.shape[-1]
        return (1 - n * eps) * probs + eps

    def _compute_distance(
        self,
        target: numpy.typing.NDArray,  # type: ignore[type-arg]
        reference: numpy.typing.NDArray,  # type: ignore[type-arg]
    ) -> numpy.typing.NDArray:  # type: ignore[type-arg]
        return rel_entr(
            self._smooth_probs(softmax(reference, -1)),
            self._smooth_probs(softmax(target, -1)),
        ).sum(-1)


class LPIPSValidator(ValidatorBase):
    """Validator to check Learned Perceptual Image Patch Similarity (LPIPS).

    LPIPS uses deep features from pre-trained networks to measure perceptual similarity.
    Lower LPIPS = more similar images (range: [0, ∞), typically 0-1). Pass when lpips <= threshold.
    """

    def __init__(self, lpips_threshold: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._lpips_threshold = lpips_threshold

    @staticmethod
    def short_name() -> str:
        return "lpips"

    @staticmethod
    def _column_names() -> list[str]:
        return ["lpips"]

    @staticmethod
    def _indices_to_sort_by() -> list[int]:
        return [0]

    @staticmethod
    def _pretty_names() -> list[str]:
        return ["LPIPS"]

    def threshold_str(self) -> str:
        return f"lpips<={self._lpips_threshold}"

    def _print_suggested_tolerances(
        self,
        targets: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        references: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        metrics: list[list[numpy.typing.NDArray]],  # type: ignore[type-arg]
    ) -> None:
        assert len(metrics) == 1
        max_lpips = np.array(metrics[0]).max()
        if math.isfinite(max_lpips):
            base = math.pow(10, math.floor(math.log10(max_lpips)) - 1)
            max_lpips = math.ceil(max_lpips / base) * base
        CONSOLE.print(
            f"Suggested {self._pretty_names()[0]} threshold: {max_lpips:.1e}\n"
        )

    def validate(
        self,
        target: numpy.typing.NDArray,  # type: ignore[type-arg]
        reference: numpy.typing.NDArray,  # type: ignore[type-arg]
        **kwargs,
    ) -> ValidationResultCollection:
        """Validate LPIPS: lower is better, pass when max(lpips) <= threshold."""
        if target.ndim < 3 or reference.ndim < 3:
            raise ValueError(
                f"LPIPS requires 3D (H, W, C) or 4D (B, H, W, C) images, "
                f"got shapes: target={target.shape}, reference={reference.shape}"
            )
        lpips_values = compute_lpips(target, reference)
        max_lpips = float(np.max(lpips_values))

        if max_lpips <= self._lpips_threshold:
            return ValidationResultCollection(
                ValidationResult(self.short_name(), True)
            )

        err_msg = (
            f"LPIPS check failed: {max_lpips:.6f} > {self._lpips_threshold:.6f}"
        )
        result = ValidationResult(
            self.short_name(),
            False,
            err_msg,
            target,
            reference,
            np.array([[0]]),
            [lpips_values],
        )
        return ValidationResultCollection(result)


class SSIMValidator(ValidatorBase):
    """Validator to check Structural Similarity Index (SSIM) for images.

    SSIM measures perceptual similarity (range: [-1, 1], 1 = identical).
    Higher SSIM is better; no distance conversion.
    """

    def __init__(self, ssim_threshold: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ssim_threshold = ssim_threshold

    @staticmethod
    def short_name() -> str:
        return "ssim"

    @staticmethod
    def _column_names() -> list[str]:
        return ["ssim"]

    @staticmethod
    def _indices_to_sort_by() -> list[int]:
        return [0]

    @staticmethod
    def _pretty_names() -> list[str]:
        return ["SSIM"]

    def threshold_str(self) -> str:
        return f"ssim>={self._ssim_threshold}"

    def _print_suggested_tolerances(
        self,
        targets: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        references: list[numpy.typing.NDArray],  # type: ignore[type-arg]
        metrics: list[list[numpy.typing.NDArray]],  # type: ignore[type-arg]
    ) -> None:
        assert len(metrics) == 1
        min_ssim = np.array(metrics[0]).min()
        if math.isfinite(min_ssim):
            CONSOLE.print(
                f"Suggested {self._pretty_names()[0]} threshold: {min_ssim:.2e}\n"
            )

    def validate(
        self,
        target: numpy.typing.NDArray,  # type: ignore[type-arg]
        reference: numpy.typing.NDArray,  # type: ignore[type-arg]
        **kwargs,
    ) -> ValidationResultCollection:
        """Validate SSIM: higher is better, pass when ssim >= threshold."""
        if target.ndim < 3 or reference.ndim < 3:
            raise ValueError(
                f"SSIM requires 3D (H, W, C) or 4D (B, H, W, C) images, "
                f"got shapes: target={target.shape}, reference={reference.shape}"
            )
        ssim_score = compute_ssim(target, reference)

        if ssim_score >= self._ssim_threshold:
            return ValidationResultCollection(
                ValidationResult(self.short_name(), True)
            )

        err_msg = (
            f"SSIM check failed: {ssim_score:.6f} < {self._ssim_threshold:.6f}"
        )
        result = ValidationResult(
            self.short_name(),
            False,
            err_msg,
            target,
            reference,
            np.array([[0]]),
            [np.array([ssim_score])],
        )
        return ValidationResultCollection(result)


_VALIDATORS_BY_NAME: dict[str, type[ValidatorBase]] = {
    v.short_name(): v  # type: ignore
    for v in [
        ToleranceValidator,
        CosineSimilarityValidator,
        KLDivergenceValidator,
        LPIPSValidator,
        SSIMValidator,
    ]
}


def construct_validator(
    names: Sequence[str] | CommaSeparatedList, **kwargs
) -> ValidatorBase:
    validators: list[ValidatorBase] = []
    for name in names:
        validators.append(_VALIDATORS_BY_NAME[name](**kwargs))

    if len(validators) == 1:
        return validators[0]

    return MultiValidator(*validators)


def _print_pareto_tolerances(
    a: np.typing.NDArray,  # type: ignore[type-arg]
    b: np.typing.NDArray,  # type: ignore[type-arg]
    min_atol: float,
    max_atol: float,
    min_rtol: float,
    max_rtol: float,
) -> None:
    """Prints the pareto frontier combinations of absolute and relative tolerance.

    This is a list of passing tolerances that have the minimal overall tolerances such that they are on the pareto frontier.
    The values chosen for this list will always be an integer times a power of 10 with 2 places of precision (E.G. 1.1e-04, 9.3e-01).

    Args:
      a: The first value to compare.
      b: The second value to compare.
      min_atol: The minimum absolute tolerance to consider.
      max_atol: The maximum absolute tolerance to consider.
      min_rtol: The minimum relative tolerance to consider.
      max_rtol: The maximum relative tolerance to consider.
    """
    atol_step = math.pow(10, math.floor(math.log10(max_atol)) - 1)
    rtol_step = math.pow(10, math.floor(math.log10(max_rtol)) - 1)

    atol_mul = math.ceil(max_atol / atol_step)
    rtol_mul = math.ceil(max_rtol / rtol_step)

    possible_atols = []
    while atol_mul >= math.floor(min_atol / atol_step):
        possible_atols.append(max(atol_mul * atol_step, min_atol))
        atol_mul -= 1
        # Reset upon reaching 9: goes from 1.0e-1 To 9.9e-2.
        # If we reset at 0, it would go from 0.1e-1 (aka 1.0e-2) to 9.9e-2.
        if atol_mul == 9:
            atol_mul = 99
            atol_step /= 10

    possible_rtols = []
    while rtol_mul >= math.floor(min_rtol / rtol_step):
        possible_rtols.append(max(rtol_mul * rtol_step, min_rtol))
        rtol_mul -= 1
        # Reset upon reaching 9: goes from 1.0e-1 To 9.9e-2.
        # If we reset at 0, it would go from 0.1e-1 (aka 1.0e-2) to 9.9e-2.
        if rtol_mul == 9:
            rtol_mul = 99
            rtol_step /= 10

    def within_tolerance(atol: float, rtol: float) -> bool:
        return bool(
            np.all(
                _is_close(
                    a,
                    b,
                    atol,
                    rtol,
                    equal_nan=True,
                )
            )
        )

    valid = []
    # atol is iterated min to max
    for atol in reversed(possible_atols):
        # rtol is iterated max to min
        while possible_rtols:
            rtol = possible_rtols[0]
            if not within_tolerance(atol, rtol):
                # Once not within tolerance, nothing else will be in tolerance until atol is increased.
                break

            # possible pareto candidate
            valid.append([atol, rtol])

            # restrict future rtol checks.
            # Anything in the future with this rtol will not be in the pareto frontier.
            possible_rtols = possible_rtols[1:]

    valid_arr = np.array(valid)
    pareto = valid_arr[_is_pareto(valid_arr)]

    def percent_passing_rtol_only(vals: tuple[float, float]) -> float:
        (_atol, rtol) = vals
        return (
            np.sum(
                _is_close(
                    a,
                    b,
                    absolute_tolerance=0,
                    relative_tolerance=rtol,
                    equal_nan=True,
                )
            )
            * 100
            / len(a)
        )

    CONSOLE.print(
        "Pareto frontier of possible tolerances contains"
        f" {len(pareto)} solutions."
    )
    CONSOLE.print("Solutions are:")
    percents = map(percent_passing_rtol_only, pareto)
    for (atol, rtol), percent in zip(pareto, percents, strict=True):
        CONSOLE.print(
            f"\tAbsolute Tolerance: {atol:.1e}\n\tRelative Tolerance:"
            f" {rtol:.1e}\n\tPercent passing from relative tolerance alone:"
            f" {percent:.1f}%\n"
        )


def _is_pareto(costs: np.typing.NDArray) -> np.typing.NDArray:  # type: ignore[type-arg]
    """
    Find the pareto-efficient points

    Args:
      costs: An (n_points, n_costs) array

    Returns:
        A boolean vector, indicating whether each point is pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def _lookup(value: dict[str, Any], keys: Sequence[str]) -> list[Any]:
    return [value[k] for k in keys]


def _print_diff_table(
    tensor_indices: numpy.typing.NDArray,  # type: ignore[type-arg]
    element_indices: numpy.typing.NDArray,  # type: ignore[type-arg]
    data: list[numpy.typing.NDArray],  # type: ignore[type-arg]
    column_names: list[str],
    max_shown: int,
    sort_by_idx: int,
) -> None:
    header = ["position", *column_names]
    # We print the first N entries where the results do not match. We take the
    # top N/2 entries and the bottom N/2 entries.
    #
    # Only take the max_shown // 2 entries from either side (separated by the
    # separator_entry). If the max_shown is an odd number, then we take
    # max_shown // 2 + 1 entries from the top side and max_shown // 2 from the
    # bottom.
    #
    # To avoid the cost of sorting all elements, first partition to grab just
    # the elements that will be printed.
    eliding_elements = max_shown < len(data[0])
    if eliding_elements:
        half, extra = divmod(max_shown, 2)
        grab_first = half + bool(extra)
        grab_last = half
        first_indices = np.argpartition(data[sort_by_idx], grab_first)
        last_indices = np.argpartition(data[sort_by_idx], -grab_last)
        indices = np.concatenate(
            (first_indices[:grab_first], last_indices[-grab_last:])
        )
        tensor_indices = tensor_indices[indices]
        element_indices = np.array(
            [list(element_indices[idx]) for idx in indices]
        )
        elided_metrics = []
        for i in range(len(data)):
            elided_metrics.append(data[i][indices])
        data = list(elided_metrics)

    # We collect all the dif information. The diff information is a hash table
    # containing the index of the tensor, the index of the element in the
    # tensor, the value with the reference framework, the value with the target
    # framework, and the value of the difference.
    diff_information: list[dict[str, Any]] = []
    diff_recs = list(zip(tensor_indices, element_indices, *data, strict=True))

    diff_recs = sorted(diff_recs, key=lambda el: el[2 + sort_by_idx])

    for tensor_i, elem_i, *data in diff_recs:
        diff_information.append(
            {
                "position": (tensor_i, elem_i),
                **{
                    name: f"{m:.5e}"
                    for name, m in zip(column_names, data, strict=True)
                },
            }
        )

    # If we are eliding elements, add in the separator.
    separator_entry = {k: "..." for k in header}
    if eliding_elements:
        diff_information = (
            diff_information[:grab_first]
            + [separator_entry]
            + diff_information[-grab_last:]
        )
    table = PrettyTable(headers=header, paginate=False)
    # Get the values to print associated with the header
    # keys.
    table.add_rows([_lookup(v, header) for v in diff_information])
    table.print()


def _is_close(  # noqa: ANN202
    a: numpy.typing.NDArray,  # type: ignore[type-arg]
    b: numpy.typing.NDArray,  # type: ignore[type-arg]
    absolute_tolerance: float,
    relative_tolerance: float,
    equal_nan: bool,
):
    """Checks if the two input values are numerically within a tolerance.

    When the type is integral, then equality is checked. When the type is
    floating point, then this checks if the two input values are numerically the
    close using the $abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)$
    formula.

    Unlike Pythons's `math.isclose`, this implementation is symmetric. I.e.
    `isclose(a,b) == isclose(b,a)`.

    Args:
      a: The first value to compare.
      a: The second value to compare.
      absolute_tolerance: The absolute tolerance.
      relative_tolerance: The relative tolerance.
      equal_nan: Whether to compare NaN's as equal.

    Returns:
      A boolean vector where a and b are equal within the specified tolerance.
    """
    finite = np.isfinite(a) & np.isfinite(b)

    result = np.zeros_like(finite)
    result[finite] = (
        np.abs(a - b)
        <= np.maximum(
            absolute_tolerance,
            relative_tolerance * np.maximum(np.abs(a), np.abs(b)),
        )
    )[finite]
    # Explicit check for equality of infinite values.
    result[~finite] = a[~finite] == b[~finite]

    if equal_nan:
        both_nan = np.isnan(a) & np.isnan(b)
        result[both_nan] = both_nan[both_nan]

    return result
