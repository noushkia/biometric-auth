"""Iris segmentation algorithms for locating pupil and iris boundaries."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    """Result of iris segmentation."""

    pupil_center: tuple[int, int]
    pupil_radius: int
    iris_center: tuple[int, int]
    iris_radius: int
    mask: np.ndarray  # Binary mask of iris region (excluding pupil)
    confidence: float = 1.0

    def is_valid(self) -> bool:
        """Check if segmentation result is valid."""
        return (
            self.pupil_radius > 0
            and self.iris_radius > self.pupil_radius
            and self.confidence > 0.5
        )


class HoughCircleSegmenter:
    """
    Iris segmentation using Hough Circle Transform.

    Fast and simple approach suitable for high-quality images.
    """

    def __init__(
        self,
        pupil_radius_range: tuple[int, int] = (20, 80),
        iris_radius_range: tuple[int, int] = (80, 150),
        canny_threshold: int = 50,
        accumulator_threshold: int = 30,
    ):
        self.pupil_radius_range = pupil_radius_range
        self.iris_radius_range = iris_radius_range
        self.canny_threshold = canny_threshold
        self.accumulator_threshold = accumulator_threshold

    def segment(self, image: np.ndarray) -> Optional[SegmentationResult]:
        """
        Segment iris from eye image.

        Args:
            image: Grayscale eye image.

        Returns:
            SegmentationResult or None if segmentation fails.
        """
        # Preprocess
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Detect pupil (dark circle, smaller)
        pupil_circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=image.shape[0] // 4,
            param1=self.canny_threshold,
            param2=self.accumulator_threshold,
            minRadius=self.pupil_radius_range[0],
            maxRadius=self.pupil_radius_range[1],
        )

        if pupil_circles is None:
            return None

        # Take the first (strongest) pupil detection
        pupil = pupil_circles[0, 0]
        pupil_center = (int(pupil[0]), int(pupil[1]))
        pupil_radius = int(pupil[2])

        # Detect iris (larger circle around pupil)
        # Use edge detection on the region around the pupil
        iris_circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=image.shape[0] // 4,
            param1=self.canny_threshold,
            param2=self.accumulator_threshold + 10,
            minRadius=max(self.iris_radius_range[0], pupil_radius + 20),
            maxRadius=self.iris_radius_range[1],
        )

        if iris_circles is None:
            # Estimate iris as 2.5x pupil radius if detection fails
            iris_center = pupil_center
            iris_radius = int(pupil_radius * 2.5)
        else:
            iris = iris_circles[0, 0]
            iris_center = (int(iris[0]), int(iris[1]))
            iris_radius = int(iris[2])

        # Create binary mask
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, iris_center, iris_radius, 255, -1)
        cv2.circle(mask, pupil_center, pupil_radius, 0, -1)

        return SegmentationResult(
            pupil_center=pupil_center,
            pupil_radius=pupil_radius,
            iris_center=iris_center,
            iris_radius=iris_radius,
            mask=mask,
            confidence=0.8,
        )


class IntegroDifferentialSegmenter:
    """
    Daugman's Integro-Differential Operator for iris segmentation.

    More accurate than Hough but computationally expensive.
    """

    def __init__(
        self,
        pupil_radius_range: tuple[int, int] = (20, 80),
        iris_radius_range: tuple[int, int] = (80, 150),
        sigma: float = 0.5,
        step: int = 2,
    ):
        self.pupil_radius_range = pupil_radius_range
        self.iris_radius_range = iris_radius_range
        self.sigma = sigma
        self.step = step

    def _circular_integral(
        self,
        image: np.ndarray,
        center: tuple[int, int],
        radius: int,
        num_points: int = 360,
    ) -> float:
        """Compute integral along a circle."""
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)

        # Clip to image bounds
        x = np.clip(x, 0, image.shape[1] - 1).astype(int)
        y = np.clip(y, 0, image.shape[0] - 1).astype(int)

        return np.mean(image[y, x])

    def _search_circle(
        self,
        image: np.ndarray,
        search_center: tuple[int, int],
        radius_range: tuple[int, int],
        search_radius: int = 30,
    ) -> tuple[tuple[int, int], int, float]:
        """
        Search for optimal circle parameters.

        Returns:
            (center, radius, max_response)
        """
        best_response = 0
        best_center = search_center
        best_radius = radius_range[0]

        # Gaussian kernel for smoothing derivative
        kernel_size = int(4 * self.sigma + 1) | 1
        gaussian = cv2.getGaussianKernel(kernel_size, self.sigma)

        for cx in range(search_center[0] - search_radius, search_center[0] + search_radius, self.step):
            for cy in range(search_center[1] - search_radius, search_center[1] + search_radius, self.step):
                prev_integral = None

                for r in range(radius_range[0], radius_range[1], self.step):
                    integral = self._circular_integral(image, (cx, cy), r)

                    if prev_integral is not None:
                        # Approximate derivative
                        deriv = abs(integral - prev_integral)

                        if deriv > best_response:
                            best_response = deriv
                            best_center = (cx, cy)
                            best_radius = r

                    prev_integral = integral

        return best_center, best_radius, best_response

    def segment(self, image: np.ndarray) -> Optional[SegmentationResult]:
        """
        Segment iris using integro-differential operator.

        Args:
            image: Grayscale eye image.

        Returns:
            SegmentationResult or None if segmentation fails.
        """
        # Preprocess
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Start search from image center
        height, width = image.shape
        initial_center = (width // 2, height // 2)

        # Search for pupil (dark circle, high gradient at boundary)
        pupil_center, pupil_radius, pupil_conf = self._search_circle(
            blurred,
            initial_center,
            self.pupil_radius_range,
            search_radius=width // 4,
        )

        if pupil_conf < 1:  # Threshold for minimum contrast
            return None

        # Search for iris around pupil
        iris_center, iris_radius, iris_conf = self._search_circle(
            blurred,
            pupil_center,
            self.iris_radius_range,
            search_radius=30,
        )

        # Create binary mask
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, iris_center, iris_radius, 255, -1)
        cv2.circle(mask, pupil_center, pupil_radius, 0, -1)

        # Normalize confidence
        confidence = min(pupil_conf, iris_conf) / 50  # Approximate normalization

        return SegmentationResult(
            pupil_center=pupil_center,
            pupil_radius=pupil_radius,
            iris_center=iris_center,
            iris_radius=iris_radius,
            mask=mask,
            confidence=min(confidence, 1.0),
        )


def create_segmenter(method: str = "hough", **kwargs) -> HoughCircleSegmenter | IntegroDifferentialSegmenter:
    """
    Factory function to create a segmenter.

    Args:
        method: "hough" or "integro_differential"
        **kwargs: Additional arguments for the segmenter.

    Returns:
        Segmenter instance.
    """
    if method == "hough":
        return HoughCircleSegmenter(**kwargs)
    elif method == "integro_differential":
        return IntegroDifferentialSegmenter(**kwargs)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
