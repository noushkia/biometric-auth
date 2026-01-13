"""Iris normalization using Daugman's rubber sheet model."""

import numpy as np
from scipy.ndimage import map_coordinates


def normalize_iris(
    image: np.ndarray,
    pupil_center: tuple[int, int],
    pupil_radius: int,
    iris_center: tuple[int, int],
    iris_radius: int,
    radial_resolution: int = 64,
    angular_resolution: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize iris region using Daugman's rubber sheet model.

    Maps iris region from Cartesian to polar coordinates, unwrapping the
    annular iris region into a rectangular strip.

    Args:
        image: Grayscale eye image.
        pupil_center: (x, y) center of pupil.
        pupil_radius: Radius of pupil in pixels.
        iris_center: (x, y) center of iris.
        iris_radius: Radius of iris in pixels.
        radial_resolution: Height of output (radial samples).
        angular_resolution: Width of output (angular samples).

    Returns:
        Tuple of (normalized_iris, mask).
        - normalized_iris: Rectangular iris strip of shape (radial_resolution, angular_resolution).
        - mask: Binary mask indicating valid pixels.
    """
    # Create output arrays
    normalized = np.zeros((radial_resolution, angular_resolution), dtype=np.float32)
    mask = np.ones((radial_resolution, angular_resolution), dtype=np.uint8)

    # Angular samples (0 to 2π)
    theta = np.linspace(0, 2 * np.pi, angular_resolution, endpoint=False)

    # Radial samples (0 to 1, normalized distance from pupil to iris)
    r = np.linspace(0, 1, radial_resolution)

    # Create meshgrid
    theta_grid, r_grid = np.meshgrid(theta, r)

    # Handle non-concentric pupil and iris
    # Pupil boundary points
    pupil_x = pupil_center[0] + pupil_radius * np.cos(theta_grid)
    pupil_y = pupil_center[1] + pupil_radius * np.sin(theta_grid)

    # Iris boundary points
    iris_x = iris_center[0] + iris_radius * np.cos(theta_grid)
    iris_y = iris_center[1] + iris_radius * np.sin(theta_grid)

    # Linear interpolation between pupil and iris boundary
    x_coords = (1 - r_grid) * pupil_x + r_grid * iris_x
    y_coords = (1 - r_grid) * pupil_y + r_grid * iris_y

    # Sample the image using bilinear interpolation
    # Note: map_coordinates expects (row, col) = (y, x)
    normalized = map_coordinates(
        image.astype(np.float32),
        [y_coords, x_coords],
        order=1,  # Bilinear interpolation
        mode="constant",
        cval=0,
    )

    # Create mask for out-of-bounds pixels
    height, width = image.shape
    out_of_bounds = (
        (x_coords < 0) | (x_coords >= width) |
        (y_coords < 0) | (y_coords >= height)
    )
    mask[out_of_bounds] = 0

    return normalized.astype(np.uint8), mask


def enhance_normalized_iris(
    normalized: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """
    Enhance normalized iris using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        normalized: Normalized iris strip.
        clip_limit: Contrast limit for CLAHE.
        tile_size: Size of grid for CLAHE.

    Returns:
        Enhanced iris strip.
    """
    import cv2

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(normalized.astype(np.uint8))


def denoise_normalized_iris(
    normalized: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Apply Gaussian denoising to normalized iris.

    Args:
        normalized: Normalized iris strip.
        kernel_size: Size of Gaussian kernel (must be odd).

    Returns:
        Denoised iris strip.
    """
    import cv2

    return cv2.GaussianBlur(normalized, (kernel_size, kernel_size), 0)


class IrisNormalizer:
    """
    Iris normalizer that handles the complete normalization pipeline.

    For pre-normalized images (e.g., from IITD Normalized_Images), this class
    can still be used to apply enhancement.
    """

    def __init__(
        self,
        radial_resolution: int = 64,
        angular_resolution: int = 512,
        enhance: bool = True,
        denoise: bool = True,
    ):
        self.radial_resolution = radial_resolution
        self.angular_resolution = angular_resolution
        self.enhance = enhance
        self.denoise = denoise

    def normalize(
        self,
        image: np.ndarray,
        pupil_center: tuple[int, int],
        pupil_radius: int,
        iris_center: tuple[int, int],
        iris_radius: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Full normalization pipeline: unwrap + enhance + denoise.

        Returns:
            Tuple of (normalized_iris, mask).
        """
        normalized, mask = normalize_iris(
            image,
            pupil_center,
            pupil_radius,
            iris_center,
            iris_radius,
            self.radial_resolution,
            self.angular_resolution,
        )

        if self.denoise:
            normalized = denoise_normalized_iris(normalized)

        if self.enhance:
            normalized = enhance_normalized_iris(normalized)

        return normalized, mask

    def process_prenormalized(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process pre-normalized iris image (e.g., from IITD Normalized_Images).

        Only applies enhancement and resizing if needed.

        Args:
            image: Pre-normalized iris strip.

        Returns:
            Tuple of (processed_iris, mask).
        """
        import cv2

        # Resize if dimensions don't match expected
        if image.shape != (self.radial_resolution, self.angular_resolution):
            image = cv2.resize(
                image,
                (self.angular_resolution, self.radial_resolution),
                interpolation=cv2.INTER_LINEAR,
            )

        mask = np.ones(image.shape, dtype=np.uint8)

        if self.denoise:
            image = denoise_normalized_iris(image)

        if self.enhance:
            image = enhance_normalized_iris(image)

        return image, mask
