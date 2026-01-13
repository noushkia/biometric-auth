"""Gabor filter-based IrisCode encoder (Daugman's algorithm)."""

import numpy as np
from typing import Optional


def create_gabor_kernel(
    size: int,
    wavelength: float,
    orientation: float,
    sigma: float,
    aspect_ratio: float = 0.5,
) -> np.ndarray:
    """
    Create a 2D Gabor kernel.

    Args:
        size: Kernel size (will be size x size).
        wavelength: Wavelength of sinusoidal factor (in pixels).
        orientation: Orientation of Gabor in radians.
        sigma: Standard deviation of Gaussian envelope.
        aspect_ratio: Spatial aspect ratio (ellipticity).

    Returns:
        Complex Gabor kernel (real + imaginary).
    """
    half = size // 2
    x, y = np.meshgrid(
        np.arange(-half, half + 1),
        np.arange(-half, half + 1),
    )

    # Rotation
    x_theta = x * np.cos(orientation) + y * np.sin(orientation)
    y_theta = -x * np.sin(orientation) + y * np.cos(orientation)

    # Gaussian envelope
    gaussian = np.exp(
        -(x_theta**2 + (aspect_ratio * y_theta) ** 2) / (2 * sigma**2)
    )

    # Complex sinusoid
    frequency = 2 * np.pi / wavelength
    sinusoid = np.exp(1j * frequency * x_theta)

    return gaussian * sinusoid


def create_gabor_bank(
    num_scales: int = 4,
    num_orientations: int = 6,
    kernel_size: int = 21,
    min_wavelength: float = 6.0,
    wavelength_ratio: float = 2.0,
    bandwidth: float = 1.0,
) -> list[np.ndarray]:
    """
    Create a bank of Gabor filters with multiple scales and orientations.

    Args:
        num_scales: Number of wavelength scales.
        num_orientations: Number of orientation angles.
        kernel_size: Size of each kernel.
        min_wavelength: Minimum wavelength.
        wavelength_ratio: Ratio between consecutive wavelengths.
        bandwidth: Bandwidth in octaves.

    Returns:
        List of complex Gabor kernels.
    """
    # Compute sigma from bandwidth
    sigma_on_wavelength = (
        (1 / np.pi) *
        np.sqrt(np.log(2) / 2) *
        (2**bandwidth + 1) / (2**bandwidth - 1)
    )

    kernels = []

    for scale in range(num_scales):
        wavelength = min_wavelength * (wavelength_ratio ** scale)
        sigma = sigma_on_wavelength * wavelength

        for orient_idx in range(num_orientations):
            orientation = orient_idx * np.pi / num_orientations

            kernel = create_gabor_kernel(
                kernel_size,
                wavelength,
                orientation,
                sigma,
            )
            kernels.append(kernel)

    return kernels


def create_log_gabor_filter(
    rows: int,
    cols: int,
    wavelength: float,
    sigma_on_f: float = 0.55,
) -> np.ndarray:
    """
    Create a Log-Gabor filter in frequency domain.

    Log-Gabor filters have zero DC component and better frequency response.

    Args:
        rows: Height of filter.
        cols: Width of filter.
        wavelength: Center wavelength.
        sigma_on_f: Ratio of sigma to center frequency.

    Returns:
        Log-Gabor filter in frequency domain.
    """
    # Create frequency coordinates
    u = np.fft.fftfreq(cols)
    v = np.fft.fftfreq(rows)
    u, v = np.meshgrid(u, v)

    # Radius from center
    radius = np.sqrt(u**2 + v**2)
    radius[0, 0] = 1  # Avoid log(0)

    # Center frequency
    fo = 1.0 / wavelength

    # Log-Gabor transfer function
    log_gabor = np.exp(
        -(np.log(radius / fo) ** 2) / (2 * (np.log(sigma_on_f)) ** 2)
    )

    # Set DC component to zero
    log_gabor[0, 0] = 0

    return log_gabor


class GaborIrisEncoder:
    """
    Daugman-style IrisCode encoder using 2D Gabor filters.

    Encodes normalized iris texture into a binary code by quantizing
    the phase of Gabor filter responses.
    """

    def __init__(
        self,
        num_scales: int = 4,
        num_orientations: int = 6,
        kernel_size: int = 21,
        block_size: tuple[int, int] = (8, 8),
    ):
        """
        Initialize Gabor encoder.

        Args:
            num_scales: Number of wavelength scales.
            num_orientations: Number of orientation angles.
            kernel_size: Size of Gabor kernels.
            block_size: Block size for pooling (height, width).
        """
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.kernel_size = kernel_size
        self.block_size = block_size

        self.gabor_bank = create_gabor_bank(
            num_scales,
            num_orientations,
            kernel_size,
        )

    def encode(
        self,
        normalized_iris: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode normalized iris into binary IrisCode.

        Args:
            normalized_iris: Normalized iris strip (e.g., 64x512).
            mask: Optional noise mask (1=valid, 0=occluded).

        Returns:
            Tuple of (iris_code, code_mask).
            - iris_code: Binary array of shape (num_blocks, 2*num_filters).
            - code_mask: Binary mask for valid code bits.
        """
        from scipy.signal import convolve2d

        h, w = normalized_iris.shape
        bh, bw = self.block_size

        # Number of blocks
        num_blocks_h = h // bh
        num_blocks_w = w // bw
        num_filters = len(self.gabor_bank)

        # Initialize output (2 bits per filter: real > 0, imag > 0)
        code_length = num_blocks_h * num_blocks_w * num_filters * 2
        iris_code = np.zeros(code_length, dtype=np.uint8)
        code_mask = np.ones(code_length, dtype=np.uint8)

        # Default mask if not provided
        if mask is None:
            mask = np.ones_like(normalized_iris)

        bit_idx = 0

        for kernel in self.gabor_bank:
            # Convolve with Gabor filter
            response = convolve2d(
                normalized_iris.astype(np.float32),
                kernel,
                mode='same',
                boundary='wrap',  # Handle circular boundary
            )

            # Process each block
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    # Extract block
                    block_response = response[
                        i * bh:(i + 1) * bh,
                        j * bw:(j + 1) * bw,
                    ]
                    block_mask = mask[
                        i * bh:(i + 1) * bh,
                        j * bw:(j + 1) * bw,
                    ]

                    # Sum over block
                    real_sum = block_response.real.sum()
                    imag_sum = block_response.imag.sum()

                    # Quantize phase to 2 bits
                    iris_code[bit_idx] = 1 if real_sum >= 0 else 0
                    iris_code[bit_idx + 1] = 1 if imag_sum >= 0 else 0

                    # Mark masked regions
                    if block_mask.sum() < block_mask.size * 0.5:
                        code_mask[bit_idx] = 0
                        code_mask[bit_idx + 1] = 0

                    bit_idx += 2

        return iris_code, code_mask

    @property
    def code_length(self) -> int:
        """Return the expected code length for a given iris size."""
        # Default for 64x512 with 8x8 blocks
        return 8 * 64 * len(self.gabor_bank) * 2


class LogGaborIrisEncoder:
    """
    IrisCode encoder using Log-Gabor filters.

    Log-Gabor filters have no DC component and provide better
    frequency response than standard Gabor filters.
    """

    def __init__(
        self,
        num_scales: int = 4,
        min_wavelength: float = 6.0,
        wavelength_ratio: float = 2.0,
        sigma_on_f: float = 0.55,
        num_angular_sectors: int = 8,
    ):
        """
        Initialize Log-Gabor encoder.

        Args:
            num_scales: Number of wavelength scales.
            min_wavelength: Minimum wavelength.
            wavelength_ratio: Ratio between consecutive wavelengths.
            sigma_on_f: Ratio of sigma to center frequency.
            num_angular_sectors: Number of angular sectors for encoding.
        """
        self.num_scales = num_scales
        self.min_wavelength = min_wavelength
        self.wavelength_ratio = wavelength_ratio
        self.sigma_on_f = sigma_on_f
        self.num_angular_sectors = num_angular_sectors

    def encode(
        self,
        normalized_iris: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode normalized iris using Log-Gabor filters.

        Args:
            normalized_iris: Normalized iris strip.
            mask: Optional noise mask.

        Returns:
            Tuple of (iris_code, code_mask).
        """
        h, w = normalized_iris.shape

        if mask is None:
            mask = np.ones_like(normalized_iris)

        # FFT of input
        fft_iris = np.fft.fft2(normalized_iris.astype(np.float32))

        code_bits = []
        mask_bits = []

        for scale in range(self.num_scales):
            wavelength = self.min_wavelength * (self.wavelength_ratio ** scale)

            # Create Log-Gabor filter
            log_gabor = create_log_gabor_filter(h, w, wavelength, self.sigma_on_f)

            # Apply filter in frequency domain
            filtered = np.fft.ifft2(fft_iris * log_gabor)

            # Divide into angular sectors
            sector_width = w // self.num_angular_sectors
            radial_samples = h

            for sector in range(self.num_angular_sectors):
                start_col = sector * sector_width
                end_col = (sector + 1) * sector_width

                # Average response in sector
                sector_response = filtered[:, start_col:end_col]
                sector_mask = mask[:, start_col:end_col]

                for row in range(radial_samples):
                    response = sector_response[row, :].mean()

                    # 2-bit phase quantization
                    code_bits.append(1 if response.real >= 0 else 0)
                    code_bits.append(1 if response.imag >= 0 else 0)

                    # Mask
                    valid = sector_mask[row, :].sum() > sector_width * 0.5
                    mask_bits.extend([1 if valid else 0, 1 if valid else 0])

        return np.array(code_bits, dtype=np.uint8), np.array(mask_bits, dtype=np.uint8)


def create_encoder(method: str = "gabor", **kwargs) -> GaborIrisEncoder | LogGaborIrisEncoder:
    """
    Factory function to create an iris encoder.

    Args:
        method: "gabor" or "log_gabor"
        **kwargs: Additional arguments for the encoder.

    Returns:
        Encoder instance.
    """
    if method == "gabor":
        return GaborIrisEncoder(**kwargs)
    elif method == "log_gabor":
        return LogGaborIrisEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoding method: {method}")
