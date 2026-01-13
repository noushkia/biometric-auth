"""Main iris recognition pipeline combining all components."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .preprocessing.normalization import IrisNormalizer
from .preprocessing.segmentation import HoughCircleSegmenter, SegmentationResult, create_segmenter
from .encoding.gabor_encoder import GaborIrisEncoder, LogGaborIrisEncoder, create_encoder
from .matching.matcher import IrisMatcher, MatchResult, GalleryDatabase


@dataclass
class EncodingResult:
    """Result of iris encoding."""

    code: np.ndarray
    mask: np.ndarray
    normalized_iris: np.ndarray
    segmentation: Optional[SegmentationResult] = None
    subject_id: Optional[str] = None


class IrisPipeline:
    """
    Complete iris recognition pipeline.

    Combines segmentation, normalization, encoding, and matching
    into a single easy-to-use interface.
    """

    def __init__(
        self,
        segmentation_method: str = "hough",
        encoding_method: str = "gabor",
        use_prenormalized: bool = False,
        threshold: float = 0.32,
        **kwargs,
    ):
        """
        Initialize pipeline.

        Args:
            segmentation_method: "hough" or "integro_differential".
            encoding_method: "gabor" or "log_gabor".
            use_prenormalized: If True, skip segmentation/normalization.
            threshold: Matching threshold (HD < threshold = match).
            **kwargs: Additional arguments passed to components.
        """
        self.use_prenormalized = use_prenormalized
        self.threshold = threshold

        # Initialize components
        if not use_prenormalized:
            self.segmenter = create_segmenter(segmentation_method)
        else:
            self.segmenter = None

        self.normalizer = IrisNormalizer(
            radial_resolution=64,
            angular_resolution=512,
            enhance=True,
            denoise=True,
        )

        self.encoder = create_encoder(encoding_method)
        self.matcher = IrisMatcher(
            code_type="iriscode",
            threshold=threshold,
        )
        self.gallery = GalleryDatabase()

    def encode(
        self,
        image: np.ndarray,
        subject_id: Optional[str] = None,
    ) -> Optional[EncodingResult]:
        """
        Encode an iris image into a binary template.

        Args:
            image: Grayscale iris/eye image.
            subject_id: Optional subject identifier.

        Returns:
            EncodingResult or None if encoding fails.
        """
        if self.use_prenormalized:
            # Image is already normalized
            normalized, mask = self.normalizer.process_prenormalized(image)
            segmentation = None
        else:
            # Full pipeline: segment → normalize
            if self.segmenter is None:
                raise RuntimeError("Segmenter not initialized")

            segmentation = self.segmenter.segment(image)
            if segmentation is None or not segmentation.is_valid():
                return None

            normalized, mask = self.normalizer.normalize(
                image,
                segmentation.pupil_center,
                segmentation.pupil_radius,
                segmentation.iris_center,
                segmentation.iris_radius,
            )

        # Encode
        code, code_mask = self.encoder.encode(normalized, mask)

        return EncodingResult(
            code=code,
            mask=code_mask,
            normalized_iris=normalized,
            segmentation=segmentation,
            subject_id=subject_id,
        )

    def enroll(
        self,
        image: np.ndarray,
        identity_id: str,
    ) -> bool:
        """
        Enroll a new identity into the gallery.

        Args:
            image: Grayscale iris/eye image.
            identity_id: Unique identifier for the person.

        Returns:
            True if enrollment succeeded, False otherwise.
        """
        result = self.encode(image, identity_id)
        if result is None:
            return False

        self.gallery.enroll(identity_id, result.code, result.mask)
        return True

    def verify(
        self,
        query_image: np.ndarray,
        claimed_identity: str,
    ) -> tuple[bool, float]:
        """
        1:1 verification against claimed identity.

        Args:
            query_image: Query iris image.
            claimed_identity: Identity being claimed.

        Returns:
            Tuple of (is_match, confidence).
        """
        query = self.encode(query_image)
        if query is None:
            return False, 0.0

        templates = self.gallery.get_identity_templates(claimed_identity)
        if not templates:
            return False, 0.0

        # Compare against all enrolled templates for this identity
        best_match = None
        for template, mask in templates:
            result = self.matcher.compare(
                query.code, template,
                query.mask, mask,
            )
            if best_match is None or result.distance < best_match.distance:
                best_match = result

        return best_match.is_match, best_match.confidence

    def identify(
        self,
        query_image: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str, MatchResult]]:
        """
        1:N identification against gallery.

        Args:
            query_image: Query iris image.
            top_k: Number of top matches to return.

        Returns:
            List of (identity_id, MatchResult) sorted by best match.
        """
        query = self.encode(query_image)
        if query is None:
            return []

        return self.matcher.identify(
            query.code,
            self.gallery.get_gallery(),
            query.mask,
            top_k,
        )


def quick_demo():
    """Quick demonstration of the pipeline."""
    import cv2
    from pathlib import Path

    print("=== Iris Biometric Encoding Demo ===\n")

    # Initialize pipeline for pre-normalized images
    pipeline = IrisPipeline(
        encoding_method="gabor",
        use_prenormalized=True,
        threshold=0.32,
    )

    # Look for IITD normalized images
    iitd_path = Path("datasets/IITD_database/Normalized_Images")
    if not iitd_path.exists():
        print(f"Dataset not found at {iitd_path}")
        print("Please run from the biometric-auth directory.")
        return

    # Load first few images
    images = sorted(iitd_path.glob("*.bmp"))[:20]  # First 20 images

    print(f"Found {len(images)} images. Processing first 20...\n")

    # Enroll first 10 as gallery
    print("Enrolling subjects into gallery...")
    for img_path in images[:10]:
        subject_id = img_path.stem.split("_")[0]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if pipeline.enroll(img, subject_id):
            print(f"  Enrolled: {subject_id}")

    print(f"\nGallery size: {pipeline.gallery.num_identities()} identities, "
          f"{len(pipeline.gallery)} templates\n")

    # Test identification with remaining images
    print("Testing identification...")
    correct = 0
    total = 0

    for img_path in images[10:]:
        true_subject = img_path.stem.split("_")[0]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        results = pipeline.identify(img, top_k=3)

        if results:
            top_match = results[0]
            is_correct = top_match[0] == true_subject
            correct += is_correct
            total += 1

            print(f"  Query: {true_subject} → Top match: {top_match[0]} "
                  f"(HD={top_match[1].distance:.3f}) "
                  f"{'✓' if is_correct else '✗'}")

    if total > 0:
        print(f"\nRank-1 accuracy: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    quick_demo()
