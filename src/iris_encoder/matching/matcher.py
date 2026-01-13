"""Iris matching algorithms for verification and identification."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MatchResult:
    """Result of an iris match comparison."""

    distance: float  # Lower is more similar
    is_match: bool
    rotation_shift: int  # Optimal rotation applied
    confidence: float  # 1 - distance (for compatible confidence score)

    @property
    def similarity(self) -> float:
        """Return similarity score (1 = identical, 0 = completely different)."""
        return 1.0 - self.distance


def hamming_distance(
    code1: np.ndarray,
    mask1: np.ndarray,
    code2: np.ndarray,
    mask2: np.ndarray,
) -> float:
    """
    Compute normalized Hamming distance between two IrisCodes.

    Only compares bits where both masks are valid.

    Args:
        code1: First binary IrisCode.
        mask1: Mask for code1 (1=valid, 0=masked).
        code2: Second binary IrisCode.
        mask2: Mask for code2.

    Returns:
        Normalized Hamming distance in [0, 1].
    """
    # Combined mask
    valid_mask = mask1 & mask2

    num_valid = valid_mask.sum()
    if num_valid == 0:
        return 1.0  # No valid bits to compare

    # XOR to find different bits
    different = (code1 ^ code2) & valid_mask

    return different.sum() / num_valid


def hamming_distance_with_rotation(
    code1: np.ndarray,
    mask1: np.ndarray,
    code2: np.ndarray,
    mask2: np.ndarray,
    max_rotation: int = 15,
    bits_per_row: Optional[int] = None,
) -> tuple[float, int]:
    """
    Compute Hamming distance with rotation compensation.

    Iris codes may be misaligned due to head tilt. This function
    tries multiple rotational shifts and returns the minimum distance.

    Args:
        code1: First binary IrisCode.
        mask1: Mask for code1.
        code2: Second binary IrisCode.
        mask2: Mask for code2.
        max_rotation: Maximum rotation in each direction (number of shifts).
        bits_per_row: Number of bits in one row (for proper circular shift).

    Returns:
        Tuple of (min_distance, best_shift).
    """
    min_distance = 1.0
    best_shift = 0

    for shift in range(-max_rotation, max_rotation + 1):
        # Circular shift
        shifted_code2 = np.roll(code2, shift)
        shifted_mask2 = np.roll(mask2, shift)

        dist = hamming_distance(code1, mask1, shifted_code2, shifted_mask2)

        if dist < min_distance:
            min_distance = dist
            best_shift = shift

    return min_distance, best_shift


def cosine_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector.
        embedding2: Second embedding vector.

    Returns:
        Cosine similarity in [-1, 1], where 1 is identical.
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(embedding1, embedding2) / (norm1 * norm2)


def euclidean_distance(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Compute Euclidean distance between two embeddings.

    Args:
        embedding1: First embedding vector.
        embedding2: Second embedding vector.
        normalize: If True, normalize embeddings before computing distance.

    Returns:
        Euclidean distance (normalized if requested).
    """
    if normalize:
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 > 0:
            embedding1 = embedding1 / norm1
        if norm2 > 0:
            embedding2 = embedding2 / norm2

    return float(np.linalg.norm(embedding1 - embedding2))


class IrisMatcher:
    """
    Iris template matcher supporting both IrisCodes and CNN embeddings.
    """

    def __init__(
        self,
        code_type: str = "iriscode",
        threshold: float = 0.32,
        max_rotation: int = 15,
    ):
        """
        Initialize matcher.

        Args:
            code_type: "iriscode" for binary codes, "embedding" for CNN vectors.
            threshold: Decision threshold (HD < threshold = match for iriscode,
                       similarity > threshold = match for embeddings).
            max_rotation: Maximum rotation shifts for IrisCode matching.
        """
        self.code_type = code_type
        self.threshold = threshold
        self.max_rotation = max_rotation

    def compare(
        self,
        query: np.ndarray,
        template: np.ndarray,
        query_mask: Optional[np.ndarray] = None,
        template_mask: Optional[np.ndarray] = None,
    ) -> MatchResult:
        """
        Compare query against template.

        Args:
            query: Query iris code or embedding.
            template: Template iris code or embedding.
            query_mask: Mask for query (IrisCode only).
            template_mask: Mask for template (IrisCode only).

        Returns:
            MatchResult with distance and decision.
        """
        if self.code_type == "iriscode":
            if query_mask is None:
                query_mask = np.ones_like(query)
            if template_mask is None:
                template_mask = np.ones_like(template)

            distance, shift = hamming_distance_with_rotation(
                query, query_mask,
                template, template_mask,
                self.max_rotation,
            )

            return MatchResult(
                distance=distance,
                is_match=distance < self.threshold,
                rotation_shift=shift,
                confidence=1.0 - distance,
            )

        else:  # embedding
            similarity = cosine_similarity(query, template)
            distance = 1.0 - similarity

            return MatchResult(
                distance=distance,
                is_match=similarity > self.threshold,
                rotation_shift=0,
                confidence=similarity,
            )

    def verify(
        self,
        query: np.ndarray,
        template: np.ndarray,
        query_mask: Optional[np.ndarray] = None,
        template_mask: Optional[np.ndarray] = None,
    ) -> bool:
        """
        1:1 verification - is this the same person?

        Returns:
            True if match, False otherwise.
        """
        result = self.compare(query, template, query_mask, template_mask)
        return result.is_match

    def identify(
        self,
        query: np.ndarray,
        gallery: list[tuple[str, np.ndarray, Optional[np.ndarray]]],
        query_mask: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> list[tuple[str, MatchResult]]:
        """
        1:N identification - find matching identities from gallery.

        Args:
            query: Query iris code or embedding.
            gallery: List of (identity_id, template, mask) tuples.
            query_mask: Mask for query.
            top_k: Number of top matches to return.

        Returns:
            List of (identity_id, MatchResult) sorted by best match.
        """
        results = []

        for identity_id, template, template_mask in gallery:
            result = self.compare(query, template, query_mask, template_mask)
            results.append((identity_id, result))

        # Sort by distance (ascending for IrisCode, descending similarity for embeddings)
        results.sort(key=lambda x: x[1].distance)

        return results[:top_k]


class GalleryDatabase:
    """
    Simple in-memory database for enrolled iris templates.
    """

    def __init__(self, code_type: str = "iriscode"):
        self.code_type = code_type
        self.templates: dict[str, list[tuple[np.ndarray, Optional[np.ndarray]]]] = {}

    def enroll(
        self,
        identity_id: str,
        template: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ):
        """
        Enroll a new template for an identity.

        Multiple templates per identity are supported for improved accuracy.
        """
        if identity_id not in self.templates:
            self.templates[identity_id] = []

        self.templates[identity_id].append((template, mask))

    def get_gallery(self) -> list[tuple[str, np.ndarray, Optional[np.ndarray]]]:
        """
        Get all templates as a flat list for identification.

        Returns:
            List of (identity_id, template, mask) tuples.
        """
        gallery = []
        for identity_id, templates in self.templates.items():
            for template, mask in templates:
                gallery.append((identity_id, template, mask))
        return gallery

    def get_identity_templates(
        self,
        identity_id: str,
    ) -> list[tuple[np.ndarray, Optional[np.ndarray]]]:
        """Get all templates for a specific identity."""
        return self.templates.get(identity_id, [])

    def __len__(self) -> int:
        """Return total number of templates."""
        return sum(len(t) for t in self.templates.values())

    def num_identities(self) -> int:
        """Return number of enrolled identities."""
        return len(self.templates)
