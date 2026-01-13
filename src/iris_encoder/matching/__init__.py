"""Matching module for iris template comparison and authentication."""

from .matcher import (
    IrisMatcher,
    MatchResult,
    GalleryDatabase,
    hamming_distance,
    hamming_distance_with_rotation,
    cosine_similarity,
)

__all__ = [
    "IrisMatcher",
    "MatchResult",
    "GalleryDatabase",
    "hamming_distance",
    "hamming_distance_with_rotation",
    "cosine_similarity",
]
