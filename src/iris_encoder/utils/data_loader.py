"""Data loading utilities for iris datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class IrisImage:
    """Container for an iris image with metadata."""

    image: np.ndarray
    subject_id: str
    eye: str  # 'L' or 'R'
    sample_id: int
    source_path: Path
    is_normalized: bool = False


class IITDDatasetLoader:
    """Loader for IIT Delhi Iris Database v1.0."""

    def __init__(self, dataset_path: str | Path):
        self.path = Path(dataset_path)
        self.raw_path = self.path
        self.normalized_path = self.path / "Normalized_Images"

    def load_normalized(self, subject_id: str | None = None) -> Iterator[IrisImage]:
        """
        Load pre-normalized iris images.

        Args:
            subject_id: Optional specific subject ID (e.g., "001") to load.

        Yields:
            IrisImage objects with normalized iris strips.
        """
        if not self.normalized_path.exists():
            raise FileNotFoundError(f"Normalized images not found at {self.normalized_path}")

        pattern = f"{subject_id}_*.bmp" if subject_id else "*.bmp"

        for img_path in sorted(self.normalized_path.glob(pattern)):
            # Format: 001_1.bmp -> subject 001, sample 1
            parts = img_path.stem.split("_")
            subj_id = parts[0]
            sample_idx = int(parts[1])

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            yield IrisImage(
                image=img,
                subject_id=subj_id,
                eye="L",  # Normalized images are from left eye
                sample_id=sample_idx,
                source_path=img_path,
                is_normalized=True,
            )

    def load_raw(self, subject_id: str | None = None) -> Iterator[IrisImage]:
        """
        Load raw iris images requiring segmentation.

        Args:
            subject_id: Optional specific subject ID to load.

        Yields:
            IrisImage objects with raw eye images.
        """
        if subject_id:
            subject_dirs = [self.raw_path / subject_id]
        else:
            subject_dirs = sorted(
                d for d in self.raw_path.iterdir()
                if d.is_dir() and d.name.isdigit()
            )

        for subj_dir in subject_dirs:
            subj_id = subj_dir.name

            for img_path in sorted(subj_dir.glob("*.bmp")):
                # Format: 01_L.bmp -> sample 1, left eye
                parts = img_path.stem.split("_")
                sample_idx = int(parts[0])
                eye = parts[1] if len(parts) > 1 else "L"

                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                yield IrisImage(
                    image=img,
                    subject_id=subj_id,
                    eye=eye,
                    sample_id=sample_idx,
                    source_path=img_path,
                    is_normalized=False,
                )

    def get_subject_ids(self) -> list[str]:
        """Get list of all subject IDs in the dataset."""
        return sorted(
            d.name for d in self.raw_path.iterdir()
            if d.is_dir() and d.name.isdigit()
        )

    def __len__(self) -> int:
        """Return number of subjects."""
        return len(self.get_subject_ids())


class CorneaIrisMultimodalLoader:
    """Loader for Data_CORNEA_IRIS_Multimodal dataset."""

    def __init__(self, dataset_path: str | Path):
        self.path = Path(dataset_path)

    def load_iris(self, subject_id: str | None = None, eye: str = "L") -> Iterator[IrisImage]:
        """
        Load iris images from the multimodal dataset.

        Args:
            subject_id: Optional specific subject ID to load.
            eye: 'L' for left eye, 'R' for right eye.

        Yields:
            IrisImage objects.
        """
        if subject_id:
            subject_dirs = [self.path / subject_id]
        else:
            subject_dirs = sorted(
                d for d in self.path.iterdir()
                if d.is_dir() and d.name.isdigit()
            )

        for subj_dir in subject_dirs:
            subj_id = subj_dir.name
            iris_dir = subj_dir / "Iris" / eye

            if not iris_dir.exists():
                continue

            for idx, img_path in enumerate(sorted(iris_dir.glob("*.*")), 1):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                yield IrisImage(
                    image=img,
                    subject_id=subj_id,
                    eye=eye,
                    sample_id=idx,
                    source_path=img_path,
                    is_normalized=False,
                )


def load_dataset(dataset_name: str, dataset_path: str | Path) -> IITDDatasetLoader | CorneaIrisMultimodalLoader:
    """
    Factory function to load a dataset by name.

    Args:
        dataset_name: One of 'IITD', 'cornea_iris'
        dataset_path: Path to the dataset directory.

    Returns:
        Appropriate dataset loader instance.
    """
    loaders = {
        "IITD": IITDDatasetLoader,
        "iitd": IITDDatasetLoader,
        "cornea_iris": CorneaIrisMultimodalLoader,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")

    return loaders[dataset_name](dataset_path)
