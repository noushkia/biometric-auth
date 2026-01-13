#!/usr/bin/env python3
"""
Encode a dataset of iris images into binary templates.

Usage:
    python scripts/encode_dataset.py --dataset IITD --encoder gabor --output outputs/codes.npz
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Encode iris dataset into templates")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["IITD", "cornea_iris"],
        help="Dataset to encode",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset (default: datasets/{dataset})",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="gabor",
        choices=["gabor", "log_gabor"],
        help="Encoding method",
    )
    parser.add_argument(
        "--prenormalized",
        action="store_true",
        help="Use pre-normalized images (IITD only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/iris_codes.npz",
        help="Output file path",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from iris_encoder.utils.data_loader import IITDDatasetLoader, CorneaIrisMultimodalLoader
    from iris_encoder.pipeline import IrisPipeline

    # Determine dataset path
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        if args.dataset == "IITD":
            dataset_path = Path("datasets/IITD_database")
        elif args.dataset == "cornea_iris":
            dataset_path = Path("datasets/Data_CORNEA_IRIS_Multimodal")

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"Loading dataset from: {dataset_path}")

    # Initialize pipeline
    pipeline = IrisPipeline(
        encoding_method=args.encoder,
        use_prenormalized=args.prenormalized,
    )

    # Initialize loader
    if args.dataset == "IITD":
        loader = IITDDatasetLoader(dataset_path)
        if args.prenormalized:
            images = list(loader.load_normalized())
        else:
            images = list(loader.load_raw())
    else:
        loader = CorneaIrisMultimodalLoader(dataset_path)
        images = list(loader.load_iris())

    print(f"Found {len(images)} images")

    # Limit subjects if requested
    if args.max_subjects:
        subject_ids = sorted(set(img.subject_id for img in images))[:args.max_subjects]
        images = [img for img in images if img.subject_id in subject_ids]
        print(f"Limited to {len(images)} images from {len(subject_ids)} subjects")

    # Encode all images
    codes = []
    masks = []
    labels = []
    eyes = []
    failed = 0

    print("\nEncoding images...")
    for iris_image in tqdm(images, desc="Encoding"):
        result = pipeline.encode(iris_image.image, iris_image.subject_id)

        if result is None:
            failed += 1
            continue

        codes.append(result.code)
        masks.append(result.mask)
        labels.append(iris_image.subject_id)
        eyes.append(iris_image.eye)

    print(f"\nEncoded: {len(codes)} images")
    print(f"Failed: {failed} images")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        codes=np.array(codes),
        masks=np.array(masks),
        labels=np.array(labels),
        eyes=np.array(eyes),
        encoder=args.encoder,
        dataset=args.dataset,
    )

    print(f"\nSaved to: {output_path}")
    print(f"Code shape: {np.array(codes).shape}")


if __name__ == "__main__":
    main()
