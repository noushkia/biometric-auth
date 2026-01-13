#!/usr/bin/env python3
"""
Evaluate iris recognition performance.

Computes:
- Genuine vs imposter distributions
- ROC curve (FAR vs FRR)
- Equal Error Rate (EER)
- d' (decidability)

Usage:
    python scripts/evaluate.py --codes outputs/iris_codes.npz --report outputs/report.html
"""

import argparse
from pathlib import Path
import sys

import numpy as np


def compute_genuine_imposter_scores(codes, masks, labels):
    """
    Compute genuine (same person) and imposter (different person) scores.

    Returns:
        Tuple of (genuine_scores, imposter_scores) as lists of Hamming distances.
    """
    from iris_encoder.matching.matcher import hamming_distance_with_rotation

    genuine_scores = []
    imposter_scores = []

    n = len(codes)
    unique_labels = list(set(labels))

    print(f"Computing pairwise distances for {n} templates...")

    # Group by label
    label_to_indices = {label: [] for label in unique_labels}
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)

    # Compute genuine scores (same identity pairs)
    for label in unique_labels:
        indices = label_to_indices[label]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                dist, _ = hamming_distance_with_rotation(
                    codes[idx1], masks[idx1],
                    codes[idx2], masks[idx2],
                )
                genuine_scores.append(dist)

    # Compute imposter scores (different identity pairs)
    # Sample to avoid O(n²) complexity
    max_imposter_pairs = 10000
    imposter_pairs = []

    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i < j:
                for idx1 in label_to_indices[label1][:2]:  # Max 2 per identity
                    for idx2 in label_to_indices[label2][:2]:
                        imposter_pairs.append((idx1, idx2))

    if len(imposter_pairs) > max_imposter_pairs:
        import random
        imposter_pairs = random.sample(imposter_pairs, max_imposter_pairs)

    for idx1, idx2 in imposter_pairs:
        dist, _ = hamming_distance_with_rotation(
            codes[idx1], masks[idx1],
            codes[idx2], masks[idx2],
        )
        imposter_scores.append(dist)

    return genuine_scores, imposter_scores


def compute_eer(genuine_scores, imposter_scores, num_thresholds=1000):
    """
    Compute Equal Error Rate.

    Returns:
        Tuple of (EER, threshold at EER, FAR array, FRR array, thresholds).
    """
    all_scores = genuine_scores + imposter_scores
    min_score = min(all_scores)
    max_score = max(all_scores)

    thresholds = np.linspace(min_score, max_score, num_thresholds)
    far_arr = []
    frr_arr = []

    for threshold in thresholds:
        # FAR: imposter accepted (score < threshold)
        far = np.mean([s < threshold for s in imposter_scores])
        # FRR: genuine rejected (score >= threshold)
        frr = np.mean([s >= threshold for s in genuine_scores])

        far_arr.append(far)
        frr_arr.append(frr)

    far_arr = np.array(far_arr)
    frr_arr = np.array(frr_arr)

    # Find EER (where FAR ≈ FRR)
    diff = np.abs(far_arr - frr_arr)
    eer_idx = np.argmin(diff)
    eer = (far_arr[eer_idx] + frr_arr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    return eer, eer_threshold, far_arr, frr_arr, thresholds


def compute_decidability(genuine_scores, imposter_scores):
    """
    Compute d' (d-prime) decidability index.

    d' = |μ_imposter - μ_genuine| / sqrt(0.5 * (σ_genuine² + σ_imposter²))
    """
    mu_genuine = np.mean(genuine_scores)
    mu_imposter = np.mean(imposter_scores)
    sigma_genuine = np.std(genuine_scores)
    sigma_imposter = np.std(imposter_scores)

    d_prime = abs(mu_imposter - mu_genuine) / np.sqrt(
        0.5 * (sigma_genuine**2 + sigma_imposter**2)
    )

    return d_prime


def main():
    parser = argparse.ArgumentParser(description="Evaluate iris recognition performance")
    parser.add_argument(
        "--codes",
        type=str,
        required=True,
        help="Path to encoded templates (.npz file)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="HTML report output path (optional)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show matplotlib plots",
    )

    args = parser.parse_args()

    # Load encoded templates
    data = np.load(args.codes, allow_pickle=True)
    codes = data["codes"]
    masks = data["masks"]
    labels = data["labels"]

    print(f"Loaded {len(codes)} templates")
    print(f"Unique identities: {len(set(labels))}")
    print(f"Code length: {codes.shape[1]} bits")

    # Compute scores
    genuine_scores, imposter_scores = compute_genuine_imposter_scores(codes, masks, labels)

    print(f"\nGenuine pairs: {len(genuine_scores)}")
    print(f"Imposter pairs: {len(imposter_scores)}")

    # Compute metrics
    eer, eer_threshold, far_arr, frr_arr, thresholds = compute_eer(
        genuine_scores, imposter_scores
    )
    d_prime = compute_decidability(genuine_scores, imposter_scores)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Equal Error Rate (EER): {100 * eer:.2f}%")
    print(f"EER Threshold: {eer_threshold:.4f}")
    print(f"Decidability (d'): {d_prime:.3f}")
    print(f"Genuine mean HD: {np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}")
    print(f"Imposter mean HD: {np.mean(imposter_scores):.4f} ± {np.std(imposter_scores):.4f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Distribution plot
            axes[0].hist(genuine_scores, bins=50, alpha=0.7, label="Genuine", density=True)
            axes[0].hist(imposter_scores, bins=50, alpha=0.7, label="Imposter", density=True)
            axes[0].axvline(eer_threshold, color="red", linestyle="--", label=f"EER={100*eer:.1f}%")
            axes[0].set_xlabel("Hamming Distance")
            axes[0].set_ylabel("Density")
            axes[0].set_title("Score Distributions")
            axes[0].legend()

            # ROC curve
            axes[1].plot(far_arr, 1 - frr_arr)
            axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
            axes[1].set_xlabel("False Accept Rate (FAR)")
            axes[1].set_ylabel("True Accept Rate (1 - FRR)")
            axes[1].set_title("ROC Curve")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("\nMatplotlib not available for plotting.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
