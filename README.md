# Iris Biometric Encoding System

A Python-based iris recognition system for encoding iris scans into feature vectors for biometric authentication.

## Features

- **Multiple Encoding Algorithms**:

  - Daugman IrisCode (2D Gabor wavelets)
  - Log-Gabor encoder
  - CNN-based deep learning encoder (ResNet/VGG backbone)

- **Preprocessing Pipeline**:

  - Iris segmentation (Integro-Differential Operator, Hough Transform)
  - Rubber sheet normalization
  - Image enhancement (CLAHE, noise filtering)

- **Authentication System**:
  - Hamming distance matching for IrisCodes
  - Cosine similarity for CNN embeddings
  - Verification (1:1) and Identification (1:N) modes

## Datasets

| Dataset                     | Subjects | Images | Description                  |
| --------------------------- | -------- | ------ | ---------------------------- |
| IITD_database               | 218      | 2,180  | IIT Delhi Iris Database v1.0 |
| Data_CORNEA_IRIS_Multimodal | 39       | ~312   | Cornea and Iris multimodal   |

## Quick Start

```bash
# Install dependencies
pip install -e .

# Encode a dataset using Gabor IrisCode
python scripts/encode_dataset.py --dataset IITD --encoder gabor

# Evaluate performance
python scripts/evaluate.py --codes outputs/iitd_codes.npz
```

## Project Structure

```
biometric-auth/
├── src/iris_encoder/
│   ├── preprocessing/    # Segmentation, normalization
│   ├── encoding/         # IrisCode, CNN encoders
│   └── matching/         # Hamming distance, similarity
├── tests/                # Unit and integration tests
├── scripts/              # CLI tools
└── notebooks/            # Jupyter exploration
```

## References

- Kumar, A., & Passi, A. (2010). Comparison and combination of iris matchers for reliable personal identification. _Pattern Recognition_, 43(3), 1016-1026.
- Daugman, J. (2004). How iris recognition works. _IEEE Transactions on Circuits and Systems for Video Technology_, 14(1), 21-30.

## License

For research and non-commercial use only. See dataset-specific licenses.
