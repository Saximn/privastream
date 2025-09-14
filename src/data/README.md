# Data Directory

This directory is used for storing model weights, datasets, and other data files.

## Structure

```
data/
├── models/          # Pre-trained model weights
│   ├── face_best.pt    # Face detection model
│   ├── plate_best.pt   # License plate detection model  
│   └── pii_clf.joblib  # PII classifier model
├── samples/         # Sample videos and audio files
├── embeddings/      # Face embedding files
└── temp/           # Temporary processing files
```

## Model Downloads

Models are automatically downloaded on first run, or you can manually place them in the `models/` subdirectory:

- **face_best.pt** (~50MB) - Face detection weights
- **plate_best.pt** (~15MB) - License plate detection weights  
- **pii_clf.joblib** (~5MB) - PII text classification model

## Usage

This directory is automatically created and managed by Privastream. You don't need to manually create the subdirectories - they will be created as needed.