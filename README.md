# Ojo Upload Magic 🪄

A Python script for uploading multi-split image datasets to Hugging Face Hub. This tool processes image datasets organized with nested directories for each card/class (train/validation/test splits) and uploads them as sharded parquet files to the Hugging Face Hub with proper dataset cards and metadata.

## Dataset Structure

The script expects a nested dataset structure where each card/class has its own directory containing multiple images:

```
data/
├── train/
│   ├── card-001/
│   │   ├── 0000.jpg
│   │   ├── 0001.jpg
│   │   └── 0002.jpg
│   └── card-002/
│       └── 0000.jpg
├── test/
│   └── card-001/
│       └── 0000.jpg
└── validation/
    ├── card-001/
    │   ├── 0000.jpg
    │   └── 0001.jpg
    └── card-002/
        └── 0000.jpg
```

## Features

- Supports multi-split datasets (train/validation/test)
- Automatically generates class labels from directory names (card-ids)
- Creates sharded parquet files for efficient storage and loading
- Generates comprehensive dataset cards with statistics
- Handles large datasets with automatic sharding
- Embeds images as bytes for optimal Hub compatibility

## Command Line Options

| Option | Required | Description | Example |
|--------|----------|-------------|---------|
| `data_path` | ✅ | Path to the data directory containing train/validation/test folders | `./data` |
| `repo_id` | ✅ | Repository ID on Hugging Face Hub | `username/dataset-name` |
| `--token` | ❌ | Hugging Face token (optional if logged in via CLI) | `--token hf_xxxxx` |
| `--output-dir` | ❌ | Directory to save processed dataset (uses temp dir if not specified) | `--output-dir ./processed` |

## Setup

Before running:

Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv ojo

# Activate virtual environment
# On Linux/macOS:
source ojo/bin/activate
# On Windows:
ojo\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Login to Hugging Face (if not using token parameter):
```
huggingface-cli login
```

Usage:
# Basic usage
```
python upload_dataset.py ./data your-username/your-dataset-name
```

# With additional options
```
python upload_dataset.py ./data your-username/your-dataset-name \
    --token your_hf_token
```

Or use it directly in code:
```
from upload_dataset import upload_dataset

upload_dataset(
    data_path="./data",
    repo_id="your-username/card-dataset"
)
```
