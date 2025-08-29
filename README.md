# Ojo Upload Magic 🪄

A high-performance Python script for uploading multi-split image datasets to Hugging Face Hub with streaming processing, resume functionality, and incremental processing capabilities.

## ✨ Recent Improvements

**🔄 Resume Functionality**: Never lose progress again! The script now automatically tracks completion at split and shard levels, allowing you to resume interrupted uploads exactly where they left off.

**📊 Incremental Processing**: New `--amount` flag lets you process only the first N classes, perfect for testing pipelines, building datasets incrementally, and managing large uploads in controlled batches.

**🛡️ Enhanced Reliability**: Combined resume + incremental processing makes this tool incredibly robust for datasets of any size, from small tests to massive production datasets.

## Dataset Structure

The script expects a nested dataset structure where each class/card has its own directory:

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

- **Streaming processing**: Memory-efficient batch processing for all dataset sizes
- **Multi-processing**: Parallel image processing using all CPU cores  
- **Smart sharding**: Automatic sharding based on file sizes for optimal Hub performance
- **Resume functionality**: Continue interrupted uploads from where they left off
- **Incremental processing**: Process only first N classes with `--amount` flag for testing and incremental builds
- **Comprehensive logging**: Detailed progress tracking and system resource monitoring
- **Robust error handling**: Graceful handling of corrupted images and processing errors
- **Works with any size**: From 100 images to 500k+ images efficiently

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv ojo
# On Windows: ojo\Scripts\activate
# On Linux/macOS: source ojo/bin/activate

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face
huggingface-cli login
```

### 2. Basic Usage

```bash
python upload_dataset.py ./data your-username/dataset-name
```

### 3. Advanced Usage

```bash
python upload_dataset.py ./data your-username/dataset-name \
    --batch-size 2000 \
    --max-shard-size-mb 500 \
    --num-workers 8 \
    --token your_hf_token \
    --resume \
    --amount 100
```

## Command Line Options

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `data_path` | ✅ | - | Path to the data directory |
| `repo_id` | ✅ | - | Repository ID (e.g., `username/dataset-name`) |
| `--token` | ❌ | None | Hugging Face token |
| `--output-dir` | ❌ | Temp | Directory to save processed dataset |
| `--batch-size` | ❌ | 1000 | Images per processing batch |
| `--max-shard-size-mb` | ❌ | 420 | Maximum shard size in MB |
| `--num-workers` | ❌ | Auto | Number of worker processes |
| `--resume` | ❌ | False | Resume from previous incomplete run |
| `--amount` | ❌ | All | Maximum number of classes/cards to process |

## Python API

```python
from upload_dataset import upload_dataset

# Basic usage
upload_dataset(
    data_path="./data",
    repo_id="your-username/dataset-name"
)

# Advanced usage
upload_dataset(
    data_path="./data",
    repo_id="your-username/large-dataset",
    batch_size=2000,
    max_shard_size_mb=500,
    num_workers=8,
    resume=True,
    amount=100
)
```

## Output

The script generates:
1. **Sharded parquet files** - `split-00000-of-00010.parquet` format for optimal Hub loading
2. **Dataset card** - Comprehensive README.md with statistics and metadata  
3. **Label mappings** - JSON file mapping label IDs to class names
4. **Progress tracking** - `progress.json` file for resume functionality (local only, not uploaded)

## Resume Functionality 🔄

The script now supports resuming interrupted uploads, making it much more reliable for large datasets:

### How It Works
- Automatically tracks progress at the split and shard level
- Skips already completed shards when resuming
- Validates existing files before skipping them
- Handles configuration changes between runs

### Usage
```bash
# Start initial upload
python upload_dataset.py ./large-data username/my-dataset --batch-size 2000

# If interrupted, resume from where it left off
python upload_dataset.py ./large-data username/my-dataset --batch-size 2000 --resume
```

### Progress File
The script creates a `progress.json` file in the output directory that tracks:
- Completed splits and shards
- Configuration hash for validation
- Upload completion status
- Timestamps for debugging

**Note**: The progress.json file is automatically excluded from the Hugging Face upload and remains local only for resume functionality.

### Safety Features
- Validates configuration hasn't changed between runs
- Confirms existing shard files are complete and valid
- Prompts user if configuration changes are detected
- Preserves progress file even after successful completion

## Incremental Processing 🔢

Process only a subset of classes for testing or incremental dataset building:

### Use Cases
- **Testing**: Validate your pipeline with just a few classes first
- **Incremental builds**: Add classes to your dataset progressively  
- **Resource management**: Process large datasets in smaller chunks
- **Quick validation**: Check data quality with a sample

### Usage
```bash
# Process only the first 10 classes (alphabetically sorted)
python upload_dataset.py ./data username/dataset-name --amount 10

# Test with 5 classes first, then expand
python upload_dataset.py ./data username/test-dataset --amount 5
python upload_dataset.py ./data username/full-dataset --amount 50
```

### How It Works
- Classes are processed alphabetically for consistent ordering
- Applied across all splits (train/validation/test)  
- Perfect for incremental dataset expansion
- Compatible with resume functionality

### Example Workflow
```bash
# 1. Test with 10 classes first
python upload_dataset.py ./magic-cards username/magic-test --amount 10

# 2. Expand to 100 classes  
python upload_dataset.py ./magic-cards username/magic-100 --amount 100

# 3. Finally process all classes
python upload_dataset.py ./magic-cards username/magic-complete
```

## Performance Tips

### For Large Datasets (100k+ images)
- Increase `--batch-size 2000` for better throughput
- Use `--max-shard-size-mb 500` to reduce shard count
- Ensure SSD storage for faster I/O
- **Use `--resume` for reliability** during long uploads

### System Requirements
- **RAM**: 4GB minimum (streaming uses minimal memory)
- **CPU**: Multi-core recommended for parallel processing
- **Storage**: SSD recommended for better I/O performance

## Examples

```bash
# Magic card dataset with 50k images
python upload_dataset.py ./magic-data acidtib/magic-cards \
    --batch-size 2000 \
    --max-shard-size-mb 500

# Test with first 20 classes only
python upload_dataset.py ./magic-data acidtib/magic-test \
    --amount 20

# Small test dataset  
python upload_dataset.py ./test-data acidtib/test-cards

# Resume interrupted upload with amount limit
python upload_dataset.py ./magic-data acidtib/magic-cards \
    --batch-size 2000 \
    --amount 100 \
    --resume
```

## Troubleshooting

**Memory Issues**: Reduce `--batch-size` to 500-1000

**Upload Failures**: Check internet connection and HF authentication. Use `--resume` to continue.

**Processing Errors**: Verify directory structure and image file integrity

**Resume Issues**: 
- Ensure same output directory is used when resuming
- Check `progress.json` file exists in output directory
- Configuration changes will prompt for confirmation

**Corrupted Progress**: Delete `progress.json` and start fresh if needed

The script automatically handles dataset processing, sharding, and uploading with detailed progress information throughout the process.