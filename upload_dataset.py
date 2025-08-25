#!/usr/bin/env python3
"""
Script to upload a multi-split image dataset to Hugging Face Hub.
Dataset structure:
├── data/
│   ├── train/
│   │   └── <card-id>.jpg    # Original image
│   ├── test/
│   │   └── <card-id>.jpg    # Test image
│   ├── validation/
│   │   └── <card-id>.jpg    # Validation image
"""

import os
import io
import json
import tempfile
import shutil
from pathlib import Path
from datasets import load_dataset, Features, ClassLabel, Image, Dataset, DatasetDict
from huggingface_hub import HfApi
from PIL import Image as PILImage
import argparse


def get_directory_size(path):
    """Get total size of directory, return 0 if directory doesn't exist"""
    if not os.path.exists(path):
        return 0
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):  # Check if file still exists
                total_size += os.path.getsize(fp)
    return total_size


def count_examples(path):
    """Count total examples in directory, return 0 if directory doesn't exist"""
    if not os.path.exists(path):
        return 0
    count = 0
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            count += 1
    return count


def create_embed_images_function(label_to_id):
    """Create embed_images function with access to label_to_id mapping"""
    def embed_images(example):
        """Embed images as bytes and convert labels to numeric IDs"""
        # Handle both file paths (from manual loading) and PIL Image objects
        if isinstance(example["image"], str):
            # File path from manual loading
            img = PILImage.open(example["image"])
            image_filename = os.path.basename(example["image"])
        else:
            # PIL Image object (from imagefolder loader)
            img = example["image"]
            image_filename = getattr(example["image"], 'filename', 'unknown.jpg')
            if hasattr(example["image"], 'filename'):
                image_filename = os.path.basename(example["image"].filename)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format or 'JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Return new example with embedded image and original label (already converted)
        return {
            "image": {
                "bytes": img_byte_arr,
                "path": image_filename
            },
            "label": example["label"]  # Label is already the numeric ID
        }
    return embed_images


def upload_dataset(
    data_path: str,
    repo_id: str,
    token: str = None,
    output_dir: str = None
):
    """
    Upload the dataset to Hugging Face Hub using parquet files.

    Args:
        data_path: Path to the data directory
        repo_id: Repository ID on Hugging Face Hub (e.g., "username/dataset-name")
        token: Hugging Face token (optional if logged in via CLI)
        output_dir: Directory to save processed dataset (temporary if None)
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_path} does not exist!")

    # Create temporary directory if output_dir not specified
    cleanup_temp = output_dir is None
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="hf_dataset_")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Processing dataset from {data_path}...")

        # Initialize HfApi
        if token:
            api = HfApi(token=token)
        else:
            api = HfApi()

        # Define split directories
        train_dir = data_path / "train"
        val_dir = data_path / "validation"
        test_dir = data_path / "test"

        output_data_dir = Path(output_dir) / "data"
        output_data_dir.mkdir(parents=True, exist_ok=True)

        # Ensure train directory exists and get labels
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found at {train_dir}")

        # Get labels from image filenames (card-ids without extensions)
        labels = sorted(list(set([
            os.path.splitext(f)[0] for f in os.listdir(train_dir)
            if os.path.isfile(train_dir / f) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
        ])))
        num_labels = len(labels)
        print(f"Found {num_labels} unique labels: {labels[:5]}...")

        # Create label mappings
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        id_to_label = {idx: label for idx, label in enumerate(labels)}

        # Calculate dataset sizes and counts
        train_size = get_directory_size(train_dir)
        val_size = get_directory_size(val_dir)
        test_size = get_directory_size(test_dir)
        total_size = train_size + val_size + test_size

        train_examples = count_examples(train_dir)
        val_examples = count_examples(val_dir)
        test_examples = count_examples(test_dir)

        print(f"\nDataset statistics:")
        print(f"Train: {train_examples} examples, {train_size/1024/1024:.2f}MB")
        print(f"Validation: {val_examples} examples, {val_size/1024/1024:.2f}MB")
        print(f"Test: {test_examples} examples, {test_size/1024/1024:.2f}MB")
        print(f"Total: {train_examples + val_examples + test_examples} examples, {total_size/1024/1024:.2f}MB")

        # Save label mappings
        with open(Path(output_dir) / 'label_mapping.json', 'w') as f:
            json.dump(id_to_label, f, indent=2)

        # Create dataset_info content
        dataset_info_content = "dataset_info:\n"
        dataset_info_content += "  features:\n"
        dataset_info_content += "  - name: image\n    dtype: image\n"
        dataset_info_content += "  - name: label\n    dtype:\n        class_label:\n          names:\n"

        # Add label mappings (limit to prevent huge README files)
        for i, (idx, label) in enumerate(id_to_label.items()):
            if i < 4999:  # Limit to first 4999 labels
                dataset_info_content += f"            '{idx}': {label}\n"
            else:
                break

        dataset_info_content += "  splits:\n"

        if train_examples > 0:
            dataset_info_content += f"  - name: train\n    num_bytes: {train_size}\n    num_examples: {train_examples}\n"
        if val_examples > 0:
            dataset_info_content += f"  - name: validation\n    num_bytes: {val_size}\n    num_examples: {val_examples}\n"
        if test_examples > 0:
            dataset_info_content += f"  - name: test\n    num_bytes: {test_size}\n    num_examples: {test_examples}\n"

        dataset_info_content += f"  download_size: {total_size}\n"
        dataset_info_content += f"  dataset_size: {total_size}\n"

        # Create or update README.md
        readme_path = Path(output_dir) / "README.md"
        if readme_path.exists():
            with open(readme_path, "r") as f:
                existing_content = f.read()

            # Find and replace dataset_info section
            start_marker = "---\ndataset_info:"
            end_marker = "\nconfigs:"

            if start_marker in existing_content and end_marker in existing_content:
                start_idx = existing_content.find(start_marker)
                end_idx = existing_content.find(end_marker)
                new_content = existing_content[:start_idx] + "---\n" + dataset_info_content + existing_content[end_idx:]
            else:
                new_content = "---\n" + dataset_info_content + "configs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/train-*\n  - split: validation\n    path: data/validation-*\n  - split: test\n    path: data/test-*\n---\n\n" + existing_content
        else:
            new_content = "---\n" + dataset_info_content + "configs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/train-*\n  - split: validation\n    path: data/validation-*\n  - split: test\n    path: data/test-*\n---\n\n# Dataset\n\nThis dataset was automatically generated and uploaded.\n"

        with open(readme_path, "w") as f:
            f.write(new_content)

        # Create dataset manually for flat structure
        def load_images_from_dir(split_dir, split_name):
            """Load images from a flat directory structure"""
            if not split_dir.exists():
                print(f"Warning: {split_name} directory not found at {split_dir}")
                return []

            examples = []
            for img_file in split_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    # Extract label from filename (card-id without extension)
                    label_name = img_file.stem
                    if label_name in label_to_id:
                        examples.append({
                            "image": str(img_file),
                            "label": label_to_id[label_name]
                        })
                    else:
                        print(f"Warning: Unknown label '{label_name}' in {split_name}, skipping {img_file.name}")
            return examples

        print(f"\nLoading dataset manually...")

        # Load each split manually
        dataset_dict = {}

        # Load train split
        train_examples = load_images_from_dir(train_dir, "train")
        if train_examples:
            dataset_dict["train"] = Dataset.from_list(train_examples).cast_column("image", Image())

        # Load validation split
        val_examples = load_images_from_dir(val_dir, "validation")
        if val_examples:
            dataset_dict["validation"] = Dataset.from_list(val_examples).cast_column("image", Image())

        # Load test split
        test_examples = load_images_from_dir(test_dir, "test")
        if test_examples:
            dataset_dict["test"] = Dataset.from_list(test_examples).cast_column("image", Image())

        # Create DatasetDict
        dataset = DatasetDict(dataset_dict)
        print("Loaded dataset:", dataset)

        # Process and embed images in each split
        embedded_dataset = {}
        for split_name, split_dataset in dataset.items():
            if len(split_dataset) == 0:
                continue

            print(f"\nProcessing {split_name} split...")

            # Show label distribution sample
            label_counts = {}
            for example in split_dataset:
                label = example["label"]
                label_name = id_to_label[label]
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
            print(f"Label distribution in {split_name}: {dict(list(label_counts.items())[:5])}...")

            # Create embed function with access to label mappings
            embed_images_func = create_embed_images_function(label_to_id)

            embedded_dataset[split_name] = split_dataset.map(
                embed_images_func,
                desc=f"Embedding images for {split_name}"
            )

            # Debug: Check the structure of the embedded dataset
            if len(embedded_dataset[split_name]) > 0:
                first_example = embedded_dataset[split_name][0]
                print(f"DEBUG {split_name}: Image type = {type(first_example['image'])}")
                if isinstance(first_example['image'], dict):
                    print(f"DEBUG {split_name}: Image keys = {first_example['image'].keys()}")

        # Convert each split to sharded parquet files
        for split_name, split_dataset in embedded_dataset.items():
            if len(split_dataset) == 0:
                continue

            # Calculate number of shards based on target shard size of 420MB
            target_shard_size = 420 * 1024 * 1024  # 420MB in bytes

            # Get a sample of images to estimate average size
            sample_size = min(100, len(split_dataset))
            sample_total = 0
            for example in split_dataset.select(range(sample_size)):
                # Handle the PIL Image object directly
                img = example["image"]
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format or 'JPEG')
                sample_total += len(img_byte_arr.getvalue())
            avg_image_size = sample_total / sample_size

            # Estimate total size and calculate number of shards
            estimated_total_size = avg_image_size * len(split_dataset)
            num_shards = max(1, int(estimated_total_size // target_shard_size + (1 if estimated_total_size % target_shard_size else 0)))

            print(f"\nSaving {split_name} split into {num_shards} shards...")
            print(f"Estimated total split size: {estimated_total_size / (1024 * 1024):.2f}MB")
            print(f"Average image size: {avg_image_size / 1024:.2f}KB")

            for index in range(num_shards):
                shard = split_dataset.shard(index=index, num_shards=num_shards, contiguous=True)
                # Use Hugging Face naming convention: split-XXXXX-of-YYYYY.parquet
                output_path = output_data_dir / f"{split_name}-{index:05d}-of-{num_shards:05d}.parquet"
                shard.to_parquet(output_path)
                print(f"Saved shard {index + 1}/{num_shards} to {output_path}")

        print(f"\nDataset processing complete. Files saved to: {output_dir}")
        print("Uploading to Hugging Face Hub...")

        # Upload the dataset
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(output_dir),
        )

        print(f"✅ Successfully uploaded dataset to https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"❌ Error processing/uploading dataset: {e}")
        raise
    finally:
        # Clean up temporary directory if we created it
        if cleanup_temp and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Cleaned up temporary directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Upload multi-split image dataset to Hugging Face Hub")
    parser.add_argument("data_path", help="Path to the data directory")
    parser.add_argument("repo_id", help="Repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--token", help="Hugging Face token (optional if logged in)")

    parser.add_argument("--output-dir", help="Directory to save processed dataset (temporary if not specified)")

    args = parser.parse_args()

    upload_dataset(
        data_path=args.data_path,
        repo_id=args.repo_id,
        token=args.token,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    # Example usage (uncomment and modify as needed):
    # upload_dataset(
    #     data_path="./data",
    #     repo_id="your-username/your-dataset-name"
    # )

    main()
