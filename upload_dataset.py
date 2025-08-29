#!/usr/bin/env python3
"""
Optimized script to upload multi-split image datasets to Hugging Face Hub.
Handles both small and large datasets (500k+ images) efficiently with streaming and multiprocessing.

Dataset structure:
├── data/
│   ├── train/
│   │   └── <card-id>/
│   │       └── 0000.jpg
│   │       └── 0001.jpg
│   │       └── 0002.jpg
│   ├── test/
│   │   └── <card-id>/
│   │       └── 0000.jpg
│   ├── validation/
│   │   └── <card-id>/
│   │       └── 0000.jpg
│   │       └── 0001.jpg
"""

import os
import io
import json
import tempfile
import shutil
import gc
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi
from PIL import Image as PILImage
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count
import logging
import psutil
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetUploader:
    def __init__(self,
                 data_path: str,
                 repo_id: str,
                 token: str = None,
                 output_dir: str = None,
                 batch_size: int = 1000,
                 max_shard_size_mb: int = 420,
                 num_workers: int = None,
                 resume: bool = False,
                 amount: int = None):
        """
        Initialize the dataset uploader.

        Args:
            data_path: Path to the data directory
            repo_id: Repository ID on Hugging Face Hub
            token: Hugging Face token
            output_dir: Directory to save processed dataset
            batch_size: Number of images to process in each batch
            max_shard_size_mb: Maximum size per shard in MB
            num_workers: Number of worker processes (None = auto-detect)
            resume: Enable resume functionality to continue from previous run
            amount: Maximum number of classes/cards to process (None = process all)
        """
        self.data_path = Path(data_path)
        self.repo_id = repo_id
        self.token = token
        self.batch_size = batch_size
        self.max_shard_size_bytes = max_shard_size_mb * 1024 * 1024
        self.num_workers = num_workers or min(8, cpu_count())
        self.resume = resume
        self.amount = amount

        # Always use streaming processing mode
        total_images = self._count_total_images()
        logger.info(f"Processing {total_images:,} images with streaming mode")

        # Setup output directory
        self.cleanup_temp = output_dir is None
        if output_dir is None:
            self.output_dir = Path(tempfile.mkdtemp(prefix="hf_dataset_"))
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_data_dir = self.output_dir / "data"
        self.output_data_dir.mkdir(parents=True, exist_ok=True)

        # Setup resume functionality
        self.progress_file = self.output_dir / "progress.json"
        self.progress = self._load_progress()

        if self.resume:
            self._validate_resume_config()

        # Initialize HfApi
        if token:
            self.api = HfApi(token=token)
        else:
            self.api = HfApi()

        # Initialize labels
        self._setup_labels()

    def _generate_config_hash(self):
        """Generate a hash of the configuration to detect changes"""
        config_str = f"{self.data_path}_{self.batch_size}_{self.max_shard_size_bytes}_{self.num_workers}_{self.amount}"
        return hashlib.md5(config_str.encode()).hexdigest()

    def _load_progress(self):
        """Load progress from file or create new progress tracking"""
        if self.resume and self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)

                # Convert lists back to sets for internal use
                if 'completed_splits' in progress:
                    progress['completed_splits'] = set(progress['completed_splits'])
                if 'completed_shards' in progress:
                    progress['completed_shards'] = {
                        split: set(shards) for split, shards in progress['completed_shards'].items()
                    }

                total_shards = sum(len(shards) for shards in progress.get('completed_shards', {}).values())
                logger.info(f"📋 Loaded progress file with {total_shards} completed shards")
                return progress
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"⚠️ Could not load progress file: {e}. Starting fresh.")

        return {
            'config_hash': self._generate_config_hash(),
            'completed_splits': set(),
            'completed_shards': {},  # {split_name: set of shard_indices}
            'upload_completed': False,
            'created_at': None,
            'last_updated': None
        }

    def _save_progress(self):
        """Save current progress to file"""
        import datetime

        # Convert sets to lists for JSON serialization
        progress_to_save = self.progress.copy()
        progress_to_save['completed_splits'] = list(self.progress.get('completed_splits', set()))
        progress_to_save['completed_shards'] = {
            split: list(shards) for split, shards in self.progress.get('completed_shards', {}).items()
        }
        progress_to_save['last_updated'] = datetime.datetime.now().isoformat()

        if not progress_to_save.get('created_at'):
            progress_to_save['created_at'] = progress_to_save['last_updated']

        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_to_save, f, indent=2)
        except IOError as e:
            logger.warning(f"⚠️ Could not save progress file: {e}")

    def _validate_resume_config(self):
        """Validate that the configuration matches the previous run"""
        if not self.progress_file.exists():
            logger.info("🆕 No previous progress found, starting fresh")
            return

        current_hash = self._generate_config_hash()
        previous_hash = self.progress.get('config_hash')

        if current_hash != previous_hash:
            logger.warning("⚠️ Configuration has changed since last run!")
            logger.warning("Previous run used different parameters. Continuing may cause issues.")
            response = input("Continue anyway? (y/N): ").lower().strip()
            if response != 'y':
                logger.info("Aborting resume. Remove --resume flag to start fresh.")
                raise SystemExit(1)
            else:
                # Update config hash
                self.progress['config_hash'] = current_hash
                self._save_progress()

    def _is_split_completed(self, split_name):
        """Check if a split has been completed"""
        return split_name in self.progress.get('completed_splits', set())

    def _is_shard_completed(self, split_name, shard_idx):
        """Check if a specific shard has been completed"""
        completed_shards = self.progress.get('completed_shards', {})
        return shard_idx in completed_shards.get(split_name, set())

    def _mark_shard_completed(self, split_name, shard_idx):
        """Mark a shard as completed"""
        if 'completed_shards' not in self.progress:
            self.progress['completed_shards'] = {}
        if split_name not in self.progress['completed_shards']:
            self.progress['completed_shards'][split_name] = set()

        self.progress['completed_shards'][split_name].add(shard_idx)
        self._save_progress()

    def _mark_split_completed(self, split_name):
        """Mark an entire split as completed"""
        if 'completed_splits' not in self.progress:
            self.progress['completed_splits'] = set()

        self.progress['completed_splits'].add(split_name)
        self._save_progress()

    def _validate_existing_shard(self, shard_path):
        """Validate that an existing shard file is complete and valid"""
        try:
            if not shard_path.exists():
                return False

            # Try to load the parquet file to ensure it's valid
            dataset = Dataset.from_parquet(str(shard_path))
            return len(dataset) > 0
        except Exception as e:
            logger.warning(f"⚠️ Existing shard {shard_path.name} appears corrupted: {e}")
            return False

    def _count_total_images(self):
        """Count total number of images across all splits"""
        total = 0
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

        for split in ['train', 'validation', 'test']:
            split_dir = self.data_path / split
            if split_dir.exists():
                for card_dir in split_dir.iterdir():
                    if card_dir.is_dir():
                        for img_file in card_dir.iterdir():
                            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                                total += 1
        return total

    def _setup_labels(self):
        """Setup label mappings from train directory"""
        train_dir = self.data_path / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found at {train_dir}")

        # Get labels from card directory names
        all_labels = sorted(list(set([
            f for f in os.listdir(train_dir)
            if os.path.isdir(train_dir / f)
        ])))

        # Apply amount limit if specified
        if self.amount is not None:
            original_count = len(all_labels)
            self.labels = all_labels[:self.amount]
            logger.info(f"📊 Amount limit applied: Processing {len(self.labels)} out of {original_count} classes")
            logger.info(f"Selected classes: {self.labels[:10]}{'...' if len(self.labels) > 10 else ''}")
        else:
            self.labels = all_labels

        self.num_labels = len(self.labels)
        logger.info(f"Found {self.num_labels} unique labels: {self.labels[:5]}...")

        # Create label mappings
        self.label_to_id = {label: idx for idx, label in enumerate(self.labels)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.labels)}

    def get_directory_size_and_count(self, path):
        """Get total size and count of images in directory"""
        if not path.exists():
            return 0, 0

        total_size = 0
        count = 0
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                if Path(f).suffix.lower() in image_extensions:
                    fp = Path(dirpath) / f
                    if fp.exists():
                        total_size += fp.stat().st_size
                        count += 1
        return total_size, count

    def collect_image_paths(self, split_dir, split_name):
        """Collect all image paths for a split without loading images"""
        if not split_dir.exists():
            logger.warning(f"{split_name} directory not found at {split_dir}")
            return []

        image_data = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

        # Sort directories to ensure consistent ordering when using amount limit
        sorted_card_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda x: x.name)

        for card_dir in sorted_card_dirs:
            label_name = card_dir.name
            if label_name in self.label_to_id:
                for img_file in card_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                        image_data.append({
                            "image_path": str(img_file),
                            "label": self.label_to_id[label_name],
                            "label_name": label_name
                        })
            else:
                # Only log warning for skipped classes if we're not using amount limit
                # or if the class would have been included without the limit
                if self.amount is None:
                    logger.warning(f"Unknown label '{label_name}' in {split_name}, skipping")
                # If using amount limit, silently skip classes beyond the limit

        return image_data

    def process_image_batch(self, batch_data):
        """Process a batch of images with multiprocessing"""
        def process_single_image(item):
            try:
                img_path, label = item["image_path"], item["label"]

                # Load and process image
                img = PILImage.open(img_path)
                if img.mode == 'RGBA':
                    # Convert RGBA to RGB with white background
                    background = PILImage.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Convert to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=95, optimize=True)
                img_bytes = img_byte_arr.getvalue()

                return {
                    "image": {"bytes": img_bytes, "path": os.path.basename(img_path)},
                    "label": label
                }
            except Exception as e:
                logger.error(f"Error processing {item['image_path']}: {e}")
                return None

        # Process batch
        processed_batch = []
        for item in batch_data:
            result = process_single_image(item)
            if result:
                processed_batch.append(result)

        return processed_batch

    def estimate_shard_count(self, image_data, split_name):
        """Estimate number of shards needed based on sample images"""
        if not image_data:
            return 1

        # Sample a subset to estimate average size
        sample_size = min(100, len(image_data))
        sample_data = image_data[:sample_size]

        total_sample_size = 0
        successful_samples = 0

        for item in sample_data:
            try:
                img_path = item["image_path"]
                file_size = Path(img_path).stat().st_size
                total_sample_size += file_size
                successful_samples += 1
            except Exception as e:
                logger.warning(f"Error sampling {img_path}: {e}")
                continue

        if successful_samples == 0:
            logger.warning(f"No successful samples for {split_name}, using 1 shard")
            return 1

        avg_size = total_sample_size / successful_samples
        estimated_total = avg_size * len(image_data)
        num_shards = max(1, int(estimated_total / self.max_shard_size_bytes) + 1)

        logger.info(f"{split_name}: Estimated {estimated_total/1024/1024:.1f}MB total, using {num_shards} shards")
        return num_shards

    def process_split_streaming(self, split_dir, split_name):
        """Process a split using streaming for memory efficiency"""
        # Check if split is already completed
        if self._is_split_completed(split_name):
            logger.info(f"✅ Split '{split_name}' already completed, skipping")
            return

        logger.info(f"Processing {split_name} split with streaming...")

        # Collect image paths
        image_data = self.collect_image_paths(split_dir, split_name)
        if not image_data:
            logger.warning(f"No images found in {split_name}")
            return

        logger.info(f"Found {len(image_data):,} images in {split_name}")

        # Estimate shard count
        num_shards = self.estimate_shard_count(image_data, split_name)

        # Process in batches and save shards
        images_per_shard = len(image_data) // num_shards
        remainder = len(image_data) % num_shards

        start_idx = 0
        completed_shards = 0

        for shard_idx in range(num_shards):
            # Calculate shard size (distribute remainder across first shards)
            shard_size = images_per_shard + (1 if shard_idx < remainder else 0)
            end_idx = start_idx + shard_size

            # Check if shard is already completed
            output_path = self.output_data_dir / f"{split_name}-{shard_idx:05d}-of-{num_shards:05d}.parquet"

            if self._is_shard_completed(split_name, shard_idx) and self._validate_existing_shard(output_path):
                logger.info(f"✅ {split_name} shard {shard_idx + 1}/{num_shards} already completed, skipping")
                completed_shards += 1
                start_idx = end_idx
                continue

            shard_data = image_data[start_idx:end_idx]
            logger.info(f"Processing {split_name} shard {shard_idx + 1}/{num_shards} ({len(shard_data):,} images)")

            # Process shard in batches
            processed_examples = []

            with tqdm(total=len(shard_data), desc=f"Processing {split_name} shard {shard_idx + 1}") as pbar:
                for batch_start in range(0, len(shard_data), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(shard_data))
                    batch = shard_data[batch_start:batch_end]

                    # Process batch
                    processed_batch = self.process_image_batch(batch)
                    processed_examples.extend(processed_batch)

                    pbar.update(len(batch))

                    # Memory management
                    if batch_start % (self.batch_size * 10) == 0:
                        gc.collect()

            # Create dataset from processed examples
            if processed_examples:
                shard_dataset = Dataset.from_list(processed_examples)

                # Save shard
                shard_dataset.to_parquet(output_path)
                logger.info(f"Saved {split_name} shard {shard_idx + 1}/{num_shards} ({len(processed_examples):,} examples)")

                # Mark shard as completed
                self._mark_shard_completed(split_name, shard_idx)
                completed_shards += 1

                # Clear memory
                del shard_dataset, processed_examples
                gc.collect()

            start_idx = end_idx

        # Mark entire split as completed if all shards are done
        if completed_shards == num_shards:
            self._mark_split_completed(split_name)
            logger.info(f"✅ Split '{split_name}' completed ({completed_shards}/{num_shards} shards)")



    def create_dataset_info(self):
        """Create dataset info and README"""
        # Calculate statistics
        train_dir = self.data_path / "train"
        val_dir = self.data_path / "validation"
        test_dir = self.data_path / "test"

        train_size, train_examples = self.get_directory_size_and_count(train_dir)
        val_size, val_examples = self.get_directory_size_and_count(val_dir)
        test_size, test_examples = self.get_directory_size_and_count(test_dir)
        total_size = train_size + val_size + test_size

        logger.info("\nDataset statistics:")
        logger.info(f"Train: {train_examples:,} examples, {train_size/1024/1024:.2f}MB")
        logger.info(f"Validation: {val_examples:,} examples, {val_size/1024/1024:.2f}MB")
        logger.info(f"Test: {test_examples:,} examples, {test_size/1024/1024:.2f}MB")
        logger.info(f"Total: {train_examples + val_examples + test_examples:,} examples, {total_size/1024/1024:.2f}MB")

        # Save label mappings
        with open(self.output_dir / 'label_mapping.json', 'w') as f:
            json.dump(self.id_to_label, f, indent=2)

        # Create dataset_info content
        dataset_info_content = "dataset_info:\n"
        dataset_info_content += "  features:\n"
        dataset_info_content += "  - name: image\n    dtype: image\n"
        dataset_info_content += "  - name: label\n    dtype:\n        class_label:\n          names:\n"

        # Add label mappings (limit to prevent huge README files)
        for i, (idx, label) in enumerate(self.id_to_label.items()):
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
        readme_path = self.output_dir / "README.md"
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

    def upload_dataset(self):
        """Main method to process and upload the dataset"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory {self.data_path} does not exist!")

        try:
            logger.info(f"Processing dataset from {self.data_path}...")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"Batch size: {self.batch_size}, Workers: {self.num_workers}")

            if self.amount is not None:
                logger.info(f"📊 Processing limited dataset: {self.amount} classes")

            if self.resume:
                logger.info("🔄 Resume mode enabled")

            # Log system resources
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count_val = psutil.cpu_count()
            logger.info(f"System: {memory_gb:.1f}GB RAM, {cpu_count_val} CPU cores")

            # Create dataset info
            self.create_dataset_info()

            # Process each split
            splits = [
                (self.data_path / "train", "train"),
                (self.data_path / "validation", "validation"),
                (self.data_path / "test", "test")
            ]

            for split_dir, split_name in splits:
                if split_dir.exists():
                    self.process_split_streaming(split_dir, split_name)

            logger.info(f"Dataset processing complete. Files saved to: {self.output_dir}")

            # Check if upload is already completed
            if self.progress.get('upload_completed', False):
                logger.info("✅ Upload already completed")
                return

            logger.info("Uploading to Hugging Face Hub...")
            logger.info("📤 Uploading dataset files (excluding progress.json)")

            # Upload the dataset (exclude progress.json from upload)
            self.api.upload_large_folder(
                repo_id=self.repo_id,
                repo_type="dataset",
                folder_path=str(self.output_dir),
                ignore_patterns=["progress.json"]
            )

            # Mark upload as completed
            self.progress['upload_completed'] = True
            self._save_progress()

            logger.info(f"✅ Successfully uploaded dataset to https://huggingface.co/datasets/{self.repo_id}")

        except Exception as e:
            logger.error(f"❌ Error processing/uploading dataset: {e}")
            raise
        finally:
            # Clean up temporary directory if we created it and upload completed successfully
            if self.cleanup_temp and self.output_dir.exists() and self.progress.get('upload_completed', False):
                logger.info("✅ Upload completed successfully. Cleaning up temporary files.")
                logger.info("📝 Note: progress.json was excluded from upload to Hugging Face")
                shutil.rmtree(self.output_dir)
                logger.info(f"🗑️ Cleaned up temporary directory: {self.output_dir}")
            elif self.cleanup_temp and self.output_dir.exists():
                # Upload didn't complete, preserve everything including progress file
                logger.info(f"⚠️ Upload incomplete. Temporary directory preserved: {self.output_dir}")
                logger.info("💡 Use --resume flag to continue from where you left off")


def upload_dataset(data_path: str,
                   repo_id: str,
                   token: str = None,
                   output_dir: str = None,
                   batch_size: int = 1000,
                   max_shard_size_mb: int = 420,
                   num_workers: int = None,
                   resume: bool = False,
                   amount: int = None,
                  ):
    """
    Upload the dataset to Hugging Face Hub using streaming processing.

    Args:
        data_path: Path to the data directory
        repo_id: Repository ID on Hugging Face Hub (e.g., "username/dataset-name")
        token: Hugging Face token (optional if logged in via CLI)
        output_dir: Directory to save processed dataset (temporary if None)
        batch_size: Number of images to process in each batch
        max_shard_size_mb: Maximum size per shard in MB
        num_workers: Number of worker processes (None = auto-detect)
        resume: Enable resume functionality to continue from previous run
        amount: Maximum number of classes/cards to process (None = process all)
    """
    uploader = DatasetUploader(
        data_path=data_path,
        repo_id=repo_id,
        token=token,
        output_dir=output_dir,
        batch_size=batch_size,
        max_shard_size_mb=max_shard_size_mb,
        num_workers=num_workers,
        resume=resume,
        amount=amount
    )
    uploader.upload_dataset()


def main():
    parser = argparse.ArgumentParser(description="Upload multi-split image dataset to Hugging Face Hub")
    parser.add_argument("data_path", help="Path to the data directory")
    parser.add_argument("repo_id", help="Repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--token", help="Hugging Face token (optional if logged in)")
    parser.add_argument("--output-dir", help="Directory to save processed dataset (temporary if not specified)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of images to process in each batch")
    parser.add_argument("--max-shard-size-mb", type=int, default=420, help="Maximum size per shard in MB")
    parser.add_argument("--num-workers", type=int, help="Number of worker processes (default: auto-detect)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous incomplete run")
    parser.add_argument("--amount", type=int, help="Maximum number of classes/cards to process (default: all)")

    args = parser.parse_args()

    upload_dataset(
        data_path=args.data_path,
        repo_id=args.repo_id,
        token=args.token,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_shard_size_mb=args.max_shard_size_mb,
        num_workers=args.num_workers,
        resume=args.resume,
        amount=args.amount
    )


if __name__ == "__main__":
    main()
