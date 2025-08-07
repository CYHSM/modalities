import json
import random
import time
from pathlib import Path

import psutil
import requests
from datasets import load_dataset
from tqdm import tqdm

# Config
LANGUAGES = ["eng_Latn", "deu_Latn", "fra_Latn", "ita_Latn", "spa_Latn", "nob_Latn"]
TARGET_TOKENS = 30_000_000_000  # 30B tokens
TOKENS_PER_LANG = TARGET_TOKENS // len(LANGUAGES)  # ~5B each
VAL_SPLIT = 0.001
OUTPUT_DIR = "/raid/s3/opengptx/mfrey/fineweb-30B"
WEIGHTS_URL = "https://raw.githubusercontent.com/huggingface/fineweb-2/main/misc/rehydration/weights/up_to_10_reps.json"
BATCH_SIZE = 1000  # Write in batches for better I/O performance


def print_memory_usage():
    """Print current memory usage"""
    memory = psutil.virtual_memory()
    process = psutil.Process()
    print(f"💾 System RAM: {memory.percent:.1f}% used ({memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB)")
    print(f"🔧 Process RAM: {process.memory_info().rss/1e9:.1f}GB")


def print_stats_header():
    """Print a nice header for stats"""
    print("\n" + "=" * 80)
    print("🚀 MULTILINGUAL DATASET PROCESSOR")
    print("=" * 80)
    print(f"📊 Target: {TARGET_TOKENS/1e9:.1f}B tokens total ({TOKENS_PER_LANG/1e9:.1f}B per language)")
    print(f"🌍 Languages: {', '.join(LANGUAGES)}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"📂 Structure: Separate files per language ({len(LANGUAGES)} train + {len(LANGUAGES)} val files)")
    print_memory_usage()
    print("=" * 80 + "\n")


class Rehydrater:
    def __init__(self, weights):
        self.weights = weights or {1: 1}
        max_key = max(self.weights.keys()) if self.weights else 1
        self.expanded = [1]  # cluster size 0 -> weight 1

        for i in range(1, max_key + 1):
            if i in self.weights:
                self.expanded.append(self.weights[i])
            else:
                self.expanded.append(self.expanded[-1])

    def get_reps(self, cluster_size):
        if cluster_size >= len(self.expanded):
            return self.expanded[-1]
        return self.expanded[cluster_size]


class LanguageBatchWriter:
    """Batched writer for a specific language with separate train/val files"""

    def __init__(self, lang, train_file, val_file, batch_size=BATCH_SIZE):
        self.lang = lang
        self.train_file = train_file
        self.val_file = val_file
        self.batch_size = batch_size
        self.train_batch = []
        self.val_batch = []

    def add_doc(self, doc, is_val=False):
        if is_val:
            self.val_batch.append(doc)
            if len(self.val_batch) >= self.batch_size:
                self._flush_val()
        else:
            self.train_batch.append(doc)
            if len(self.train_batch) >= self.batch_size:
                self._flush_train()

    def _flush_train(self):
        if self.train_batch:
            for doc in self.train_batch:
                self.train_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
            self.train_batch = []

    def _flush_val(self):
        if self.val_batch:
            for doc in self.val_batch:
                self.val_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
            self.val_batch = []

    def flush_all(self):
        self._flush_train()
        self._flush_val()


def process_language(lang, target_tokens, weights_data, writer, stats):
    """Process a single language and write directly to its files"""
    print(f"\n🌐 Processing {lang}")
    print(f"🎯 Target: {target_tokens/1e9:.1f}B tokens")

    # Setup dataset and rehydrater
    if lang == "eng_Latn":
        print("📚 Loading FineWeb-Edu...")
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        source_name = "fineweb-edu"
        rehydrater = None
        print("✅ FineWeb-Edu loaded (no rehydration needed)")
    else:
        print(f"📚 Loading FineWeb-2 for {lang}...")
        dataset = load_dataset("HuggingFaceFW/fineweb-2", name=lang, split="train", streaming=True)
        source_name = "fineweb-2"
        weights = weights_data.get(lang, {1: 1})
        if weights and isinstance(list(weights.keys())[0], str):
            weights = {int(k): v for k, v in weights.items()}
        rehydrater = Rehydrater(weights)
        print("✅ FineWeb-2 loaded with rehydration weights")

    # Process documents
    total_tokens = 0
    processed_docs = 0
    train_docs = 0
    val_docs = 0
    start_time = time.time()

    # Create progress bar focused on token progress
    pbar = tqdm(
        total=target_tokens,
        desc=f"🎯 {lang} tokens",
        unit="B",
        unit_scale=True,
        bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} tokens [{elapsed}<{remaining}, {rate_fmt}tokens/s]",  # noqa: E501
    )

    for item in dataset:
        text = item["text"]
        tokens = len(text) // 5  # rough estimate on the conservative side to get more tokens

        # Skip very short documents
        if tokens < 100:
            continue

        # Apply rehydration if available
        if rehydrater:
            cluster_size = item.get("minhash_cluster_size", 1)
            reps = rehydrater.get_reps(cluster_size)
        else:
            reps = 1

        # Create base document
        doc = {"text": text, "url": item.get("url", ""), "language": lang, "source": source_name}

        # Add rehydrated copies
        tokens_added_this_batch = 0
        for _ in range(reps):
            # Randomly assign to train or val
            is_val = random.random() < VAL_SPLIT
            writer.add_doc(doc.copy(), is_val)

            # Update counters
            processed_docs += 1
            total_tokens += tokens
            tokens_added_this_batch += tokens
            if is_val:
                val_docs += 1
            else:
                train_docs += 1

            if total_tokens >= target_tokens:
                break

        # Update progress bar with token progress and doc info in postfix
        pbar.set_postfix({"docs": f"{processed_docs:,}", "train": f"{train_docs:,}", "val": f"{val_docs:,}"})
        pbar.update(tokens_added_this_batch)

        if total_tokens >= target_tokens:
            break

    pbar.close()

    # Update final stats
    elapsed = time.time() - start_time
    docs_per_sec = processed_docs / elapsed if elapsed > 0 else 0

    print(f"✅ {lang} complete!")
    print(f"   📄 Documents: {processed_docs:,}")
    print(f"   🎯 Tokens: {total_tokens/1e9:.2f}B")
    print(f"   🚂 Train: {train_docs:,}")
    print(f"   🧪 Val: {val_docs:,}")
    print(f"   ⏱️  Time: {elapsed:.1f}s ({docs_per_sec:.0f} docs/sec)")

    stats[lang] = {
        "total_docs": processed_docs,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "tokens": total_tokens,
        "elapsed_time": elapsed,
    }

    return processed_docs, total_tokens


def main():
    start_time = time.time()

    # Setup
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print_stats_header()

    # Download rehydration weights
    print("🔄 Downloading rehydration weights...")
    try:
        response = requests.get(WEIGHTS_URL, timeout=30)
        response.raise_for_status()
        weights_data = response.json()
        print("✅ Rehydration weights downloaded successfully")
    except Exception as e:
        print(f"⚠️  Failed to download weights: {e}")
        print("ℹ️  Continuing without rehydration...")
        weights_data = {}

    # Process each language with its own files
    total_docs = 0
    total_tokens = 0
    stats = {}
    created_files = []

    for i, lang in enumerate(LANGUAGES, 1):
        print(f"\n{'='*60}")
        print(f"🌍 LANGUAGE {i}/{len(LANGUAGES)}: {lang}")
        print(f"{'='*60}")

        # Create language-specific file paths
        train_path = Path(OUTPUT_DIR) / f"{lang}_train.jsonl"
        val_path = Path(OUTPUT_DIR) / f"{lang}_val.jsonl"

        print(f"📁 Creating files for {lang}:")
        print(f"   🚂 Train: {train_path.name}")
        print(f"   🧪 Val: {val_path.name}")

        created_files.extend([train_path, val_path])
        with open(train_path, "w", encoding="utf-8") as train_f, open(val_path, "w", encoding="utf-8") as val_f:
            writer = LanguageBatchWriter(lang, train_f, val_f)
            lang_docs, lang_tokens = process_language(lang, TOKENS_PER_LANG, weights_data, writer, stats)

            # Flush any remaining batches
            writer.flush_all()

            total_docs += lang_docs
            total_tokens += lang_tokens

        # Show file sizes and memory after each language
        train_size_mb = train_path.stat().st_size / (1024 * 1024)
        val_size_mb = val_path.stat().st_size / (1024 * 1024)
        print(f"💾 File sizes: Train {train_size_mb:.1f}MB, Val {val_size_mb:.1f}MB")
        print_memory_usage()

        # Show running totals
        print("\n📊 Running totals:")
        print(f"   📄 Total docs: {total_docs:,}")
        print(f"   🎯 Total tokens: {total_tokens/1e9:.2f}B")
        print(f"   📈 Progress: {((i/len(LANGUAGES))*100):.1f}%")

    # Calculate final stats
    elapsed = time.time() - start_time
    total_train = sum(s["train_docs"] for s in stats.values())
    total_val = sum(s["val_docs"] for s in stats.values())

    print(f"\n{'='*80}")
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"⏱️  Total time: {elapsed/3600:.1f} hours ({elapsed:.0f} seconds)")
    print(f"📄 Total documents: {total_docs:,}")
    print(f"🚂 Train documents: {total_train:,}")
    print(f"🧪 Val documents: {total_val:,}")
    print(f"🎯 Total tokens: {total_tokens/1e9:.2f}B")
    print(f"📊 Processing rate: {total_docs/elapsed:.0f} docs/sec")
    print(f"📂 Files created: {len(created_files)} ({len(LANGUAGES)} train + {len(LANGUAGES)} val)")

    print("\n📋 Language breakdown:")
    for lang, lang_stats in stats.items():
        pct = (lang_stats["total_docs"] / total_docs) * 100
        tokens_gb = lang_stats["tokens"] / 1e9
        train_file = Path(OUTPUT_DIR) / f"{lang}_train.jsonl"
        val_file = Path(OUTPUT_DIR) / f"{lang}_val.jsonl"
        train_size = train_file.stat().st_size / (1024 * 1024)
        val_size = val_file.stat().st_size / (1024 * 1024)
        print(f"   {lang}: {lang_stats['total_docs']:,} docs ({pct:.1f}%) - {tokens_gb:.2f}B tokens")
        print(f"      📂 {train_file.name} ({train_size:.1f}MB), {val_file.name} ({val_size:.1f}MB)")

    print("\n📁 All output files:")
    for file_path in sorted(created_files):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"   {file_path.name} ({size_mb:.1f}MB)")

    # Save metadata with file information
    file_info = {}
    for lang in LANGUAGES:
        train_path = Path(OUTPUT_DIR) / f"{lang}_train.jsonl"
        val_path = Path(OUTPUT_DIR) / f"{lang}_val.jsonl"
        file_info[lang] = {
            "train_file": train_path.name,
            "val_file": val_path.name,
            "train_size_bytes": train_path.stat().st_size,
            "val_size_bytes": val_path.stat().st_size,
        }

    metadata = {
        "total_docs": total_docs,
        "train_docs": total_train,
        "val_docs": total_val,
        "total_tokens": total_tokens,
        "target_tokens": TARGET_TOKENS,
        "processing_time_seconds": elapsed,
        "rehydration_applied": len(weights_data) > 0,
        "file_structure": "per_language",
        "files_created": len(created_files),
        "language_stats": stats,
        "file_info": file_info,
        "config": {"languages": LANGUAGES, "val_split": VAL_SPLIT, "batch_size": BATCH_SIZE},
    }

    metadata_path = Path(OUTPUT_DIR) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 Metadata saved to: {metadata_path}")
    print_memory_usage()
    print("\n🚀 All done! Happy training! 🎯")


if __name__ == "__main__":
    main()
