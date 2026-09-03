"""
download_frontier_corpus.py — NOESIS-Σ Real-World Data Downloader.

Downloads high-density, reasoning-rich open-source corpora to be ingested 
into the SIM database. We bypass "internet slop" and focus purely on:
1. FineWeb-Edu (High-grade textbooks and educational content)
2. Wikipedia (Structured encyclopedic facts)
3. ArXiv (Scientific papers and physics/math concepts)
"""

import os
import argparse
from pathlib import Path

# Ensure the user has the required library
try:
    from datasets import load_dataset
except ImportError:
    print("CRITICAL: The 'datasets' library is required to download frontier corpora.")
    print("Please run: pip install datasets")
    exit(1)

def download_corpus(dataset_name: str, config_name: str, split: str, num_samples: int, out_file: Path):
    print(f"Downloading {num_samples} samples from {dataset_name} ({config_name})...")
    
    # Streaming allows us to download just what we need without downloading a 500GB file
    dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for item in dataset:
            if count >= num_samples:
                break
            
            # Extract text (different datasets use different keys)
            text = item.get("text", item.get("content", item.get("abstract", "")))
            
            # Clean up the text
            text = text.replace("\n", " ").strip()
            if len(text) > 100: # Only keep substantial facts
                f.write(text + "\n")
                count += 1
                
                if count % 10000 == 0:
                    print(f"  ... downloaded {count} high-density facts ...")
                    
    print(f"✅ Success! Saved {count} facts to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100000, help="Number of documents per domain")
    args = parser.parse_args()
    
    base_dir = Path("/home/sephi-asi/NOESIS-Σ/Runtime/Data/Raw_Corpus")
    
    print("=== NOESIS-Σ FRONTIER CORPUS DOWNLOADER ===")
    
    # 1. Download Educational Textbooks (FineWeb-Edu)
    download_corpus(
        dataset_name="HuggingFaceFW/fineweb-edu", 
        config_name="sample-10BT", 
        split="train", 
        num_samples=args.samples, 
        out_file=base_dir / "textbooks_fineweb.txt"
    )
    
    # 2. Download Wikipedia (Structured Facts)
    download_corpus(
        dataset_name="wikimedia/wikipedia", 
        config_name="20231101.en", 
        split="train", 
        num_samples=args.samples, 
        out_file=base_dir / "facts_wikipedia.txt"
    )
    
    print("\n🚀 All downloads complete. The Raw_Corpus directory is populated.")
    print("Next Step: Run ingest_world_knowledge.py to vectorize these into the SIM.")
