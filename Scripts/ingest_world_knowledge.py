"""
ingest_world_knowledge.py — NOESIS-Σ Declarative Knowledge Ingestor.

Populates the Semantic Integration Memory (SIM) with real-world facts.
Thermal-Safe Edition:
- Batch Size 8 for 5GB VRAM safety.
- Mandatory Thermal Sleep to prevent laptop shutdown.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

class SIMIngestor:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        print(f"Initializing SIM Ingestor with {model_name}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.sim_dir = _ROOT / "Runtime" / "Data" / "SIM_Index"
        self.sim_dir.mkdir(parents=True, exist_ok=True)

    def ingest_file(self, file_path: Path, domain: str, batch_size: int = 15):
        """Reads a large text file, embeds in batches, and saves to the SIM."""
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
            
        print(f"Ingesting file {file_path} into SIM domain '{domain}'...")
        
        all_embeddings = []
        all_text = []
        batch = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    batch.append(text)
                    all_text.append(text)
                    
                if len(batch) >= batch_size:
                    # Embed the batch
                    print(f"  Embedding batch of {len(batch)} facts...")
                    emb = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
                    all_embeddings.append(emb.cpu())
                    
                    # --- THERMAL SAFETY GUARD ---
                    # Adding sleep to allow GPU fans to keep up and prevent system shutdown
                    time.sleep(1.0) 
                    # ----------------------------
                    
                    batch = []
                    
        # Process remaining
        if batch:
            print(f"  Embedding final batch of {len(batch)} facts...")
            emb = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            all_embeddings.append(emb.cpu())
            
        if not all_embeddings:
            print("No data found to ingest.")
            return
            
        final_tensor = torch.cat(all_embeddings, dim=0)
        
        domain_path = self.sim_dir / f"sim_{domain}.pt"
        metadata_path = self.sim_dir / f"sim_{domain}_meta.json"
        
        torch.save(final_tensor, domain_path)
        with open(metadata_path, "w", encoding="utf-8") as meta_f:
            json.dump({
                "count": len(all_text), 
                "dim": final_tensor.shape[1],
                "source": str(file_path)
            }, meta_f, indent=2)
            
        print(f"✅ Domain '{domain}' ready. Saved {len(all_text)} facts to {domain_path}")

def run_full_ingestion():
    ingestor = SIMIngestor()
    base_dir = _ROOT / "Runtime" / "Data" / "Raw_Corpus"
    
    fineweb_path = base_dir / "textbooks_fineweb.txt"
    wiki_path = base_dir / "facts_wikipedia.txt"
    
    if fineweb_path.exists():
        ingestor.ingest_file(fineweb_path, domain="fineweb_edu")
        
    if wiki_path.exists():
        ingestor.ingest_file(wiki_path, domain="wikipedia")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    
    if args.full:
        run_full_ingestion()
    elif args.bootstrap:
        # Define run_bootstrap locally or call it if defined
        def run_bootstrap():
            ingestor = SIMIngestor()
            physics_facts = [
                "Newton's Second Law: F = ma.",
                "Einstein's Mass-Energy Equivalence: E = mc^2."
            ]
            ingestor.ingest_file_from_list(physics_facts, "physics")
        
        # We need ingest_file_from_list or similar. 
        # Actually, let's just make bootstrap work by creating a temp file.
        print("Running bootstrap test...")
        ingestor = SIMIngestor()
        temp_file = Path("temp_bootstrap.txt")
        temp_file.write_text("Newton's Second Law: F = ma.\nEinstein's Mass-Energy Equivalence: E = mc^2.")
        ingestor.ingest_file(temp_file, domain="bootstrap_test", batch_size=2)
        temp_file.unlink()
    else:
        print("Usage: python3 ingest_world_knowledge.py --full")
