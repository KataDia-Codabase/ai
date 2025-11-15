#!/usr/bin/env python3
"""
Download and setup ML models for pronunciation assessment.
Uses wav2vec2-base for Railway free plan compatibility (RAM: ~1-2 GB).
"""

import os
import sys
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def download_wav2vec2_base():
    """
    Download wav2vec2-base model for English pronunciation assessment.
    
    Model: facebook/wav2vec2-base-960h
    - Size: ~360 MB (Railway free plan compatible)
    - RAM: ~1-2 GB during inference
    - Task: automatic-speech-recognition
    - Fine-tunable: Yes (with speechocean762 dataset)
    """
    print("=" * 70)
    print("Downloading Wav2Vec2 Base Model")
    print("=" * 70)
    print()
    
    model_name = "facebook/wav2vec2-base-960h"
    output_dir = BASE_DIR / "models" / "wav2vec2-base"
    
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Expected size: ~360 MB")
    print(f"RAM usage: ~1-2 GB (Railway free plan compatible)")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Downloading processor...")
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        processor.save_pretrained(output_dir)
        print("Processor downloaded")
        
        print("Downloading model (this may take a few minutes)...")
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        model.save_pretrained(output_dir)
        print("Model downloaded")
        
        # Test model loading
        print()
        print("Testing model...")
        test_model = Wav2Vec2ForCTC.from_pretrained(output_dir)
        test_processor = Wav2Vec2Processor.from_pretrained(output_dir)
        
        # Get model info
        num_params = sum(p.numel() for p in test_model.parameters())
        print(f"Model loaded successfully")
        print(f"   Parameters: {num_params:,}")
        print(f"   Size on disk: ~{num_params * 4 / (1024**2):.1f} MB")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")
        
        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        
        print()
        print("=" * 70)
        print("Wav2Vec2 Base Model Ready for Fine-tuning!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Fine-tune with speechocean762 dataset")
        print("2. Save to: models/wav2vec2-english-finetuned/")
        print("3. Update config.py WAV2VEC_MODEL_PATH")
        
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def check_disk_space():
    """Check if there's enough disk space for model download."""
    import shutil
    
    total, used, free = shutil.disk_usage(BASE_DIR)
    free_gb = free / (1024**3)
    
    print(f"Disk space check:")
    print(f"   Free: {free_gb:.2f} GB")
    
    if free_gb < 2:
        print(f"Warning: Low disk space. Need at least 2 GB free.")
        return False
    
    print(f"Sufficient disk space")
    return True

def main():
    print()
    print("ML Model Download Script")
    print("   For KataDia AI - English Pronunciation Assessment")
    print()
    
    # Check disk space
    if not check_disk_space():
        print()
        print("Insufficient disk space. Please free up space and try again.")
        sys.exit(1)
    
    print()
    
    # Download wav2vec2-base
    success = download_wav2vec2_base()
    
    if success:
        print()
        print("All models downloaded successfully!")
        print()
        print("Model locations:")
        print(f"   Wav2Vec2 Base: {BASE_DIR / 'models' / 'wav2vec2-base'}")
        print()
        print("To use the model, update your .env file:")
        print("   WAV2VEC_MODEL_PATH=./models/wav2vec2-base")
        print()
    else:
        print()
        print("Model download failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
