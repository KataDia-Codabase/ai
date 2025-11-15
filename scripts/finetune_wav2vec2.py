#!/usr/bin/env python3
"""
Fine-tune Wav2Vec2-base model with speechocean762 dataset for English pronunciation assessment.

This script:
1. Loads speechocean762 dataset (5000 English sentences with pronunciation scores)
2. Prepares data for Wav2Vec2 training
3. Fine-tunes wav2vec2-base model
4. Saves fine-tuned model to models/wav2vec2-english-finetuned/

Dataset info:
- 5000 English sentences
- Non-native speakers (Chinese L1)
- Phoneme-level, word-level, sentence-level scores
- Train/test split already provided
"""

import os
import sys
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datasets import Dataset, DatasetDict
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
from transformers import logging as transformers_logging

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Suppress some warnings
transformers_logging.set_verbosity_error()

# Paths
DATASET_PATH = BASE_DIR / "datasets" / "speechocean762-1.2.0"
MODEL_INPUT_PATH = BASE_DIR / "models" / "wav2vec2-base"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "wav2vec2-english-finetuned"
SCORES_PATH = DATASET_PATH / "resource" / "scores.json"

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
        
        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        batch["labels"] = labels
        
        return batch


def load_speechocean762_data(split="train"):
    """
    Load speechocean762 dataset for the given split.
    
    Args:
        split: "train" or "test"
        
    Returns:
        List of dicts with audio_path, text, and scores
    """
    print(f"Loading {split} data...")
    
    # Load scores
    with open(SCORES_PATH, 'r') as f:
        scores_data = json.load(f)
    
    # Load split info
    split_path = DATASET_PATH / split
    text_file = split_path / "text"
    wav_scp_file = split_path / "wav.scp"
    
    # Read text file (format: utt_id text)
    texts = {}
    with open(text_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, text = parts
                texts[utt_id] = text
    
    # Read wav.scp file (format: utt_id path)
    wav_paths = {}
    with open(wav_scp_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, rel_path = parts
                # Convert relative path to absolute
                abs_path = DATASET_PATH / rel_path
                wav_paths[utt_id] = str(abs_path)
    
    # Combine data
    data = []
    for utt_id in texts.keys():
        if utt_id in wav_paths and utt_id in scores_data:
            data.append({
                "utt_id": utt_id,
                "audio_path": wav_paths[utt_id],
                "text": texts[utt_id],
                "accuracy": scores_data[utt_id].get("accuracy", 0),
                "fluency": scores_data[utt_id].get("fluency", 0),
                "completeness": scores_data[utt_id].get("completeness", 0),
                "total": scores_data[utt_id].get("total", 0)
            })
    
    print(f"Loaded {len(data)} samples from {split} split")
    return data


def prepare_dataset(processor, split="train", max_samples=None):
    """
    Prepare dataset for training.
    
    Args:
        processor: Wav2Vec2Processor
        split: "train" or "test"
        max_samples: Maximum number of samples to use (for testing)
        
    Returns:
        HuggingFace Dataset
    """
    data = load_speechocean762_data(split)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Using only {max_samples} samples for {split}")
    
    def speech_file_to_array_fn(path):
        """Load audio file using soundfile."""
        speech_array, sampling_rate = sf.read(path)
        
        # Convert to mono if stereo
        if len(speech_array.shape) > 1:
            speech_array = speech_array.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sampling_rate != 16000:
            # Simple resampling using librosa
            import librosa
            speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
        
        return speech_array
    
    def prepare_example(example):
        """Prepare single example for training."""
        # Load and process audio
        audio = speech_file_to_array_fn(example["audio_path"])
        example["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
        
        # Prepare text labels using tokenizer directly (without as_target_processor)
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        
        return example
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(data)
    
    # Process dataset
    print(f"Processing {split} dataset...")
    dataset = dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
        desc=f"Processing {split}"
    )
    
    return dataset


def compute_metrics(pred):
    """Compute WER metric."""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    # Just a placeholder - in real scenario you'd compute WER
    # For pronunciation assessment, we focus on accuracy scores
    return {"accuracy": 0.0}


def main():
    print("=" * 70)
    print("Fine-tuning Wav2Vec2 for English Pronunciation Assessment")
    print("=" * 70)
    print()
    
    # Check if base model exists
    if not MODEL_INPUT_PATH.exists():
        print(f"Base model not found at: {MODEL_INPUT_PATH}")
        print("   Please run: python scripts/download_models.py")
        sys.exit(1)
    
    print(f"Dataset: {DATASET_PATH}")
    print(f"📥 Base model: {MODEL_INPUT_PATH}")
    print(f"Output: {MODEL_OUTPUT_PATH}")
    print()
    
    # Force GPU usage for local training
    if not torch.cuda.is_available():
        print("ERROR: GPU (CUDA) not available!")
        print("   This script requires GPU for training.")
        print("   Please ensure CUDA is properly installed.")
        sys.exit(1)
    
    device = "cuda"
    print(f"Device: {device}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print()
    
    # Load processor and model
    print("Loading processor and model...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_INPUT_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_INPUT_PATH)
    
    # Move model to GPU (forced)
    model = model.to(device)
    print(f"Model loaded and moved to GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Prepare datasets
    print("Preparing datasets...")
    # For quick testing, use subset. For full training, remove max_samples
    train_dataset = prepare_dataset(processor, split="train")  # Use 100 for quick test
    test_dataset = prepare_dataset(processor, split="test")     # Use 50 for quick test
    print()
    
    # Training arguments - optimized for GPU
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_PATH),
        group_by_length=True,
        per_device_train_batch_size=8,   # Increased batch size for GPU
        per_device_eval_batch_size=8,
        eval_strategy="no",              # Disable evaluation during training (avoid multiprocessing issues on Windows)
        num_train_epochs=3,
        fp16=True,                       # Force mixed precision training on GPU
        save_steps=100,
        logging_steps=10,
        learning_rate=3e-5,
        warmup_steps=100,
        save_total_limit=2,
        push_to_hub=False,
        dataloader_num_workers=0,        # CRITICAL: Disable multiprocessing on Windows to avoid pickle errors
        gradient_accumulation_steps=2,   # Effective batch size = 8 * 2 = 16
    )
    
    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,  # Use tokenizer instead of processing_class
    )
    
    # Train
    print("Starting training...")
    print("=" * 70)
    print()
    
    try:
        trainer.train()
        
        print()
        print("=" * 70)
        print("Training completed!")
        print("=" * 70)
        print()
        
        # Save model
        print("Saving fine-tuned model...")
        trainer.save_model(MODEL_OUTPUT_PATH)
        processor.save_pretrained(MODEL_OUTPUT_PATH)
        print(f"Model saved to: {MODEL_OUTPUT_PATH}")
        print()
        
        print("🎉 Fine-tuning completed successfully!")
        print()
        print("Next steps:")
        print("1. Update .env file:")
        print("   WAV2VEC_MODEL_PATH=./models/wav2vec2-english-finetuned")
        print("2. Test the model with pronunciation scoring")
        print("3. Evaluate on test set")
        
    except Exception as e:
        print()
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
