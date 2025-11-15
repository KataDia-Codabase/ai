"""
Test Fine-tuned Wav2Vec2 Model for Pronunciation Assessment
"""

import os
import sys
from pathlib import Path
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "wav2vec2-english-finetuned"
DATASET_PATH = BASE_DIR / "datasets" / "speechocean762-1.2.0"


def load_model():
    """Load the fine-tuned model."""
    print("=" * 70)
    print("Testing Fine-tuned Wav2Vec2 Model")
    print("=" * 70)
    print()
    
    if not MODEL_PATH.exists():
        print(f"Fine-tuned model not found at: {MODEL_PATH}")
        print("   Please run fine-tuning first: python scripts/finetune_wav2vec2.py")
        sys.exit(1)
    
    print(f"Loading model from: {MODEL_PATH}")
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load processor and model
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    print()
    
    return processor, model, device


def transcribe_audio(audio_path, processor, model, device, expected_text=None):
    """Transcribe audio file using the model."""
    # Load audio
    speech, sr = librosa.load(audio_path, sr=16000)
    
    # Process audio
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Move to device
    input_values = inputs.input_values.to(device)
    
    # Get logits
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    # Calculate confidence (average max probability)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_probs = torch.max(probs, dim=-1)[0]
    confidence = max_probs.mean().item()
    
    # Print results
    print(f"File: {Path(audio_path).name}")
    if expected_text:
        print(f"Expected: {expected_text}")
    print(f"Transcription: {transcription}")
    print(f"Confidence: {confidence:.2%}")
    
    # Calculate accuracy if expected text provided
    if expected_text:
        # Simple word-level accuracy
        expected_words = expected_text.lower().split()
        transcribed_words = transcription.lower().split()
        
        correct = sum(1 for e, t in zip(expected_words, transcribed_words) if e == t)
        total = max(len(expected_words), len(transcribed_words))
        accuracy = correct / total if total > 0 else 0
        
        print(f"Word Accuracy: {accuracy:.2%}")
    
    print("-" * 70)
    print()
    
    return transcription, confidence


def test_from_dataset(processor, model, device, num_samples=5):
    """Test model on samples from the test dataset."""
    import json
    
    print("Testing on dataset samples...")
    print()
    
    # Load test data
    test_info_path = DATASET_PATH / "test" / "all-info.json"
    utt2spk_path = DATASET_PATH / "test" / "utt2spk"
    
    if not test_info_path.exists():
        print(f"Test data not found at: {test_info_path}")
        return
    
    # Load utt2spk mapping
    utt2spk = {}
    if utt2spk_path.exists():
        with open(utt2spk_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    utt_id, spk_id = parts
                    utt2spk[utt_id] = spk_id
    
    with open(test_info_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Test on first N samples
    samples = list(test_data.items())[:num_samples]
    
    total_confidence = 0
    total_accuracy = 0
    
    for i, (audio_id, info) in enumerate(samples, 1):
        print(f"Sample {i}/{num_samples}")
        
        # Get speaker ID from utt2spk mapping
        speaker_id = utt2spk.get(audio_id, None)
        if not speaker_id:
            print(f"Speaker ID not found for: {audio_id}")
            print()
            continue
        
        # Construct audio path: WAVE/SPEAKER{speaker_id}/{audio_id}.WAV
        speaker_folder = f"SPEAKER{speaker_id.zfill(4)}"
        audio_path = DATASET_PATH / "WAVE" / speaker_folder / f"{audio_id}.WAV"
        expected_text = info.get('text', '')
        
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            print()
            continue
        
        transcription, confidence = transcribe_audio(
            str(audio_path), 
            processor, 
            model, 
            device, 
            expected_text
        )
        
        total_confidence += confidence
    
    # Summary
    avg_confidence = total_confidence / num_samples
    
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Samples tested: {num_samples}")
    print(f"Average confidence: {avg_confidence:.2%}")
    print()


def test_custom_audio(audio_path, processor, model, device):
    """Test model on a custom audio file."""
    print("Testing custom audio file...")
    print()
    
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        return
    
    transcription, confidence = transcribe_audio(audio_path, processor, model, device)
    
    print("=" * 70)
    print("Test completed")
    print("=" * 70)
    print()


def main():
    """Main test function."""
    # Load model
    processor, model, device = load_model()
    
    # Test on dataset samples
    print("Testing on 5 samples from test dataset...")
    print()
    test_from_dataset(processor, model, device, num_samples=5)
    
    # Instructions for custom testing
    print()
    print("=" * 70)
    print("To test on your own audio file:")
    print("=" * 70)
    print()
    print("  1. Prepare an audio file (WAV, MP3, etc.)")
    print("  2. Run this script with the audio path:")
    print()
    print("     python scripts/test_finetuned_model.py --audio <path_to_audio.wav>")
    print()
    print("  Or use it in Python code:")
    print()
    print("     from scripts.test_finetuned_model import load_model, transcribe_audio")
    print("     processor, model, device = load_model()")
    print("     transcription, confidence = transcribe_audio('audio.wav', processor, model, device)")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test fine-tuned Wav2Vec2 model")
    parser.add_argument("--audio", type=str, help="Path to audio file to test")
    parser.add_argument("--samples", type=int, default=5, help="Number of dataset samples to test")
    
    args = parser.parse_args()
    
    # Load model
    processor, model, device = load_model()
    
    if args.audio:
        # Test custom audio
        test_custom_audio(args.audio, processor, model, device)
    else:
        # Test on dataset
        test_from_dataset(processor, model, device, num_samples=args.samples)
        
        # Show instructions
        print()
        print("=" * 70)
        print("To test on your own audio file:")
        print("=" * 70)
        print()
        print("  python scripts/test_finetuned_model.py --audio <path_to_audio.wav>")
        print()
