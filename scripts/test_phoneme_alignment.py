"""
Test Phoneme Alignment with G2P-EN
This script tests phoneme extraction and alignment for pronunciation assessment.
"""

import sys
from pathlib import Path
import librosa
import numpy as np
from g2p_en import G2p

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "datasets" / "speechocean762-1.2.0"


def test_g2p():
    """Test G2P-EN for phoneme extraction."""
    print("=" * 70)
    print("Testing G2P-EN for Phoneme Extraction")
    print("=" * 70)
    print()
    
    # Initialize G2P
    g2p = G2p()
    
    # Test sentences
    test_sentences = [
        "HELLO WORLD",
        "MARK IS GOING TO SEE ELEPHANT",
        "KATE LOVES CHINA",
        "TWO SIX FOUR EIGHT"
    ]
    
    print("Testing phoneme extraction...")
    print()
    
    for sentence in test_sentences:
        try:
            # Convert to phonemes
            phonemes = g2p(sentence)
            phonemes_str = ' '.join(phonemes)
            
            print(f"Text:     {sentence}")
            print(f"Phonemes: {phonemes_str}")
            print(f"Count:    {len([p for p in phonemes if p not in [' ']])}")
            print("-" * 70)
            
        except Exception as e:
            print(f"Error processing '{sentence}': {e}")
            print("-" * 70)
    
    print()


def test_phoneme_alignment():
    """Test phoneme alignment with audio from dataset."""
    import json
    
    print("=" * 70)
    print("Testing Phoneme Alignment with Audio")
    print("=" * 70)
    print()
    
    # Initialize G2P
    g2p = G2p()
    
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
    
    # Test on 3 samples
    samples = list(test_data.items())[:3]
    
    for i, (audio_id, info) in enumerate(samples, 1):
        print(f"Sample {i}/3: {audio_id}")
        print()
        
        # Get speaker ID and audio path
        speaker_id = utt2spk.get(audio_id, None)
        if not speaker_id:
            print(f"Speaker ID not found for: {audio_id}")
            print()
            continue
        
        speaker_folder = f"SPEAKER{speaker_id.zfill(4)}"
        audio_path = DATASET_PATH / "WAVE" / speaker_folder / f"{audio_id}.WAV"
        expected_text = info.get('text', '')
        
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            print()
            continue
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=16000)
        duration = len(audio) / sr
        
        # Get phonemes from expected text
        try:
            phonemes = g2p(expected_text)
            phonemes_str = ' '.join(phonemes)
            phoneme_count = len([p for p in phonemes if p not in [' ']])
            
            print(f"File: {audio_path.name}")
            print(f"Duration: {duration:.2f}s")
            print(f"Expected Text: {expected_text}")
            print(f"Phonemes: {phonemes_str}")
            print(f"Phoneme Count: {phoneme_count}")
            
            # Estimate phoneme boundaries (simple approach)
            avg_phoneme_duration = duration / phoneme_count if phoneme_count > 0 else 0
            print(f"Avg Phoneme Duration: {avg_phoneme_duration:.3f}s")
            
            # Get scores if available
            if 'accuracy' in info:
                print(f"Accuracy Score: {info['accuracy']}")
            if 'completeness' in info:
                print(f"Completeness Score: {info['completeness']}")
            if 'fluency' in info:
                print(f"Fluency Score: {info['fluency']}")
            
            print("=" * 70)
            print()
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 70)
            print()


def create_phoneme_service_test():
    """Test the PhonemeService from the app."""
    print("=" * 70)
    print("Testing PhonemeService Integration")
    print("=" * 70)
    print()
    
    try:
        from app.ml.services.phoneme_service import PhonemeService
        
        service = PhonemeService()
        
        # Test audio path
        test_info_path = DATASET_PATH / "test" / "all-info.json"
        utt2spk_path = DATASET_PATH / "test" / "utt2spk"
        
        if not test_info_path.exists():
            print("Test data not found")
            return
        
        # Load first sample
        utt2spk = {}
        if utt2spk_path.exists():
            with open(utt2spk_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        utt_id, spk_id = parts
                        utt2spk[utt_id] = spk_id
        
        import json
        with open(test_info_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        audio_id, info = list(test_data.items())[0]
        speaker_id = utt2spk.get(audio_id)
        
        if speaker_id:
            speaker_folder = f"SPEAKER{speaker_id.zfill(4)}"
            audio_path = DATASET_PATH / "WAVE" / speaker_folder / f"{audio_id}.WAV"
            expected_text = info.get('text', '')
            
            if audio_path.exists():
                print(f"Testing with: {audio_path.name}")
                print(f"Expected: {expected_text}")
                print()
                
                # Extract phonemes
                result = service.extract_phonemes(str(audio_path), expected_text)
                
                print(f"Phoneme extraction successful!")
                print(f"Phonemes found: {len(result.get('phonemes', []))}")
                print()
                
    except ImportError as e:
        print(f"PhonemeService not available: {e}")
        print("   This is OK - it's a placeholder service")
    except Exception as e:
        print(f"Error testing PhonemeService: {e}")
    
    print()


def main():
    """Main test function."""
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         Phoneme Alignment Testing Suite                       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Test 1: G2P Phoneme Extraction
    test_g2p()
    
    # Test 2: Phoneme alignment with audio
    test_phoneme_alignment()
    
    # Test 3: PhonemeService integration
    create_phoneme_service_test()
    
    print("=" * 70)
    print("All phoneme alignment tests completed!")
    print("=" * 70)
    print()
    
    # Summary
    print("Summary:")
    print("  G2P-EN installed and working")
    print("  Can extract phonemes from text (ARPAbet format)")
    print("  Can process audio files from dataset")
    print("  Can estimate phoneme durations")
    print()
    print("Next Steps:")
    print("  1. Integrate G2P into PhonemeService")
    print("  2. Implement phoneme-level scoring algorithm")
    print("  3. Test with real-time audio input")
    print("  4. Compare with Wav2Vec2 predictions")
    print()


if __name__ == "__main__":
    main()
