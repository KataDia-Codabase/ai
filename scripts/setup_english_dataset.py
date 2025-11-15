#!/usr/bin/env python3
"""
Setup script for English pronunciation dataset from speechocean762.
This script preprocesses the dataset for English pronunciation scoring training.
"""

import os
import json
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
import structlog
from typing import List, Dict, Tuple
import zipfile
import shutil

logger = structlog.get_logger()

class EnglishDatasetProcessor:
    """Process speechocean762 dataset for English pronunciation scoring."""
    
    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # English-specific phoneme inventory
        self.english_phonemes = {
            'vowels': ['i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɒ', 'ʌ', 'ɔ', 'ʊ', 'u', 'ɪə', 'eə', 'ɔə', 'aɪ', 'aʊ', 'ɔɪ', 'ə', 'ɚ'],
            'consonants': ['b', 'd', 'g', 'k', 'p', 't', 'f', 'h', 's', 'ʃ', 'θ', 'ð', 'v', 'z', 'ʒ', 'dʒ', 'tʃ', 'm', 'n', 'ŋ', 'l', 'ɹ', 'w', 'j', 'x'],
            'diphthongs': ['aɪ', 'aʊ', 'ɔɪ', 'ɪə', 'eə', 'ʊə']
        }
    
    def extract_dataset(self) -> bool:
        """Extract speechocean762 dataset if compressed."""
        try:
            # Look for zip file
            zip_files = list(self.dataset_path.glob("*.zip"))
            if zip_files:
                logger.info("Found zip file, extracting...")
                with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                    zip_ref.extractall(self.dataset_path)
                logger.info("Dataset extraction completed")
                return True
            else:
                logger.info("Dataset already extracted")
                return True
        except Exception as e:
            logger.error(f"Failed to extract dataset: {e}")
            return False
    
    def find_dataset_structure(self) -> Dict:
        """Find and analyze dataset structure."""
        structure = {
            'audio_files': [],
            'transcripts': [],
            'phonetic_annotations': []
        }
        
        # Common patterns in speech datasets
        audio_patterns = ['**/*.wav', '**/*.mp3', '**/*.flac']
        transcript_patterns = ['**/transcript*', '**/*.txt', '**/*.json']
        
        for pattern in audio_patterns:
            structure['audio_files'].extend(self.dataset_path.glob(pattern))
        
        for pattern in transcript_patterns:
            structure['transcripts'].extend(self.dataset_path.glob(pattern))
        
        logger.info(f"Found {len(structure['audio_files'])} audio files")
        logger.info(f"Found {len(structure['transcripts'])} transcript files")
        
        return structure
    
    def preprocess_audio_files(self, audio_files: List[Path]) -> List[Dict]:
        """Preprocess audio files for training."""
        processed_data = []
        
        for i, audio_file in enumerate(audio_files[:1000]):  # Limit for testing
            try:
                # Load audio
                y, sr = librosa.load(audio_file, sr=16000)
                
                # Normalize audio
                y = librosa.util.normalize(y)
                
                # Extract basic features
                duration = len(y) / sr
                rms_energy = librosa.feature.rms(y=y)[0].mean()
                zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
                
                # Get MFCC features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                
                processed_data.append({
                    'file_id': audio_file.stem,
                    'file_path': str(audio_file),
                    'duration': duration,
                    'sample_rate': sr,
                    'rms_energy': float(rms_energy),
                    'zero_crossing_rate': float(zcr),
                    'mfcc_features': mfccs.tolist(),
                    'audio_length': len(y)
                })
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} audio files")
                    
            except Exception as e:
                logger.warning(f"Failed to process {audio_file}: {e}")
                continue
        
        return processed_data
    
    def load_transcripts(self, transcript_files: List[Path]) -> Dict[str, str]:
        """Load transcripts from various formats."""
        transcripts = {}
        
        for transcript_file in transcript_files:
            try:
                if transcript_file.suffix == '.txt':
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            transcripts[transcript_file.stem] = content
                            
                elif transcript_file.suffix == '.json':
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Handle JSON transcript formats
                        if isinstance(data, dict):
                            if 'text' in data:
                                transcripts[transcript_file.stem] = data['text']
                            elif 'transcript' in data:
                                transcripts[transcript_file.stem] = data['transcript']
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'text' in item:
                                    transcripts[item.get('id', f"item_{len(transcripts)}")] = item['text']
                                
            except Exception as e:
                logger.warning(f"Failed to load transcript {transcript_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(transcripts)} transcripts")
        return transcripts
    
    def create_training_data(
        self, 
        processed_audio: List[Dict], 
        transcripts: Dict[str, str]
    ) -> pd.DataFrame:
        """Create training dataset by matching audio and transcripts."""
        training_data = []
        
        for audio_info in processed_audio:
            file_id = audio_info['file_id']
            
            # Find matching transcript
            transcript = transcripts.get(file_id)
            if not transcript:
                # Try fuzzy matching
                for trans_id, text in transcripts.items():
                    if file_id in trans_id or trans_id in file_id:
                        transcript = text
                        break
            
            if transcript:
                training_data.append({
                    'file_id': file_id,
                    'audio_path': audio_info['file_path'],
                    'transcript': transcript,
                    'duration': audio_info['duration'],
                    'rms_energy': audio_info['rms_energy'],
                    'zero_crossing_rate': audio_info['zero_crossing_rate'],
                    'mfcc_features': audio_info['mfcc_features'],
                    'audio_length': audio_info['audio_length']
                })
        
        if training_data:
            df = pd.DataFrame(training_data)
            logger.info(f"Created training dataset with {len(df)} samples")
            return df
        else:
            logger.error("No training data could be created")
            return pd.DataFrame()
    
    def generate_phonetic_transcriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate phonetic transcriptions for English text."""
        # Basic phonetic mapping for English
        basic_phonetics = {
            'a': 'ə', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'ɪ',
            'f': 'f', 'g': 'g', 'h': 'h', 'i': 'ɪ', 'j': 'dʒ',
            'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'ɒ',
            'p': 'p', 'q': 'k', 'r': 'ɹ', 's': 's', 't': 't',
            'u': 'ʌ', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j', 'z': 'z'
        }
        
        def text_to_phonetics(text: str) -> str:
            """Convert text to basic phonetic transcription."""
            phonetic = []
            for char in text.lower():
                if char.isalpha():
                    phonetic.append(basic_phonetics.get(char, char))
                elif char.isspace():
                    phonetic.append(' ')
            return ''.join(phonetic)
        
        df['phonetic_transcript'] = df['transcript'].apply(text_to_phonetics)
        df['word_count'] = df['transcript'].apply(lambda x: len(x.split()))
        df['phoneme_count'] = df['phonetic_transcript'].apply(lambda x: len([p for p in x.replace(' ', '')]))
        
        return df
    
    def save_processed_dataset(self, df: pd.DataFrame):
        """Save processed dataset to files."""
        # Save main dataset
        output_csv = self.output_path / "english_pronunciation_dataset.csv"
        df.to_csv(output_csv, index=False)
        
        # Save audio metadata
        metadata = {
            'total_samples': len(df),
            'avg_duration': df['duration'].mean(),
            'total_duration_hours': df['duration'].sum() / 3600,
            'unique_words': len(set(' '.join(df['transcript'].tolist()).split())),
            'avg_words_per_utterance': df['word_count'].mean(),
            'dataset_info': {
                'target_language': 'en-US',
                'task': 'pronunciation_scoring',
                'features': ['mfcc', 'rms', 'zcr', 'timing'],
                'phonemes_used': list(self.english_phonemes.values())
            }
        }
        
        with open(self.output_path / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save sample files for validation
        sample_size = min(100, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        sample_df.to_csv(self.output_path / "sample_dataset.csv", index=False)
        
        logger.info(f"Dataset saved to {self.output_path}")
        logger.info(f"Total samples: {metadata['total_samples']}")
        logger.info(f"Total duration: {metadata['total_duration_hours']:.2f} hours")
    
    def run(self):
        """Run the complete dataset processing pipeline."""
        logger.info("Starting English dataset processing...")
        
        # Step 1: Extract dataset if needed
        if not self.extract_dataset():
            raise RuntimeError("Failed to extract dataset")
        
        # Step 2: Find dataset structure
        structure = self.find_dataset_structure()
        
        if not structure['audio_files']:
            raise RuntimeError("No audio files found")
        
        # Step 3: Preprocess audio
        logger.info("Preprocessing audio files...")
        processed_audio = self.preprocess_audio_files(structure['audio_files'])
        
        # Step 4: Load transcripts
        logger.info("Loading transcripts...")
        transcripts = self.load_transcripts(structure['transcripts'])
        
        # Step 5: Create training dataset
        logger.info("Creating training dataset...")
        df = self.create_training_data(processed_audio, transcripts)
        
        if df.empty:
            raise RuntimeError("Failed to create training dataset")
        
        # Step 6: Generate phonetic transcriptions
        logger.info("Generating phonetic transcriptions...")
        df = self.generate_phonetic_transcriptions(df)
        
        # Step 7: Save processed dataset
        self.save_processed_dataset(df)
        
        logger.info("English dataset processing completed successfully!")
        return df

def main():
    """Main function to run dataset processing."""
    # Configuration
    dataset_path = "../../../datasets/speechocean762-1.2.0"  # Adjust path as needed
    output_path = "./data/processed_english"
    
    # Create processor and run
    processor = EnglishDatasetProcessor(dataset_path, output_path)
    try:
        df = processor.run()
        logger.info("Dataset processing completed successfully!")
        return df
    except Exception as e:
        logger.error(f"Dataset processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
