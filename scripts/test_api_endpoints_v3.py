import requests
import json
from pathlib import Path
import time
import sys

BASE_URL = "http://localhost:8000/api/v1"

# Get the project root directory (parent of scripts directory)
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "speechocean762-1.2.0"

def test_health():
    """Test health endpoint."""
    print("=" * 70)
    print("Testing Health Endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def test_transcribe():
    """Test transcribe endpoint dengan audio file."""
    print("=" * 70)
    print("Testing Transcribe Endpoint")
    print("=" * 70)
    
    # Load first audio file
    test_info_path = DATASET_PATH / "test" / "all-info.json"
    utt2spk_path = DATASET_PATH / "test" / "utt2spk"
    
    print(f"Looking for test data at: {test_info_path}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Path exists: {test_info_path.exists()}")
    
    if not test_info_path.exists():
        print(f"Test data not found at {test_info_path}")
        print()
        return False
    
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
    
    audio_id, info = list(test_data.items())[0]
    speaker_id = utt2spk.get(audio_id)
    
    if not speaker_id:
        print("Speaker ID not found")
        print()
        return False
    
    speaker_folder = f"SPEAKER{speaker_id.zfill(4)}"
    audio_path = DATASET_PATH / "WAVE" / speaker_folder / f"{audio_id}.WAV"
    expected_text = info.get('text', '')
    
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        print()
        return False
    
    try:
        print(f"Testing with: {audio_id}.WAV")
        print(f"Expected: {expected_text}")
        
        # Try sending Pydantic model fields individually as form fields
        # FastAPI will reconstruct the model from the flattened fields
        with open(audio_path, 'rb') as f:
            # Create proper multipart request with form fields and file
            form_data = {
                'language': (None, 'en-US'),  # (filename, data) - None filename for regular form field
                'audio_file': (f'{audio_id}.WAV', f, 'audio/wav')
            }
            
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/transcribe",
                files=form_data
            )
            elapsed = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Transcript: {result.get('transcript', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            print(f"Language: {result.get('language', 'N/A')}")
            print()
            return True
        else:
            error_detail = response.json().get('detail', response.text)
            print(f"Error: {str(error_detail)[:200]}")
            print()
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_score():
    """Test score endpoint."""
    print("=" * 70)
    print("Testing Score Endpoint")
    print("=" * 70)
    
    # Load first audio file
    test_info_path = DATASET_PATH / "test" / "all-info.json"
    utt2spk_path = DATASET_PATH / "test" / "utt2spk"
    
    print(f"Looking for test data at: {test_info_path}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Path exists: {test_info_path.exists()}")
    
    if not test_info_path.exists():
        print(f"Test data not found at {test_info_path}")
        print()
        return False
    
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
    
    audio_id, info = list(test_data.items())[0]
    speaker_id = utt2spk.get(audio_id)
    
    if not speaker_id:
        print("Speaker ID not found")
        print()
        return False
    
    speaker_folder = f"SPEAKER{speaker_id.zfill(4)}"
    audio_path = DATASET_PATH / "WAVE" / speaker_folder / f"{audio_id}.WAV"
    expected_text = info.get('text', '')
    
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        print()
        return False
    
    try:
        print(f"Testing with: {audio_id}.WAV")
        print(f"Expected: {expected_text}")
        
        # Send all ScoreRequest fields individually
        with open(audio_path, 'rb') as f:
            form_data = {
                'transcript': (None, expected_text),
                'language': (None, 'en-US'),
                'user_id': (None, 'test_user_001'),
                'session_id': (None, 'test_session_001'),
                'audio_file': (f'{audio_id}.WAV', f, 'audio/wav')
            }
            
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/score",
                files=form_data
            )
            elapsed = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Overall Score: {result.get('overall_score', 'N/A')}")
            print(f"Dimensions: {result.get('dimensions', {})}")
            print(f"Errors Found: {len(result.get('errors', []))}")
            print()
            return True
        else:
            error_detail = response.json().get('detail', response.text)
            print(f"Error: {str(error_detail)[:200]}")
            print()
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_feedback():
    """Test feedback endpoint."""
    print("=" * 70)
    print("Testing Feedback Endpoint")
    print("=" * 70)
    
    try:
        # Feedback endpoint does NOT require file upload
        # It only requires JSON body
        payload = {
            "pronunciation_data": {
                "overall_score": 85.5,
                "dimensions": {
                    "accuracy": 0.85,
                    "fluency": 0.80,
                    "prosody": 0.88,
                    "stress": 0.82
                },
                "errors": [
                    {
                        "type": "substitution",
                        "expected": "th",
                        "actual": "s",
                        "position": 2,
                        "confidence": 0.72
                    }
                ],
                "cefr_level": "B1"
            },
            "user_id": "test_user_001",
            "language": "en-US"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/feedback",
            json=payload
        )
        elapsed = time.time() - start_time
        
        print(f"Overall Score: {payload['pronunciation_data']['overall_score']}")
        print(f"Errors: {len(payload['pronunciation_data']['errors'])}")
        print(f"Status: {response.status_code}")
        print(f"Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"AI Feedback: {result.get('ai_feedback', 'N/A')[:100]}...")
            print(f"Suggestions: {len(result.get('improvement_suggestions', []))}")
            print(f"Next Steps: {len(result.get('next_steps', []))}")
            print()
            return True
        else:
            error_detail = response.json().get('detail', response.text)
            print(f"Error: {str(error_detail)[:200]}")
            print()
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Main test function."""
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║           API Endpoints Testing - Version 3                    ║")
    print("║        (Using correct FastAPI multipart form approach)         ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    results = {}
    
    # Test endpoints
    results['health'] = test_health()
    results['transcribe'] = test_transcribe()
    results['score'] = test_score()
    results['feedback'] = test_feedback()
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print()
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for endpoint, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status}: {endpoint.upper()}")
    
    print()
    print(f"Total: {passed}/{total} endpoints working")
    
    if passed == total:
        print("All endpoints working!")
    else:
        print(f"{total - passed} endpoint(s) need fixing")
    
    print()


if __name__ == "__main__":
    main()
