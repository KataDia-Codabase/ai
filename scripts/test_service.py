import requests
import json
import time

def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("Health check passed")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def create_test_audio():
    """Create a simple test audio file."""
    import numpy as np
    import soundfile as sf
    import tempfile
    
    # Generate 2 seconds of test audio
    duration = 2.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    
    t = np.linspace(0, duration, samples)
    # Generate signal around 200Hz (male voice range)
    audio = 0.5 * np.sin(2 * np.pi * 150 * t)
    audio = audio + 0.2 * np.sin(2 * np.pi * 300 * t)
    
    # Add envelope variation
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    audio = audio * envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Add small noise
    noise = 0.02 * np.random.normal(0, 1, samples)
    audio = audio + noise
    
    # Save file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, sample_rate)
    temp_file.close()
    
    return temp_file.name

def test_scoring():
    """Test pronunciation scoring endpoint."""
    print("Testing pronunciation scoring...")
    
    try:
        # Create test audio
        audio_file = create_test_audio()
        print(f"Created test audio: {audio_file}")
        
        # Prepare request
        url = "http://localhost:8000/api/v1/score"
        
        with open(audio_file, 'rb') as f:
            files = {
                'audio': f,
                'transcript': (None, 'hello how are you today'),
                'language': (None, 'en-US'),
                'user_id': (None, 'test_user_123'),
                'session_id': (None, 'test_session_456')
            }
            
            print("Sending scoring request...")
            start_time = time.time()
            
            response = requests.post(url, files=files, timeout=30)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"Response time: {duration:.3f}s")
            
        if response.status_code == 200:
            print("Scoring test passed!")
            result = response.json()
            
            print("\nResults:")
            print(f"Overall Score: {result.get('overall_score', 'N/A')}")
            print(f"Analysis Level: {result.get('analysis_level', 'N/A')}")
            print(f"Processing Time: {result.get('processing_time', 'N/A')}s")
            
            # Show dimensions
            dimensions = result.get('dimensions', {})
            if dimensions:
                print("\nDimension Scores:")
                for dim, score in dimensions.items():
                    print(f"  {dim.title()}: {score}")
            
            # Show features if available  
            features = result.get('features', {})
            if 'cefr_assessment' in features:
                cefr = features['cefr_assessment']
                print(f"\nCEFR Assessment:")
                print(f"  Level: {cefr.get('level', 'N/A')}")
                print(f"  Confidence: {cefr.get('confidence', 'N/A')}")
            
            print(f"\nFeedback: {result.get('feedback', 'No feedback generated')}") 
            
            return True
            
        else:
            print(f"Scoring failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Scoring test error: {e}")
        return False
    finally:
        # Cleanup test audio file
        try:
            import os
            if 'audio_file' in locals():
                os.unlink(audio_file)
        except:
            pass

def test_transcription():
    """Test transcription endpoint."""
    print("Testing transcription...")
    
    try:
        audio_file = create_test_audio()
        
        url = "http://localhost:8000/api/v1/transcribe"
        
        with open(audio_file, 'rb') as f:
            files = {'audio': f}
            data = {'language': 'en-US'}
            
            print("Sending transcription request...")
            response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            print("Transcription test passed!")
            result = response.json()
            
            print(f"Transcript: {result.get('transcript', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            print(f"Engine: {result.get('engine', 'N/A')}")
            
            return True
        else:
            print(f"Transcription failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Transcription test error: {e}")
        return False
    finally:
        try:
            import os
            if 'audio_file' in locals():
                os.unlink(audio_file)
        except:
            pass

def main():
    """Run all tests."""
    print("=" * 60)
    print("English Pronunciation AI Service - Test Suite")
    print("=" * 60)
    print()
    
    # Check if service is running
    print("Checking if service is available...")
    if not test_health():
        print("\nService is not running or not responding.")
        print("Please start the service first:")
        print("   python run.py")
        print("   or: python -m uvicorn app.main:app --reload")
        return
    
    print()
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Transcription", test_transcription), 
        ("Pronunciation Scoring", test_scoring)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print('-' * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        
        print()
        
        # Add delay between tests
        time.sleep(1)
    
    # Summary
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("All tests passed! Service is working perfectly!")
        print("\nService is ready for use!")
        print("Open http://localhost:8000/docs to explore the API")
    else:
        print("Some tests failed. Check the logs for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
