import os
import sys
import subprocess
import argparse
from pathlib import Path
import warnings

# Suppress common deprecation warnings from ML libraries
warnings.filterwarnings("ignore", category=UserWarning, module="praatio")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")

# Base directory where this script lives. Use this to resolve .env, venv, and
# requirements paths so the script works regardless of the current working dir.
BASE_DIR = Path(__file__).resolve().parent

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("Error: Python 3.9+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    
    print(f"Python {version.major}.{version.minor}.{version.micro} detected")

def check_environment():
    """Check if environment is properly set up.

    This looks for `.env` next to this script (BASE_DIR). If `.env` is missing
    it attempts to create it from `.env.example` (also next to this script).
    """
    env_file = BASE_DIR / '.env'

    if not env_file.exists():
        print(".env file not found. Creating from template...")
        template_file = BASE_DIR / '.env.example'
        if template_file.exists():
            import shutil
            shutil.copy(template_file, env_file)
            print(".env file created from template")
            print("Please edit .env and set your OPENAI_API_KEY")
        else:
            print(".env.example not found. Please create .env manually")
            return False

    # Check for API key (warn but don't fail)
    try:
        with open(env_file, 'r') as f:
            env_content = f.read()
            if 'OPENAI_API_KEY=' not in env_content or 'your_gemini' in env_content:
                print("WARNING: Gemini API key not configured!")
                print("Please edit .env file and set OPENAI_API_KEY")
                print("You can get a free key from: https://aistudio.google.com/app/apikey")
                print("")
    except Exception as e:
        print(f"Failed to read .env file: {e}")
        return False

    return True

def setup_environment():
    """Setup virtual environment if needed."""
    venv_dir = BASE_DIR / '.venv'

    if not venv_dir.exists():
        print("Creating virtual environment...")
        result = subprocess.run([sys.executable, '-m', 'venv', '.venv'], cwd=str(BASE_DIR), capture_output=True)
        if result.returncode != 0:
            print("Failed to create virtual environment")
            print(result.stderr.decode() if result.stderr else "")
            return False
        print("Virtual environment created")
    
    # Activate virtual environment and install dependencies
    print("Installing/Updating dependencies...")

    if sys.platform == "win32":
        pip_path = venv_dir / 'Scripts' / 'pip.exe'
    else:
        pip_path = venv_dir / 'bin' / 'pip'

    if pip_path.exists():
        subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], capture_output=True)
        req_file = BASE_DIR / 'requirements.txt'
        try:
            result2 = subprocess.run([str(pip_path), 'install', '-r', str(req_file)], capture_output=True, text=True, timeout=300)
            if result2.returncode != 0:
                print("Failed to install dependencies")
                if result2.stderr:
                    print("STDERR:", result2.stderr[:500])
                if result2.stdout:
                    print("STDOUT:", result2.stdout[-500:])
                return False
        except Exception as e:
            print(f"Exception during pip install: {e}")
            import traceback
            traceback.print_exc()
            return False

        print("Dependencies installed successfully")
    else:
        print("Virtual environment not set up correctly")
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        'data/processed_english',
        'models/wav2vec2-english-finetuned', 
        'logs',
        'temp_audio'
    ]
    
    for directory in directories:
        (BASE_DIR / directory).mkdir(parents=True, exist_ok=True)
    
    print("Directories created")

def run_service(host='127.0.0.1', port=8000, debug=False):
    """Run the FastAPI service."""
    print(f"Starting KataDia AI Service...")
    print(f"Service will be available at: http://{host}:{port}")
    print(f"API Docs: http://{host}:{port}/docs")
    print("")
    
    # Set environment variables
    env_vars = os.environ.copy()
    # Suppress deprecation warnings from ML libraries
    env_vars['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::FutureWarning'
    if debug:
        env_vars['LOG_LEVEL'] = 'DEBUG'
        env_vars['DEBUG'] = 'true'
    
    # Prepare uvicorn command
    venv_dir = BASE_DIR / '.venv'
    if sys.platform == "win32":
        python_path = venv_dir / 'Scripts' / 'python.exe'
    else:
        python_path = venv_dir / 'bin' / 'python'

    if not python_path.exists():
        python_path = Path(sys.executable)

    command = [
        str(python_path), '-m', 'uvicorn', 'app.main:app',
        '--host', host,
        '--port', str(port),
        '--reload'
    ]

    if debug:
        command.extend(['--log-level', 'debug'])

    try:
        print(f"Running command: {' '.join(command)}")
        # Run uvicorn with the service directory as cwd so the app module resolves
        subprocess.run(command, env=env_vars, cwd=str(BASE_DIR))
    except KeyboardInterrupt:
        print("\nService stopped by user")
    except Exception as e:
        print(f"Failed to start service: {e}")

def test_service(host='127.0.0.1', port=8000):
    """Test if service is running properly."""
    import requests
    import time
    
    print(f"Testing service at http://{host}:{port}")
    
    # Wait a moment for service to start
    time.sleep(2)
    
    try:
        # Test health endpoint
        response = requests.get(f"http://{host}:{port}/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("Service is responding correctly")
            return True
        else:
            print(f"Service responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("Cannot connect to service. Is it running?")
        return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='KataDia AI Service')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--test', action='store_true', help='Test service after startup')
    parser.add_argument('--setup-only', action='store_true', help='Only setup environment, don\'t start service')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("KataDia AI Service")
    print("=" * 60)
    print()
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Check environment setup
    if not check_environment():
        print("Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Step 3: Setup if needed
    if not args.skip_deps:
        if not setup_environment():
            print("Environment setup failed.")
            sys.exit(1)
        
        create_directories()
    
    # Step 4: Check if we should only do setup
    if args.setup_only:
        print("Environment setup completed successfully!")
        print("Run this script again without --setup-only to start service")
        return
    
    # Step 5: Start service
    run_service(args.host, args.port, args.debug)
    
    # Step 6: Test if requested
    if args.test:
        print("\nTesting service...")
        if test_service(args.host, args.port):
            print("Service is working perfectly!")
        else:
            print("Service test failed. Check the logs.")

if __name__ == "__main__":
    main()
