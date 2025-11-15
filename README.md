# KataDia AI - ML Service

AI/ML microservice for pronunciation analysis and feedback in the KataDia AI language learning platform.

## Features

- **Speech-to-Text**: Google Cloud STT with Whisper fallback
- **Phoneme Analysis**: Montreal Forced Aligner integration
- **Pronunciation Scoring**: Multi-dimensional scoring (accuracy, fluency, prosody, stress)
- **CEFR Assessment**: Automatic level placement (A1-C2)
- **Personalized Feedback**: AI-powered feedback using GPT-4
- **Adaptive Learning**: Difficulty adjustment based on performance

## Architecture

```
ai-ml-service/
├── app/
│   ├── api/
│   │   └── endpoints/          # API endpoints
│   │       ├── score.py       # Pronunciation scoring
│   │       ├── transcribe.py  # Speech-to-text
│   │       ├── feedback.py    # Feedback generation
│   │       └── health.py      # Health checks
│   ├── core/
│   │   ├── config.py          # App configuration
│   │   └── logging.py         # Structured logging
│   ├── ml/
│   │   ├── services/          # ML services
│   │   ├── models/            # ML model implementations
│   │   └── utils/             # ML utilities
│   └── main.py                # FastAPI app entry point
├── models/                    # Downloaded ML models
├── tests/                     # Test suite
├── requirements.txt           # Python dependencies
├── Dockerfile                # Docker configuration
└── .env.example              # Environment variables template
```

## Setup

### Prerequisites
- Python 3.9+
- Docker (optional)
- Google Cloud credentials (for STT/TTS)
- OpenAI API key (for feedback generation)

### Local Development

1. Clone and navigate to the ML service directory:
```bash
cd ai-ml-service
```

2. Create and activate virtual environment:
```bash
conda create -n katadia-ml python=3.9
conda activate katadia-ml
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Setup environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the service:
```bash
uvicorn app.main:app --reload
```

### Docker Setup

1. Build the image:
```bash
docker build -t katadia-ai-ml .
```

2. Run the container:
```bash
docker run -p 8000:8000 katadia-ai-ml
```

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

- `POST /api/v1/score` - Score pronunciation from audio
- `POST /api/v1/transcribe` - Transcribe audio to text  
- `POST /api/v1/feedback` - Generate personalized feedback
- `GET /api/v1/health` - Service health check

## Model Downloads

The service requires large model files to be downloaded:

### Montreal Forced Aligner (MFA)
```bash
# Install MFA (run once)
pip install montreal-forced-aligner

# Download Indonesian acoustic model
mfa model download acoustic indonesian_mfa

# Download Indonesian dictionary
mfa model download dictionary indonesian_mfa
```

### Whisper Model
Download automatically on first use or manually:
```bash
# Base model (faster, less accurate)
huggingface-cli download openai/whisper-base

# Large model (slower, more accurate)
huggingface-cli download openai/whisper-large-v2
```

### Wav2Vec 2.0
Download pretrained model:
```bash
huggingface-cli download facebook/wav2vec2-large-xlsr-53
```

## Development Workflow

### Phase 0: Setup & Planning (Week 1)
- [x] Environment configuration
- [x] FastAPI project structure
- [x] API endpoint scaffolds
- [ ] Model download and setup
- [ ] Basic testing framework

### Phase 1: Core Foundation (Weeks 2-3)
- [ ] Google Cloud STT integration
- [ ] Whisper fallback implementation
- [ ] Phoneme extraction (MFA)
- [ ] Basic GOP scoring
- [ ] API development

### Phase 2: Enhanced Scoring (Weeks 4-5)
- [ ] Wav2Vec 2.0 fine-tuning
- [ ] Multi-dimensional scoring
- [ ] Error detection enhancement
- [ ] Performance optimization
- [ ] Mobile integration support

### Phase 3: Advanced AI (Weeks 6-7)
- [ ] CEFR assessment system
- [ ] GPT-4 feedback integration
- [ ] Adaptive learning algorithms
- [ ] Advanced analytics
- [ ] Bilingual support

### Phase 4: Production Ready (Week 8)
- [ ] Model optimization
- [ ] A/B testing framework
- [ ] Monitoring and alerting
- [ ] Performance tuning
- [ ] Documentation completion

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=app --cov-report=html
```

## Performance Targets

- **Response Time**: <3 seconds for scoring
- **Accuracy**: >85% correlation with human raters
- **Concurrent Users**: Support for 100+ simultaneous users
- **Test Coverage**: >70% across all components

## Configuration

Key environment variables:

```bash
# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=./google-credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id

# Database
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_DATABASE=katadia_ml
REDIS_URL=redis://localhost:6379

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Models
WAV2VEC_MODEL_PATH=./models/wav2vec2-indonesian
WHISPER_MODEL_SIZE=base
```

## Logging

The service uses structured logging with JSON format by default. Logs are written to:
- Console (structured output)
- File (`./logs/app.log` with rotation)

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Monitoring

Health checks available at:
- `/api/v1/health` - Basic status
- `/api/v1/health/detailed` - Component status (Phase 1+)

## Troubleshooting

### Common Issues

1. **Google Cloud authentication failure**
   - Ensure `GOOGLE_APPLICATION_CREDENTIALS` is set correctly
   - Verify the service account has STT/TTS permissions

2. **Model download timeouts**
   - Check internet connection
   - Try manual download using `huggingface-cli`

3. **Memory errors with large models**
   - Increase system RAM or use smaller models
   - Enable model quantization (Phase 4)

4. **Docker build failures**
   - Ensure all system dependencies are installed
   - Check Dockerfile permissions

## License

[Add license information]

## Contributing

[Add contribution guidelines]
