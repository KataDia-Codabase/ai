# KataDia AI - ML Service

AI/ML microservice for pronunciation analysis and feedback in the KataDia AI language learning platform.

## Features

- **Speech-to-Text**: Google Cloud STT with Whisper fallback
- **Phoneme Analysis**: Montreal Forced Aligner integration
- **Pronunciation Scoring**: Multi-dimensional scoring (accuracy, fluency, prosody, stress) (currently tuned for English en-US)
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

### Full Stack (FastAPI + MySQL + Redis)

The production VM now hosts MySQL and Redis inside the same Compose stack to avoid external cloud fees. Use the provided `docker-compose.prod.yml` to run everything locally or on the VPS.

```bash
# Copy env template and set secrets (MySQL, Redis, API keys)
cp .env.example .env

# Start FastAPI, MySQL, Redis, schema migrator, and Nginx
docker compose -f docker-compose.prod.yml up -d

# Tail the one-shot schema migrator until it reports success
docker compose -f docker-compose.prod.yml logs -f schema-migrator

# Stream logs (optional)
docker compose -f docker-compose.prod.yml logs -f fastapi

# Stop the stack
docker compose -f docker-compose.prod.yml down
```

Data is persisted through named volumes:

| Service | Volume | Path |
|---------|--------|------|
| MySQL   | `mysql-data` | `/var/lib/mysql` |
| Redis   | `redis-data` | `/data` |

FastAPI reaches the internal DB/cache via hostnames `mysql` and `redis` on the shared `katadia-network` bridge.

The `schema-migrator` service waits for MySQL to accept connections and replays `database/schema.sql` on every deployment, ensuring migrations stay in sync without manual intervention.

### HTTPS Termination (Cloudflare)

Public traffic is served through Cloudflare, which terminates TLS at the edge. Keep the VM's Nginx listener on HTTP (port 80) only; Cloudflare forwards traffic over HTTP while presenting its managed certificate to end users. If you ever bypass Cloudflare (orange-cloud off), reintroduce a certificate on the VM before exposing port 443.

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

- `POST /api/v1/score` - Score pronunciation from audio and return DB-ready payload
- `POST /api/v1/transcribe` - Transcribe audio to text  
- `POST /api/v1/feedback` - Generate personalized feedback (same engine used by `/score`)
- `GET /api/v1/health` - Service health check

#### `/api/v1/score` Response Snapshot

The scoring endpoint now bundles all columns required by the `pronunciation_submissions` table:

| Field | Source |
|-------|--------|
| `language_code` | Request `language` (currently `en-US` only) |
| `overall_score`, `accuracy_score`, `fluency_score`, `prosody_score`, `stress_score` | Enhanced scoring engine |
| `phoneme_errors_json` | Serialized phoneme-level mistakes |
| `generated_transcript` | STT verification output |
| `audio_url` | Uploaded audio reference (local `file://` or mobile URL) |
| `cefr_level_assessment` | CEFR module result |
| `personalized_feedback` & `feedback_details` | Gemini-driven (or template) feedback |
| `lesson_vocab_id`, `user_id` | Request context |
| `created_at` | Server-side UTC timestamp |

Mobile/backend services can persist the response directly without calling a separate feedback endpoint, although `/api/v1/feedback` remains available for standalone use or historical comparisons.

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

# Fine-tuned English checkpoint (place files under ./models/wav2vec2-english)
huggingface-cli download <organization>/<english-model-id> -d models/wav2vec2-english
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

# Database (Docker Compose defaults)
MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_USER=katadia
MYSQL_PASSWORD=katadia_password
MYSQL_DATABASE=katadia_ml
MYSQL_ROOT_PASSWORD=rootpassword
MYSQL_URI=mysql+pymysql://katadia:katadia_password@mysql:3306/katadia_ml
REDIS_URL=redis://redis:6379/0

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Models
WAV2VEC_MODEL_PATH=./models/wav2vec2-indonesian
WAV2VEC_ENGLISH_MODEL_PATH=./models/wav2vec2-english
WHISPER_MODEL_SIZE=base
```

### Enhanced Scoring Notes

- The `/api/v1/score` endpoint now always uses the enhanced English pipeline (Wav2Vec2 + fluency/prosody/stress analyzers). Requests must set `language=en-US`.
- Uploaded scoring audio is stored under `temp_audio/scoring_uploads/` so the response can include a real local `audio_url` (e.g., `file:///.../temp_audio/scoring_uploads/xyz.wav`). Clean this folder periodically if disk usage grows.
- When testing manually, send the same local path in the `audio_url` field (see the provided curl/Postman examples) so downstream services receive the actual file location used for scoring.

## Logging

The service uses structured logging with JSON format by default. Logs are written to:
- Console (structured output)
- File (`./logs/app.log` with rotation)

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Monitoring

Health checks available at:
- `/api/v1/health` - Basic status
- `/api/v1/health/detailed` - Component status (Phase 1+)

## Data Backup & Restore

Both stateful services run inside Docker with named volumes. Create regular dumps so the VPS disk can be rebuilt safely.

### MySQL (`mysql-data`)

```bash
# Ad-hoc backup
docker exec katadia-mysql mysqldump -u root -p"$MYSQL_ROOT_PASSWORD" --databases katadia_ml \
   > backups/katadia_ml-$(date +%F).sql

# Or reuse the helper script (also scheduled nightly on the VPS)
./scripts/backup_data.sh

# Restore
docker exec -i katadia-mysql mysql -u root -p"$MYSQL_ROOT_PASSWORD" < backups/katadia_ml-YYYY-MM-DD.sql
```

Schedule this via cron on the VM and sync the `backups/` folder to blob/object storage if possible.

### Redis (`redis-data`)

Redis writes an append-only file inside `/data`. Take periodic snapshots before maintenance:

```bash
# Trigger snapshot
docker exec katadia-redis redis-cli save

# Copy rdb/aof off the server
docker cp katadia-redis:/data .
```

For point-in-time recovery, keep multiple copies of the `/data` directory or enable `redis-cli --rdb` streaming to external storage.

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
