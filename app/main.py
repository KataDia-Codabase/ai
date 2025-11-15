from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import score, transcribe, feedback, health
from app.core.logging import setup_logging
import structlog
import warnings

# Suppress deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="praatio")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")

# Setup logging
setup_logging()
logger = structlog.get_logger()

# Create FastAPI app
app = FastAPI(
    title="KataDia AI - ML Service",
    description="AI/ML microservice for pronunciation analysis and feedback",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(score.router, prefix=settings.API_V1_STR, tags=["scoring"])
app.include_router(transcribe.router, prefix=settings.API_V1_STR, tags=["transcription"])
app.include_router(feedback.router, prefix=settings.API_V1_STR, tags=["feedback"])
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["health"])

@app.on_event("startup")
async def startup_event():
    logger.info("KataDia AI ML Service starting up...")
    # Initialize ML models, database connections, etc.
    # This will be implemented in subsequent sprints
    pass

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("KataDia AI ML Service shutting down...")
    # Cleanup resources
    pass

@app.get("/")
async def root():
    return {
        "service": "KataDia AI - ML Service",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
