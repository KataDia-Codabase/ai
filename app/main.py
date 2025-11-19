from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse
from app.core.config import settings
from app.api.endpoints import score, transcribe, feedback, health, analytics, ab_testing
from app.core.logging import setup_logging
import structlog
import warnings
import json
from datetime import datetime

# Suppress deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="praatio")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")

# Setup logging
setup_logging()
logger = structlog.get_logger()

# Custom JSON encoder for pretty printing
class PrettyJSONResponse(JSONResponse):
    def render(self, content):
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=False,
        ).encode("utf-8")

# Create FastAPI app with custom JSON encoder
app = FastAPI(
    title="KataDia AI - ML Service",
    description="AI/ML microservice for pronunciation analysis and feedback",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None  # Disable FastAPI default ReDoc, gunakan custom endpoint
)

# Override the default JSONResponse
app.default_response_class = PrettyJSONResponse

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
app.include_router(analytics.router, prefix=settings.API_V1_STR, tags=["analytics"])
app.include_router(ab_testing.router, prefix=settings.API_V1_STR, tags=["experiments"])

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

# Custom ReDoc endpoint dengan fallback untuk localStorage issue
@app.get("/redoc", response_class=HTMLResponse)
async def redoc_html():
    """
    Custom ReDoc dengan sessionStorage fallback untuk browser tracking prevention.
    Mengatasi issue: "Tracking Prevention blocked access to storage"
    """
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <title>KataDia AI - ML Service API Documentation</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
        <style>
          body {
            margin: 0;
            padding: 0;
          }
        </style>
      </head>
      <body>
        <redoc spec-url='/openapi.json'></redoc>
        <script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"></script>
        <script>
          // Workaround untuk browser tracking prevention
          // Gunakan sessionStorage jika localStorage tidak tersedia
          (function() {
            const originalLocalStorage = window.localStorage;
            
            // Override localStorage dengan sessionStorage fallback
            window.localStorage = {
              getItem: function(key) {
                try {
                  return originalLocalStorage.getItem(key);
                } catch (e) {
                  return sessionStorage.getItem(key);
                }
              },
              setItem: function(key, value) {
                try {
                  originalLocalStorage.setItem(key, value);
                } catch (e) {
                  sessionStorage.setItem(key, value);
                }
              },
              removeItem: function(key) {
                try {
                  originalLocalStorage.removeItem(key);
                } catch (e) {
                  sessionStorage.removeItem(key);
                }
              },
              clear: function() {
                try {
                  originalLocalStorage.clear();
                } catch (e) {
                  sessionStorage.clear();
                }
              }
            };
          })();
        </script>
      </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
