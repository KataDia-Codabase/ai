"""
Azure Web App Entry Point
This module serves as the WSGI application entry point for Azure App Service.
"""

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
