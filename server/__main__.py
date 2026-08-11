"""Entry point: ``python -m server`` runs the FastAPI backend on port 8000."""
from . import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
