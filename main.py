import logging
import os
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.core.logging import setup_logging
from app.core.storage import init_minio

# Configure logging once at application startup
setup_logging(os.getenv("LOG_LEVEL", "INFO"))

# Get a bound logger instance for this specific module
logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events gracefully.
    """
    # Startup logic
    logger.info("Starting application...")
    
    # Initialize MinIO client
    try:
        init_minio()
    except Exception as e:
        logger.error("Application failed to start due to MinIO initialization error", error=str(e))
        raise

    # Yield control back to FastAPI so it can start accepting requests
    yield

    # Shutdown logic
    logger.info("Shutting down application...")
    
    # Explicitly flush logs before exit to prevent missing log lines in containerized environments.
    # While Python's logging module registers an atexit handler to do this, fast async shutdowns 
    # (like receiving a SIGTERM in Kubernetes) can sometimes bypass it.
    for handler in logging.root.handlers:
        handler.flush()
    sys.stdout.flush()

app = FastAPI(
    title="Multimodal Lecture Evaluation Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """
    Health check endpoint to verify the service is running.
    Useful for Kubernetes probes or load balancers.
    """
    return {"status": "ok"}

# Configuration for running the app directly for local development
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)