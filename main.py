import logging
import sys
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.storage import get_minio_client, init_minio
from app.evaluation.handler import router as evaluation_router
from app.media.handler import router as media_router

# Load configuration once, then configure logging from it.
settings = get_settings()
setup_logging(settings.log_level)

# Get a bound logger instance for this specific module
logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Bind a request ID to the structlog context for the lifetime of each request.

    Every log line emitted while handling a request then carries the same
    ``request_id`` (taken from an inbound ``X-Request-ID`` header if present, or
    generated otherwise), which makes it possible to trace a single upload across
    all pipeline stages in an aggregator. The ID is echoed back in the response.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("method", "path")
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events gracefully.
    """
    # Startup logic
    logger.info(
        "Starting application",
        llm_provider=settings.llm_provider,
        whisper_model=settings.whisper_model_size,
        whisper_device=settings.whisper_device,
        bucket=settings.minio_default_bucket,
    )

    # Initialize MinIO client
    try:
        init_minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            default_bucket=settings.minio_default_bucket,
        )
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

app.add_middleware(RequestIDMiddleware)

# Register feature routers
app.include_router(media_router)
app.include_router(evaluation_router)


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """
    Liveness probe — returns OK as long as the process is up and serving.
    Useful for Kubernetes livenessProbe or a load balancer.
    """
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
async def readiness_check() -> dict:
    """
    Readiness probe — verifies the service's critical dependency (MinIO) is
    reachable before declaring the instance ready to receive traffic.

    Returns 200 with per-dependency status. A failing dependency is reported as
    "down" with the reason, so an orchestrator can keep the pod out of rotation.
    """
    dependencies: dict[str, str] = {}
    all_ok = True

    try:
        client = get_minio_client()
        client.bucket_exists(settings.minio_default_bucket)
        dependencies["minio"] = "ok"
    except Exception as e:  # noqa: BLE001 — report any failure as not-ready
        dependencies["minio"] = f"down: {e}"
        all_ok = False

    return {"status": "ready" if all_ok else "not_ready", "dependencies": dependencies}


# Configuration for running the app directly for local development
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=settings.port)
