"""
Persistence for asynchronous evaluation jobs.

Job records are stored as small JSON objects in MinIO under ``jobs/{job_id}.json``.
Using the object store (rather than in-process memory) means a job submitted to
one worker can be polled from any worker, and status survives restarts — a
requirement once ``/evaluate`` runs in the background instead of blocking the
request.

This is deliberately a thin, dependency-light job store. For very high job
volumes a dedicated queue/DB (Redis + RQ/Arq, Postgres, …) would be the next
step; the interface here (create/update/get) is intentionally small so it can be
swapped without touching call sites.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from minio.error import S3Error

from app.core.config import get_settings
from app.core.storage import get_bytes, put_bytes
from app.evaluation.schemas import EvaluateJobStatus, EvaluateResponse, JobStatus

logger = structlog.get_logger(__name__)

_JOB_PREFIX = "jobs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_key(job_id: str) -> str:
    return f"{_JOB_PREFIX}/{job_id}.json"


def _save(record: EvaluateJobStatus) -> None:
    settings = get_settings()
    put_bytes(
        bucket=settings.minio_default_bucket,
        object_name=_job_key(record.job_id),
        data=record.model_dump_json().encode("utf-8"),
        content_type="application/json",
    )


def create_job(job_id: str, person_name: str, subject: str, timing: str) -> EvaluateJobStatus:
    """Persist a new job in the QUEUED state and return the record."""
    now = _now_iso()
    record = EvaluateJobStatus(
        job_id=job_id,
        status=JobStatus.QUEUED,
        person_name=person_name,
        subject=subject,
        timing=timing,
        created_at=now,
        updated_at=now,
    )
    _save(record)
    logger.info("Created evaluation job", job_id=job_id, subject=subject)
    return record


def get_job(job_id: str) -> EvaluateJobStatus | None:
    """Load a job record, or return None if no such job exists."""
    settings = get_settings()
    try:
        raw = get_bytes(bucket=settings.minio_default_bucket, object_name=_job_key(job_id))
    except S3Error as e:
        if e.code in ("NoSuchKey", "NoSuchObject"):
            return None
        raise
    return EvaluateJobStatus.model_validate_json(raw)


def _update(job_id: str, **changes) -> EvaluateJobStatus:
    record = get_job(job_id)
    if record is None:
        raise KeyError(f"Job '{job_id}' not found")
    updated = record.model_copy(update={**changes, "updated_at": _now_iso()})
    _save(updated)
    return updated


def mark_running(job_id: str, stage: str, upload_id: int | None = None) -> None:
    """Move a job into RUNNING and record the current stage (and upload_id once known)."""
    changes: dict = {"status": JobStatus.RUNNING, "stage": stage}
    if upload_id is not None:
        changes["upload_id"] = upload_id
    _update(job_id, **changes)
    logger.info("Job progress", job_id=job_id, stage=stage, upload_id=upload_id)


def mark_completed(job_id: str, result: EvaluateResponse) -> None:
    """Mark a job COMPLETED with its final result."""
    _update(job_id, status=JobStatus.COMPLETED, stage=None, result=result)
    logger.info("Job completed", job_id=job_id)


def mark_failed(job_id: str, error: str) -> None:
    """Mark a job FAILED with the failure reason."""
    _update(job_id, status=JobStatus.FAILED, error=error)
    logger.error("Job failed", job_id=job_id, error=error)
