import os
import tempfile
import uuid

import structlog
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile, status

from app.evaluation import jobs
from app.evaluation.schemas import EvaluateJobAccepted, EvaluateJobStatus, JobStatus
from app.evaluation.usecase import process_evaluation_job

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/evaluate",
    tags=["Evaluation"],
)

_UPLOAD_CHUNK = 1024 * 1024  # 1 MiB


@router.post(
    "",
    response_model=EvaluateJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a lecture video for asynchronous multimodal evaluation",
    description=(
        "Accepts a video upload along with metadata (person name, subject, timing) and "
        "runs the full extraction + evaluation pipeline (split, frames, transcribe, OCR, "
        "consolidate, score) in the background. Returns 202 with a job_id immediately — "
        "poll GET /api/v1/evaluate/{job_id} for progress and the final scores. "
        "Long lectures can take many minutes, so the work is not done inline."
    ),
)
async def submit_evaluation(
    background_tasks: BackgroundTasks,
    video: UploadFile,
    person_name: str = Form(..., description="Name of the person delivering the lecture."),
    subject: str = Form(..., description="The main subject or topic of the lecture."),
    timing: str = Form(..., description="Timing or duration context."),
    reference_material: str | None = Form(
        default=None,
        description=(
            "Optional authoritative source text (syllabus, textbook excerpt, notes). "
            "When provided, the technical score is grounded against the most relevant "
            "passages instead of the model's prior knowledge alone."
        ),
    ),
) -> EvaluateJobAccepted:
    """
    Accept an upload, persist it to disk, register a job, and schedule background
    processing. The heavy pipeline never runs inside the request itself.
    """
    if not video.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No video file provided.",
        )

    job_id = uuid.uuid4().hex

    # Persist the upload to a temp file that outlives the request. BackgroundTasks
    # run in this same worker process, so a local path is valid; the background
    # job deletes the file when it finishes.
    fd, input_path = tempfile.mkstemp(prefix=f"eval_job_{job_id}_", suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := await video.read(_UPLOAD_CHUNK):
                f.write(chunk)
    except Exception as e:
        os.unlink(input_path)
        logger.error("Failed to persist upload for evaluation job", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store the uploaded video.",
        )

    try:
        jobs.create_job(job_id=job_id, person_name=person_name, subject=subject, timing=timing)
    except Exception as e:
        os.unlink(input_path)
        logger.error("Failed to create evaluation job record", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not register the evaluation job (storage unavailable).",
        )

    background_tasks.add_task(
        process_evaluation_job,
        job_id=job_id,
        input_path=input_path,
        person_name=person_name,
        subject=subject,
        timing=timing,
        reference_material=reference_material,
    )

    logger.info("Accepted evaluation job", job_id=job_id, filename=video.filename, subject=subject)
    return EvaluateJobAccepted(job_id=job_id, status=JobStatus.QUEUED)


@router.get(
    "/{job_id}",
    response_model=EvaluateJobStatus,
    status_code=status.HTTP_200_OK,
    summary="Get the status and result of an evaluation job",
    description=(
        "Returns the current lifecycle state of a submitted evaluation job. While "
        "running, includes the current pipeline stage; on completion, includes the "
        "final scores; on failure, includes the error reason."
    ),
)
async def get_evaluation_status(job_id: str) -> EvaluateJobStatus:
    """Return the persisted status record for a job, or 404 if unknown."""
    try:
        record = jobs.get_job(job_id)
    except Exception as e:
        logger.error("Failed to load evaluation job", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not read job status (storage unavailable).",
        )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No evaluation job found with id '{job_id}'.",
        )
    return record
