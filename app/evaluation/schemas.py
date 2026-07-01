from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states for an asynchronous evaluation job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluateResponse(BaseModel):
    """
    Final evaluation response for a lecture video.

    Contains all four scores produced by the end-to-end evaluation pipeline:
    technical accuracy, grammar quality, and language distribution percentages.
    """

    technical_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Technical accuracy score (0–100) based on subject correctness.",
    )
    grammatical_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Grammar quality score (0–100) based on transcript analysis.",
    )
    english_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of the lecture delivered in English (0–100).",
    )
    tamil_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of the lecture delivered in Tamil (0–100).",
    )


class EvaluateJobAccepted(BaseModel):
    """Returned by POST /evaluate — the job was accepted and is running in the background."""

    job_id: str = Field(..., description="Identifier used to poll for this job's status/result.")
    status: JobStatus = Field(..., description="Current job status (initially 'queued').")


class EvaluateJobStatus(BaseModel):
    """Full status record for an asynchronous evaluation job."""

    job_id: str = Field(..., description="Job identifier.")
    status: JobStatus = Field(..., description="Current lifecycle state.")
    stage: Optional[str] = Field(
        default=None, description="Human-readable current pipeline stage (while running)."
    )
    person_name: Optional[str] = Field(default=None, description="Lecturer name supplied at submission.")
    subject: Optional[str] = Field(default=None, description="Lecture subject supplied at submission.")
    timing: Optional[str] = Field(default=None, description="Timing/duration context supplied at submission.")
    upload_id: Optional[int] = Field(
        default=None, description="Internal upload ID assigned once processing begins."
    )
    created_at: str = Field(..., description="ISO-8601 UTC timestamp when the job was created.")
    updated_at: str = Field(..., description="ISO-8601 UTC timestamp of the last status update.")
    result: Optional[EvaluateResponse] = Field(
        default=None, description="Final scores, present only when status is 'completed'."
    )
    error: Optional[str] = Field(
        default=None, description="Failure reason, present only when status is 'failed'."
    )
