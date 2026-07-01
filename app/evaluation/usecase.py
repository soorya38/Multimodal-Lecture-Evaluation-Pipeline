import asyncio
import json
import os
import tempfile
import shutil
import time
from typing import Awaitable, Callable, Optional

import structlog
from fastapi import UploadFile

from app.core.config import get_settings
from app.core.storage import download_file
from app.evaluation import jobs
from app.evaluation.evaluate import evaluate_grammar, evaluate_language_mix, evaluate_technical
from app.evaluation.schemas import EvaluateResponse
from app.media.usecase import (
    consolidate_and_store,
    extract_frames_and_store,
    extract_text_and_store,
    split_and_store,
    split_stored_file,
    transcribe_and_store,
)

logger = structlog.get_logger(__name__)

# An optional async hook invoked at the start of each stage with (stage_label,
# upload_id). Used by the async job runner to persist progress; None in the
# direct/synchronous path.
StageHook = Optional[Callable[[str, Optional[int]], Awaitable[None]]]


async def run_full_pipeline(
    file: UploadFile,
    person_name: str,
    subject: str,
    timing: str,
) -> EvaluateResponse:
    """
    Orchestrates the entire end-to-end evaluation pipeline for an uploaded file.

    1. Splits the uploaded video into video and audio streams.
    2. Extracts frames and transcribes audio concurrently.
    3. Runs OCR on the extracted frames.
    4. Consolidates transcript and OCR results.
    5. Evaluates the consolidated document (technical, grammar, language mix) concurrently.
    """
    logger.info("Starting full end-to-end pipeline", filename=file.filename, person=person_name, subject=subject)
    split_result = await split_and_store(file)
    return await _run_pipeline_after_split(split_result.upload_id, subject)


async def process_evaluation_job(
    job_id: str,
    input_path: str,
    person_name: str,
    subject: str,
    timing: str,
) -> None:
    """
    Run the full pipeline for a previously-persisted upload as a background job,
    recording progress and the final result/error in the job store.

    ``input_path`` is a local file the request handler already streamed to disk;
    it is removed once processing finishes (success or failure).
    """
    async def on_stage(stage: str, upload_id: Optional[int]) -> None:
        # Offload the (blocking, MinIO-backed) status write to a thread.
        await asyncio.to_thread(jobs.mark_running, job_id, stage, upload_id)

    logger.info("Processing evaluation job", job_id=job_id, subject=subject)
    try:
        await on_stage("split", None)
        split_result = await split_stored_file(input_path)
        upload_id = split_result.upload_id

        result = await _run_pipeline_after_split(upload_id, subject, on_stage=on_stage)

        await asyncio.to_thread(jobs.mark_completed, job_id, result)
    except Exception as e:  # noqa: BLE001 — any failure must be recorded on the job
        logger.error("Evaluation job failed", job_id=job_id, error=str(e), exc_info=True)
        await asyncio.to_thread(jobs.mark_failed, job_id, str(e))
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


async def _run_pipeline_after_split(
    upload_id: int,
    subject: str,
    on_stage: StageHook = None,
) -> EvaluateResponse:
    """
    Shared stages 2–6 of the pipeline, given an ``upload_id`` whose video/audio
    have already been split and stored. Invokes ``on_stage`` before each stage
    when provided (used to persist async-job progress).
    """
    pipeline_start = time.monotonic()

    async def stage(label: str) -> None:
        if on_stage is not None:
            await on_stage(label, upload_id)

    # Step 2: Extract frames + Transcribe audio (run concurrently)
    await stage("frames_and_transcription")
    step_start = time.monotonic()
    logger.info("Pipeline Step 2: Frame Extraction & Transcription (parallel)", upload_id=upload_id)
    await asyncio.gather(
        extract_frames_and_store(upload_id=upload_id),
        transcribe_and_store(upload_id=upload_id),
    )
    logger.info("Pipeline Step 2 completed", upload_id=upload_id, wall_time=f"{time.monotonic() - step_start:.1f}s")

    # Step 3: OCR on frames
    await stage("ocr")
    step_start = time.monotonic()
    logger.info("Pipeline Step 3: OCR on frames", upload_id=upload_id)
    await extract_text_and_store(upload_id=upload_id)
    logger.info("Pipeline Step 3 completed", upload_id=upload_id, wall_time=f"{time.monotonic() - step_start:.1f}s")

    # Step 4: Consolidate
    await stage("consolidation")
    step_start = time.monotonic()
    logger.info("Pipeline Step 4: Consolidation", upload_id=upload_id)
    consolidate_result = await consolidate_and_store(upload_id=upload_id)
    logger.info("Pipeline Step 4 completed", upload_id=upload_id, wall_time=f"{time.monotonic() - step_start:.1f}s")

    # Step 5: Download the consolidated JSON to feed into evaluations
    step_start = time.monotonic()
    logger.info("Pipeline Step 5: Preparing data for evaluation", upload_id=upload_id)
    consolidated_data = _download_consolidated(upload_id, consolidate_result.consolidated_object_key)
    logger.info("Pipeline Step 5 completed", upload_id=upload_id, wall_time=f"{time.monotonic() - step_start:.1f}s")

    # Step 6: Run evaluations concurrently via the configured LLM
    await stage("evaluation")
    step_start = time.monotonic()
    logger.info("Pipeline Step 6: Running LLM evaluations (parallel)", upload_id=upload_id)
    technical_score, grammatical_score, language_mix = await asyncio.gather(
        asyncio.to_thread(evaluate_technical, consolidated_data, subject),
        asyncio.to_thread(evaluate_grammar, consolidated_data),
        asyncio.to_thread(evaluate_language_mix, consolidated_data),
    )
    logger.info("Pipeline Step 6 completed", upload_id=upload_id, wall_time=f"{time.monotonic() - step_start:.1f}s")

    total_elapsed = time.monotonic() - pipeline_start
    logger.info("Full pipeline completed successfully", upload_id=upload_id, total_wall_time=f"{total_elapsed:.1f}s")

    return EvaluateResponse(
        technical_score=technical_score,
        grammatical_score=grammatical_score,
        english_percentage=language_mix["english_percentage"],
        tamil_percentage=language_mix["tamil_percentage"],
    )


def _download_consolidated(upload_id: int, object_key: str) -> dict:
    """Helper to download and parse the consolidated JSON from MinIO."""
    tmp_dir = tempfile.mkdtemp(prefix=f"eval_{upload_id}_")
    try:
        local_path = os.path.join(tmp_dir, "consolidated.json")
        download_file(
            bucket=get_settings().minio_default_bucket,
            object_name=object_key,
            file_path=local_path,
        )
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
