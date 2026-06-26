import asyncio
import json
import os
import tempfile
import shutil
import time

import structlog
from fastapi import UploadFile

from app.core.storage import download_file
from app.evaluation.evaluate import evaluate_grammar, evaluate_language_mix, evaluate_technical
from app.evaluation.schemas import EvaluateResponse
from app.media.usecase import (
    consolidate_and_store,
    extract_frames_and_store,
    extract_text_and_store,
    split_and_store,
    transcribe_and_store,
)

logger = structlog.get_logger(__name__)

# Default bucket — sourced from the same env var used by storage init
_DEFAULT_BUCKET = os.getenv("MINIO_DEFAULT_BUCKET", "lectures")


async def run_full_pipeline(
    file: UploadFile,
    person_name: str,
    subject: str,
    timing: str,
) -> EvaluateResponse:
    """
    Orchestrates the entire end-to-end evaluation pipeline.

    1. Splits the uploaded video into video and audio streams.
    2. Extracts frames and transcribes audio concurrently.
    3. Runs OCR on the extracted frames.
    4. Consolidates transcript and OCR results.
    5. Evaluates the consolidated document (technical, grammar, language mix) concurrently.
    """
    pipeline_start = time.monotonic()
    logger.info("Starting full end-to-end pipeline", filename=file.filename, person=person_name, subject=subject)

    # Step 1: Split video/audio
    step_start = time.monotonic()
    logger.info("Pipeline Step 1: Split Media")
    split_result = await split_and_store(file)
    upload_id = split_result.upload_id
    logger.info("Pipeline Step 1 completed", upload_id=upload_id, wall_time=f"{time.monotonic() - step_start:.1f}s")

    # Step 2: Extract frames + Transcribe audio (run concurrently)
    step_start = time.monotonic()
    logger.info("Pipeline Step 2: Frame Extraction & Transcription (parallel)", upload_id=upload_id)
    await asyncio.gather(
        extract_frames_and_store(upload_id=upload_id),
        transcribe_and_store(upload_id=upload_id),
    )
    logger.info("Pipeline Step 2 completed", upload_id=upload_id, wall_time=f"{time.monotonic() - step_start:.1f}s")

    # Step 3: OCR on frames
    step_start = time.monotonic()
    logger.info("Pipeline Step 3: OCR on frames", upload_id=upload_id)
    await extract_text_and_store(upload_id=upload_id)
    logger.info("Pipeline Step 3 completed", upload_id=upload_id, wall_time=f"{time.monotonic() - step_start:.1f}s")

    # Step 4: Consolidate
    step_start = time.monotonic()
    logger.info("Pipeline Step 4: Consolidation", upload_id=upload_id)
    consolidate_result = await consolidate_and_store(upload_id=upload_id)
    logger.info("Pipeline Step 4 completed", upload_id=upload_id, wall_time=f"{time.monotonic() - step_start:.1f}s")

    # Step 5: Download the consolidated JSON to feed into evaluations
    step_start = time.monotonic()
    logger.info("Pipeline Step 5: Preparing data for evaluation", upload_id=upload_id)
    consolidated_data = _download_consolidated(upload_id, consolidate_result.consolidated_object_key)
    logger.info("Pipeline Step 5 completed", upload_id=upload_id, wall_time=f"{time.monotonic() - step_start:.1f}s")

    # Step 6: Run evaluations concurrently via Ollama
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
            bucket=_DEFAULT_BUCKET,
            object_name=object_key,
            file_path=local_path,
        )
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
