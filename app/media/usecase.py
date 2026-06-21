import os
import shutil
import tempfile
import uuid

import structlog
from fastapi import UploadFile

from app.core.storage import upload_file
from app.media.ffmpeg import split_video_audio
from app.media.schemas import SplitMediaResponse

logger = structlog.get_logger(__name__)

# Default bucket — sourced from the same env var used by storage init
_DEFAULT_BUCKET = os.getenv("MINIO_DEFAULT_BUCKET", "lectures")


async def split_and_store(file: UploadFile) -> SplitMediaResponse:
    """
    Orchestrate the full split-and-store workflow:

    1. Persist the uploaded file to a temporary directory.
    2. Run FFmpeg to split it into video-only and audio-only streams.
    3. Upload both outputs to MinIO.
    4. Clean up all temporary files.

    Args:
        file: The uploaded video file from the request.

    Returns:
        SplitMediaResponse with the MinIO object keys for both outputs.
    """
    upload_id = uuid.uuid4().hex
    tmp_dir = tempfile.mkdtemp(prefix=f"media_split_{upload_id}_")

    logger.info(
        "Starting split_and_store",
        upload_id=upload_id,
        original_filename=file.filename,
        tmp_dir=tmp_dir,
    )

    try:
        # --- 1. Save uploaded file to disk ---
        input_path = os.path.join(tmp_dir, "input.mp4")
        with open(input_path, "wb") as f:
            # Stream the upload in 1 MiB chunks to avoid loading the entire file into memory
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        logger.info("Saved uploaded file to disk", path=input_path)

        # --- 2. Run FFmpeg ---
        video_out = os.path.join(tmp_dir, "video.mp4")
        audio_out = os.path.join(tmp_dir, "audio.mp3")

        await split_video_audio(input_path, video_out, audio_out)

        # --- 3. Upload results to MinIO ---
        video_key = f"{upload_id}/video.mp4"
        audio_key = f"{upload_id}/audio.mp3"

        upload_file(
            bucket=_DEFAULT_BUCKET,
            object_name=video_key,
            file_path=video_out,
            content_type="video/mp4",
        )
        upload_file(
            bucket=_DEFAULT_BUCKET,
            object_name=audio_key,
            file_path=audio_out,
            content_type="audio/mpeg",
        )

        logger.info(
            "Split and store completed",
            upload_id=upload_id,
            video_key=video_key,
            audio_key=audio_key,
        )

        return SplitMediaResponse(
            upload_id=upload_id,
            video_object_key=video_key,
            audio_object_key=audio_key,
            bucket=_DEFAULT_BUCKET,
        )

    finally:
        # --- 4. Clean up temp files regardless of success or failure ---
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary directory", tmp_dir=tmp_dir)
