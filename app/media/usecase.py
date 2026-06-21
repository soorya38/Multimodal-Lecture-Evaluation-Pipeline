import asyncio
import os
import re
import shutil
import tempfile
import uuid

import structlog
from fastapi import UploadFile

from app.core.storage import download_file, upload_file
from app.media.ffmpeg import split_video_audio
from app.media.scene_detect import detect_scenes_and_extract_frames
from app.media.schemas import (
    ExtractFramesResponse,
    FrameInfo,
    SplitMediaResponse,
)

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


async def extract_frames_and_store(
    upload_id: str,
    threshold: float = 27.0,
    num_images: int = 1,
) -> ExtractFramesResponse:
    """
    Orchestrate frame extraction from a previously split video:

    1. Download the video-only file from MinIO.
    2. Run PySceneDetect to detect scenes and extract representative frames.
    3. Upload each extracted frame to MinIO.
    4. Clean up all temporary files.

    Args:
        upload_id: The upload ID from a prior /split response.
        threshold: ContentDetector sensitivity (lower = more sensitive).
        num_images: Number of frames to extract per detected scene.

    Returns:
        ExtractFramesResponse with metadata for all extracted frames.

    Raises:
        FileNotFoundError: If the video object does not exist in MinIO.
        RuntimeError: If scene detection or frame extraction fails.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"frame_extract_{upload_id}_")
    video_object_key = f"{upload_id}/video.mp4"

    logger.info(
        "Starting extract_frames_and_store",
        upload_id=upload_id,
        threshold=threshold,
        num_images=num_images,
    )

    try:
        # --- 1. Download video from MinIO ---
        local_video_path = os.path.join(tmp_dir, "video.mp4")
        download_file(
            bucket=_DEFAULT_BUCKET,
            object_name=video_object_key,
            file_path=local_video_path,
        )

        # --- 2. Run scene detection + frame extraction ---
        # PySceneDetect is CPU-bound (decodes video frames with OpenCV),
        # so offload to a thread to avoid blocking the async event loop.
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        frame_paths = await asyncio.to_thread(
            detect_scenes_and_extract_frames,
            video_path=local_video_path,
            output_dir=frames_dir,
            threshold=threshold,
            num_images=num_images,
        )

        # --- 3. Upload each frame to MinIO ---
        frames: list[FrameInfo] = []

        for frame_path in frame_paths:
            filename = os.path.basename(frame_path)
            object_key = f"{upload_id}/frames/{filename}"

            upload_file(
                bucket=_DEFAULT_BUCKET,
                object_name=object_key,
                file_path=frame_path,
                content_type="image/jpeg",
            )

            # Parse scene number from the filename pattern:
            # e.g. "video-Scene-001-01.jpg" → scene_number = 1
            scene_number = _parse_scene_number(filename)

            frames.append(FrameInfo(
                scene_number=scene_number,
                object_key=object_key,
            ))

        logger.info(
            "Frame extraction and upload completed",
            upload_id=upload_id,
            frame_count=len(frames),
        )

        return ExtractFramesResponse(
            upload_id=upload_id,
            bucket=_DEFAULT_BUCKET,
            frame_count=len(frames),
            frames=frames,
        )

    finally:
        # --- 4. Clean up temp files regardless of success or failure ---
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary directory", tmp_dir=tmp_dir)


def _parse_scene_number(filename: str) -> int:
    """
    Extract the scene number from a PySceneDetect filename.

    PySceneDetect's save_images produces filenames like:
        video-Scene-001-01.jpg  (Scene 1, image 1)
        video-Scene-012-03.jpg  (Scene 12, image 3)

    Returns:
        1-based scene number, or 0 if the pattern doesn't match.
    """
    match = re.search(r"Scene-(\d+)", filename)
    return int(match.group(1)) if match else 0

