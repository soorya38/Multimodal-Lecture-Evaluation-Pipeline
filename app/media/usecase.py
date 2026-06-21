import asyncio
import json
import os
import re
import shutil
import tempfile
import uuid

import structlog
from fastapi import UploadFile

from app.core.storage import download_file, list_objects, upload_file
from app.media.ffmpeg import split_video_audio
from app.media.scene_detect import detect_scenes_and_extract_frames
from app.media.schemas import (
    ExtractFramesResponse,
    FrameInfo,
    OcrResponse,
    SplitMediaResponse,
    TranscribeResponse,
)
from app.media.transcribe import transcribe_audio
from app.media.ocr import extract_text_from_frames

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


async def transcribe_and_store(
    upload_id: str,
    model_size: str = "small",
    language: str | None = None,
) -> TranscribeResponse:
    """
    Orchestrate audio transcription from a previously split audio file:

    1. Download the audio-only file from MinIO.
    2. Run faster-whisper to transcribe the audio.
    3. Upload the resulting transcript JSON to MinIO.
    4. Clean up all temporary files.

    Args:
        upload_id: The upload ID from a prior /split response.
        model_size: Size of the Whisper model to use.
        language: ISO language code (e.g. 'en') to force, or None for auto-detect.

    Returns:
        TranscribeResponse with metadata for the transcript.

    Raises:
        FileNotFoundError: If the audio object does not exist in MinIO.
        RuntimeError: If transcription fails.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"transcribe_{upload_id}_")
    audio_object_key = f"{upload_id}/audio.mp3"

    logger.info(
        "Starting transcribe_and_store",
        upload_id=upload_id,
        model_size=model_size,
        language=language,
    )

    try:
        # --- 1. Download audio from MinIO ---
        local_audio_path = os.path.join(tmp_dir, "audio.mp3")
        download_file(
            bucket=_DEFAULT_BUCKET,
            object_name=audio_object_key,
            file_path=local_audio_path,
        )

        # --- 2. Run faster-whisper transcription ---
        # Inference is CPU-bound (unless using GPU), so we offload to a thread
        transcript_result = await asyncio.to_thread(
            transcribe_audio,
            audio_path=local_audio_path,
            model_size=model_size,
            language=language,
        )

        # --- 3. Upload transcript JSON to MinIO ---
        transcript_path = os.path.join(tmp_dir, "transcript.json")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_result, f, ensure_ascii=False, indent=2)

        transcript_object_key = f"{upload_id}/transcript.json"

        upload_file(
            bucket=_DEFAULT_BUCKET,
            object_name=transcript_object_key,
            file_path=transcript_path,
            content_type="application/json",
        )

        logger.info(
            "Transcription and upload completed",
            upload_id=upload_id,
            duration=transcript_result["duration"],
        )

        return TranscribeResponse(
            upload_id=upload_id,
            bucket=_DEFAULT_BUCKET,
            transcript_object_key=transcript_object_key,
            language_detected=transcript_result["language"],
            duration=transcript_result["duration"],
        )

    finally:
        # --- 4. Clean up temp files regardless of success or failure ---
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary directory", tmp_dir=tmp_dir)


async def extract_text_and_store(
    upload_id: str,
) -> OcrResponse:
    """
    Orchestrate multimodal text extraction from previously extracted video frames:

    1. Discover all frame keys for this upload_id in MinIO.
    2. Download them to a temporary directory.
    3. Run Gemini OCR (in a separate thread) across all frames.
    4. Save the combined OCR results to a JSON file.
    5. Upload the JSON to MinIO.
    6. Clean up temporary files.

    Args:
        upload_id: The upload ID from a prior /split response.

    Returns:
        OcrResponse with metadata about the extraction.

    Raises:
        FileNotFoundError: If no frames are found for the upload_id.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"ocr_{upload_id}_")

    logger.info("Starting extract_text_and_store via Gemini", upload_id=upload_id)

    try:
        # --- 1. Find all frames in MinIO ---
        prefix = f"{upload_id}/frames/"
        frame_keys = list_objects(bucket=_DEFAULT_BUCKET, prefix=prefix)
        
        if not frame_keys:
            raise FileNotFoundError(f"No frames found in MinIO for upload_id '{upload_id}'. Ensure /extract-frames was called first.")

        # --- 2. Download frames ---
        local_frame_paths = []
        for key in frame_keys:
            filename = os.path.basename(key)
            local_path = os.path.join(tmp_dir, filename)
            download_file(
                bucket=_DEFAULT_BUCKET,
                object_name=key,
                file_path=local_path,
            )
            local_frame_paths.append(local_path)

        # Sort the paths so OCR happens sequentially by scene/frame number
        local_frame_paths.sort()

        # --- 3. Run Gemini OCR ---
        # Network IO bound, offload to thread
        ocr_results = await asyncio.to_thread(
            extract_text_from_frames,
            frame_paths=local_frame_paths,
        )

        # --- 4. Save results to JSON ---
        ocr_path = os.path.join(tmp_dir, "ocr.json")
        with open(ocr_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)

        # --- 5. Upload JSON to MinIO ---
        ocr_object_key = f"{upload_id}/ocr.json"
        upload_file(
            bucket=_DEFAULT_BUCKET,
            object_name=ocr_object_key,
            file_path=ocr_path,
            content_type="application/json",
        )

        logger.info(
            "Gemini OCR and upload completed",
            upload_id=upload_id,
            frames_processed=len(ocr_results),
        )

        return OcrResponse(
            upload_id=upload_id,
            bucket=_DEFAULT_BUCKET,
            ocr_object_key=ocr_object_key,
            frames_processed=len(ocr_results),
        )

    finally:
        # --- 6. Clean up temp files ---
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary directory", tmp_dir=tmp_dir)

