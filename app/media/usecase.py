import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import structlog
from fastapi import UploadFile

from app.core.config import get_settings
from app.core.storage import download_file, list_objects, upload_file
from app.media.ffmpeg import split_video_audio
from app.media.scene_detect import detect_scenes_and_extract_frames
from app.media.schemas import (
    ConsolidateResponse,
    ExtractFramesResponse,
    FrameInfo,
    OcrResponse,
    SplitMediaResponse,
    TranscribeResponse,
)
from app.media.transcribe import transcribe_audio
from app.media.ocr import extract_text_from_frames

logger = structlog.get_logger(__name__)

MB_TO_BYTES = 1024 * 1024
INPUT_FILE_NAME = "input.mp4"
EXTRACTED_VIDEO_FILE_NAME = "video.mp4"
EXTRACTED_AUDIO_FILE_NAME = "audio.mp3"
CONTENT_TYPE_VIDEO = "video/mp4"
CONTENT_TYPE_AUDIO = "audio/mpeg"
CONTENT_TYPE_IMAGE = "image/jpeg"
EXTRACTED_FRAMES_DIR = "frames"

# Dynamic I/O thread pool size for parallel MinIO uploads/downloads.
# Scales with available CPUs, capped at 16 to avoid overwhelming the network.
_IO_MAX_WORKERS = min(os.cpu_count() or 4, 16)


def _default_bucket() -> str:
    """Resolve the configured MinIO bucket at call time (not import time)."""
    return get_settings().minio_default_bucket


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
    upload_id = uuid.uuid4().int
    tmp_dir = tempfile.mkdtemp(prefix=f"media_split_{upload_id}_")

    logger.info(
        "Starting split_and_store",
        upload_id=upload_id,
        original_filename=file.filename,
        tmp_dir=tmp_dir,
    )

    try:
        # --- 1. Save uploaded file to disk ---
        input_path = os.path.join(tmp_dir, INPUT_FILE_NAME)
        with open(input_path, "wb") as f:
            # Stream the upload in 1 MiB chunks to avoid loading the entire file into memory
            while chunk := await file.read(MB_TO_BYTES):
                f.write(chunk)

        logger.info("Saved uploaded file to disk", path=input_path)

        # --- 2. Run FFmpeg ---
        video_out = os.path.join(tmp_dir, EXTRACTED_VIDEO_FILE_NAME)
        audio_out = os.path.join(tmp_dir, EXTRACTED_AUDIO_FILE_NAME)

        await split_video_audio(input_path, video_out, audio_out)

        # --- 3. Upload results to MinIO ---
        video_key = f"{upload_id}/{EXTRACTED_VIDEO_FILE_NAME}"
        audio_key = f"{upload_id}/{EXTRACTED_AUDIO_FILE_NAME}"

        upload_file(
            bucket=_default_bucket(),
            object_name=video_key,
            file_path=video_out,
            content_type=CONTENT_TYPE_VIDEO,
        )
        upload_file(
            bucket=_default_bucket(),
            object_name=audio_key,
            file_path=audio_out,
            content_type=CONTENT_TYPE_AUDIO,
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
            bucket=_default_bucket(),
        )

    finally:
        # --- 4. Clean up temp files regardless of success or failure ---
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary directory", tmp_dir=tmp_dir)


async def extract_frames_and_store(
    upload_id: int,
    threshold: float = 27.0,
    num_images: int = 1,
    frame_skip: int = 0,
) -> ExtractFramesResponse:
    """
    Orchestrate the extraction of frames from a previously split video:

    1. Download the video-only file from MinIO.
    2. Run PySceneDetect to find scenes and extract frames.
    3. Upload the resulting frames to MinIO under the same upload_id.
    4. Clean up all temporary files.

    Args:
        upload_id: The upload ID from a prior /split response.
        threshold: The threshold for PySceneDetect.
        num_images: The number of images to extract per scene.
        frame_skip: Number of frames to skip during detection.

    Returns:
        ExtractFramesResponse with metadata about the uploaded frames.

    Raises:
        FileNotFoundError: If the video object does not exist in MinIO.
        RuntimeError: If scene detection fails.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"frame_extract_{upload_id}_")
    video_object_key = f"{upload_id}/{EXTRACTED_VIDEO_FILE_NAME}"

    logger.info(
        "Starting extract_frames_and_store",
        upload_id=upload_id,
        threshold=threshold,
        num_images=num_images,
        frame_skip=frame_skip,
    )

    try:
        # --- 1. Download video from MinIO ---
        local_video_path = os.path.join(tmp_dir, EXTRACTED_VIDEO_FILE_NAME)
        download_file(
            bucket=_default_bucket(),
            object_name=video_object_key,
            file_path=local_video_path,
        )

        # --- 2. Run scene detection + frame extraction ---
        # PySceneDetect is CPU-bound (decodes video frames with OpenCV),
        # so offload to a thread to avoid blocking the async event loop.
        frames_dir = os.path.join(tmp_dir, EXTRACTED_FRAMES_DIR)
        os.makedirs(frames_dir, exist_ok=True)

        frame_paths = await asyncio.to_thread(
            detect_scenes_and_extract_frames,
            video_path=local_video_path,
            output_dir=frames_dir,
            threshold=threshold,
            num_images=num_images,
            frame_skip=frame_skip,
        )

        # --- 3. Upload each frame to MinIO (parallel) ---
        total_frames = len(frame_paths)
        logger.info(
            "Starting parallel frame upload to MinIO",
            upload_id=upload_id,
            total_frames=total_frames,
        )

        upload_start = time.monotonic()
        frames: list[FrameInfo] = [None] * total_frames  # type: ignore[list-item]

        def _upload_single_frame(index: int, frame_path: str) -> FrameInfo:
            filename = os.path.basename(frame_path)
            object_key = f"{upload_id}/{EXTRACTED_FRAMES_DIR}/{filename}"

            upload_file(
                bucket=_default_bucket(),
                object_name=object_key,
                file_path=frame_path,
                content_type=CONTENT_TYPE_IMAGE,
            )

            logger.info(
                "Uploaded frame to MinIO",
                frame_index=f"{index + 1}/{total_frames}",
                object_key=object_key,
            )

            scene_number = _parse_scene_number(filename)
            return FrameInfo(scene_number=scene_number, object_key=object_key)

        with ThreadPoolExecutor(max_workers=_IO_MAX_WORKERS) as executor:
            futures = {
                executor.submit(_upload_single_frame, i, fp): i
                for i, fp in enumerate(frame_paths)
            }
            for future in futures:
                idx = futures[future]
                frames[idx] = future.result()

        upload_elapsed = time.monotonic() - upload_start
        logger.info(
            "Frame extraction and upload completed",
            upload_id=upload_id,
            frame_count=total_frames,
            upload_wall_time=f"{upload_elapsed:.1f}s",
        )

        return ExtractFramesResponse(
            upload_id=upload_id,
            bucket=_default_bucket(),
            frame_count=total_frames,
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
    upload_id: int,
    model_size: str = "base",
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
    audio_object_key = f"{upload_id}/{EXTRACTED_AUDIO_FILE_NAME}"

    logger.info(
        "Starting transcribe_and_store",
        upload_id=upload_id,
        model_size=model_size,
        language=language,
    )

    try:
        # --- 1. Download audio from MinIO ---
        local_audio_path = os.path.join(tmp_dir, EXTRACTED_AUDIO_FILE_NAME)
        download_file(
            bucket=_default_bucket(),
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
            bucket=_default_bucket(),
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
            bucket=_default_bucket(),
            transcript_object_key=transcript_object_key,
            language_detected=transcript_result["language"],
            duration=transcript_result["duration"],
        )

    finally:
        # --- 4. Clean up temp files regardless of success or failure ---
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary directory", tmp_dir=tmp_dir)


async def extract_text_and_store(
    upload_id: int,
) -> OcrResponse:
    """
    Orchestrate multimodal text extraction from previously extracted video frames:

    1. Discover all frame keys for this upload_id in MinIO.
    2. Download them to a temporary directory.
    3. Run vision-LLM OCR (in a separate thread) across all frames.
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

    logger.info("Starting extract_text_and_store via vision LLM", upload_id=upload_id)

    try:
        # --- 1. Find all frames in MinIO ---
        prefix = f"{upload_id}/frames/"
        frame_keys = list_objects(bucket=_default_bucket(), prefix=prefix)

        if not frame_keys:
            raise FileNotFoundError(f"No frames found in MinIO for upload_id '{upload_id}'. Ensure /extract-frames was called first.")

        total_frames = len(frame_keys)
        logger.info(
            "Found frames in MinIO, starting parallel download",
            upload_id=upload_id,
            total_frames=total_frames,
        )

        # --- 2. Download frames (parallel) ---
        download_start = time.monotonic()
        local_frame_paths: list[str | None] = [None] * total_frames  # type: ignore[list-item]

        def _download_single_frame(index: int, key: str) -> str:
            filename = os.path.basename(key)
            local_path = os.path.join(tmp_dir, filename)
            download_file(
                bucket=_default_bucket(),
                object_name=key,
                file_path=local_path,
            )
            logger.info(
                "Downloaded frame from MinIO",
                frame_index=f"{index + 1}/{total_frames}",
                object_key=key,
            )
            return local_path

        with ThreadPoolExecutor(max_workers=_IO_MAX_WORKERS) as executor:
            futures = {
                executor.submit(_download_single_frame, i, key): i
                for i, key in enumerate(frame_keys)
            }
            for future in futures:
                idx = futures[future]
                local_frame_paths[idx] = future.result()

        download_elapsed = time.monotonic() - download_start
        logger.info(
            "Frame download completed",
            upload_id=upload_id,
            total_frames=total_frames,
            download_wall_time=f"{download_elapsed:.1f}s",
        )

        # Sort the paths so OCR happens sequentially by scene/frame number
        sorted_paths = sorted(p for p in local_frame_paths if p is not None)

        # --- 3. Run OCR (already parallelised internally) ---
        # extract_text_from_frames uses its own ThreadPoolExecutor for Ollama calls,
        # so we offload the orchestrating call to a thread to avoid blocking the event loop.
        ocr_results = await asyncio.to_thread(
            extract_text_from_frames,
            frame_paths=sorted_paths,
        )

        # --- 4. Save results to JSON ---
        ocr_path = os.path.join(tmp_dir, "ocr.json")
        with open(ocr_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)

        # --- 5. Upload JSON to MinIO ---
        ocr_object_key = f"{upload_id}/ocr.json"
        upload_file(
            bucket=_default_bucket(),
            object_name=ocr_object_key,
            file_path=ocr_path,
            content_type="application/json",
        )

        logger.info(
            "OCR and upload completed",
            upload_id=upload_id,
            frames_processed=len(ocr_results),
        )

        return OcrResponse(
            upload_id=upload_id,
            bucket=_default_bucket(),
            ocr_object_key=ocr_object_key,
            frames_processed=len(ocr_results),
        )

    finally:
        # --- 6. Clean up temp files ---
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary directory", tmp_dir=tmp_dir)


async def consolidate_and_store(
    upload_id: int,
) -> ConsolidateResponse:
    """
    Orchestrate the consolidation of transcript and OCR data into a single
    unified knowledge representation:

    1. Download transcript.json and ocr.json from MinIO.
    2. Parse and merge them into a structured consolidated document.
    3. Upload the consolidated JSON to MinIO.
    4. Clean up temporary files.

    The consolidated output has three top-level sections:
    - `transcript`: Full speech data with a concatenated full_text field.
    - `visual_content`: Normalized OCR extraction results per frame.
    - `summary`: Computed statistics and boolean flags for downstream evaluators.

    Args:
        upload_id: The upload ID from a prior /split response.

    Returns:
        ConsolidateResponse with metadata about the consolidated output.

    Raises:
        FileNotFoundError: If transcript.json or ocr.json is missing in MinIO.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"consolidate_{upload_id}_")

    logger.info("Starting consolidate_and_store", upload_id=upload_id)

    try:
        # --- 1. Download transcript.json ---
        transcript_object_key = f"{upload_id}/transcript.json"
        local_transcript_path = os.path.join(tmp_dir, "transcript.json")

        try:
            download_file(
                bucket=_default_bucket(),
                object_name=transcript_object_key,
                file_path=local_transcript_path,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Transcript not found for upload_id '{upload_id}'. "
                f"Ensure /transcribe was called first."
            ) from e

        # --- 2. Download ocr.json ---
        ocr_object_key = f"{upload_id}/ocr.json"
        local_ocr_path = os.path.join(tmp_dir, "ocr.json")

        try:
            download_file(
                bucket=_default_bucket(),
                object_name=ocr_object_key,
                file_path=local_ocr_path,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"OCR results not found for upload_id '{upload_id}'. "
                f"Ensure /ocr was called first."
            ) from e

        # --- 3. Parse both JSON files ---
        with open(local_transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        with open(local_ocr_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)

        # --- 4. Build the consolidated document ---
        # Concatenate all transcript segments into a single string for easy downstream use
        segments = transcript_data.get("segments", [])
        full_text = " ".join(seg.get("text", "") for seg in segments).strip()

        # Normalize visual content from OCR results
        visual_content = []
        has_handwriting = False
        has_diagrams = False
        ocr_error_count = 0

        for frame_entry in ocr_data:
            content = frame_entry.get("content", {})

            # Track OCR failures — a failed frame yields empty text, which would
            # otherwise silently degrade the technical score with no visible cause.
            if content.get("error"):
                ocr_error_count += 1

            # Check for presence of handwriting and diagrams across all frames
            handwritten = content.get("handwritten_text", "").strip()
            diagrams = content.get("diagram_descriptions", "").strip()

            if handwritten:
                has_handwriting = True
            if diagrams:
                has_diagrams = True

            visual_content.append({
                "frame": frame_entry.get("frame_filename", "unknown"),
                "typed_text": content.get("typed_text", "").strip(),
                "handwritten_text": handwritten,
                "diagram_descriptions": diagrams,
            })

        consolidated = {
            "upload_id": upload_id,
            "transcript": {
                "language": transcript_data.get("language", "unknown"),
                "language_probability": transcript_data.get("language_probability", 0.0),
                "duration": transcript_data.get("duration", 0.0),
                "full_text": full_text,
                "segments": segments,
            },
            "visual_content": visual_content,
            "summary": {
                "total_segments": len(segments),
                "total_frames": len(visual_content),
                "total_duration_seconds": transcript_data.get("duration", 0.0),
                "detected_language": transcript_data.get("language", "unknown"),
                "has_handwriting": has_handwriting,
                "has_diagrams": has_diagrams,
            },
        }

        if ocr_error_count:
            logger.warning(
                "Some frames failed OCR — visual evidence will be incomplete, "
                "which lowers technical-score reliability",
                upload_id=upload_id,
                failed_frames=ocr_error_count,
                total_frames=len(visual_content),
            )

        # --- 5. Write and upload consolidated.json ---
        consolidated_path = os.path.join(tmp_dir, "consolidated.json")
        with open(consolidated_path, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, ensure_ascii=False, indent=2)

        consolidated_object_key = f"{upload_id}/consolidated.json"
        upload_file(
            bucket=_default_bucket(),
            object_name=consolidated_object_key,
            file_path=consolidated_path,
            content_type="application/json",
        )

        logger.info(
            "Consolidation and upload completed",
            upload_id=upload_id,
            transcript_segments=len(segments),
            frames_processed=len(visual_content),
            detected_language=transcript_data.get("language", "unknown"),
        )

        return ConsolidateResponse(
            upload_id=upload_id,
            bucket=_default_bucket(),
            consolidated_object_key=consolidated_object_key,
            transcript_segments=len(segments),
            frames_processed=len(visual_content),
            detected_language=transcript_data.get("language", "unknown"),
            duration=transcript_data.get("duration", 0.0),
        )

    finally:
        # --- 6. Clean up temp files ---
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary directory", tmp_dir=tmp_dir)
