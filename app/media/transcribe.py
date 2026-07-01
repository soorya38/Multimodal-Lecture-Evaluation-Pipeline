import os
import subprocess
import time

import structlog
from faster_whisper import WhisperModel

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Whisper configuration is sourced from app.core.config.Settings:
#   WHISPER_DEVICE        "cuda" (GPU) | "cpu" | "auto" (let CTranslate2 decide)
#   WHISPER_COMPUTE_TYPE  "float16" (GPU, fastest) | "int8" (CPU) | "default"
#   WHISPER_MODEL_SIZE    e.g. "large-v3" (best quality on GPU) | "base" | "small"
#   WHISPER_CPU_THREADS   0 = use all cores (CPU fallback / pre-post processing)
# ---------------------------------------------------------------------------

# Global cache for the WhisperModel instance to avoid reloading it for every request.
# In production, you might manage this differently (e.g. per-worker model pool or external service).
_model_instance: WhisperModel | None = None
_current_model_size: str | None = None


def get_whisper_model(
    model_size: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> WhisperModel:
    """
    Get or initialize the faster-whisper model.

    All parameters default to their corresponding environment variable values,
    enabling zero-config GPU acceleration when WHISPER_DEVICE=cuda is set.
    """
    global _model_instance, _current_model_size

    settings = get_settings()
    model_size = model_size or settings.whisper_model_size
    device = device or settings.whisper_device
    compute_type = compute_type or settings.whisper_compute_type
    cpu_threads = settings.resolved_cpu_threads()

    if _model_instance is None or _current_model_size != model_size:
        logger.info(
            "Loading faster-whisper model",
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        _model_instance = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        _current_model_size = model_size
        logger.info(
            "faster-whisper model loaded successfully",
            model_size=model_size,
            device=device,
            compute_type=compute_type,
        )

    return _model_instance


def _get_audio_duration_seconds(audio_path: str) -> float:
    """Get the duration of an audio file in seconds using ffprobe, or 0 on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def transcribe_audio(
    audio_path: str,
    model_size: str | None = None,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> dict:
    """
    Transcribe audio using faster-whisper.

    Args:
        audio_path: Absolute path to the input audio file.
        model_size: Size of the Whisper model to use. Defaults to WHISPER_MODEL_SIZE env var.
        language: ISO code of the language to force (e.g. "en"). If None, auto-detects.
        initial_prompt: Optional text prepended as context to bias decoding toward
            the lecture's domain vocabulary (e.g. the subject), improving accuracy
            on technical terms and proper nouns.

    Returns:
        A dictionary containing:
        - "language": detected language (or forced language)
        - "language_probability": probability of the detected language
        - "duration": total duration of the audio in seconds
        - "segments": list of dictionaries with "start", "end", and "text"

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        RuntimeError: If transcription fails.
    """
    settings = get_settings()
    model_size = model_size or settings.whisper_model_size

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Probe audio duration upfront to decide on beam_size and for progress logging
    audio_duration = _get_audio_duration_seconds(audio_path)
    is_long_audio = audio_duration > 3600  # > 1 hour

    # On GPU, beam_size=5 is fast enough even for long audio.
    # On CPU, use beam_size=1 (greedy) for long audio — it's 3-5x faster
    # with minimal quality loss for lecture content.
    is_gpu = settings.whisper_device.lower() in ("cuda", "auto")
    beam_size = 5 if is_gpu else (1 if is_long_audio else 5)

    logger.info(
        "Starting audio transcription",
        audio_path=audio_path,
        model_size=model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=language,
        audio_duration_seconds=round(audio_duration, 1),
        beam_size=beam_size,
        biased=bool(initial_prompt),
    )

    try:
        model = get_whisper_model(model_size=model_size)

        # Transcribe
        segments_generator, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,  # bias decoding toward domain vocabulary
            vad_filter=True,  # Filters out parts without speech using VAD
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments = []
        start_time = time.monotonic()
        last_log_time = start_time

        for segment in segments_generator:
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })

            # Log progress every 30 seconds of wall-clock time
            now = time.monotonic()
            if now - last_log_time >= 30:
                elapsed = now - start_time
                progress_pct = (segment.end / audio_duration * 100) if audio_duration > 0 else 0
                logger.info(
                    "Transcription in progress",
                    segments_so_far=len(segments),
                    audio_position=f"{segment.end:.1f}s / {audio_duration:.1f}s",
                    progress_percent=f"{progress_pct:.1f}%",
                    wall_time_elapsed=f"{elapsed:.0f}s",
                )
                last_log_time = now

        logger.info(
            "Transcription completed",
            audio_path=audio_path,
            detected_language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segment_count=len(segments),
            wall_time=f"{time.monotonic() - start_time:.1f}s",
        )

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segments,
        }

    except Exception as e:
        logger.error("Transcription failed", audio_path=audio_path, error=str(e), exc_info=True)
        raise RuntimeError(f"Transcription failed: {e}") from e
