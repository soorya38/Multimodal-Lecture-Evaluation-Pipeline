import structlog
from faster_whisper import WhisperModel

logger = structlog.get_logger(__name__)

# Global cache for the WhisperModel instance to avoid reloading it for every request.
# In production, you might manage this differently (e.g. per-worker model pool or external service).
_model_instance: WhisperModel | None = None
_current_model_size: str | None = None


def get_whisper_model(model_size: str = "small", device: str = "auto", compute_type: str = "default") -> WhisperModel:
    """
    Get or initialize the faster-whisper model.
    """
    global _model_instance, _current_model_size

    if _model_instance is None or _current_model_size != model_size:
        logger.info(
            "Loading faster-whisper model",
            model_size=model_size,
            device=device,
            compute_type=compute_type,
        )
        _model_instance = WhisperModel(model_size, device=device, compute_type=compute_type)
        _current_model_size = model_size
        logger.info("faster-whisper model loaded successfully")

    return _model_instance


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    language: str | None = None,
) -> dict:
    """
    Transcribe audio using faster-whisper.

    Args:
        audio_path: Absolute path to the input audio file.
        model_size: Size of the Whisper model to use ("tiny", "base", "small", "medium", "large-v3").
        language: ISO code of the language to force (e.g. "en"). If None, auto-detects.

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
    import os
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(
        "Starting audio transcription",
        audio_path=audio_path,
        model_size=model_size,
        language=language,
    )

    try:
        model = get_whisper_model(model_size=model_size)

        # Transcribe
        segments_generator, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,  # Filters out parts without speech using VAD
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        segments = []
        for segment in segments_generator:
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })

        logger.info(
            "Transcription completed",
            audio_path=audio_path,
            detected_language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segment_count=len(segments),
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
