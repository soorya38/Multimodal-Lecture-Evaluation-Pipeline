import os
import time
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.core.config import get_settings
from app.core.llm import get_llm_client

logger = structlog.get_logger(__name__)

# System prompt tuned to extract ALL text (typed, handwritten, and diagram descriptions)
OCR_SYSTEM_PROMPT = """
You are an expert OCR and image analysis system. Your job is to extract all text from the provided lecture slide frame.
Pay close attention to:
1. Typed text (bullet points, titles, body text).
2. Handwritten text (annotations, equations, notes).
3. Diagrams, charts, and figures (describe what they show and extract any text within them).

You must return a raw JSON object (without markdown wrappers like ```json) with the following structure:
{
  "typed_text": "All typed text extracted",
  "handwritten_text": "All handwritten text extracted",
  "diagram_descriptions": "Detailed descriptions of diagrams and their labels"
}
If a specific type of content is not present, set its value to an empty string.
"""


def _ocr_single_frame(
    path: str,
    index: int,
    total: int,
    model_name: str,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    """
    Run OCR on a single frame image. Designed to be called from a thread pool.

    Each invocation builds its own LLM client to avoid sharing state across threads.
    """
    frame_start = time.monotonic()

    if not os.path.isfile(path):
        logger.warning("Frame file not found during OCR, skipping", path=path, frame_index=f"{index}/{total}")
        return {
            "frame_filename": os.path.basename(path),
            "content": {
                "typed_text": "",
                "handwritten_text": "",
                "diagram_descriptions": "",
                "error": "File not found",
            },
        }

    try:
        logger.info(
            "OCR processing frame",
            frame_index=f"{index}/{total}",
            frame_path=os.path.basename(path),
            model=model_name,
        )

        # Each thread gets its own client to avoid shared-state issues
        client = get_llm_client()

        parsed_json = client.chat_json(
            OCR_SYSTEM_PROMPT,
            model=model_name,
            images=[path],
            temperature=temperature,
            seed=seed,  # fixed seed → reproducible OCR output
        )

        elapsed = time.monotonic() - frame_start
        logger.info(
            "OCR completed for frame",
            frame_index=f"{index}/{total}",
            frame=os.path.basename(path),
            wall_time=f"{elapsed:.1f}s",
        )

        return {
            "frame_filename": os.path.basename(path),
            "content": parsed_json,
        }

    except Exception as e:
        elapsed = time.monotonic() - frame_start
        logger.error(
            "Failed to run local OCR on frame",
            frame_index=f"{index}/{total}",
            path=path,
            error=str(e),
            wall_time=f"{elapsed:.1f}s",
            exc_info=True,
        )
        return {
            "frame_filename": os.path.basename(path),
            "content": {
                "typed_text": "",
                "handwritten_text": "",
                "diagram_descriptions": "",
                "error": str(e),
            },
        }


def extract_text_from_frames(frame_paths: list[str]) -> list[dict[str, Any]]:
    """
    Extract text and diagram descriptions from a list of frame images using the
    configured vision LLM (Ollama or any OpenAI-compatible endpoint).

    Uses a ThreadPoolExecutor to process multiple frames concurrently, since each
    call is network I/O-bound. The concurrency level is controlled by the
    OCR_MAX_WORKERS setting (0 = scale with CPUs, capped at 8).

    Args:
        frame_paths: List of absolute paths to the frame JPEG images.

    Returns:
        A list of dictionaries containing the extracted rich text for each frame,
        ordered to match the input frame_paths order.
    """
    settings = get_settings()
    total = len(frame_paths)
    workers = min(settings.resolved_ocr_workers(), total) if total > 0 else 1

    # Vision model for OCR. Configurable via LLM_OCR_MODEL (or legacy OLLAMA_OCR_MODEL).
    # The default "llava-phi3" (3.8B) must be pulled first (`ollama pull llava-phi3`)
    # or every frame will fail with "model not found" and visual_content ends up empty.
    model_name = settings.ocr_model

    logger.info(
        "Starting parallel multimodal extraction from frames via configured LLM",
        frame_count=total,
        max_workers=workers,
        model=model_name,
        provider=settings.llm_provider,
    )

    overall_start = time.monotonic()

    # Submit all frames to the thread pool. We use a dict keyed by future → index
    # so we can reassemble results in the original order.
    results: list[dict[str, Any] | None] = [None] * total

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                _ocr_single_frame,
                path=path,
                index=i + 1,
                total=total,
                model_name=model_name,
                temperature=settings.ocr_temperature,
                seed=settings.eval_seed,
            ): i
            for i, path in enumerate(frame_paths)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()

    # Filter out any None entries (shouldn't happen, but defensive)
    final_results = [r for r in results if r is not None]

    overall_elapsed = time.monotonic() - overall_start
    logger.info(
        "Completed parallel multimodal extraction from frames",
        frames_processed=len(final_results),
        total_wall_time=f"{overall_elapsed:.1f}s",
        workers_used=workers,
    )
    return final_results
