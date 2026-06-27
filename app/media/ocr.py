import json
import os
import time
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import ollama

logger = structlog.get_logger(__name__)

# Configurable concurrency for parallel OCR. Each worker sends an independent
# HTTP request to the Ollama server, so this is safe to parallelise.
# Default scales with available CPUs (capped at 8) since GPU-backed Ollama
# can handle higher concurrency than CPU-only setups.
_OCR_MAX_WORKERS = int(os.getenv("OCR_MAX_WORKERS", str(min(os.cpu_count() or 4, 8))))

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
    ollama_host: str,
    model_name: str,
) -> dict[str, Any]:
    """
    Run OCR on a single frame image. Designed to be called from a thread pool.

    Each invocation creates its own Ollama client to avoid sharing state across threads.
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
        client = ollama.Client(host=ollama_host)

        response = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": OCR_SYSTEM_PROMPT,
                    "images": [path],
                }
            ],
            format="json",
            options={
                "temperature": 0.0,
                "seed": 42,  # Fixed seed → reproducible OCR output
            },
        )

        content = response["message"]["content"]
        if isinstance(content, dict):
            parsed_json = content
        else:
            response_text = str(content or "{}").strip()
            parsed_json = json.loads(response_text)

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
    Extract text and diagram descriptions from a list of frame images using local Ollama model.

    Uses a ThreadPoolExecutor to process multiple frames concurrently, since each
    Ollama call is network I/O-bound. The concurrency level is controlled by the
    OCR_MAX_WORKERS environment variable (default: 4).

    Args:
        frame_paths: List of absolute paths to the frame JPEG images.

    Returns:
        A list of dictionaries containing the extracted rich text for each frame,
        ordered to match the input frame_paths order.
    """
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    total = len(frame_paths)
    workers = min(_OCR_MAX_WORKERS, total) if total > 0 else 1

    # Vision model for OCR. Configurable so it can point at an installed tag.
    # Default "llava-phi3" (3.8B) must be pulled first (`ollama pull llava-phi3`)
    # or every frame will fail with "model not found" and visual_content ends up empty.
    model_name = os.getenv("OLLAMA_OCR_MODEL", "llava-phi3")

    logger.info(
        "Starting parallel multimodal extraction from frames via local Ollama",
        frame_count=total,
        max_workers=workers,
        model=model_name,
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
                ollama_host=ollama_host,
                model_name=model_name,
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
