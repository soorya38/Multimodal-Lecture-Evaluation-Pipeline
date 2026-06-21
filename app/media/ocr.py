import json
import os
import structlog
from typing import Any
import ollama

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

def extract_text_from_frames(frame_paths: list[str]) -> list[dict[str, Any]]:
    """
    Extract text and diagram descriptions from a list of frame images using local Ollama model.

    Args:
        frame_paths: List of absolute paths to the frame JPEG images.

    Returns:
        A list of dictionaries containing the extracted rich text for each frame.
    """
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client = ollama.Client(host=ollama_host)
    
    logger.info("Starting multimodal extraction from frames via local Ollama", frame_count=len(frame_paths))

    results = []
    
    # Use llava-phi3 (3.8B) for OCR
    model_name = "llava-phi3"

    for path in frame_paths:
        if not os.path.isfile(path):
            logger.warning("Frame file not found during OCR, skipping", path=path)
            continue

        try:
            logger.info("Requesting multimodal OCR from Ollama", frame_path=path, model=model_name)
            
            # Ollama expects the image path or bytes
            response = client.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": OCR_SYSTEM_PROMPT,
                        "images": [path]
                    }
                ],
                format="json",
                options={
                    "temperature": 0.0,
                }
            )
            
            content = response['message']['content']
            if isinstance(content, dict):
                parsed_json = content
            else:
                response_text = str(content or "{}").strip()
                parsed_json = json.loads(response_text)
            
            results.append({
                "frame_filename": os.path.basename(path),
                "content": parsed_json
            })

        except Exception as e:
            logger.error("Failed to run local OCR on frame", path=path, error=str(e), exc_info=True)
            results.append({
                "frame_filename": os.path.basename(path),
                "content": {
                    "typed_text": "",
                    "handwritten_text": "",
                    "diagram_descriptions": "",
                    "error": str(e)
                }
            })

    logger.info("Completed multimodal extraction from frames", frames_processed=len(results))
    return results
