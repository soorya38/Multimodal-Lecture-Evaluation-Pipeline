import json
import os
from typing import Any

import structlog
from google import genai
from google.genai import types

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
    Extract text and diagram descriptions from a list of frame images using Gemini.

    Args:
        frame_paths: List of absolute paths to the frame JPEG images.

    Returns:
        A list of dictionaries containing the extracted rich text for each frame.
    """
    # Initialize the Gemini client. It automatically picks up GEMINI_API_KEY from env.
    try:
        client = genai.Client()
    except Exception as e:
        logger.error("Failed to initialize Gemini Client. Check GEMINI_API_KEY.", error=str(e))
        raise RuntimeError("Missing or invalid GEMINI_API_KEY.") from e

    logger.info("Starting multimodal extraction from frames via Gemini", frame_count=len(frame_paths))

    results = []
    
    # We use gemini-2.5-flash as it is fast and excellent at multimodal OCR.
    model_name = "gemini-2.5-flash"

    for path in frame_paths:
        if not os.path.isfile(path):
            logger.warning("Frame file not found during Gemini OCR, skipping", path=path)
            continue

        try:
            # Upload the image file to the Gemini File API.
            # Using File API is safer for large/multiple images than inline base64.
            logger.info("Uploading frame to Gemini", frame_path=path)
            uploaded_file = client.files.upload(file=path)
            
            # Request content from the model
            logger.info("Requesting multimodal OCR from Gemini", frame_path=path, model=model_name)
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_uri(file_uri=uploaded_file.uri, mime_type="image/jpeg"),
                        types.Part.from_text(text=OCR_SYSTEM_PROMPT),
                    ])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,  # low temperature for deterministic OCR
                ),
            )
            
            # Clean up the file from Gemini storage
            client.files.delete(name=uploaded_file.name)

            response_text = response.text or "{}"
            
            # Gemini might return wrapped json despite instructions, strip it if necessary
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
                
            parsed_json = json.loads(response_text)
            
            results.append({
                "frame_filename": os.path.basename(path),
                "content": parsed_json
            })

        except Exception as e:
            logger.error("Failed to run Gemini OCR on frame", path=path, error=str(e), exc_info=True)
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
