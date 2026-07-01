from typing import Any

import structlog

from app.core.config import get_settings
from app.core.llm import get_llm_client

logger = structlog.get_logger(__name__)


def _call_llm(prompt: str) -> dict[str, Any]:
    """
    Send a prompt to the configured LLM provider and return the parsed JSON reply.

    Uses low temperature + a fixed seed for deterministic, rubric-based scoring.
    The provider (Ollama or any OpenAI-compatible endpoint) and model are chosen
    via configuration — see ``app.core.config.Settings``.

    NOTE: the default eval model ``llama3.2`` resolves to ``llama3.2:latest`` (3B)
    on Ollama — if that tag is not pulled, the call fails with "model not found".
    Run ``ollama pull llama3.2`` or set ``LLM_EVAL_MODEL`` to an installed tag.
    """
    settings = get_settings()
    client = get_llm_client(settings)

    return client.chat_json(
        prompt,
        model=settings.eval_model,
        temperature=settings.eval_temperature,  # low temperature for consistent scoring
        seed=settings.eval_seed,  # fixed seed → reproducible scores
    )


def _require_score(result: dict[str, Any], key: str) -> float:
    """
    Extract a numeric score from the model's parsed JSON reply.

    Raises ValueError if the key is missing or non-numeric, instead of silently
    defaulting to 0.0 — a malformed reply is a failure to surface, not a real
    zero score.
    """
    if key not in result:
        raise ValueError(
            f"Model reply missing required key '{key}'. Got keys: {list(result.keys())}"
        )
    try:
        return float(result[key])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Model reply key '{key}' is not numeric: {result[key]!r}") from e


def evaluate_technical(consolidated: dict[str, Any], subject: str) -> float:
    """
    Evaluate the technical accuracy of the lecture content against the given subject.

    Sends the full transcript and visual content to Ollama with a rubric that scores:
    - Correctness of concepts explained
    - Depth and completeness of coverage
    - Accuracy of examples, diagrams, and formulas

    Args:
        consolidated: The parsed consolidated.json document.
        subject: The subject/topic the lecture is supposed to cover.

    Returns:
        A technical accuracy score from 0 to 100.
    """
    transcript_text = consolidated.get("transcript", {}).get("full_text", "")
    visual_content = consolidated.get("visual_content", [])

    # Build a summary of the visual content for context
    visual_summary = ""
    for frame in visual_content:
        parts = []
        if frame.get("typed_text"):
            parts.append(f"Typed: {frame['typed_text']}")
        if frame.get("handwritten_text"):
            parts.append(f"Handwritten: {frame['handwritten_text']}")
        if frame.get("diagram_descriptions"):
            parts.append(f"Diagram: {frame['diagram_descriptions']}")
        if parts:
            visual_summary += f"\n[{frame.get('frame', 'unknown')}]\n" + "\n".join(parts) + "\n"

    prompt = f"""You are an expert academic evaluator. Evaluate the technical accuracy of the following lecture on the subject: "{subject}".

## Transcript (what the lecturer said):
{transcript_text}

## Visual Content (what was on the slides/board):
{visual_summary if visual_summary else "No visual content available."}

## Evaluation Rubric:
Score the lecture on a scale of 0-100 based on:
1. **Correctness** (40%): Are the concepts, definitions, and explanations technically accurate?
2. **Completeness** (30%): Does the lecture cover the key aspects of "{subject}" adequately?
3. **Clarity** (30%): Are examples, diagrams, and explanations clear and well-structured?

Return ONLY a JSON object with this exact structure:
{{"technical_score": <number between 0 and 100>}}
"""

    logger.info("Running technical evaluation via configured LLM", subject=subject)

    try:
        result = _call_llm(prompt)
        score = _require_score(result, "technical_score")
        # Clamp to valid range
        score = max(0.0, min(100.0, score))
        logger.info("Technical evaluation completed", score=score)
        return score
    except Exception as e:
        logger.error("Technical evaluation failed", error=str(e), exc_info=True)
        raise RuntimeError(f"Technical evaluation failed: {e}") from e


def evaluate_grammar(consolidated: dict[str, Any]) -> float:
    """
    Evaluate the grammatical quality of the lecture transcript.

    Analyzes sentence structure, verb agreement, tense consistency,
    and overall language proficiency.

    Args:
        consolidated: The parsed consolidated.json document.

    Returns:
        A grammar quality score from 0 to 100.
    """
    transcript_text = consolidated.get("transcript", {}).get("full_text", "")

    prompt = f"""You are an expert English language evaluator. Evaluate the grammatical quality of the following lecture transcript.

## Transcript:
{transcript_text}

## Important Context:
This is a spoken lecture transcript generated by speech recognition. Some apparent errors may be transcription artifacts rather than actual grammar mistakes. Focus on:
- Patterns of grammatical errors (not one-off transcription glitches)
- Sentence structure and coherence
- Verb tense consistency
- Subject-verb agreement
- Proper use of articles and prepositions

## Scoring Rubric (0-100):
- **90-100**: Excellent grammar with minimal errors. Native or near-native proficiency.
- **70-89**: Good grammar with occasional errors that don't impede understanding.
- **50-69**: Moderate grammar with noticeable errors. Meaning is generally clear.
- **30-49**: Poor grammar with frequent errors that sometimes impede understanding.
- **0-29**: Very poor grammar with pervasive errors.

Return ONLY a JSON object with this exact structure:
{{"grammatical_score": <number between 0 and 100>}}
"""

    logger.info("Running grammar evaluation via configured LLM")

    try:
        result = _call_llm(prompt)
        score = _require_score(result, "grammatical_score")
        score = max(0.0, min(100.0, score))
        logger.info("Grammar evaluation completed", score=score)
        return score
    except Exception as e:
        logger.error("Grammar evaluation failed", error=str(e), exc_info=True)
        raise RuntimeError(f"Grammar evaluation failed: {e}") from e


def evaluate_language_mix(consolidated: dict[str, Any]) -> dict[str, float]:
    """
    Analyze the language distribution of the lecture transcript.

    Detects the percentage of English vs Tamil (including Tanglish — Tamil written
    in English script or code-mixed speech).

    Args:
        consolidated: The parsed consolidated.json document.

    Returns:
        A dictionary with "english_percentage" and "tamil_percentage" (both 0-100,
        summing to 100).
    """
    transcript_text = consolidated.get("transcript", {}).get("full_text", "")

    prompt = f"""You are an expert linguist specializing in multilingual analysis of Indian languages. Analyze the language distribution of the following lecture transcript.

## Transcript:
{transcript_text}

## Instructions:
Classify the language used in this transcript into:
1. **English**: Sentences, phrases, or words spoken in English.
2. **Tamil**: Sentences, phrases, or words spoken in Tamil. This includes:
   - Pure Tamil (even if transliterated in English script by the speech recognizer)
   - Tanglish (Tamil-English code-mixing where Tamil sentence structure uses English technical terms)
   - Tamil filler words and connectors (e.g., "appo", "idhu", "paarunga")

Estimate the percentage breakdown. The two percentages must sum to 100.
If the transcript is entirely in one language, set the other to 0.
If you cannot detect any Tamil, set tamil to 0 and english to 100.

Return ONLY a JSON object with this exact structure:
{{"english_percentage": <number between 0 and 100>, "tamil_percentage": <number between 0 and 100>}}
"""

    logger.info("Running language mix evaluation via configured LLM")

    try:
        result = _call_llm(prompt)

        english_pct = _require_score(result, "english_percentage")
        tamil_pct = _require_score(result, "tamil_percentage")

        # Clamp and normalize so they sum to 100
        english_pct = max(0.0, min(100.0, english_pct))
        tamil_pct = max(0.0, min(100.0, tamil_pct))
        total = english_pct + tamil_pct
        if total > 0:
            english_pct = round(english_pct / total * 100, 1)
            tamil_pct = round(tamil_pct / total * 100, 1)
        else:
            english_pct = 100.0
            tamil_pct = 0.0

        logger.info("Language mix evaluation completed", english=english_pct, tamil=tamil_pct)
        return {"english_percentage": english_pct, "tamil_percentage": tamil_pct}

    except Exception as e:
        logger.error("Language mix evaluation failed", error=str(e), exc_info=True)
        raise RuntimeError(f"Language mix evaluation failed: {e}") from e
