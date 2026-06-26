import structlog
from fastapi import APIRouter, Form, HTTPException, UploadFile, status

from app.evaluation.schemas import EvaluateResponse
from app.evaluation.usecase import run_full_pipeline

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/evaluate",
    tags=["Evaluation"],
)


@router.post(
    "",
    response_model=EvaluateResponse,
    status_code=status.HTTP_200_OK,
    summary="Run full end-to-end multimodal evaluation",
    description=(
        "Accepts a video upload along with metadata (person name, subject, timing), "
        "runs the full extraction pipeline (split, frames, transcribe, OCR, consolidate), "
        "and evaluates the technical accuracy, grammar, and language mix using Gemini."
    ),
)
async def evaluate_lecture(
    video: UploadFile,
    person_name: str = Form(..., description="Name of the person delivering the lecture."),
    subject: str = Form(..., description="The main subject or topic of the lecture."),
    timing: str = Form(..., description="Timing or duration context."),
) -> EvaluateResponse:
    """
    Handler for the unified evaluation endpoint.
    """
    # Simple validation of input file
    if not video.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No video file provided.",
        )

    try:
        # executes the entire evaluvation pipeline
        result = await run_full_pipeline(
            file=video,
            person_name=person_name,
            subject=subject,
            timing=timing,
        )
        return result

    except Exception as e:
        logger.error("Full pipeline evaluation failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation pipeline failed: {str(e)}",
        )
