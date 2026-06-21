import structlog
from fastapi import APIRouter, HTTPException, UploadFile, status

from app.media.schemas import SplitMediaResponse
from app.media.usecase import split_and_store

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/media",
    tags=["Media"],
)


@router.post(
    "/split",
    response_model=SplitMediaResponse,
    status_code=status.HTTP_200_OK,
    summary="Split a video into separate video and audio files",
    description=(
        "Accepts a video upload, uses FFmpeg to extract the video stream "
        "(codec copy, no re-encoding) and encode the audio stream to MP3, "
        "then stores both artifacts in MinIO."
    ),
)
async def split_media(file: UploadFile) -> SplitMediaResponse:
    """
    Handler for the media split endpoint.

    Accepts a multipart file upload, delegates processing to the usecase layer,
    and returns the MinIO object keys for the resulting video and audio files.
    """
    # Validate content type — accept common video MIME types
    allowed_types = {
        "video/mp4",
        "video/x-matroska",
        "video/avi",
        "video/quicktime",
        "video/webm",
    }

    content_type = file.content_type
    # If client sends a generic application/octet-stream or content-type is missing,
    # attempt to guess it based on the file extension for a better user experience.
    if not content_type or content_type == "application/octet-stream":
        import mimetypes
        guessed_type, _ = mimetypes.guess_type(file.filename or "")
        if guessed_type:
            content_type = guessed_type

    if content_type not in allowed_types:
        logger.warning(
            "Rejected upload with unsupported content type",
            content_type=file.content_type,
            resolved_type=content_type,
            filename=file.filename,
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {file.content_type or 'unknown'}. "
                   f"Accepted types: {', '.join(sorted(allowed_types))}",
        )

    try:
        result = await split_and_store(file)
    except FileNotFoundError as e:
        logger.error("Input file not found during processing", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        logger.error("FFmpeg processing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Video processing failed: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error during media split", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during processing.",
        )

    return result
