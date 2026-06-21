import structlog
from fastapi import APIRouter, HTTPException, UploadFile, status
from minio.error import S3Error

from app.media.schemas import (
    ExtractFramesRequest,
    ExtractFramesResponse,
    OcrRequest,
    OcrResponse,
    SplitMediaResponse,
    TranscribeRequest,
    TranscribeResponse,
)
from app.media.usecase import (
    extract_frames_and_store,
    extract_text_and_store,
    split_and_store,
    transcribe_and_store,
)

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


@router.post(
    "/extract-frames",
    response_model=ExtractFramesResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract key frames from a previously split video",
    description=(
        "Uses PySceneDetect to detect scene changes in the video-only file "
        "produced by /split, extracts representative frames as JPEGs, "
        "and stores them in MinIO."
    ),
)
async def extract_frames(request: ExtractFramesRequest) -> ExtractFramesResponse:
    """
    Handler for the frame extraction endpoint.

    Takes an upload_id from a previous /split call, downloads the video from MinIO,
    runs scene detection, extracts frames, and uploads them back to MinIO.
    """
    try:
        result = await extract_frames_and_store(
            upload_id=request.upload_id,
            threshold=request.threshold,
            num_images=request.num_images,
        )
    except S3Error as e:
        # The video object for this upload_id doesn't exist in MinIO
        logger.error(
            "Video not found in MinIO for upload_id",
            upload_id=request.upload_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video not found for upload_id '{request.upload_id}'. "
                   f"Ensure /split was called first.",
        )
    except FileNotFoundError as e:
        logger.error("File not found during frame extraction", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except RuntimeError as e:
        logger.error("Scene detection failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Frame extraction failed: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error during frame extraction", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during frame extraction.",
        )

    return result


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe audio using Whisper",
    description=(
        "Downloads the audio file associated with the upload_id, "
        "runs faster-whisper to generate a transcript, and stores "
        "the resulting JSON in MinIO."
    ),
)
async def transcribe(request: TranscribeRequest) -> TranscribeResponse:
    """
    Handler for the transcription endpoint.
    """
    try:
        result = await transcribe_and_store(
            upload_id=request.upload_id,
            model_size=request.model_size,
            language=request.language,
        )
    except S3Error as e:
        logger.error(
            "Audio not found in MinIO for upload_id",
            upload_id=request.upload_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio not found for upload_id '{request.upload_id}'. Ensure /split was called.",
        )
    except FileNotFoundError as e:
        logger.error("File not found during transcription", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except RuntimeError as e:
        logger.error("Transcription failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Transcription failed: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error during transcription", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during transcription.",
        )
    return result


@router.post(
    "/ocr",
    response_model=OcrResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract rich text and diagrams from video frames using Gemini",
    description=(
        "Downloads all frames associated with the upload_id, "
        "uploads them to the Gemini API to extract typed text, handwriting, and diagram descriptions, "
        "and stores the resulting consolidated JSON in MinIO."
    ),
)
async def ocr_frames(request: OcrRequest) -> OcrResponse:
    """
    Handler for the multimodal OCR endpoint.
    """
    try:
        result = await extract_text_and_store(
            upload_id=request.upload_id,
        )
    except S3Error as e:
        logger.error(
            "Frames not found in MinIO for upload_id",
            upload_id=request.upload_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frames not found for upload_id '{request.upload_id}'. Ensure /extract-frames was called.",
        )
    except FileNotFoundError as e:
        logger.error("File not found during OCR", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except RuntimeError as e:
        logger.error("Gemini OCR extraction failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"OCR extraction failed: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error during OCR", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during OCR extraction.",
        )

    return result
