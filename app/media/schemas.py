from pydantic import BaseModel, Field


class SplitMediaResponse(BaseModel):
    """
    Response model for the media split endpoint.
    Contains the MinIO object keys for the separated video and audio files.
    """

    upload_id: str = Field(
        ...,
        description="Unique identifier for this upload session.",
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
    )
    video_object_key: str = Field(
        ...,
        description="MinIO object key for the video-only file.",
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890/video.mp4"],
    )
    audio_object_key: str = Field(
        ...,
        description="MinIO object key for the audio-only file.",
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890/audio.mp3"],
    )
    bucket: str = Field(
        ...,
        description="MinIO bucket where the objects are stored.",
        examples=["lectures"],
    )


class ExtractFramesRequest(BaseModel):
    """
    Request model for the frame extraction endpoint.
    Takes an upload_id from a previous /split response and optional tuning parameters.
    """

    upload_id: str = Field(
        ...,
        description="Upload ID from the /split endpoint whose video will be processed.",
        examples=["a1b2c3d4e5f67890abcdef1234567890"],
    )
    threshold: float = Field(
        default=27.0,
        ge=1.0,
        le=100.0,
        description=(
            "ContentDetector sensitivity for scene change detection. "
            "Lower values detect subtler transitions (e.g., slide changes). "
            "Range: 1.0–100.0."
        ),
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "Number of representative frames to extract per detected scene. "
            "1 = middle frame only (recommended for OCR pipelines)."
        ),
    )


class FrameInfo(BaseModel):
    """Metadata for a single extracted frame."""

    scene_number: int = Field(
        ...,
        description="1-based index of the scene this frame belongs to.",
    )
    object_key: str = Field(
        ...,
        description="MinIO object key for this frame image.",
        examples=["abc123/frames/scene-001-01.jpg"],
    )


class ExtractFramesResponse(BaseModel):
    """Response model for the frame extraction endpoint."""

    upload_id: str = Field(
        ...,
        description="Upload ID that was processed.",
    )
    bucket: str = Field(
        ...,
        description="MinIO bucket where frames are stored.",
        examples=["lectures"],
    )
    frame_count: int = Field(
        ...,
        description="Total number of frames extracted.",
    )
    frames: list[FrameInfo] = Field(
        ...,
        description="List of extracted frame metadata.",
    )
