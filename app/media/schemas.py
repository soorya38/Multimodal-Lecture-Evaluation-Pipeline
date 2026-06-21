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
