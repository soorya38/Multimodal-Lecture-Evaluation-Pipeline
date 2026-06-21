import asyncio
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


async def split_video_audio(
    input_path: str,
    video_out_path: str,
    audio_out_path: str,
) -> None:
    """
    Use FFmpeg to split a media file into a video-only MP4 and an audio-only MP3.

    Equivalent to:
        ffmpeg -i input.mp4 -map 0:v -c:v copy video.mp4 -map 0:a -c:a libmp3lame audio.mp3

    Args:
        input_path: Path to the source media file.
        video_out_path: Destination path for the video-only output.
        audio_out_path: Destination path for the audio-only output.

    Raises:
        RuntimeError: If the FFmpeg process exits with a non-zero return code.
        FileNotFoundError: If the input file does not exist.
    """
    # Validate that the input file actually exists before invoking FFmpeg
    if not Path(input_path).is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cmd = [
        "ffmpeg",
        "-y",               # Overwrite output files without prompting
        "-i", input_path,
        "-map", "0:v",      # Select the video stream
        "-c:v", "copy",     # Copy video codec (no re-encoding)
        video_out_path,
        "-map", "0:a",      # Select the audio stream
        "-c:a", "libmp3lame",  # Encode audio to MP3
        audio_out_path,
    ]

    logger.info(
        "Starting FFmpeg split",
        input=input_path,
        video_out=video_out_path,
        audio_out=audio_out_path,
    )

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_output = stderr.decode(errors="replace")
        logger.error(
            "FFmpeg process failed",
            return_code=process.returncode,
            stderr=error_output,
        )
        raise RuntimeError(
            f"FFmpeg exited with code {process.returncode}: {error_output}"
        )

    logger.info(
        "FFmpeg split completed successfully",
        video_out=video_out_path,
        audio_out=audio_out_path,
    )
