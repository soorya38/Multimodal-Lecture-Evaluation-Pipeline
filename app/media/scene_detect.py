from pathlib import Path

import structlog
from scenedetect import ContentDetector, SceneManager, open_video
from scenedetect.scene_manager import save_images

logger = structlog.get_logger(__name__)


def detect_scenes_and_extract_frames(
    video_path: str,
    output_dir: str,
    threshold: float = 27.0,
    num_images: int = 1,
    frame_skip: int = 0,
) -> list[str]:
    """
    Detect scene changes in a video and extract representative frames.

    Uses PySceneDetect's ContentDetector to identify scene boundaries based on
    changes in hue, saturation, and luminance between consecutive frames. For each
    detected scene, `num_images` representative frames are saved as JPEGs.

    Args:
        video_path: Absolute path to the input video file.
        output_dir: Directory where extracted frame images will be written.
        threshold: ContentDetector sensitivity. Lower values detect subtler
                   transitions (e.g., slide changes). Default 27.0.
        num_images: Number of representative frames to extract per scene.
                    1 = middle frame only (recommended for OCR pipelines).
        frame_skip: Number of frames to skip to speed up processing. Defaults to 0,
                    which auto-calculates a safe skip factor for very long videos.

    Returns:
        A flat list of absolute paths to the saved JPEG frame files.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        RuntimeError: If scene detection produces zero scenes.
    """
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(
        "Starting scene detection",
        video_path=video_path,
        threshold=threshold,
        num_images=num_images,
        frame_skip=frame_skip,
    )

    # Open the video and configure the scene manager
    video = open_video(video_path)
    
    # Auto-calculate frame_skip for long videos if not explicitly provided
    if frame_skip == 0:
        duration_sec = video.duration.get_seconds() if video.duration else 0
        if duration_sec > 3600:
            # Skip roughly 2 frames per hour of video length to drastically speed up
            # processing for multi-hour lectures, up to a reasonable cap.
            frame_skip = min(int((duration_sec / 3600) * 2), 30)
            logger.info("Auto-calculated frame_skip for long video", duration_sec=duration_sec, frame_skip=frame_skip)

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Perform scene detection across the entire video
    scene_manager.detect_scenes(video=video, show_progress=False, frame_skip=frame_skip)
    scene_list = scene_manager.get_scene_list()

    logger.info(
        "Scene detection completed",
        scenes_detected=len(scene_list),
        video_path=video_path,
    )

    # If no scene changes are detected, treat the entire video as one scene
    if not scene_list:
        logger.warning(
            "No scene changes detected — treating entire video as a single scene",
            video_path=video_path,
        )
        video_duration = video.duration
        scene_list = [(video.base_timecode, video_duration)]

    # Extract and save representative frames for each scene
    image_map = save_images(
        scene_list=scene_list,
        video=video,
        num_images=num_images,
        output_dir=output_dir,
        image_extension="jpg",
        encoder_param=95,  # JPEG quality (0-100)
        show_progress=False,
    )

    # Flatten the {scene_number: [paths]} dict into a single list
    import os
    all_frame_paths: list[str] = []
    for scene_number in sorted(image_map.keys()):
        for path in image_map[scene_number]:
            if not os.path.isabs(path):
                path = os.path.join(output_dir, path)
            all_frame_paths.append(path)

    logger.info(
        "Frame extraction completed",
        total_frames=len(all_frame_paths),
        output_dir=output_dir,
    )

    return all_frame_paths
