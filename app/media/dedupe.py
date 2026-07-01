"""
Perceptual-hash de-duplication of extracted slide frames.

Scene detection often captures the *same* slide more than once — a presenter
returns to a previous slide, a camera jitters across a boundary, or a long slide
spans several detected scenes. Those duplicates waste OCR calls and, worse,
over-weight whatever text is on the repeated slide when the technical evaluator
reads the consolidated visual content.

We collapse them with a difference hash (dHash): each frame is reduced to a
64-bit fingerprint, and frames whose fingerprints are within a small Hamming
distance of an already-kept frame are dropped. This is robust to compression
noise and minor camera movement while still distinguishing genuinely different
slides. Uses OpenCV (already a dependency via PySceneDetect) — no new packages.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

_HASH_SIDE = 8  # produces an 8x8 = 64-bit hash


def dhash(image_path: str) -> int | None:
    """
    Compute a 64-bit difference hash for an image, or None if it can't be read.

    The image is converted to grayscale and resized to 9x8; each of the 8x8
    horizontal adjacent-pixel comparisons contributes one bit.
    """
    import cv2

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning("Could not read frame for hashing; keeping it", path=image_path)
        return None

    resized = cv2.resize(img, (_HASH_SIDE + 1, _HASH_SIDE), interpolation=cv2.INTER_AREA)
    bits = 0
    for row in range(_HASH_SIDE):
        for col in range(_HASH_SIDE):
            bits <<= 1
            if resized[row, col] < resized[row, col + 1]:
                bits |= 1
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def deduplicate_frames(frame_paths: list[str], hamming_threshold: int) -> list[str]:
    """
    Return the subset of ``frame_paths`` with near-duplicate slides removed,
    preserving input order.

    A frame is kept if its hash is more than ``hamming_threshold`` bits away from
    every previously-kept frame's hash. Frames that fail to hash are always kept
    (fail-open: never silently drop content we couldn't inspect).
    """
    if hamming_threshold <= 0 and len(frame_paths) <= 1:
        return list(frame_paths)

    kept: list[str] = []
    kept_hashes: list[int] = []

    for path in frame_paths:
        h = dhash(path)
        if h is None:
            kept.append(path)
            continue

        is_dup = any(_hamming(h, kh) <= hamming_threshold for kh in kept_hashes)
        if is_dup:
            logger.info("Dropping near-duplicate frame", path=path)
            continue

        kept.append(path)
        kept_hashes.append(h)

    return kept
