from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# def read_imgs(img_list):
#     frames = []
#     logger.info('reading images...')
#     for img_path in tqdm(img_list):
#         frame = cv2.imread(img_path)
#         frames.append(frame)
#     return frames

def _rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[..., ::-1])


def read_bgr(img_path) -> np.ndarray:
    """Read an image as an owned BGR uint8 array without importing OpenCV."""

    with Image.open(img_path) as image:
        return _rgb_to_bgr(np.asarray(image.convert("RGB"), dtype=np.uint8))


def decode_bgr(payload: bytes) -> np.ndarray:
    """Decode encoded image bytes into the BGR layout expected by LiveTalking."""

    with Image.open(BytesIO(payload)) as image:
        return _rgb_to_bgr(np.asarray(image.convert("RGB"), dtype=np.uint8))


def resize_bgr(
    frame: np.ndarray, size: tuple[int, int], *, resample="bilinear"
) -> np.ndarray:
    """Resize a BGR frame using Pillow while preserving OpenCV-style layout."""

    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    try:
        resize_filter = filters[resample]
    except KeyError as exc:
        raise ValueError(f"Unsupported image resample mode: {resample}") from exc
    rgb = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[..., ::-1])
    resized = Image.fromarray(rgb).resize(size, resize_filter)
    return _rgb_to_bgr(np.asarray(resized, dtype=np.uint8))


@lru_cache(maxsize=256)
def _lower_face_alpha(
    height: int, width: int, split: float, feather: int, edge: int
) -> np.ndarray:
    """Build the read-only alpha mask used to composite a regenerated mouth.

    Lip-sync models rebuild the whole face crop even though only the masked
    lower half carries new information, so pasting the full crop back also
    replaces sharp eye/hair pixels with a softer reconstruction.  The mask is
    zero above ``split``, ramps to one over ``feather`` rows, and fades out
    again over ``edge`` pixels at the left/right/bottom borders so the crop
    boundary does not read as a rectangle.

    Frame-to-frame box sizes barely move, so caching keeps this off the
    25-fps render path entirely.
    """

    alpha = np.zeros((height, width, 1), dtype=np.float32)
    mid = min(height, max(0, int(round(height * split))))
    ramp_end = min(height, mid + max(0, feather))
    alpha[ramp_end:] = 1.0
    if ramp_end > mid:
        alpha[mid:ramp_end, :, 0] = np.linspace(
            0.0, 1.0, ramp_end - mid, dtype=np.float32
        )[:, None]

    edge = max(0, min(edge, width // 2, height // 2))
    if edge:
        ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
        alpha[:, :edge] *= ramp[None, :, None]
        alpha[:, width - edge:] *= ramp[::-1][None, :, None]
        alpha[height - edge:, :] *= ramp[::-1][:, None, None]

    alpha.setflags(write=False)
    return alpha


def blend_lower_face_bgr(
    canvas: np.ndarray,
    patch: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    split: float = 0.5,
    feather: int = 24,
    edge: int = 12,
    resample: str = "lanczos",
) -> np.ndarray:
    """Composite only the regenerated lower face of ``patch`` into ``canvas``.

    ``box`` is LiveTalking's ``(y1, y2, x1, x2)`` face rectangle.  ``canvas``
    is modified in place and returned.  Rows above the blend line are never
    touched, which is both the point of this function and why it costs about
    half of a full-crop paste.
    """

    y1, y2, x1, x2 = box
    height, width = y2 - y1, x2 - x1
    if height <= 0 or width <= 0:
        return canvas

    resized = resize_bgr(patch, (width, height), resample=resample)
    alpha = _lower_face_alpha(height, width, split, feather, edge)
    mid = min(height, max(0, int(round(height * split))))

    region = canvas[y1 + mid:y2, x1:x2]
    weight = alpha[mid:]
    blended = region * (1.0 - weight) + resized[mid:] * weight
    canvas[y1 + mid:y2, x1:x2] = np.rint(blended).astype(np.uint8)
    return canvas


def blend_bgr(
    first: np.ndarray,
    first_weight: float,
    second: np.ndarray,
    second_weight: float,
) -> np.ndarray:
    """Blend equally-shaped BGR frames without loading OpenCV's FFmpeg wheel."""

    if first.shape != second.shape:
        raise ValueError(f"Cannot blend frame shapes {first.shape} and {second.shape}")
    blended = (
        first.astype(np.float32) * first_weight
        + second.astype(np.float32) * second_weight
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_debug_label_bgr(frame: np.ndarray, text="LiveTalking") -> np.ndarray:
    """Draw the small upstream debug label while touching only a small crop."""

    result = np.asarray(frame, dtype=np.uint8).copy()
    crop_height = min(26, result.shape[0])
    crop_width = min(128, result.shape[1])
    if crop_height == 0 or crop_width == 0:
        return result
    crop_rgb = np.ascontiguousarray(result[:crop_height, :crop_width, ::-1])
    crop = Image.fromarray(crop_rgb)
    ImageDraw.Draw(crop).text((10, 4), text, fill=(128, 128, 128))
    result[:crop_height, :crop_width] = np.asarray(crop, dtype=np.uint8)[..., ::-1]
    return result


def read_imgs(img_list):
    def load_image(index, img_path):
        return index, read_bgr(img_path)

    frames = [None] * len(img_list)  # Initialize a list with the same length as img_list
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_image, idx, img_path): idx for idx, img_path in enumerate(img_list)}
        for future in tqdm(as_completed(futures), total=len(img_list)):
            idx, img = future.result()
            frames[idx] = img
    return frames

def mirror_index(size, index):
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1
