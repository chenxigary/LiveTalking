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
