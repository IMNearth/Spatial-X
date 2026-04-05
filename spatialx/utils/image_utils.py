""" Modified from https://github.com/kq-chen/qwen-vl-utils/blob/main/src/qwen_vl_utils/vision_process.py """
from typing import Union
import math
import base64
import requests
import io
import numpy as np
from PIL import Image


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _encode_pil_image(PIL_image: Image.Image, format: str = "JPEG", max_mb: int=10) -> str:
    max_bytes = max_mb * 1024 * 1024

    output_bytes = io.BytesIO()
    PIL_image = PIL_image.convert("RGB")
    PIL_image.save(output_bytes, format=format)
    bytes_data = output_bytes.getvalue()

    org_w, org_h = PIL_image.width, PIL_image.height
    while len(bytes_data) > max_bytes:
        ratio = (max_bytes / len(bytes_data)) ** 0.5
        new_w = int(PIL_image.width * ratio)
        new_h = int(PIL_image.height * ratio)
        PIL_image = PIL_image.resize((new_w, new_h))

        output_bytes = io.BytesIO()
        PIL_image.save(output_bytes, 'JPEG')
        bytes_data = output_bytes.getvalue()
        print(f"Image size after compression: {len(bytes_data) / 1024 / 1024:.2f} MB [From {org_w}x{org_h} To {new_w}x{new_h}]")
    base64str = base64.b64encode(bytes_data).decode('utf-8')

    mime = f"image/{format.lower()}"
    return f"data:{mime};base64,{base64str}"


def encode_image(
    image: Union[str, Image.Image],
    format: str = "JPEG",
    max_side: int = 1280,
    size_factor: int = IMAGE_FACTOR,
    detail: str = "high",
) -> dict:
    """ Convert various inputs into the format expected by OpenAI/Claude interfaces, which is
        {"type": "image_url", "image_url": {"url": ...}}  
    """
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        data = image.split(";", 1)[1]
        if data.startswith("base64,"):
            data = base64.b64decode(data[7:])
            image_obj = Image.open(io.BytesIO(data))
    else:
        image_obj = Image.open(image)

    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    
    image = image_obj.convert("RGB")
    width, height = image.size
    if max_side is not None and max(width, height) > max_side:
        resize_ratio = min(1.0, max_side / float(max(width, height)))
        resized_width = int(width * resize_ratio)
        resized_height = int(height * resize_ratio)
        
        resized_width, resized_height = smart_resize(resized_height, resized_width, factor=size_factor)
    else: resized_width, resized_height = width, height

    # resized_width, resized_height = 2048, 2048
    image = image.resize((resized_width, resized_height))
    new_width, new_height = image.size
    print(f"Resized image to {new_width}x{new_height} -- Detail: {detail}")

    image_url = _encode_pil_image(image, format)
    return {"type": "image_url", "image_url": {"url": image_url}, "detail": detail}


def avoid_overlap(
    px, py,
    past_circle_coords,
    min_distance,
    image_width,
    image_height,
    margin=0,
    max_radius_mul=6,
    num_angles=16,
):
    """ Adjust (px, py), avoid overlap with existing circle centers, and ensure it falls within the image bounds.

    Parameters:
        px, py               - Initial circle center coordinates (int)
        past_circle_coords   - List of previously drawn circle centers [(x1, y1), ...]
        min_distance         - Minimum distance between circle centers (pixels)
        image_width          - Image width (pixels)
        image_height         - Image height (pixels)
        margin               - Minimum space to leave from the edges (pixels), typically set to the circle radius
        max_radius_mul       - Maximum search radius multiplier, maximum search radius=max_radius_mul * min_distance
        num_angles           - Number of angles sampled per circle during spiral search
    
    Returns:
        (new_px, new_py)     - Adjusted circle center coordinates (int)
    """

    # Boundary constraints (ensure the entire circle is within the image, margin can be set to the circle radius)
    min_x = margin
    max_x = image_width - 1 - margin
    min_y = margin
    max_y = image_height - 1 - margin

    px = int(np.clip(px, min_x, max_x))
    py = int(np.clip(py, min_y, max_y))

    if not past_circle_coords: return px, py

    def in_bounds(x, y):
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)

    def is_ok(x, y):
        if not in_bounds(x, y):
            return False
        for (cx, cy) in past_circle_coords:
            if np.hypot(x - cx, y - cy) < min_distance:
                return False
        return True

    # 1. Check if the original position is valid
    if is_ok(px, py): return px, py

    max_radius = max_radius_mul * min_distance

    # 2. Try moving left and right
    for dx in range(1, max_radius + 1):
        # Move left
        nx = px - dx
        if is_ok(nx, py):
            return nx, py
        # Move right
        nx = px + dx
        if is_ok(nx, py):
            return nx, py

    # 3. Then move up and down
    for dy in range(1, max_radius + 1):
        # Move up (note that the y-axis is positive downward, so "up" means decreasing y)
        ny = py - dy
        if is_ok(px, ny):
            return px, ny
        # Move down
        ny = py + dy
        if is_ok(px, ny):
            return px, ny

    # 4. Finally, spiral search
    for radius in range(min_distance, max_radius + 1, 2):
        angles = np.linspace(0, 2 * np.pi, num=num_angles, endpoint=False)
        for ang in angles:
            nx = int(round(px + radius * np.cos(ang)))
            ny = int(round(py + radius * np.sin(ang)))
            if is_ok(nx, ny):
                return nx, ny

    # 5. Fall back to original position if no valid position found
    return px, py



def adjust_title_y(
    y_coord,
    circle_ys,
    image_height,
    min_distance_pix=44,
    step_pix=2,
):
    """
    Adjust the y-coordinate of the title (0~1), avoiding being too close to circle centers in the y-direction.
    Only checks the y-axis, ignoring the x-axis.

    Parameters:
        y_coord         - Initial title y (axes coordinate, 0~1)
        circle_ys       - List of all circle center y-coordinates (0~1, already converted to 1 - y/HEIGHT format)
        image_height    - Original image height (pixels, used to convert pixel distances to 0~1)
        min_distance_pix- Minimum pixel distance between title and circle centers in the y-direction
        step_pix        - Pixel step size for each attempt (smaller values are more detailed, larger values are faster)
    """

    if not circle_ys:
        return y_coord

    # Pixel distance -> Normalized distance
    min_dy = min_distance_pix / float(image_height)
    step_dy = step_pix / float(image_height)

    def collides(y):
        return any(abs(y - cy) < min_dy for cy in circle_ys)

    if not collides(y_coord): return y_coord

    # Initial position is low: move up (y increases)
    if y_coord <= 0.5:
        y = y_coord
        while y < 1.0:
            y += step_dy
            if not collides(y):
                return y
    else:
        # Initial position is high: move down (y decreases)
        y = y_coord
        while y > 0.0:
            y -= step_dy
            if not collides(y):
                return y

    return y_coord  # Fall back to original position if no valid position found

