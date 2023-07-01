# modified from https://github.com/skimai/DragGAN

import copy
import math
import os
import urllib.request
from typing import List, Optional, Tuple

import numpy as np
import PIL
import PIL.Image
import PIL.ImageDraw
import torch
import torch.optim
from tqdm import tqdm

BASE_DIR = os.environ.get(
    'DRAGGAN_HOME',
    os.path.join(os.path.expanduser('~'), 'draggan', 'checkpoints-pkl')
)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def get_path(base_path):
    save_path = os.path.join(BASE_DIR, base_path)
    if not os.path.exists(save_path):
        url = f"https://huggingface.co/aaronb/StyleGAN2-pkl/resolve/main/{base_path}"
        print(f'{base_path} not found')
        print('Try to download from huggingface: ', url)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        download_url(url, save_path)
        print('Downloaded to ', save_path)
    return save_path


def tensor_to_PIL(img: torch.Tensor) -> PIL.Image.Image:
    """
    Converts a tensor image to a PIL Image.

    Args:
        img (torch.Tensor): The tensor image of shape [batch_size, num_channels, height, width].

    Returns:
        A PIL Image object.
    """
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")


def get_ellipse_coords(
    point: Tuple[int, int], radius: int = 5
) -> Tuple[int, int, int, int]:
    """
    Returns the coordinates of an ellipse centered at the given point.

    Args:
        point (Tuple[int, int]): The center point of the ellipse.
        radius (int): The radius of the ellipse.

    Returns:
        A tuple containing the coordinates of the ellipse in the format (x_min, y_min, x_max, y_max).
    """
    center = point
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )


def draw_handle_target_points(
        img: PIL.Image.Image,
        handle_points: List[Tuple[int, int]],
        target_points: List[Tuple[int, int]],
        radius: int = 5):
    """
    Draws handle and target points with arrow pointing towards the target point.

    Args:
        img (PIL.Image.Image): The image to draw on.
        handle_points (List[Tuple[int, int]]): A list of handle [x,y] points.
        target_points (List[Tuple[int, int]]): A list of target [x,y] points.
        radius (int): The radius of the handle and target points.
    """
    if not isinstance(img, PIL.Image.Image):
        img = PIL.Image.fromarray(img)

    if len(handle_points) == len(target_points) + 1:
        target_points = copy.deepcopy(target_points) + [None]

    draw = PIL.ImageDraw.Draw(img)
    for handle_point, target_point in zip(handle_points, target_points):
        handle_point = [handle_point[1], handle_point[0]]
        # Draw the handle point
        handle_coords = get_ellipse_coords(handle_point, radius)
        draw.ellipse(handle_coords, fill="red")

        if target_point is not None:
            target_point = [target_point[1], target_point[0]]
            # Draw the target point
            target_coords = get_ellipse_coords(target_point, radius)
            draw.ellipse(target_coords, fill="blue")

            # Draw arrow head
            arrow_head_length = 10.0

            # Compute the direction vector of the line
            dx = target_point[0] - handle_point[0]
            dy = target_point[1] - handle_point[1]
            angle = math.atan2(dy, dx)

            # Shorten the target point by the length of the arrowhead
            shortened_target_point = (
                target_point[0] - arrow_head_length * math.cos(angle),
                target_point[1] - arrow_head_length * math.sin(angle),
            )

            # Draw the arrow (main line)
            draw.line([tuple(handle_point), shortened_target_point], fill='white', width=3)

            # Compute the points for the arrowhead
            arrow_point1 = (
                target_point[0] - arrow_head_length * math.cos(angle - math.pi / 6),
                target_point[1] - arrow_head_length * math.sin(angle - math.pi / 6),
            )

            arrow_point2 = (
                target_point[0] - arrow_head_length * math.cos(angle + math.pi / 6),
                target_point[1] - arrow_head_length * math.sin(angle + math.pi / 6),
            )

            # Draw the arrowhead
            draw.polygon([tuple(target_point), arrow_point1, arrow_point2], fill='white')
    return np.array(img)


def create_circular_mask(
    h: int,
    w: int,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None,
) -> torch.Tensor:
    """
    Create a circular mask tensor.

    Args:
        h (int): The height of the mask tensor.
        w (int): The width of the mask tensor.
        center (Optional[Tuple[int, int]]): The center of the circle as a tuple (y, x). If None, the middle of the image is used.
        radius (Optional[int]): The radius of the circle. If None, the smallest distance between the center and image walls is used.

    Returns:
        A boolean tensor of shape [h, w] representing the circular mask.
    """
    if center is None:  # use the middle of the image
        center = (int(h / 2), int(w / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h - center[0], w - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((Y - center[0]) ** 2 + (X - center[1]) ** 2)

    mask = dist_from_center <= radius
    mask = torch.from_numpy(mask).bool()
    return mask


def create_square_mask(
    height: int, width: int, center: list, radius: int
) -> torch.Tensor:
    """Create a square mask tensor.

    Args:
        height (int): The height of the mask.
        width (int): The width of the mask.
        center (list): The center of the square mask as a list of two integers. Order [y,x]
        radius (int): The radius of the square mask.

    Returns:
        torch.Tensor: The square mask tensor of shape (1, 1, height, width).

    Raises:
        ValueError: If the center or radius is invalid.
    """
    if not isinstance(center, list) or len(center) != 2:
        raise ValueError(
            f"Invalid center: {center}. Center should be a list with two elements representing Y and X coordinates respectively."
        )
    if not isinstance(radius, int) or radius <= 0:
        raise ValueError(
            f"Invalid radius: {radius}. Radius should be a positive integer value."
        )
    
    # NOTE: Remember that center[0] is (y, _) and center[1] is (_, x)
    x_lower_bound, x_upper_bound = radius, (width - radius)
    y_lower_bound, y_upper_bound = radius, (height - radius)
    is_center_x_out_of_bounds = (center[0] < y_lower_bound) or (center[0] >= y_upper_bound)
    is_center_y_out_of_bounds = (center[1] < x_lower_bound) or center[1] >= x_upper_bound
    if is_center_x_out_of_bounds or is_center_y_out_of_bounds:
        eplison_percentage = 2
        source_x_percentage_lower_bound = int((radius / width) * 100) + eplison_percentage
        source_x_percentage_upper_bound = int((x_upper_bound / width) * 100) - eplison_percentage
        source_y_percentage_lower_bound = int((y_lower_bound / height) * 100) + eplison_percentage
        source_y_percentage_upper_bound = int((y_upper_bound / height) * 100) - eplison_percentage
        error_message = (
            f"Invalid center and radius values. Current values: center = ({center[0]:.2f}, {center[1]:.2f}), radius = {radius:.2f}. "
            f"The center's X coordinate must be in the range [{x_lower_bound:.2f}, {x_upper_bound:.2f}]. "
            f"The center's Y coordinate must be in the range [{y_lower_bound:.2f}, {y_upper_bound:.2f}]. "
            f"`source_x_percentage` must be a value between [{source_x_percentage_lower_bound}%, {source_x_percentage_upper_bound}%] of the total width. "
            f"`source_y_percentage` must be a value between [{source_y_percentage_lower_bound}%, {source_y_percentage_upper_bound}%] of the total height. "
            f"Please adjust the `source_x_percentage` and `source_y_percentage` values."
        )
        raise ValueError(error_message)

    mask = torch.zeros((height, width), dtype=torch.float32)
    x1 = int(center[1]) - radius
    x2 = int(center[1]) + radius
    y1 = int(center[0]) - radius
    y2 = int(center[0]) + radius
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1.0
    return mask.bool()
