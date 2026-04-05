from typing import Union, Tuple, Optional, Dict
import numpy as np
from pyntcloud import PyntCloud
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.font_manager import FontProperties
from scipy.spatial.transform import Rotation as R

from .scene_structure import House, Region
from .scene_utils import OrientedBoundingBox, AxisAlignedBoundingBox


def load_point_cloud(ply_path: str, ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    cloud = PyntCloud.from_file(ply_path)
    points = cloud.points[['x', 'y', 'z']].to_numpy()
    colors = cloud.points[['red', 'green', 'blue']].to_numpy()
    if ratio < 1.0:
        n_points = int(len(points) * ratio)
        indices = np.random.choice(len(points), size=n_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    return points, colors


def voxel_down_sample(
    points: np.ndarray,
    voxel_size: float,
    *,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    method: str = "centroid",          # "centroid" 或 "first"
    return_inverse: bool = False,
    return_voxel_coords: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """ Voxel downsample the point cloud, similar to open3d.voxel_down_sample. """
    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[1] == 3
    assert voxel_size > 0, "voxel_size must be > 0"
    N = points.shape[0]

    if colors is not None:
        colors = np.asarray(colors)
        assert colors.shape == (N, 3), f"colors must align with points (N,3), got {colors.shape}"
    if normals is not None:
        normals = np.asarray(normals)
        assert normals.shape == (N, 3), f"normals must align with points (N,3), got {normals.shape}"

    v = np.floor(points / voxel_size).astype(np.int64)  # (N,3)

    unique_v, inv, counts = np.unique(v, axis=0, return_inverse=True, return_counts=True)
    K = unique_v.shape[0]

    extras: Dict[str, np.ndarray] = {"counts": counts}

    if method == "first": 
        # for each voxel, take the first point that falls into it (order determined by np.unique)
        first_idx = np.full(K, -1, dtype=np.int64)
        for i, k in enumerate(inv):
            if first_idx[k] < 0:
                first_idx[k] = i
        down_pts = points[first_idx]
        if colors is not None:
            extras["colors"] = colors[first_idx]
        if normals is not None:
            extras["normals"] = normals[first_idx]

    elif method == "centroid":
        # for each voxel, compute the centroid of points that fall into it. Use np.add.at for efficient accumulation without Python loops.
        down_pts = np.zeros((K, 3), dtype=points.dtype)
        np.add.at(down_pts, inv, points)
        down_pts /= counts[:, None]

        if colors is not None:
            acc_c = np.zeros((K, 3), dtype=colors.dtype)
            np.add.at(acc_c, inv, colors)
            extras["colors"] = acc_c / counts[:, None]

        if normals is not None:
            acc_n = np.zeros((K, 3), dtype=normals.dtype)
            np.add.at(acc_n, inv, normals)
            n = acc_n / counts[:, None]
            # avoid division by zero for empty voxels
            norm = np.linalg.norm(n, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            extras["normals"] = n / norm
    else:
        raise ValueError("method must be 'centroid' or 'first'")

    if return_inverse:
        extras["inverse_indices"] = inv
    if return_voxel_coords:
        extras["voxel_coords"] = unique_v

    return down_pts, extras


def points_within_bbox(points: np.ndarray, 
                       bbox: Union[OrientedBoundingBox, AxisAlignedBoundingBox], 
                       expand:float=0.0, 
                       is_strict: bool = True) -> np.ndarray:
    """ Judge which points in the point cloud are within the given bounding box (optionally expanded).
    
    Args:
        points : (N,3) point cloud
        bbox : OrientedBoundingBox or AxisAlignedBoundingBox
        expand : float, the size of expansion in each direction
    """
    expand = np.array([expand, expand, 0], dtype=np.float32)
    if is_strict:
        mask = np.all(np.logical_and(points >= bbox.min_bound - expand, 
                                    points <= bbox.max_bound + expand), axis=1)
    else: 
        min_x, min_y, bbox_min_z = bbox.min_bound - expand
        max_x, max_y, bbox_max_z = bbox.max_bound + expand
        min_z = min(bbox_min_z, points[:, 2].min()) - expand[0]
        max_z = bbox_max_z + expand[0]
        min_bound = np.array([min_x, min_y, min_z], dtype=np.float32)
        max_bound = np.array([max_x, max_y, max_z], dtype=np.float32)
        mask = np.all(np.logical_and(points >= min_bound, points <= max_bound), axis=1)
    mask = np.all(np.logical_and(points >= bbox.min_bound - expand, 
                                 points <= bbox.max_bound + expand), axis=1)
    return mask


def not_ceiling_points(points: np.ndarray, 
                       bbox: Union[OrientedBoundingBox, AxisAlignedBoundingBox], 
                       height_threshold: float = 0.5):
    """Filter out points that are close to the ceiling of the bounding box.
    
    Params:
        points : (N,3) point cloud
        bbox : OrientedBoundingBox or AxisAlignedBoundingBox
        height_threshold : float keep points that are at least this far below the ceiling
    """
    ceiling_z = min(bbox.max_bound[2], points[:, 2].max())  # ceiling height cannot exceed the highest point in the point cloud
    mask = points[:, 2] <= (ceiling_z - height_threshold)
    return mask


def get_top_down_map(points:np.ndarray, colors:np.ndarray, grid_size:float, background="white") -> np.ndarray:
    """ Firstly mapping the points to a 2D grid, only keep the top points in each grid cell  """
    P = points.astype(np.float64, copy=False)
    C = colors
    if 0 <= C.max() <= 255: C = C.astype(np.uint8, copy=False)

    # calculate the x/y boundaries and grid width/height
    x_min, y_min = P[:, 0].min(), P[:, 1].min()
    x_max, y_max = P[:, 0].max(), P[:, 1].max()
    W = int(np.floor((x_max - x_min) / grid_size)) + 1
    H = int(np.floor((y_max - y_min) / grid_size)) + 1
    
    # mapping to grid coordinates (j -- x direction, i -- y direction)
    j = np.floor((P[:, 0] - x_min) / grid_size).astype(np.int64)  # [0, W-1]
    i = np.floor((P[:, 1] - y_min) / grid_size).astype(np.int64)  # [0, H-1]
    np.clip(j, 0, W - 1, out=j)
    np.clip(i, 0, H - 1, out=i)

    lin = i * W + j  # linear index
    order = np.lexsort((P[:, 2], lin))  # sort by lin, then by height (z)
    lin_sorted = lin[order]
    uniq_lin, first_idx, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
    last_idx = first_idx + counts - 1

    top_indices = order[last_idx]  # indices of top points in original array
    top_i = i[top_indices]
    top_j = j[top_indices]

    if background == "white":
        rgb_image = np.ones((H, W, 3), dtype=np.uint8) * 255    # white background
    elif background == "black":
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)          # black background
    else: raise ValueError("background must be 'white' or 'black'")
    rgb_image[top_i, top_j] = C[top_indices]

    return rgb_image, (x_min, y_min), (W, H)


# =======================================================================
def draw_top_down_map(
        points: np.ndarray, colors: np.ndarray, agent_position: np.ndarray, house: House, 
        grid_size: float = 0.025, p_stride: int = 5, h_threshold: float = 0.5, 
        font_path: Optional[str] = None, dpi: int = 100, 
        region_color: str = 'blue', **kwargs
    ):
    inside_room: Region = house.check_room_membership(agent_position)
    level = house.get_level_by_id(inside_room.level)
    majority_z_max = level.majority_z_range[1]

    # filter points within the level's regions
    point_indices = np.zeros(points.shape[0], dtype=bool)
    for region in level.regions:
        region_mask = points_within_bbox(points, region.obb, expand=0.05)
        region_mask = region_mask & (points[:, 2] <= majority_z_max + 0.01)
        region_mask = region_mask & not_ceiling_points(points, region.obb, h_threshold)
        point_indices = np.logical_or(point_indices, region_mask)
    points, colors = points[point_indices], colors[point_indices]
    if p_stride > 1: # downsample for faster plotting
        shuffled_indices = np.random.permutation(points.shape[0])
        points, colors = points[shuffled_indices], colors[shuffled_indices]
        points, colors = points[::p_stride], colors[::p_stride]
    print("Points within level regions:", points.shape)

    # draw the point cloud top-down map
    img_array, origin, (W, H) = get_top_down_map(points, colors, grid_size)
    img = Image.fromarray(img_array)
    print("Top-down map size: ", img.size)

    # draw room boundaries and labels using matplotlib for better text rendering
    # prepare canvas
    fig_w, fig_h = W // dpi, H // dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])  # full figure
    ax.imshow(img_array, origin='lower', interpolation='nearest')
    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.axis('off')

    # agent coordinates transform (world -> image)
    agent_radius = max(W, H) // 100  # set agent radius relative to map size
    agent_i = int((agent_position[0] - origin[0]) / grid_size)
    agent_j = int((agent_position[1] - origin[1]) / grid_size)

    # draw rooms
    font_size = max(W, H) // 80
    font = FontProperties(size=font_size) if font_path is None else \
        FontProperties(fname=font_path, size=font_size)
    for region in level.regions:
        room_bound_points = region.obb.get_box_points()
        room_x = ((room_bound_points[:, 0] - origin[0]) / grid_size).astype(int)
        room_y = ((room_bound_points[:, 1] - origin[1]) / grid_size).astype(int)
        imin, imax = max(room_x.min(), 0), min(room_x.max(), W-1)
        jmin, jmax = max(room_y.min(), 0), min(room_y.max(), H-1)
        rect = Rectangle(
            (imin, jmin), width=(imax - imin), height=(jmax - jmin), 
            linewidth=2, edgecolor=region_color, facecolor='none')
        ax.add_patch(rect)

        icent = (imin + imax) / 2.0
        jcent = (jmin + jmax) / 2.0
        room_label = region.label if region.label is not None else "unknown"
        ax.text(icent, jcent, room_label, fontproperties=font,
                ha='center', va='center', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # draw the agent position
    agent_circle = Circle(
        (agent_i, agent_j), radius=agent_radius,
        facecolor='red', edgecolor='yellow', linewidth=2)
    ax.add_patch(agent_circle)

    # draw the agent angle (optional)
    if 'agent_yaw' in kwargs:
        agent_yaw = kwargs['agent_yaw']  # in degrees
        yaw_rad = np.deg2rad(agent_yaw)
        arrow_length = agent_radius * 1.5
        dx = arrow_length * np.cos(yaw_rad)    
        dy = arrow_length * np.sin(yaw_rad)
        print("dx = ", dx / arrow_length, "dy = ", dy / arrow_length)
        ax.arrow(
            agent_i, agent_j, dx, dy,
            head_width=agent_radius, head_length=agent_radius * 0.7,
            fc='red', ec='red', linewidth=agent_radius//4)

    # convert back to PIL image
    map_img = fig_to_pil_image(fig)
    return map_img


def fig_to_pil_image(fig: plt.Figure, W: int, H: int) -> Image.Image:
    """ Convert a matplotlib figure to a PIL Image. """
    buf = BytesIO()
    fig.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    pil_img = Image.open(buf).convert('RGB')
    plt.close(fig)
    buf.close()
    
    pil_img = pil_img.resize((W, H))
    return pil_img


# =======================================================================

def compute_iou_2d(bbox1: np.ndarray, bbox2: np.ndarray) -> Union[float, np.ndarray]:
    """ Compute 2D IoU between two array of bounding boxes. 
    
    Inputs: 
        bbox1, bbox2 : (N, 4) arrays of (x_min, y_min, x_max, y_max) 
    """
    b1 = np.asarray(bbox1, dtype=float)
    b2 = np.asarray(bbox2, dtype=float)

    # Ensure shape is (N, 4)
    if b1.ndim == 1: b1 = b1[None, :]
    if b2.ndim == 1: b2 = b2[None, :]

    # Intersection box
    x_min = np.maximum(b1[:, 0], b2[:, 0])
    y_min = np.maximum(b1[:, 1], b2[:, 1])
    x_max = np.minimum(b1[:, 2], b2[:, 2])
    y_max = np.minimum(b1[:, 3], b2[:, 3])

    inter_w = np.clip(x_max - x_min, a_min=0.0, a_max=None)
    inter_h = np.clip(y_max - y_min, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    # Areas of the boxes
    area1 = np.clip((b1[:, 2] - b1[:, 0]), a_min=0.0, a_max=None) * \
            np.clip((b1[:, 3] - b1[:, 1]), a_min=0.0, a_max=None)
    area2 = np.clip((b2[:, 2] - b2[:, 0]), a_min=0.0, a_max=None) * \
            np.clip((b2[:, 3] - b2[:, 1]), a_min=0.0, a_max=None)

    union = area1 + area2 - inter_area
    # Avoid division by zero
    iou = inter_area / np.clip(union, a_min=1e-12, a_max=None)

    # If user passed single boxes, return a scalar
    if iou.size == 1: return float(iou[0])
    
    return iou
