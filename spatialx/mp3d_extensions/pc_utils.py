from typing import Union, Tuple, List, Optional, Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation as R

SIM2PLY_ROTATION = np.eye(3)  # Identity matrix
PLY2SIM_ROTATION = np.eye(3)  # Identity matrix


def covert_sim_to_ply_points(points: np.ndarray):
    return points


def covert_ply_to_sim_points(points: np.ndarray):
    return points


def obtain_sim_rotation(heading: float, elevation: float) -> np.ndarray:
    """Obtain rotation in Mattport Sim coordinate system. 
    
    Args:
        heading: float, angle in radians, which is defined from 
                 the y-axis with the z-axis up (turning right is positive). 
        elevation: float, angle in radians, which is measured from 
                   the horizon defined by the x-y plane (up is positive). 
    Returns:
        rotation: rotation matrix in Mattport Sim coordinate system.
    """
    # 1. convert the hearning to roll angles 
    # FROM to +y-axis (heading=0), turning clockwise (right) is positive
    # TO +y-axis, counterclockwise is positive
    rot_heading = R.from_euler('z', -heading)

    # 2. elevation is rotation around +x-axis, up is positive
    rot_elev = R.from_euler('x', elevation)
    
    # 3. first heading, then elevation
    rot_total = rot_heading * rot_elev

    return rot_total.as_matrix()


def obtain_ply_yaw(heading: float, elevation: float) -> float:
    """Obtain yaw angle in PLY coordinate system. 
    
    Args:
        heading: float, angle in radians, which is defined from 
                 the y-axis with the z-axis up (turning right is positive).
        elevation: float, angle in radians, which is measured from 
                   the horizon defined by the x-y plane (up is positive).
        
    Returns:
        yaw: float, angle in degrees relative to +X axis in PLY coordinate system.
             Counterclockwise is positive.
    """
    rot_heading = R.from_euler('z', np.pi/2 - heading)
    roll, pitch, yaw = rot_heading.as_euler('xyz', degrees=True)
    return yaw


def angle_from_y_axis_turn_right_positive(x: float, y: float, z: float) -> np.float64:
    """
    计算点 (x, y, z) 与 +Y 轴的夹角, 右转为正
    单位: randians
    """
    return np.arctan2(x, y, dtype=np.float64)   # 注意顺序是 (x, y)，不是 (y, x)


def angle_with_xy_plane_turn_up_positive(x: float, y: float, z: float) -> np.float64:
    """
    Camera elevation is measured from the horizon defined by the x-y plane (up is positive)
    计算点 (x, y, z) 与 x-y 平面的夹角, 上转为正
    单位: randians
    """
    return np.arctan2(z, np.linalg.norm(np.array([x, y]), ord=2), dtype=np.float64)


def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        x, y, z: Cartesian coordinates.
        
    Returns:
        theta: heading angle in radians
        phi: elevation angle in radians
        d: distance from the origin
    """
    theta = angle_from_y_axis_turn_right_positive(x, y, z)
    phi = angle_with_xy_plane_turn_up_positive(x, y, z)

    d = np.linalg.norm(np.array([x, y, z]), ord=2)
    return theta, phi, d



def hfov_to_vfov(hfov_deg, width, height):
    """ Convert horizontal field of view to vertical field of view."""
    aspect = width / height
    hfov_rad = np.radians(hfov_deg)
    vfov_rad = 2 * np.arctan((1 / aspect) * np.tan(hfov_rad / 2))
    vfov_deg = np.degrees(vfov_rad)
    return vfov_deg


def vfov_to_hfov(vfov_deg, width, height):
    """ Convert vertical field of view to horizontal field of view."""
    aspect = width / height
    vfov_rad = np.radians(vfov_deg)
    hfov_rad = 2 * np.arctan(aspect * np.tan(vfov_rad / 2))
    hfov_deg = np.degrees(hfov_rad)
    return hfov_deg


def spherical_to_pixel(theta, phi, d, image_width, image_height, hfov_deg):
    """ Convert spherical coordinates (in the global coordinate system) 
        to pixel coordinates (in the local image coordinate system).
    
    Args:
        theta: heading angle in radians
        phi: elevation angle in radians
        d: distance from the origin
        image_width: width of the image in pixels
        image_height: height of the image in pixels
        hfov_deg: horizontal field of view in degrees
    """
    hfov_rad = np.radians(hfov_deg)
    focal_length = (image_width / 2) / np.tan(hfov_rad / 2)
    # Convert spherical to Cartesian coordinates
    x = d * np.cos(phi) * np.sin(theta)
    y = d * np.sin(phi)
    z = d * np.cos(phi) * np.cos(theta)
    # Perspective projection to image plane
    u = (x * focal_length) / z
    v = (y * focal_length) / z
    # Convert to pixel coordinates (origin at top-left corner)
    pixel_x = int(np.round(u + (image_width / 2)))
    pixel_y = int(np.round((image_height / 2) - v))
    return pixel_x, pixel_y


def cartesian_to_pixel(x, y, z, image_width, image_height, hfov_deg):
    """ Convert Cartesian coordinates (in the global coordinate system) 
        to pixel coordinates (in the local image coordinate system).
    
    Args:
        x, y, z: Cartesian coordinates
        image_width: width of the image in pixels
        image_height: height of the image in pixels
        hfov_deg: horizontal field of view in degrees
    """
    hfov_rad = np.radians(hfov_deg)
    focal_length = (image_width / 2) / np.tan(hfov_rad / 2)
    # Perspective projection to image plane
    u = (x * focal_length) / z
    v = (y * focal_length) / z
    # Convert to pixel coordinates (origin at top-left corner)
    pixel_x = int(np.round(u + (image_width / 2)))
    pixel_y = int(np.round((image_height / 2) - v))
    return pixel_x, pixel_y


