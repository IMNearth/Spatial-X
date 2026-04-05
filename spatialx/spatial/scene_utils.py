from typing import List, Dict, Tuple, Optional, Any, Union, Set
import os, sys
from dataclasses import dataclass, field
from collections import defaultdict
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation as R

import numpy as np


@dataclass
class AxisAlignedBoundingBox:
    """
    Lightweight drop-in replacement for open3d.geometry.AxisAlignedBoundingBox.
    Stores min and max bounds and provides a few helper methods used in this file.
    """
    min_bound: np.ndarray
    max_bound: np.ndarray

    def __init__(self, 
        min_bound: Union[Tuple[float, float, float], List[float], np.ndarray], 
        max_bound: Union[Tuple[float, float, float], List[float], np.ndarray]
    ):
        self.min_bound = np.array(min_bound, dtype=np.float32)
        self.max_bound = np.array(max_bound, dtype=np.float32)
    
    def get_min_bound(self):
        return self.min_bound
    
    def get_max_bound(self):
        return self.max_bound
    
    def get_center(self):
        return self.center
    
    def get_extent(self):
        return self.extent

    @property
    def center(self):
        return (self.min_bound + self.max_bound) / 2
    
    @property
    def extent(self):
        return self.max_bound - self.min_bound
    
    def get_box_points(self) -> np.ndarray:
        """ Returns the 8 corner points of the bounding box. """
        corners = []
        for z in [self.min_bound[2], self.max_bound[2]]:
            for y in [self.min_bound[1], self.max_bound[1]]:
                for x in [self.min_bound[0], self.max_bound[0]]:
                    corners.append([x, y, z])
        return np.array(corners, dtype=np.float32)
    
    def get_oriented_bounding_box(self) -> 'OrientedBoundingBox':
        """ Returns the oriented bounding box representation of this AABB. """
        center = self.center
        R_mat = np.eye(3)
        extent = self.extent
        return OrientedBoundingBox(center=center, R=R_mat, extent=extent)


@dataclass
class OrientedBoundingBox:
    """
    Lightweight drop-in replacement for open3d.geometry.OrientedBoundingBox.
    Stores center (3,), rotation matrix R (3x3), and extent (3,).
    Provides get_box_points() used in this file.
    """
    center: np.ndarray
    R: np.ndarray
    extent: np.ndarray

    def __init__(self, 
        center: Union[Tuple[float, float, float], List[float], np.ndarray], 
        R: Union[np.ndarray, List[List[float]]],
        extent: Union[Tuple[float, float, float], List[float], np.ndarray]
    ):
        self.center = np.array(center, dtype=np.float32)
        self.R = np.array(R, dtype=np.float32)
        self.extent = np.array(extent, dtype=np.float32)
        # shape checking 
        assert self.R.shape == (3,3), f"Rotation matrix R must be of shape (3,3), got {self.R.shape}"
        assert self.center.shape == (3,), f"Center must be of shape (3,), got {self.center.shape}"
        assert self.extent.shape == (3,), f"Extent must be of shape (3,), got {self.extent.shape}"
    
    @property
    def half_extent(self):
        return self.extent / 2
    
    @property
    def volume(self):
        return self.extent[0] * self.extent[1] * self.extent[2]
    
    @property
    def min_bound(self):
        box_points = self.get_box_points()
        return box_points.min(axis=0)
    
    @property
    def max_bound(self):
        box_points = self.get_box_points()
        return box_points.max(axis=0)
    
    def get_box_points(self) -> np.ndarray:
        """ Returns the 8 corner points of the oriented bounding box. """
        xe, ye, ze = self.half_extent
        corners_local = np.array([
            [-xe, -ye, -ze],
            [ xe, -ye, -ze],
            [ xe,  ye, -ze],
            [-xe,  ye, -ze],
            [-xe, -ye,  ze],
            [ xe, -ye,  ze],
            [ xe,  ye,  ze],
            [-xe,  ye,  ze],
        ], dtype=np.float32)  # (8, 3)
        corners_world = (self.R @ corners_local.T).T + self.center  # (8, 3)
        return corners_world
    
    def get_min_bound(self) -> np.ndarray:
        """ Returns the minimum bound of the OBB. """
        box_points = self.get_box_points()
        return box_points.min(axis=0)
    
    def get_max_bound(self) -> np.ndarray:
        """ Returns the maximum bound of the OBB. """
        box_points = self.get_box_points()
        return box_points.max(axis=0)
    
    def get_axis_aligned_bounding_box(self) -> AxisAlignedBoundingBox:
        """ Returns the axis-aligned bounding box that contains this OBB. """
        box_points = self.get_box_points()
        min_bound = box_points.min(axis=0)
        max_bound = box_points.max(axis=0)
        return AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    

def get_bottom_points(obb: Union[OrientedBoundingBox, AxisAlignedBoundingBox]) -> np.ndarray:
    """ Get the 4 bottom points of the OBB in clockwise order"""
    box_points = np.asarray(obb.get_box_points())
    bottom_points = box_points[box_points[:, 2] == box_points[:, 2].min()]
    # sort the bottom points in clockwise order
    center_2d = bottom_points[:, :2].mean(axis=0)
    angles = np.arctan2(bottom_points[:,1]-center_2d[1], bottom_points[:,0]-center_2d[0])
    sort_idx = np.argsort(angles)
    # let the first point be the one with smallest x and y
    min_idx = np.argmin(bottom_points[sort_idx][:,0] + bottom_points[sort_idx][:,1])
    sort_idx = np.roll(sort_idx, -min_idx)
    bottom_points = bottom_points[sort_idx]
    return bottom_points


def obtain_z_rotated_obb(obb: OrientedBoundingBox) -> OrientedBoundingBox:
    """ Obtain the obb that is only rotated along Z axis """
    P = np.asarray(obb.get_box_points()) 
    z_min, z_max = P[:, 2].min(), P[:, 2].max()

    # Compute the minimum area rectangle in the XY plane
    xy_poly = Polygon(P[:, :2])
    rect = xy_poly.minimum_rotated_rectangle
    rect_xy = np.array(rect.exterior.coords)[:-1]  # 4 corners (closed loop)

    # Compute the angle of the rectangle
    edge_vec = rect_xy[1] - rect_xy[0]
    angle = np.arctan2(edge_vec[1], edge_vec[0])
    rotation = R.from_euler('z', angle).as_matrix()

    # Compute the size of the new OBB
    size_x = np.linalg.norm(rect_xy[1] - rect_xy[0])
    size_y = np.linalg.norm(rect_xy[2] - rect_xy[1])
    size_z = z_max - z_min
    scale = np.array([size_x, size_y, size_z])
    
    # Compute the center of the new OBB
    center_xy = rect_xy.mean(axis=0)
    center = np.array([center_xy[0], center_xy[1], (z_min + z_max)/2])

    new_obb = OrientedBoundingBox(center=center, R=rotation, extent=scale)
    return new_obb


def distance_to_plane(p: np.ndarray, n: np.ndarray, c: np.ndarray, signed: bool=True) -> float:
    """ Compute the distance from a point to a plane defined by (normal, center) """
    d = (p - c) @ n
    if not signed: d = abs(d)
    return d


def project_to_plane(points: np.ndarray, plane_normal: np.ndarray, plane_center: np.ndarray) -> np.ndarray:
    """ Project points onto a plane defined by (normal, center) """
    n = np.linalg.norm(plane_normal)
    if not np.isclose(n, 1.0):
        plane_normal = plane_normal / ( n + 1e-12 )
    
    if points.ndim == 1:
        points = points.reshape(1, -1)
    assert points.shape[1] == 3, "Points should be of shape (N, 3)"

    dots = (points - plane_center) @ plane_normal               # (N,)   投影长度(沿法向量的分量)
    projected_points = points - np.outer(dots, plane_normal)    # (N, 3) 投影点 = 原点 - 距离 * 法向量

    return projected_points


def _compute_xy_iou(box1: Union[AxisAlignedBoundingBox, OrientedBoundingBox], 
                   box2: Union[AxisAlignedBoundingBox, OrientedBoundingBox]) -> float:
    """ Compute the 2D IoU between two bounding boxes in the X-Y plane """
    bottom_points1 = get_bottom_points(box1)[:, :2]
    bottom_points2 = get_bottom_points(box2)[:, :2]
    
    poly1 = Polygon(bottom_points1)
    poly2 = Polygon(bottom_points2)
    inter_area = poly1.intersection(poly2).area
    if inter_area == 0: return 0.0
    
    area1 = poly1.area
    area2 = poly2.area
    iou = inter_area / (area1 + area2 - inter_area)
    return iou
