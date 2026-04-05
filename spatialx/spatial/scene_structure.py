from typing import List, Dict, Tuple, Optional, Any, Union, Set
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
from collections import defaultdict
from .scene_utils import (
    AxisAlignedBoundingBox, 
    OrientedBoundingBox, 
    obtain_z_rotated_obb,
    _compute_xy_iou,
)

@dataclass
class Object3D:
    """ A 3D object in the house with its properties. """
    id: int
    region: int
    category_id: int
    category: str
    obb: OrientedBoundingBox

    def __post_init__(self):
        self.label = self.category.lower().replace(" ", "_")
        self.obb = obtain_z_rotated_obb(self.obb)
    
    def rotate(self, angle_z: float):
        """ Rotate the object around Z axis by angle_z (in radians) """
        Rz = R.from_euler('z', angle_z).as_matrix()
        new_R = Rz @ self.obb.R
        self.obb.R = new_R
    
    def translate(self, translation: np.ndarray):
        """ Translate the object by the given translation vector """
        self.obb.center += translation
    
    def scale(self, scaling: float):
        """ Scale the object by the given scaling factor """
        self.obb.center *= scaling
        self.obb.extent *= scaling
    
    

@dataclass
class Portal: 
    """Portal connecting two regions"""
    id: int
    region_i: int
    region_j: int
    label: str
    obb: OrientedBoundingBox

    def rotate(self, angle_z: float):
        """ Rotate the portal around Z axis by angle_z (in radians) """
        Rz = R.from_euler('z', angle_z).as_matrix()
        new_R = Rz @ self.obb.R
        self.obb.R = new_R
    
    def translate(self, translation: np.ndarray):
        """ Translate the portal by the given translation vector """
        self.obb.center += translation
    
    def scale(self, scaling: float):
        """ Scale the portal by the given scaling factor """
        self.obb.center *= scaling
        self.obb.extent *= scaling


@dataclass
class Region:
    """ Region within a level """
    id: int
    level: int
    label: str
    center: np.ndarray
    obb: OrientedBoundingBox
    height: float
    objects: Optional[List[Object3D]] = field(default_factory=list)
    neighbors: Optional[Dict[str, Set[int]]] = field(default_factory=dict)

    def add_object(self, obj: Object3D):
        self.objects.append(obj)
    
    def compute_xy_iou(self, other: 'Region') -> float:
        """ Compute the 2D IoU between this region and another region in the X-Y plane """
        return _compute_xy_iou(self.obb, other.obb)
    
    def contains_point(self, points: np.ndarray, atol: float = 1e-6) -> bool:
        """ Check if the region contains the given point in the space """
        if points.ndim == 1: points = points.reshape(1, -1)
        # print(points.shape, points.dtype)
        q = points - self.obb.center
        R_inv = self.obb.R.T    # Inverse rotation, as the rotation matrix R is orthogonal
        q_local = q @ R_inv     # Transform point to local OBB frame
        h = (self.obb.extent * 0.5)
        inside = np.all(np.abs(q_local) <= (h + atol), axis=1)
        if points.shape[0] == 1: return inside[0]
        return inside

    def rotate(self, angle_z: float):
        """ Rotate the region around Z axis by angle_z (in radians) """
        Rz = R.from_euler('z', angle_z).as_matrix()
        new_R = Rz @ self.obb.R
        self.obb.R = new_R
        for obj in self.objects: obj.rotate(angle_z)
    
    def translate(self, translation: np.ndarray):
        """ Translate the region by the given translation vector """
        self.obb.center += translation
        for obj in self.objects: obj.translate(translation)
    
    def scale(self, scaling: float):
        """ Scale the region by the given scaling factor """
        self.obb.center *= scaling
        self.obb.extent *= scaling
        for obj in self.objects: obj.scale(scaling)


@dataclass
class Level:
    """ Level within a house """
    id: int
    height: float
    center: np.ndarray
    obb: OrientedBoundingBox
    regions: Optional[List[Region]] = field(default_factory=list)
    extra: Optional[Dict[str, Region]] = field(default_factory=dict)

    def get_all_objects(self) -> List[Object3D]:
        all_objects = []
        for region in self.regions:
            all_objects.extend(region.objects)
        return all_objects

    def add_region(self, region: Region):
        self.regions.append(region)
    
    def rotate(self, angle_z: float):
        """ Rotate the level around Z axis by angle_z (in radians) """
        Rz = R.from_euler('z', angle_z).as_matrix()
        new_R = Rz @ self.obb.R
        self.obb.R = new_R
        for region in self.regions: region.rotate(angle_z)
    
    def translate(self, translation: np.ndarray):
        """ Translate the level by the given translation vector """
        self.obb.center += translation
        for region in self.regions: region.translate(translation)
    
    def scale(self, scaling: float):
        """ Scale the level by the given scaling factor """
        self.obb.center *= scaling
        self.obb.extent *= scaling
        for region in self.regions: region.scale(scaling)
    
    @property
    def majority_z_range(self, ) -> Tuple[float, float]:
        max_z_limits = []
        min_z_limits = []
        for region in self.regions:
            box_points = region.obb.get_box_points()
            min_z = box_points[:, 2].min()
            max_z = box_points[:, 2].max()
            # round it to 0.1 precision
            max_z = round(max_z * 10) / 10.0
            min_z = round(min_z * 10) / 10.0
            max_z_limits.append(max_z)
            min_z_limits.append(min_z)
        major_z_max = max(set(max_z_limits), key = max_z_limits.count)
        major_z_min = max(set(min_z_limits), key = min_z_limits.count)
        majority_z = (major_z_min, major_z_max)
        return majority_z

    def contains_point(self, point: np.ndarray) -> bool:
        """ Check if the level contains the given point in the space """
        z_range = self.majority_z_range
        return z_range[0] <= point[2] <= z_range[1]

@dataclass
class House:
    scan_id: str
    obb: OrientedBoundingBox
    levels: Optional[List[Level]] = field(default_factory=list)
    portals: Optional[List[Portal]] = field(default_factory=list)

    def get_all_rooms(self) -> List[Region]:
        all_rooms = []
        for level in self.levels:
            all_rooms.extend(level.regions)
        return all_rooms

    def add_portal(self, portal: Portal):
        self.portals.append(portal)

    def add_level(self, level: Level):
        self.levels.append(level)
    
    def num_levels(self) -> int:
        return len(self.levels)
    
    def rotate(self, angle_z: float):
        """ Rotate the house around Z axis by angle_z (in radians) """
        Rz = R.from_euler('z', angle_z).as_matrix()
        new_R = Rz @ self.obb.R
        self.obb.R = new_R
        for level in self.levels: level.rotate(angle_z)
        for portal in self.portals: portal.rotate(angle_z)
    
    def translate(self, translation: np.ndarray):
        """ Translate the house by the given translation vector """
        self.obb.center += translation
        for level in self.levels: level.translate(translation)
        for portal in self.portals: portal.translate(translation)
    
    def scale(self, scaling: float):
        """ Scale the house by the given scaling factor """
        self.obb.center *= scaling
        self.obb.extent *= scaling
        for level in self.levels: level.scale(scaling)
        for portal in self.portals: portal.scale(scaling)
    
    def get_level_by_id(self, level_id: int) -> Optional[Level]:
        for level in self.levels:
            if level.id == level_id:
                return level
        return None

    def check_room_membership(self, point: np.ndarray) -> Optional[Region]:
        """Check which room the point belongs to."""
        for level in self.levels:
            for room in level.regions:
                if room.contains_point(point):
                    return room
        return None
    
    def check_level_membership(self, point: np.ndarray) -> Optional[Level]:
        """Check which level the point belongs to."""
        matched_level = None
        all_z_range = {}
        for level in self.levels:
            z_range = level.majority_z_range
            all_z_range[level.id] = z_range
            if z_range[0] <= point[2] <= z_range[1]:
                matched_level = level
                break

        if matched_level is None:
            min_dist, nearest_id = float('inf'), None
            for level_id, z_range in all_z_range.items():
                dist_to_level = min(abs(point[2] - z_range[0]), abs(point[2] - z_range[1]))
                if dist_to_level < min_dist:
                    min_dist, nearest_id = dist_to_level, level_id
            matched_level = self.get_level_by_id(nearest_id)
        
        return matched_level
