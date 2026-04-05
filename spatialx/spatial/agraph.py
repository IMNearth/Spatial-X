from typing import List, Dict, Tuple, Optional, Union, Any
import os, sys
import torch
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from .scene_structure import House, Level, Region, Object3D
from .scene_utils import AxisAlignedBoundingBox, OrientedBoundingBox
from .reader import (
    read_house, read_category,
    post_process_house_and_points,
    post_process_stairs
)


@dataclass
class SceneObjectConfig:
    """ Configuration for scene object graph construction and querying """
    layout_dir: str = field(default="data/scene_datasets/mp3d/v1/tasks/mp3d")
    category_file_path: str = field(default="data/scene_datasets/mp3d/category_spatiallm_mapping.csv")
    category_col: str = field(default="raw_category")
    visibility_radius: float = field(default=3.0)


class SceneObjectGraph(object):
    IGNORE_CATEGORIES = {"wall", "floor", "object", "ignore", "unknown", "ceiling", "remove", "other"}

    def __init__(self, config: SceneObjectConfig, scans: List[str], env_type: str = "mp3d", data_name: str="R2R"):
        self.config = config
        self.scans = scans
        self.env_type = env_type
        self.data_name = data_name

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("data/models/all-MiniLM-L6-v2", device=device)
        
        self.houses: Dict[str, House] = {}
        self.category_mapping = read_category(self.config.category_file_path, self.config.category_col)
        print(f"Loaded {len(self.category_mapping)} categories from {self.config.category_file_path}.")
        self.interested_objects = self.load_interested_objects()
        self.room_cache = defaultdict(dict)  # scan_id -> viewpoint_id -> Region
        self.prev_level_id: Optional[int] = None

    def load_interested_objects(self):
        import json
        file_path = f"data/tasks/{self.data_name}_category.jsonl"
        with open(file_path, "r") as f: 
            lines = [json.loads(l) for l in f.readlines()]

            if self.data_name == "R2R":
                return {l["path_id"]: sum(l["category"], []) for l in lines}
            elif self.data_name == "REVERIE":
                return {l["id"]: sum(l["category"], []) for l in lines}
            else: raise NotImplementedError
    
    def get_interest_objects(self, path_id:str, instruction:str) -> List[str]:
        """ Get the list of objects in the instruction """
        if path_id not in self.interested_objects:
            return []

        obj_proposals = self.interested_objects[path_id]
        obj_proposals = [obj["category"] for obj in obj_proposals]
        obj_proposals = list(set(obj_proposals))
        obj_proposals = [x for x in obj_proposals if x in instruction.lower()]
        return obj_proposals
    
    def get_relevant_categories(self, object_proposals: List[str], candidate_categories:List[str]) -> List[str]:
        """ Get the relevant categories based on the instruction and candidate categories """
        if len(object_proposals) == 0: return []

        proposal_embeddings = self.model.encode(object_proposals, 
                                                device=self.model.device,
                                                convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidate_categories, 
                                                 device=self.model.device,
                                                 convert_to_tensor=True)
        # print("Proposal embeddings shape:", proposal_embeddings.shape)
        # print("Candidate embeddings shape:", candidate_embeddings.shape)
        similarities = (proposal_embeddings @ candidate_embeddings.T).cpu().numpy() # P x C
        
        relevant_categories = []
        for i, prop in enumerate(object_proposals):
            sim_scores = similarities[i]
            sorted_indices = sim_scores.argsort()[::-1]
            if max(sim_scores) > 0.5: # reserve similar ones
                topk_indices = [idx for idx in sorted_indices if sim_scores[idx] > 0.5]
            else: topk_indices = sorted_indices[:2]
            relevant_categories.extend([candidate_categories[idx] for idx in topk_indices])
        relevant_categories = list(set(relevant_categories))
        return relevant_categories

    def load_layout(self, scan: str) -> House:
        """ Load layout data for the specified scan """
        if scan in self.houses:
            return self.houses[scan]
        layout_path = os.path.join(self.config.layout_dir, scan, f"{scan}.house")
        house = read_house(layout_path, category_mapping=self.category_mapping)
        house = post_process_house_and_points(house)[0]
        house = post_process_stairs(house)
        self.houses[scan] = house
        return house

    def get_current_level(self, scan_id: str, position: Union[np.ndarray, List[float]]) -> Tuple[int, Level, Optional[Region]]:
        """ Get the current level for the specified position """
        house = self.load_layout(scan_id)
        inside_room: Region = house.check_room_membership(np.asarray(position))

        if inside_room is not None: 
            if inside_room.label == "stairs": 
                level_id = house.check_level_membership(np.asarray(position)).id
            else: level_id = inside_room.level
        elif self.prev_level_id is not None: level_id = self.prev_level_id
        else: level_id = house.check_level_membership(np.asarray(position)).id
        
        self.prev_level_id = level_id
        level: Level = house.get_level_by_id(level_id)
        return level_id, level, inside_room

    def get_local_objects(self, scan_id: str, 
                          viewpoint_id: str, viewpoint_pos: List[float], 
                          visibility_radius:float=None) -> Dict[str, Any]:
        """ Get the list of objects that can be perceived from the current position """
        if visibility_radius is None: visibility_radius = self.config.visibility_radius

        try: 
            local_info = self.room_cache[scan_id][viewpoint_id]
            assert local_info["visible_radius"] == visibility_radius
        except Exception as e:
            level_id, level, inside_room = self.get_current_level(scan_id, viewpoint_pos)
            if inside_room is not None:
                room_bbox = inside_room.obb.get_box_points()
                max_z, min_z = np.max(room_bbox[:, 2]), np.min(room_bbox[:, 2])
                objects: List[Object3D] = inside_room.objects
            else:
                level_bbox = level.obb.get_box_points()
                max_z, min_z = np.max(level_bbox[:, 2]), np.min(level_bbox[:, 2])
                objects = [] # TODO: get all level objects within the visibility range
                for obj in level.get_all_objects():
                    if self.check_visibility(obj.obb, 
                                             viewpoint_pos, 
                                             visibility_radius, 
                                             min_z=min_z, max_z=max_z):
                        objects.append(obj)
            
            valid_objects: List[Object3D] = []
            for obj in objects: 
                if obj.category in self.IGNORE_CATEGORIES: continue
                if any(x in obj.category.lower() for x in self.IGNORE_CATEGORIES): continue
                valid_objects.append(obj)
            
            local_info = {
                "level_id": level_id,
                "room_id": inside_room.id if inside_room is not None else None,
                "room_type": inside_room.label if inside_room is not None else None,
                "z_range": (min_z, max_z),
                "local_objects": valid_objects, 
                "visible_radius": visibility_radius
            }
            self.room_cache[scan_id][viewpoint_id] = local_info

        return local_info
    
    @staticmethod
    def check_visibility(obb: OrientedBoundingBox, viewpoint_pos: np.ndarray, 
                         visibility_radius: float=3.0, min_z: float=None, max_z: float=None) -> bool:
        """ Check if the OBB is within the visibility range """
        surf_points = SceneObjectGraph.sample_obb_surface(obb, grid_size=0.1)
        if min_z is not None:  surf_points = surf_points[(surf_points[:, 2] >= min_z)]
        if max_z is not None:  surf_points = surf_points[(surf_points[:, 2] <= max_z)]
        if surf_points.shape[0] == 0: return False

        viewpoint_pos = np.asarray(viewpoint_pos)
        dists = np.linalg.norm(surf_points - viewpoint_pos[None, :], axis=1)
        if np.any(dists <= visibility_radius): return True
        return False

    @staticmethod
    def sample_obb_surface(obb: OrientedBoundingBox, grid_size: float = 0.025) -> np.ndarray:
        """ Uniformly sample points on the surface of the OBB, return an array of sampled points (N, 3) """
        hx, hy, hz = obb.half_extent
        # the range of local coordinates along each axis
        xs = np.arange(-hx, hx + 1e-8, grid_size)
        ys = np.arange(-hy, hy + 1e-8, grid_size)
        zs = np.arange(-hz, hz + 1e-8, grid_size)

        points_local = []
        # sample points on each face of the OBB
        for x in [-hx, hx]:
            yy, zz = np.meshgrid(ys, zs)
            face_points = np.stack([np.full_like(yy, x), yy, zz], axis=-1).reshape(-1, 3)
            points_local.append(face_points)
        for y in [-hy, hy]:
            xx, zz = np.meshgrid(xs, zs)
            face_points = np.stack([xx, np.full_like(xx, y), zz], axis=-1).reshape(-1, 3)
            points_local.append(face_points)
        for z in [-hz, hz]:
            xx, yy = np.meshgrid(xs, ys)
            face_points = np.stack([xx, yy, np.full_like(xx, z)], axis=-1).reshape(-1, 3)
            points_local.append(face_points)
        points_local = np.concatenate(points_local, axis=0)  # (N, 3)

        # avoid duplicate points on edges by rounding
        rounded = np.round(points_local / (grid_size * 0.5))
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        points_local = points_local[unique_indices]

        # convert local points to world coordinates
        points_world = obb.center[None, :] + points_local @ obb.R.T
        return points_world


