""" Scene graph mapper for drawing top-down maps with scene structure and agent state. """
from typing import List, Dict, Tuple, Optional, Union, Any
import os, sys
import numpy as np
from PIL import Image, ImageChops
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.font_manager import FontProperties
import cv2

from .scene_structure import House, Level, Region
from .scene_utils import AxisAlignedBoundingBox, OrientedBoundingBox
from .reader import (
    read_house, read_category,
    post_process_house_and_points,
    post_process_stairs,
)
from .point_cloud import (
    load_point_cloud,
    points_within_bbox,
    not_ceiling_points,
    get_top_down_map,
    fig_to_pil_image,
)

import platform
current_os = platform.system()
if current_os == "Darwin":
    MP3D_SCENE_DIR = "data/scene_datasets"
elif current_os == "Linux":
    MP3D_SCENE_DIR = "data/scene_datasets/mp3d/v1/tasks"
else: # 其他系统
    print(f"Warning: Unrecognized OS '{current_os}', please check the sys.path configuration")
    sys.exit(1)


@dataclass
class SceneGraphConfig:
    """ 配置场景图相关参数 """
    
    point_cloud_dir: str = field(default=f"{MP3D_SCENE_DIR}/mp3d")
    layout_dir: str = field(default=f"{MP3D_SCENE_DIR}/mp3d")
    category_file_path: str = field(default=f"{MP3D_SCENE_DIR}/mp3d/category_spatiallm_mapping.csv")
    category_col: str = field(default="raw_category")

    global_map_width: int = field(default=1024)     # Global map minimum width (pixels)
    global_map_height: int = field(default=1024)    # Global map minimum height (pixels)
    global_map_limit: float = field(default=2048)   # Global map maximum pixels
    use_global_limit: bool = field(default=False)   # Whether to use the global map maximum pixel limit

    h_threshold: float = field(default=0.5)         # Ceiling height threshold
    p_stride: int = field(default=1)                # Point cloud down-sampling interval
    grid_size: float = field(default=0.010)         # Map grid size (meters)
    dpi: int = field(default=100)                   # Map DPI

    # extra settings for visualization
    do_ctrl_max_level_height: bool = field(default=True)  # Whether to control maximum floor height
    do_ignore_junk_room: bool = field(default=True)       # Whether to ignore junk rooms
    do_extra_stairs: bool = field(default=True)           # Whether to extra process stairs rooms

    max_agent_radius: int = field(default=18)           # Maximum agent radius (pixels)
    agent_orient_length: float = field(default=2.0)     # Agent orientation arrow length ratio
    draw_orient: bool = field(default=True)             # Whether to draw orientation arrows
    draw_room: bool = field(default=True)               # Whether to draw room information
    draw_room_bounds: bool = field(default=False)       # Whether to draw room boundaries
    draw_room_labels: bool = field(default=True)        # Whether to draw room labels
    max_room_font_size: int = field(default=24)         # Maximum room label font size
    draw_history: bool = field(default=False)           # Whether to draw history trajectories
    history_scale: float = field(default=1.0)           # History trajectory size
    draw_history_index: bool = field(default=False)     # Whether to draw history trajectory index
    draw_navigable: bool = field(default=True)          # Whether to draw navigation candidates
    draw_navigable_index: bool = field(default=False)   # Whether to draw navigation candidate index
    draw_navigable_id: bool = field(default=False)      # Whether to draw navigation candidate ID
    navigable_scale: float = field(default=0.8)         # Navigation candidate size
    # rotation settings
    agent_front_up: bool = field(default=False)          # Agent front always up or not
    rotation_strategy: str = field(default="absolute")   # Rotation strategy, absolute / relative
    map_fill_color: str = field(default="white")         # Map rotation fill color, white / black

    # cropping settings
    crop_map: bool = field(default=False)               # Whether to crop the map
    crop_width: int = field(default=1024)               # Cropped Map width (pixels)
    crop_height: int = field(default=1024)              # Cropped Map height (pixels)

    # ablation settings
    use_pred_layouts: bool = field(default=False)       # Whether to use predicted layouts



class SceneGraphMapper:
    """ Scene graph mapper for loading and processing scene graph data (lazy_load) """
    delete_pcd = True
    
    def __init__(self, config:SceneGraphConfig, scans: List[str], env_type: str = "mp3d"):
        self.config = config
        self.scans = scans
        self.env_type = env_type

        self.houses: Dict[str, House] = {}
        self.map_data: Dict[str, Dict[str, Image.Image]] = defaultdict(dict)
        self.category_mapping = read_category(self.config.category_file_path, self.config.category_col)
        print(f"Loaded {len(self.category_mapping)} categories from {self.config.category_file_path}.")
        
        self.prev_level_id: Optional[int] = None
    
    def load_layout(self, scan: str) -> House:
        """ Load layout data for the specified scan """
        if scan in self.houses:
            return self.houses[scan]
        layout_path = os.path.join(self.config.layout_dir, scan, f"{scan}.house")
        if self.config.use_pred_layouts:
            import pickle as pkl
            # load_path = f"outputs/mp3d_objects/processed_houses/{scan}_house_labeled.pkl"
            load_path = f"outputs/ablation_anno/processed_houses_unseen/{scan}_house.pkl"
            house = pkl.load(open(load_path, "rb"))
            print(f"Loaded predicted layout for scan {scan} from {load_path}.")
        else: house = read_house(layout_path, category_mapping=self.category_mapping)

        self.houses[scan] = house
        return house
    
    def load_point_cloud(self, scan: str) -> Tuple[np.ndarray, np.ndarray]:
        """ Load point cloud data for the specified scan """
        pc_path = os.path.join(self.config.point_cloud_dir, scan, f"{scan}_semantic.ply")
        points, colors = load_point_cloud(pc_path)
        return points, colors
    
    def load_cache_point_cloud(self, scan: str) -> Tuple[np.ndarray, np.ndarray]:
        """ Load cached point cloud data for the specified scan """
        return self.load_point_cloud(scan)
    
    @staticmethod
    def special_scan_handling(scan: str, level_id: int, points: np.ndarray, colors: np.ndarray, env_type:str=None) -> Tuple[np.ndarray, np.ndarray]:
        """ Handle special scans """
        special_cases = {
            "82sE5b5pLXE": {"level_0": {"min": np.array([-16.17090034,  24.67939949,   1.45702004]),
                                        "max": np.array([-16.17090034,  24.67939949,   1.45702004])},},
            "cV4RVeZvu5T": {"level_1": {"min": np.array([1.5675,     5.73283005, 4.53318977]),
                                        "max": np.array([1.5675,     5.73283005, 4.53318977])},},
            "qoiz87JEwZ2": {"level_0": {"min": np.array([-4.26831007, 19.47430038, -1.60219002]),
                                        "max": np.array([-3.77608991, 19.57430077, -1.59661996])},},
            "uNb9QFRL6hY": {"level_0": {"min": np.array([-13.06140041,  29.44849968,   3.74470997]),
                                        "max": np.array([-13.06140041,  29.44849968,   3.74470997])},},
            "SN83YJsR3w2": {"level_2": {"min": np.array([ -3.29784012, -12.85879993,   5.3882699 ]),
                                        "max": np.array([ -3.29784012, -12.85879993,   5.3882699 ])},
                            "level_1": {"min": np.array([-12.08909988,  -7.57412004,  -0.25983801]),
                                        "max": np.array([-9.04333973, -5.55390978,  0.31209201])},},
            "p5wJjkQkbXX": {"level_1": {"min": np.array([-0.115852,   -6.33242989,  1.23716998]),
                                        "max": np.array([-0.115852,   -6.33242989,  1.23716998])},},
            "5LpN3gDmAk7": {"level_1": {"min": np.array([-0.67255998, -1.57142997,  4.45804024]),
                                        "max": np.array([-0.67255998, -1.57142997,  4.45804024])},},
            "aayBHfsNo7d": {"level_0": {"min": np.array([-1.47528994, -3.91816998,  1.47530997]),
                                        "max": np.array([-1.47528994, -3.91816998,  1.47530997])},},
            "r1Q1Z4BcV1o": {"level_0": {"min": np.array([ -7.06557989, -19.05960083,   1.00130999]),
                                        "max": np.array([ 5.37904978, 20.11499977,  1.60494006])},}, 
            "sKLMLpTHeUy": {"level_1": {"min": np.array([15.75669956, -1.97792006,  1.54086006]),
                                        "max": np.array([15.75669956, -1.97792006,  1.54086006])},},
            "gTV8FGcVJC9": {"level_1": {"min": np.array([ 13.8920002,  -13.95839977,   2.56426001]),
                                        "max": np.array([ 13.8920002,  -13.95839977,   2.56426001])},},
            "e9zR4mvMWw7": {"level_1": {"min": np.array([0.110752001, -0.0861247032, 1.3526799678]),
                                        "max": np.array([0.110752001, -0.0861247032, 1.3526799678])},},
        }
        if scan not in special_cases: return points, colors
        
        level_key = f"level_{level_id}"
        if level_key not in special_cases[scan]: return points, colors
        to_add_colors = np.array([[255, 255, 255], [255, 255, 255]])  # white color for added points
        
        min_bound = special_cases[scan][level_key]["min"]
        max_bound = special_cases[scan][level_key]["max"]
        to_add_points = np.array([
            [min_bound[0]-1.0, min_bound[1]-1.0, min_bound[2]],
            [max_bound[0]+1.0, max_bound[1]+1.0, max_bound[2]],
        ])
        points = np.vstack([points, to_add_points])
        colors =  np.vstack([colors, to_add_colors])
        return points, colors

    def load_scan_map(self, 
        scan: str,
        is_pred_pcd: bool = False
    ) -> Dict[str, Tuple[Image.Image, Tuple[float, float], Tuple[float, float]]]:
        """ Load map data for the specified scan """
        if scan in self.map_data:
            return self.map_data[scan]
        
        house = self.load_layout(scan)
        points, colors = self.load_cache_point_cloud(scan)
        if not self.config.use_pred_layouts: 
            house, points, colors = post_process_house_and_points(house, points, colors)
        if self.config.do_extra_stairs: house = post_process_stairs(house)

        # print(f"Generating top-down maps for scan {scan} with {len(house.levels)} levels.")
        for level in house.levels:
            level_id = level.id
            majority_z_max = level.majority_z_range[1]

            point_indices = np.zeros(points.shape[0], dtype=bool)
            for region in level.regions:
                if self.config.do_ignore_junk_room and region.label == "junk": continue
                region_mask = points_within_bbox(points, region.obb, expand=0.10, is_strict=(not is_pred_pcd))
                if self.config.do_ctrl_max_level_height and region.label != "stairs":
                    region_mask = region_mask & (points[:, 2] <= majority_z_max + 0.01)
                region_mask = region_mask & not_ceiling_points(points, region.obb, self.config.h_threshold)
                point_indices = np.logical_or(point_indices, region_mask)
            if self.config.do_extra_stairs:
                for region in level.extra.get("stairs", []):
                    region_mask = points_within_bbox(points, region.obb, expand=0.10, is_strict=(not is_pred_pcd))
                    region_mask = region_mask & not_ceiling_points(points, region.obb, self.config.h_threshold)
                    point_indices = np.logical_or(point_indices, region_mask)
            level_points, level_colors = points[point_indices], colors[point_indices]
            
            if self.config.p_stride > 1: # downsample for faster plotting
                shuffled_indices = np.random.permutation(level_points.shape[0])
                level_points, level_colors = level_points[shuffled_indices], level_colors[shuffled_indices]
                level_points, level_colors = level_points[::self.config.p_stride], level_colors[::self.config.p_stride]
            
            # added special handling for specific scans
            level_points, level_colors = self.special_scan_handling(
                scan, level_id, level_points, level_colors, env_type=self.env_type)

            img_array, origin, (W, H) = get_top_down_map(level_points, level_colors, self.config.grid_size)
            if W < self.config.global_map_width and H < self.config.global_map_height:
                if W > H: scale_factor = self.config.global_map_width / W
                else: scale_factor = self.config.global_map_height / H
                cur_grid_size = self.config.grid_size / scale_factor
                img_array, origin, (W, H) = get_top_down_map(level_points, level_colors, cur_grid_size)
            elif (W > self.config.global_map_limit or H > self.config.global_map_limit) and self.config.use_global_limit:
                if W > H: scale_factor = self.config.global_map_limit / W
                else: scale_factor = self.config.global_map_limit / H
                cur_grid_size = self.config.grid_size / scale_factor
                img_array, origin, (W, H) = get_top_down_map(level_points, level_colors, cur_grid_size)
            else: cur_grid_size = self.config.grid_size
            print(f"Loaded map for scan {scan}, level {level_id}: size=({W}, {H}), grid_size={cur_grid_size:.3f}m/pixel")
            level_map = Image.fromarray(img_array)
            self.map_data[scan][f"level_{level_id}"] = (level_map, (W, H), origin, cur_grid_size)
        
        del level_points, level_colors
        if self.delete_pcd: del points, colors

        return self.map_data[scan]
    
    def get_current_level(self, scan_id: str, position: Union[np.ndarray, List[float]]):
        if self.env_type == "mp3d":
            return self.get_current_level_mp3d(scan_id, position)
        elif self.env_type == "habitat":
            raise NotImplementedError("Getting current level for Habitat is not implemented yet.")
        else: raise ValueError(f"Unsupported env_type: {self.env_type}")

    def get_current_level_mp3d(self, scan_id: str, position: Union[np.ndarray, List[float]]):
        """ Get the current level for the specified position (Matterport3D Simulator) """
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

    def post_process_habitat(self, scan_id: str, position: Union[np.ndarray, List[float]], add_to="z") -> Union[np.ndarray, List[float]]:
        raise NotImplementedError("Post-processing for Habitat is not implemented yet.")
    
    def get_visual_map(self, 
        scan_id: str, 
        position: Union[np.ndarray, List[float]], 
        orientation: Union[List[float], Tuple[float, float]],
        navigable_viewpoints: Dict[str, Dict[str, Any]]=None, 
        history_viewpoints: Dict[str, Dict[str, Any]]=None, 
    ) -> Image.Image:
        """ Obtain a visual map for the given position, orientation, and optionally navigable and history viewpoints. """
        if self.env_type == "mp3d": 
            from ..mp3d_extensions import pc_utils as mp_pc_utils
            PC_UTILS = mp_pc_utils
            heading, elevation = orientation    # in degrees
            agent_xyz = PC_UTILS.covert_sim_to_ply_points(np.asarray(position))
            agent_yaw = 90.0 - heading          # convert to PLY yaw in degrees
        else: raise ValueError(f"Unsupported env_type: {self.env_type}")

        level_id, level, __ = self.get_current_level(scan_id, agent_xyz)
        print(f"Current level for scan {scan_id} at position {position}: level_id={level_id}")
        
        if scan_id not in self.map_data: self.load_scan_map(scan_id)
        level_map, (W, H), origin, cur_grid_size = self.map_data[scan_id][f"level_{level_id}"]

        agent_radius = min(max(W, H) // 100, self.config.max_agent_radius)  # agent radius in pixels
        agent_coord = self.calc_map_coord(agent_xyz, origin, cur_grid_size)
        
        cand_radius = int(agent_radius * self.config.navigable_scale)
        if self.config.draw_navigable and navigable_viewpoints is not None:
            cand_coords = dict()
            for viewpoint_id, info in navigable_viewpoints.items():
                cand_xyz = PC_UTILS.covert_sim_to_ply_points(np.asarray(info['position']))
                if self.env_type == "habitat":
                    cand_xyz = self.post_process_habitat(scan_id, cand_xyz)
                cand_coords[viewpoint_id] = {
                    "coord": self.calc_map_coord(cand_xyz, origin, cur_grid_size),
                    "index": info.get('global_order', None)
                }
        else: cand_coords = None

        hist_radius = int(agent_radius * self.config.history_scale)
        if self.config.draw_history and history_viewpoints is not None:
            hist_coords = dict()
            for viewpoint_id, info in history_viewpoints.items():
                if info["within_same_level"] is False: continue
                hist_xyz = PC_UTILS.covert_sim_to_ply_points(np.asarray(info['position']))
                if self.env_type == "habitat":
                    hist_xyz = self.post_process_habitat(scan_id, hist_xyz)
                hist_coords[viewpoint_id] = {
                    'coord': self.calc_map_coord(hist_xyz, origin, cur_grid_size), 
                    'index': info['history_order']
                }
        else: hist_coords = None

        visual_map = self.draw_scene_map(
            level, level_map, origin, W, H, cur_grid_size, 
            agent_coord, agent_yaw, agent_radius,
            cand_coords, cand_radius,
            hist_coords, hist_radius
        )
        return visual_map

    def calc_map_coord(self, agent_xyz, map_origin, cur_grid_size):
        """ Calculate the coordinates of the agent on the map """
        agent_i = int((agent_xyz[0] - map_origin[0]) / cur_grid_size)
        agent_j = int((agent_xyz[1] - map_origin[1]) / cur_grid_size)
        return agent_i, agent_j

    def draw_scene_map(self, 
        level: Level,
        level_map: Image.Image,
        origin: Tuple[float, float],
        W: int,
        H: int,
        cur_grid_size: float,
        agent_coord: Tuple[int, int],
        agent_yaw: float,
        agent_radius: int,
        cand_coords: Optional[Dict[str, Tuple[int, int]]]=None,
        cand_radius: int=5,
        hist_coords: Optional[Dict[str, Tuple[int, int]]]=None,
        hist_radius: int=7,
    ) -> Image.Image:
        """ Draw the scene map with agent position, navigable candidates, and history viewpoints. """
        fig_w, fig_h = W // self.config.dpi, H // self.config.dpi
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.config.dpi)
        ax = plt.axes([0, 0, 1, 1])  # full figure
        ax.imshow(level_map, origin='lower', interpolation='nearest')
        ax.set_xlim([0, W])
        ax.set_ylim([0, H])
        ax.axis('off')

        # draw the agent position and orientation
        agent_i, agent_j = agent_coord
        agent_circle = Circle(
            agent_coord, radius=agent_radius, 
            facecolor='red', edgecolor='red', linewidth=1,
            zorder=10,
        )
        ax.add_patch(agent_circle)
        if self.config.draw_orient:
            yaw_rad = np.deg2rad(agent_yaw)
            arrow_length = agent_radius * self.config.agent_orient_length
            dx = arrow_length * np.cos(yaw_rad)    
            dy = arrow_length * np.sin(yaw_rad)
            ax.arrow(
                agent_i, agent_j, dx, dy,
                head_width=round(agent_radius + 2), 
                head_length=round(agent_radius * 0.8),
                fc='red', ec='red', 
                linewidth=max(agent_radius//3, 2),
                zorder=10
            )
        
        # determine rotation angle
        agent_rotate_angle = None
        if self.config.agent_front_up:
            # rotate the map so that agent front is always up
            if self.config.rotation_strategy == "absolute":
                agent_rotate_angle = self.normalize_angle(90.0 - agent_yaw)
                text_rotate_deg = agent_yaw - 90.0
            elif self.config.rotation_strategy == "relative":
                agent_rotate_angle = self.normalize_angle(90.0 + 45 - agent_yaw)
                agent_rotate_angle = agent_rotate_angle // 90 * 90  # round to nearest 90 degrees
                text_rotate_deg = agent_rotate_angle - (agent_rotate_angle % 180) // 90 * 180
            else: raise ValueError(f"Unsupported rotation_strategy: {self.config.rotation_strategy}")
        else: text_rotate_deg = 0.0
        
        # draw navigable candidates
        delta_x, delta_y = 0, 0
        if self.config.draw_navigable and cand_coords is not None:
            cand_font_size = int(cand_radius * 1.2)
            circle_edge_color = 'blue' if not self.config.draw_navigable_index else 'lightblue'
            for vp_i, (viewpoint_id, vp_info) in enumerate(cand_coords.items()):
                (cand_i, cand_j) = vp_info['coord']
                ax.plot([agent_i, cand_i], [agent_j, cand_j], 
                        linestyle='--', linewidth=2, color='blue', zorder=8)
                cand_circle = Circle((cand_i, cand_j), radius=cand_radius,
                    facecolor='cyan', edgecolor=circle_edge_color, linewidth=1, zorder=9)
                ax.add_patch(cand_circle)
                if self.config.draw_navigable_index: 
                    vp_index = vp_info['index'] if vp_info['index'] is not None else vp_i
                    ax.text(
                        cand_i, cand_j, vp_index, 
                        fontsize=cand_font_size, zorder=10,
                        ha='center', va='center', color='black',
                        rotation_mode='anchor', rotation=text_rotate_deg,
                    )
                if self.config.draw_navigable_id: 
                    ax.text(
                        cand_i, cand_j, viewpoint_id, 
                        fontsize=cand_font_size, zorder=10,
                        ha='center', va='center', color='black',
                        rotation_mode='anchor', rotation=text_rotate_deg
                    )
                delta_x = max(abs(cand_i - agent_i), delta_x)
                delta_y = max(abs(cand_j - agent_j), delta_y)
        
        # draw history viewpoints
        if self.config.draw_history and hist_coords is not None:
            sorted_hist = sorted(hist_coords.items(), key=lambda x: x[1]['index'])
            for idx in range(1, len(sorted_hist)):
                prev_id, prev_info = sorted_hist[idx - 1]
                curr_id, curr_info = sorted_hist[idx]
                pi, pj = prev_info["coord"]
                ci, cj = curr_info["coord"]
                circle = Circle((pi, pj), radius=hist_radius, 
                    facecolor='orange', edgecolor='darkorange', linewidth=3, zorder=7)
                ax.add_patch(circle)
                ax.plot([pi, ci], [pj, cj], linestyle='-', linewidth=4, color='orange')
                # print(f"Draw history point {prev_id} at ({pi}, {pj})")
                if self.config.draw_history_index: 
                    hist_index = prev_info['index']
                    ax.text(
                        pi, pj, hist_index, 
                        fontsize=cand_font_size, zorder=10,
                        ha='center', va='center', color='black',
                        rotation_mode='anchor', rotation=text_rotate_deg,
                    )
        
        # draw room boundaries and labels
        if self.config.draw_room:
            room_font_size = min(max(W, H) // 80, self.config.max_room_font_size)
            to_draw_rooms = level.regions if not self.config.do_extra_stairs else \
                            level.regions + level.extra.get("stairs", [])
            for r_idx, region in enumerate(to_draw_rooms):
                if self.config.do_ignore_junk_room and region.label == "junk": continue
                if region.label == "ignore": continue
                room_bound_points = region.obb.get_box_points()
                room_x = ((room_bound_points[:, 0] - origin[0]) / cur_grid_size).astype(int)
                room_y = ((room_bound_points[:, 1] - origin[1]) / cur_grid_size).astype(int)
                imin, imax = max(room_x.min(), 0), min(room_x.max(), W-1)
                jmin, jmax = max(room_y.min(), 0), min(room_y.max(), H-1)
                if self.config.draw_room_bounds:
                    if r_idx < len(level.regions): region_edge_color = 'grey'
                    else: region_edge_color = "red"  # extra stairs in red
                    rect = Rectangle(
                        (imin, jmin), width=(imax - imin), height=(jmax - jmin), 
                        linewidth=2, edgecolor=region_edge_color, facecolor='none', linestyle='--')
                    ax.add_patch(rect)
            
                if self.config.draw_room_labels:
                    icent = (imin + imax) / 2.0
                    jcent = (jmin + jmax) / 2.0
                    room_label = region.label if region.label is not None else "unknown"
                    ax.text(icent, jcent, room_label, fontsize=room_font_size,
                            ha='center', va='center', color='black',
                            bbox=dict(boxstyle='round,pad=0.3', 
                                      facecolor='white', alpha=0.8, edgecolor='none'),
                            rotation=text_rotate_deg, zorder=8)

        # convert back to PIL image
        scane_map = fig_to_pil_image(fig, W, H)

        # crop the map if needed
        if self.config.crop_map:
            sw, sh = scane_map.size
            # make sure the candidate points are all inside the cropped map
            cur_crop_width = max((delta_x + 50) * 2, self.config.crop_width)
            cur_crop_height = max((delta_y + 50) * 2, self.config.crop_height)
            if cur_crop_width < sw or cur_crop_height < sh:
                width = min(cur_crop_width, sw)
                height = min(cur_crop_height, sh)

                left = max(agent_i - width // 2, 0)
                right = min(agent_i + width // 2, sw)
                top = max(sh - agent_j - height // 2, 0)
                bottom = min(sh - agent_j + height // 2, sh)
                if right - left < width:
                    if left == 0: right = min(width, sw)
                    if right == sw: left = max(sw - width, 0)
                if bottom - top < height:
                    if top == 0: bottom = min(height, sh)
                    elif bottom == sh: top = max(sh - height, 0)
            
                if self.config.agent_front_up and agent_rotate_angle is not None:
                    # rorate the cropping box [left, top, right, bottom] accordingly
                    corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
                    theta = np.deg2rad(agent_rotate_angle)
                    cos_t, sin_t = np.cos(theta), np.sin(theta)
                    o = np.array([agent_i, sh - agent_j])   # rotate around agent position
                    xr = (corners[:, 0] - o[0]) * cos_t - (corners[:, 1] - o[1]) * sin_t + o[0]
                    yr = (corners[:, 0] - o[0]) * sin_t + (corners[:, 1] - o[1]) * cos_t + o[1]
                    left = max(min(int(xr.min()), left), 0)
                    right = min(max(int(xr.max()), right), sw)
                    top = max(min(int(yr.min()), top), 0)
                    bottom = min(max(int(yr.max()), bottom), sh)
                    # pts_src = np.stack([xr, yr], axis=1).astype(np.float32)  # shape (4, 2)
                    # nw = int(np.linalg.norm(pts_src[1] - pts_src[0]))
                    # nh = int(np.linalg.norm(pts_src[3] - pts_src[0]))
                    # pts_dst = np.array([[0,   0], [nw-1, 0], [nw-1, nh-1], [0,   nh-1]], dtype=np.float32)
                    # M = cv2.getPerspectiveTransform(pts_src, pts_dst)
                    # img_np = np.array(scane_map)
                    # patch = cv2.warpPerspective(img_np, M, (nw, nh))
                    # scane_map = Image.fromarray(patch)
                assert left <= agent_i <= right and top <= sh - agent_j <= bottom, \
                    f"Agent ({agent_i}, {agent_j}) must be inside the cropped map: " + \
                    f"[({left}, {top}), ({right}, {bottom})] !"
                scane_map = scane_map.crop((left, top, right, bottom))
        
        # rotate the map if needed
        if self.config.agent_front_up and agent_rotate_angle is not None:
            bg_color = (255, 255, 255) if self.config.map_fill_color == "white" else (0, 0, 0)
            scane_map = scane_map.rotate(agent_rotate_angle, expand=True, fillcolor=bg_color)
            bg = Image.new("RGB", scane_map.size, bg_color)
            diff = ImageChops.difference(scane_map, bg)
            scane_map = scane_map.crop(diff.getbbox())

        return scane_map

    @staticmethod
    def normalize_angle(angle):
        while angle > 360: angle -= 360
        while angle <= 0: angle += 360
        return angle


