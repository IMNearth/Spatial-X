from typing import List, Dict, Tuple, Optional, Any, Union, Set
import os, sys
import pandas as pd
import numpy as np
from collections import defaultdict
from .region_labels import REGION_MAPPING, LINE_MAPPING
from .scene_utils import OrientedBoundingBox, AxisAlignedBoundingBox
from .scene_structure import Object3D, Portal, Region, Level, House
from .point_cloud import points_within_bbox
from .point_cloud import (
    points_within_bbox,
    compute_iou_2d
)



def read_line(line_str:str, line_type:Optional[str]=None) -> Union[List[Any], Dict[str, Any]]:
    """ read line of data, split by space """
    parsed_line = []
    for e in line_str.strip().split(" "):
        try: e = int(e)
        except: 
            try: e = float(e)
            except: pass
        if e != '': parsed_line.append(e)
    
    if line_type is None or line_type not in LINE_MAPPING:
        return parsed_line
    assert len(parsed_line) - 1 == len(LINE_MAPPING[line_type]), \
        f"Line length mismatch for type {line_type}: " + \
        f"expected {len(LINE_MAPPING[line_type])}, got {len(parsed_line)-1}"

    res = dict()
    for i, e in enumerate(parsed_line[1:]):
        k = LINE_MAPPING[line_type][i]
        if k == '-': continue
        else: res[k] = e
    return res


def read_category(category_file: str, category_col: str="raw_category") -> Dict[int, str]:
    """ read category mapping file, 
        return dict from category id to category name """
    assert os.path.exists(category_file), f"Category file {category_file} does not exist."
    
    if category_file.endswith(".tsv"):
        raw_cat_df = pd.read_csv(category_file, sep="\t")
    else: raw_cat_df = pd.read_csv(category_file)

    cat_dict = {str(int(r['index']-1)): r[category_col] for _, r in raw_cat_df.iterrows()}
    cat_dict['-1'] = "<UNK>"
    return cat_dict


def bbox_from_dict(bbox_dict: Dict[str, float], box_in_type: str="obb", box_out_type: str="obb") -> Union[AxisAlignedBoundingBox, OrientedBoundingBox]:
    """ create AABB or OBB from dict """
    if box_in_type == "aabb":
        min_bound = (bbox_dict['xlo'], bbox_dict['ylo'], bbox_dict['zlo'])
        max_bound = (bbox_dict['xhi'], bbox_dict['yhi'], bbox_dict['zhi'])
        aabb =  AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        if box_out_type == "aabb": return aabb
        elif box_out_type == "obb": return aabb.get_oriented_bounding_box()
    elif box_in_type == "obb":
        center = (bbox_dict['px'], bbox_dict['py'], bbox_dict['pz'])
        a0 = np.array([bbox_dict['a0x'], bbox_dict['a0y'], bbox_dict['a0z']])
        norm_a0 = np.linalg.norm(a0)
        a0 = a0 / norm_a0 if norm_a0 > 1e-6 else a0 / (norm_a0 + 1e-6)
        a1 = np.array([bbox_dict['a1x'], bbox_dict['a1y'], bbox_dict['a1z']])
        norm_a1 = np.linalg.norm(a1)
        a1 = a1 / norm_a1 if norm_a1 > 1e-6 else a1 / (norm_a1 + 1e-6)
        a2 = np.cross(a0, a1)
        R = np.stack([a0, a1, a2], axis=1)
        extent = (bbox_dict['r0']*2, bbox_dict['r1']*2, bbox_dict['r2']*2)
        obb = OrientedBoundingBox(center=center, R=R, extent=extent)
        if box_out_type == "obb": return obb
        elif box_out_type == "aabb": return obb.get_axis_aligned_bounding_box()
    else: 
        raise ValueError(f"Unknown box type: {box_in_type}. Use 'aabb' or 'obb'.")


def read_house(house_file: str, 
               category_mapping: Optional[Dict[int, str]]=None,
               category_file: str="data/mp3d/category_mapping.tsv", 
               category_col:str="raw_category") -> House:
    """ Read house data from .house file, return a House object.
        If category_mapping is provided, use it directly. Otherwise, read from category_file. """
    assert os.path.exists(house_file) and house_file.endswith(".house"), \
        f"House file {house_file} does not exist."
    scan_id = os.path.basename(house_file).replace(".house", "")

    if category_mapping is None:
        CAT_MAPPING = read_category(category_file, category_col)
        print(f"Loaded {len(CAT_MAPPING)} categories from {category_file}.")
    else: CAT_MAPPING = category_mapping

    collections = defaultdict(list)
    with open(house_file, 'r', encoding='utf-8', errors='ignore') as f:
        house_data = f.readlines()[1:]
        assert house_data[0].startswith("H"), "Invalid house file format."

        house_line = read_line(house_data[0], line_type="H")
        collections['H'].append(house_line)
        # print(json.dumps(house_line, indent=2))
        
        line_idx = 1
        num_level = house_line['num_level']
        for level in house_data[line_idx:line_idx+num_level]:
            assert level.startswith("L"), f"Expected level line at Line#{line_idx}, but got: {level}"
            level_line = read_line(level, line_type="L")
            collections['L'].append(level_line)
            line_idx += 1
        
        num_region = house_line['num_region']
        for region in house_data[line_idx:line_idx+num_region]:
            assert region.startswith("R"), f"Expected region line at Line#{line_idx}, but got: {region}"
            region_line = read_line(region, line_type="R")
            region_line["label_name"] = REGION_MAPPING[region_line["label"]]
            collections['R'].append(region_line)
            line_idx += 1
        
        num_portal = house_line['num_port']
        for portal in house_data[line_idx:line_idx+num_portal]:
            assert portal.startswith("P"), f"Expected portal line at Line#{line_idx}, but got: {portal}"
            portal_line = read_line(portal, line_type="Port")
            collections['P'].append(portal_line)
            line_idx += 1
        
        line_idx += house_line['num_surf'] + house_line['num_vtx'] + \
            house_line["num_pano"] + house_line["num_image"] + house_line["num_cat"]

        num_object = house_line['num_obj']
        for obj in house_data[line_idx:line_idx+num_object]:
            assert obj.startswith("O"), f"Expected object line at Line#{line_idx}, but got: {obj}"
            obj_line = read_line(obj, line_type="O")
            try: obj_line["category"] = CAT_MAPPING[str(obj_line["category_index"])]
            except Exception as e: 
                print(f"Error mapping category for object_index {obj_line['object_index']}: {obj_line}")
                raise e
            collections['O'].append(obj_line)
            line_idx += 1
        
        line_idx += house_line['num_seg']
        assert line_idx == len(house_data), "Line number mismatch." + \
            f"The house file has {len(house_data)}, but we got {line_idx}."
    
    # Organize data by Obj-Room-Level-House
    object_by_region = defaultdict(list)
    for obj_line in collections['O']:
        obj_bbox = bbox_from_dict(obj_line, box_in_type="obb", box_out_type="obb")
        try:
            obj = Object3D(
                id=obj_line['object_index'],
                region=obj_line['region_index'],
                category_id=obj_line['category_index'],
                category=obj_line['category'],
                obb=obj_bbox
            )
        except Exception as e: 
            print(f"Error creating Object3D for object_index {obj_line['object_index']}: {e}")
            continue
        object_by_region[obj.region].append(obj)

    region_by_level = defaultdict(list)
    for region_line in collections['R']:
        region_bbox = bbox_from_dict(region_line, box_in_type="aabb", box_out_type="obb")
        region = Region(
            id=region_line['region_index'],
            level=region_line['level_index'],
            label=region_line['label_name'],
            center=np.array([region_line['px'], region_line['py'], region_line['pz']]),
            obb=region_bbox,
            height=region_line['height'],
            objects=object_by_region.get(region_line['region_index'], [])
        )
        region_by_level[region.level].append(region)
    
    portal_list = []
    for portal_line in collections['P']:
        portal_bbox = bbox_from_dict(portal_line, box_in_type="aabb", box_out_type="obb")
        portal = Portal(
            id=portal_line['portal_index'],
            region_i=portal_line['region0_index'],
            region_j=portal_line['region1_index'],
            label=portal_line['label'],
            obb=portal_bbox
        )
        portal_list.append(portal)

    level_list = []
    for level_line in collections['L']:
        level_bbox = bbox_from_dict(level_line, box_in_type="aabb", box_out_type="obb")
        level = Level(
            id=level_line['level_index'],
            height=level_line['px'],  # Here we assume px represents the height
            center=np.array([level_line['px'], level_line['py'], level_line['pz']]),
            obb=level_bbox,
            regions=region_by_level.get(level_line['level_index'], [])
        )
        level_list.append(level)

    house_bbox = bbox_from_dict(house_line, box_in_type="aabb", box_out_type="obb")
    house = House(
        scan_id=scan_id,
        obb=house_bbox,
        levels=level_list,
        portals=portal_list
    )

    return house


# ===================== Post Processing Functions ===================== #

def post_process_house_and_points(house: House, points: np.ndarray=None, colors: np.ndarray=None):
    """ Post process the house and points to 
        fix issues such as staircase room level assignment."""
    special_cases = {'XcA2TqTSSAj', 'Z6MFQCViBuw', 'VLzqgDo317F'}
    
    if house.scan_id not in special_cases:
        return house, points, colors
    
    if house.scan_id == 'XcA2TqTSSAj':
        idx, special_level = [x for x in enumerate(house.levels) if x[1].id == 5][0]
        assert len(special_level.regions) == 1
        stair_room = special_level.regions[0]
        assert stair_room.id == 24
        stair_room.level = 3
        for level in house.levels:
            if level.id == 3:
                level.regions.append(stair_room)
                break
        del house.levels[idx]
        print(f"[{house.scan_id}] Moved stair room {stair_room.id} from level 5 to level 3.")
    
    if house.scan_id == "Z6MFQCViBuw":
        idx, special_level = [x for x in enumerate(house.levels) if x[1].id == 1][0]
        assert len(special_level.regions) == 1
        other_room = special_level.regions[0]
        del house.levels[idx]
        if points is not None and colors is not None:
            before_num_points = points.shape[0]
            region_mask = points_within_bbox(points, other_room.obb, expand=0)
            points = points[~region_mask]
            colors = colors[~region_mask]
            after_num_points = points.shape[0]
            print(f"[{house.scan_id}] Removed the useless {other_room.label}, deleted {before_num_points - after_num_points} points.")

    if house.scan_id == "VLzqgDo317F":
        level_idx, special_level = [x for x in enumerate(house.levels) if x[1].id == 0][0]
        idx4, stair_4 = [r for r in enumerate(special_level.regions) if r[1].id == 4][0]
        idx13, stair_13 = [r for r in enumerate(special_level.regions) if r[1].id == 13][0]
        stair_4_box_points = stair_4.obb.get_box_points()
        stair_13_box_points = stair_13.obb.get_box_points()
        min_z, max_z = stair_13_box_points[:, 2].min(), stair_13_box_points[:, 2].max()
        two_staris_bbox = np.concatenate((stair_4_box_points, stair_13_box_points), axis=0)
        x_min_share, y_min_share = np.min(two_staris_bbox[:, :2], axis=0)
        x_max_share, y_max_share = np.max(two_staris_bbox[:, :2], axis=0)
        new_center = np.array([(x_min_share + x_max_share)/2, (y_min_share + y_max_share)/2, (min_z + max_z)/2])
        new_extent = np.array([x_max_share - x_min_share, y_max_share - y_min_share, max_z - min_z])
        stair_13.obb = OrientedBoundingBox(center=new_center, R=stair_13.obb.R, extent=new_extent)
        stair_13.level = 1
        new_level = house.get_level_by_id(1)
        new_level.regions.append(stair_13)
        del house.levels[level_idx].regions[idx13]
        del house.levels[level_idx].regions[idx4]
        print(f"[{house.scan_id}] Merged stair rooms 4 and 13 into level 1.") 

        stair_17 = [r for r in special_level.regions if r.id == 17][0]
        stair_17_box_points = stair_17.obb.get_box_points()
        x_min_17, y_min_17, z_min_17 = np.min(stair_17_box_points, axis=0)
        x_max_17, y_max_17, __ = np.max(stair_17_box_points, axis=0)
        min_17_bound = np.array([x_min_17, y_min_17, z_min_17])
        max_17_bound = np.array([x_max_17, y_max_17, min_z])
        stair_17.obb = OrientedBoundingBox(
            center=(min_17_bound + max_17_bound) / 2,
            R=stair_17.obb.R,
            extent=(max_17_bound - min_17_bound)
        )
        stair_18 = [r for r in special_level.regions if r.id == 18][0]
        stair_18_box_points = stair_18.obb.get_box_points()
        x_min_18, y_min_18, z_min_18 = np.min(stair_18_box_points, axis=0)
        x_max_18, y_max_18, __ = np.max(stair_18_box_points, axis=0)
        min_18_bound = np.array([x_min_18, y_min_18, z_min_18])
        max_18_bound = np.array([x_max_18, y_max_18, min_z])
        stair_18.obb = OrientedBoundingBox(
            center=(min_18_bound + max_18_bound) / 2,
            R=stair_18.obb.R,
            extent=(max_18_bound - min_18_bound)
        )
        print(f"[{house.scan_id}] Adjusted stair rooms 17 and 18 bounding boxes.")

    return house, points, colors


def find_z_neighbor(room: Region, level: Level, label=None) -> Tuple[Optional[Region], float]:
    """ Find the z-overlapping neighbor room in the given level. """
    r1 = room
    bbox_1 = r1.obb.get_box_points()
    x1_min, y1_min, __ = np.min(bbox_1, axis=0)
    x1_max, y1_max, __ = np.max(bbox_1, axis=0)
    
    possible_neighbors = []
    for r2 in level.regions:
        if label is not None and r2.label != label: continue
        bbox_2 = r2.obb.get_box_points()
        x2_min, y2_min, __ = np.min(bbox_2, axis=0)
        x2_max, y2_max, __ = np.max(bbox_2, axis=0)
        iou = compute_iou_2d(
            [x1_min, y1_min, x1_max, y1_max],
            [x2_min, y2_min, x2_max, y2_max]
        )
        if iou > 0: possible_neighbors.append((r2, iou))
    possible_neighbors = sorted(possible_neighbors, key=lambda x: x[1], reverse=True)

    if len(possible_neighbors) > 0: return possible_neighbors[0]
    else: return None, 0.0


def compute_xy_distance(room: Region, level: Level) -> float:
    """ Compute the minimum xy-distance between the room (corners) and all rooms (edges) in the level. """
    r1 = room
    bbox_1 = r1.obb.get_box_points()
    x1_min, y1_min, __ = np.min(bbox_1, axis=0)
    x1_max, y1_max, __ = np.max(bbox_1, axis=0)
    r1_bbox = np.array([x1_min, y1_min, x1_max, y1_max])

    level_bboxes = []
    for r2 in level.regions:
        bbox_2 = r2.obb.get_box_points()
        x2_min, y2_min, __ = np.min(bbox_2, axis=0)
        x2_max, y2_max, __ = np.max(bbox_2, axis=0)
        level_bboxes.append([x2_min, y2_min, x2_max, y2_max])
    level_bboxes = np.array(level_bboxes)

    if level_bboxes.size == 0: return 0.0

    dx_left  = level_bboxes[:, 0] - r1_bbox[2]   # r2 在 r1 右侧时 > 0
    dx_right = r1_bbox[0] - level_bboxes[:, 2]   # r2 在 r1 左侧时 > 0
    dx = np.maximum(np.maximum(dx_left, dx_right), 0.0)

    dy_down  = level_bboxes[:, 1] - r1_bbox[3]   # r2 在 r1 上方时 > 0（假设 y 向上）
    dy_up    = r1_bbox[1] - level_bboxes[:, 3]   # r2 在 r1 下方时 > 0
    dy = np.maximum(np.maximum(dy_down, dy_up), 0.0)

    dist = np.hypot(dx, dy)                      # sqrt(dx**2 + dy**2)
    return float(dist.min())


def post_process_stairs(house: House, z_threshold:float=0.5, xy_threshold: float=0.2) -> House:
    """ Post process the house to handle stairs that cover multiple levels. """
    all_stair_regions: List[Region] = []
    id2room, id2level = {}, {}
    for level in house.levels:
        id2level[level.id] = level
        for room in level.regions:
            id2room[room.id] = room
            if room.label == "stairs":
                all_stair_regions.append(room)
    
    stair_heights = [stair.obb.extent[2] for stair in all_stair_regions]
    avg_stair_height = np.mean(stair_heights) if len(stair_heights) > 0 else 0.0
    # print("-"*100)
    # print("Stairs: ", stair_heights)
    # print(f"[{house.scan_id}] Average stair height: {avg_stair_height:.3f}m.")
    # print("-"*100)
    
    extra_stairs_by_level = defaultdict(list)
    for stair in all_stair_regions:
        stair_bbox = stair.obb.get_box_points()
        stair_z_min, stair_z_max = stair_bbox[:, 2].min(), stair_bbox[:, 2].max()
        
        for level in house.levels:
            if level.id == stair.level: continue
            level_z_min, level_z_max = level.majority_z_range
            overlap_in_z = max(0, min(stair_z_max, level_z_max) - max(stair_z_min, level_z_min))
            # Case 1: The stairs cover multiple levels in the house
            if overlap_in_z / (level_z_max - level_z_min) >= z_threshold:
                extra_stairs_by_level[level.id].append(stair.id)
            else: # Case 2: The stairs are between two levels, but mainly located in one level
                z_neighbor, neighbor_iou = find_z_neighbor(stair, level) # return the neighbor room with largest iou
                distance_z = level_z_min - stair_z_max if level_z_min > stair_z_max \
                        else stair_z_min - level_z_max
                if overlap_in_z > 0 or distance_z < z_threshold: # stairs -- close enough to the level
                    # Case 2.1: no overlap room found in the level, ok, directly add
                    if z_neighbor is None: # check if the stair is indeed close enough in xy-plane
                        min_xy_dist = compute_xy_distance(stair, level)
                        if min_xy_dist < xy_threshold: extra_stairs_by_level[level.id].append(stair.id)
                    # Case 2.2: overlap room found is a hallway, we allow this case
                    elif z_neighbor.label == "hallway": extra_stairs_by_level[level.id].append(stair.id)
                    # Case 2.3: overlap room found is not a hallway, only add if the iou is small enough
                    elif neighbor_iou < xy_threshold: extra_stairs_by_level[level.id].append(stair.id)

    for level_id, stair_ids in extra_stairs_by_level.items():
        stair_ids = sorted(list(set(stair_ids)))
        level: Level = id2level[level_id]
        for stair_id in stair_ids:
            stair_room: Region = id2room[stair_id]
            level.extra.setdefault("stairs", []).append(stair_room)
    
    return house

