from typing import List, Dict, Tuple, Optional, Any


REGION_MAPPING = {
    'a': 'bathroom',
    'b': 'bedroom',
    'c': 'closet',
    'd': 'dining_room',
    'e': 'entrance',        # entryway_or_foyer
    'f': 'family_room',
    'g': 'garage',
    'h': 'hallway',
    'i': 'library',
    'j': 'laundry_room',
    'k': 'kitchen',
    'l': 'living_room',
    'm': 'meeting_room',
    'n': 'lounge',
    'o': 'office',
    'p': 'outdoor',         # porch_or_terrace_or_deck_or_driveway
    'r': 'recreation_room',
    's': 'stairs',
    't': 'toilet',
    'u': 'utility_room',
    'v': 'home_theater',
    'w': 'workout_room',    # workout_room_or_gym_room
    'x': 'yard',
    'y': 'balcony',
    'z': 'other_room',
    'B': 'bar',
    'C': 'classroom',
    'D': 'booth',           # dining_booth
    'S': 'wellness_room',   # spa_room_or_sauna_room
    'Z': 'junk',
    '-': 'unknown'
}




LINE_MAPPING = {
    "H": ["name", "label", "num_image", "num_pano", "num_vtx", "num_surf", "num_seg", "num_obj", "num_cat", "num_region", "num_port", "num_level", "-", "-", "-", "-", "-", "xlo", "ylo", "zlo", "xhi", "yhi", "zhi", "-", "-", "-", "-", "-"],
    "L": ["level_index", "num_region", "label", "px", "py", "pz", "xlo", "ylo", "zlo", "xhi", "yhi", "zhi", "-", "-", "-", "-", "-"],
    "R": ["region_index", "level_index", "-", "-", "label", "px", "py", "pz", "xlo", "ylo", "zlo", "xhi", "yhi", "zhi", "height", "-", "-", "-", "-"],
    "Port": ["portal_index", "region0_index", "region1_index", "label", "xlo", "ylo", "zlo", "xhi", "yhi", "zhi", "-", "-", "-", "-"],
    "S": ["surface_index", "region_index", "-", "label", "px", "py", "pz", "nx", "ny", "nz", "xlo", "ylo", "zlo", "xhi", "yhi", "zhi", "-", "-", "-", "-", "-"],
    "V": ["vertex_index", "surface_index", "label", "px", "py", "pz", "nx", "ny", "nz", "-", "-", "-"],
    "Pano": ["name", "panorama_index", "region_index", "-", "px", "py", "pz", "-", "-", "-", "-", "-"],
    "I": ["image_index", "panorama_index", "name", "camera_index", "yaw_index", "e00", "e01", "e02", "e03", "e10", "e11", "e12", "e13", "e20", "e21", "e22", "e23", "e30", "e31", "e32", "e33", "i00", "i01", "i02", "i10", "i11", "i12", "i20", "i21", "i22", "width", "height", "px", "py", "pz", "-", "-", "-", "-", "-"],
    "C": ["category_index", "category_mapping_index", "category_mapping_name", "mpcat40_index", "mpcat40_name", "-", "-", "-", "-", "-"],
    "O": ["object_index", "region_index", "category_index", "px", "py", "pz", "a0x", "a0y", "a0z", "a1x", "a1y", "a1z", "r0", "r1", "r2", "-", "-", "-", "-", "-", "-", "-", "-"],
    "E": ["segment_index", "object_index", "id", "area", "px", "py", "pz", "xlo", "ylo", "zlo", "xhi", "yhi", "zhi", "-", "-", "-", "-", "-"],
}
