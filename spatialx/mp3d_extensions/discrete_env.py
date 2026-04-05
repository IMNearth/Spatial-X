''' Batched REVERIE navigation environment '''
from typing import List, Tuple, Dict, Union, Optional, Any
import json
import os
import math
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image
from .mp_utils import (
    NavGraph, 
    ImageObservationsDB,
    ImageTextObservationsDB,
    build_mp3d_simulator
)
import cv2
from .pc_utils import (
    obtain_sim_rotation, 
    vfov_to_hfov, 
    spherical_to_pixel, 
    cartesian_to_spherical
)
from ..utils.image_utils import (
    adjust_title_y, 
    avoid_overlap
)


ERROR_MARGIN = 3.0
ENV_ACTIONS = {
    'left':    (0,-1, 0), # left
    'right':   (0, 1, 0), # right
    'up':      (0, 0, 1), # up
    'down':    (0, 0,-1), # down
    'forward': (1, 0, 0), # forward
    '<end>':   (0, 0, 0), # <end>
    '<start>': (0, 0, 0), # <start>
    '<ignore>':(0, 0, 0)  # <ignore>
}


def normalize_angle_rad(angle):
    while angle < 0: angle += 2 * math.pi
    while angle >= 2 * math.pi: angle -= 2 * math.pi
    return angle


class DiscreteSimulator(object):
    ''' A simple simulator in Matterport3D environment '''

    def __init__(self, *args, **kwargs):
        self.heading = 0
        self.elevation = 0
        self.position = np.zeros(3)
        self.location_dict = {}
        self.scan_ID = ''
        self.viewpoint_ID = ''
        self.navigable_dict = defaultdict(dict)
        self.candidate = {}
        self.gmap = NavGraph()
        pass

    def newEpisode(self, *args, **kwargs):
        pass

    def getState(self) -> dict:
        pass

    def makeAction(self, *args, **kwargs):
        pass


class DiscreteTextSimulator(DiscreteSimulator):
    ''' A simple simulator in Matterport3D environment '''

    def __init__(self, navigable_dir: str, location_dir:str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.location_dir = location_dir
        self.navigable_dir = navigable_dir
    
    @staticmethod
    def readLocation(location_dir, scan_id: str) -> Dict[str, List[float]]:
        position_path = os.path.join(location_dir, scan_id + '_vp_locations.json')
        with open(position_path, 'r') as f: cur_location_dict = json.load(f)
        for k, v in cur_location_dict.items():
            if isinstance(v, str): cur_location_dict[k] = json.loads(v)
        return cur_location_dict
    
    @staticmethod
    def readNavigable(navigable_dir, scan_id: str) -> Dict[str, Dict[str, Any]]:
        navigable_path = os.path.join(navigable_dir, scan_id + '_navigable.json')
        with open(navigable_path, 'r') as f: cur_navigable_dict = json.load(f)
        return cur_navigable_dict
    
    def get_vp_location(self, viewpoint_ID: str) -> np.ndarray:
        if viewpoint_ID not in self.location_dict:
            raise ValueError(f"Viewpoint ID {viewpoint_ID} not found in location dict.")
        return np.array(self.location_dict[viewpoint_ID])

    def newEpisode(self, scan_ID: str, viewpoint_ID: str, heading: int, elevation: int):
        self.heading = heading
        self.elevation = elevation
        self.scan_ID = scan_ID
        self.viewpoint_ID = viewpoint_ID
        # Load navigable dict
        self.navigable_dict = self.readNavigable(self.navigable_dir, self.scan_ID)
        # Load viewpoints positions dict
        self.location_dict = self.readLocation(self.location_dir, self.scan_ID)
        self.position = np.array(self.location_dict[self.viewpoint_ID])
        # Get candidate
        self.getCandidate()

    def updateGraph(self):
        # build graph
        for candidate in self.candidate.keys():
            self.gmap.update_connection(self.viewpoint_ID, candidate)

    def getState(self) -> dict:
        self.state = {
            'scanID': self.scan_ID,
            'viewpointID': self.viewpoint_ID,
            'heading': self.heading,
            'elevation': self.elevation,
            'candidate': self.candidate, 
            # dict{ "cand_vp_id": {
            #           "heading": float, 
            #           "elevation": float, 
            #           "ang_dis": float, 
            #           "distance": float
            #      }
            # }
            'position': self.position.tolist()
        }
        return self.state

    def getCandidate(self):
        """
        Get the agent's candidate list from pre-stored navigable dict.
        """
        self.candidate = self.navigable_dict[self.viewpoint_ID]
        for cand_id, cand_attr in self.candidate.items():
            cand_attr['position'] = self.location_dict[cand_id]
        self.updateGraph()

    def makeAction(self, next_viewpoint_ID: str):
        """
        Make action and update the agent's state.
        """
        if next_viewpoint_ID == self.viewpoint_ID:
            return
        elif next_viewpoint_ID in self.candidate.keys():
            self.heading = self.candidate[next_viewpoint_ID]['heading']
            self.elevation = self.candidate[next_viewpoint_ID]['elevation']
        self.viewpoint_ID = next_viewpoint_ID
        self.position = np.array(self.location_dict[self.viewpoint_ID])
        self.getCandidate()

    def sortCandidate(self) -> Tuple[Dict[str, int], List[str]]:
        agent_heading = self.heading
        candidate_ord = {}

        headings = []
        for cand_id in self.candidate.keys():
            cand_heading = self.candidate[cand_id]['heading']
            rel_heading = normalize_angle_rad(cand_heading - agent_heading)
            headings.append((rel_heading, cand_id))
        
        sorted_headings = sorted(headings, key=lambda x: x[0]) # small to large
        for idx, (rel_heading, cand_id) in enumerate(sorted_headings):
            candidate_ord[cand_id] = idx
        sorted_cand_ids = [cand_id for (__, cand_id) in sorted_headings]
        return candidate_ord, sorted_cand_ids


# ===============================================================================


class DiscreteVisualBase(DiscreteTextSimulator):
    ''' A simple simulator in Matterport3D environment with visual observations'''

    def __init__(self, navigable_dir: str, connectivity_dir: str, scan_dir: str, *args, 
                 use_panorama: bool=False, draw_viewpoint: bool=False, 
                 adaptive_radius: bool=False, circle_radius:int=22, draw_viewpoint_id: bool=False, **kwargs):
        if "width" in kwargs: self.WIDTH = kwargs.pop("width")
        if "height" in kwargs: self.HEIGHT = kwargs.pop("height")
        if "vfov" in kwargs: self.VFOV = kwargs.pop("vfov")
        if "delta_elevation" in kwargs: self.DELTA_ELE = kwargs.pop("delta_elevation")
        super().__init__(navigable_dir, *args, **kwargs)

        self.sim = build_mp3d_simulator(connectivity_dir, scan_dir, 
                                        width=self.WIDTH, height=self.HEIGHT, vfov=self.VFOV)
        self.use_panorama = use_panorama
        self.draw_viewpoint = draw_viewpoint
        self.draw_viewpoint_id = draw_viewpoint_id
        self.adaptive_radius = adaptive_radius
        self.circle_radius = circle_radius
        
    def newEpisode(self, scan_ID: str, viewpoint_ID: str, heading: int, elevation: int):
        super().newEpisode(scan_ID, viewpoint_ID, heading, elevation)
        self.sim.newEpisode([scan_ID], [viewpoint_ID], [heading], [elevation])

    def getState(self) -> dict:
        state = super().getState()
        sim_state = self.sim.getState()[0]
        
        if self.use_panorama: 
            state['panorama'] = self.getPanorama()
        else:
            image = np.array(sim_state.rgb, copy=True)      # in BGR channel
            image = Image.fromarray(image[:, :, ::-1])      # to RGB channel
            state['rgb'] = image
        
        state["position"] = [sim_state.location.x, sim_state.location.y, sim_state.location.z]
        state["rotation"] = obtain_sim_rotation(self.heading, self.elevation)
        return state
    
    def makeAction(self, next_viewpoint_ID: str):
        """
        Make action and update the agent's state.
        """
        super().makeAction(next_viewpoint_ID)
        self.sim.newEpisode([self.scan_ID], [self.viewpoint_ID], [self.heading], [self.elevation])
    
    def getPanorama(self) -> Union[np.ndarray, Image.Image]:
        """
        Get the panoramic RGB images at current viewpoint.
        :return: np.ndarray: (num_views, H, W, 3), in RGB channel, where as num_views = n_horizonal_views * n_vertical_views
        """
        panorama = self.make_pano() # (num_views, H, W, 3)
        if self.draw_viewpoint: panorama, past_circles = self.markViewpoints(panorama)
        else: past_circles = None
        panorama_image = self.drawPanorama(panorama, past_circles)
        # finally, reset to original heading and elevation
        self.sim.newEpisode([self.scan_ID], [self.viewpoint_ID], [self.heading], [self.elevation])
        return panorama_image
    
    def markViewpoints(self, pano_images: np.ndarray) -> np.ndarray:
        """ Mark the candidate viewpoints on the panorama images. """
        lightblue = (153, 210, 255)
        candidates = self.candidate
        agent_heading = self.heading
        agent_elevation = self.elevation

        past_circle_coords = defaultdict(list)
        visualize_info = defaultdict(dict)
        for i, cand_vid in enumerate(candidates.keys()):
            canx, cany, canz = candidates[cand_vid]['position']
            agx, agy, agz = self.position.tolist()
            theta, phi, d = cartesian_to_spherical(canx - agx, cany - agy, canz - agz)
            view_index, adj_theta, adj_phi = self.assign_view(
                theta, phi, agent_heading, agent_elevation)
            px, py = spherical_to_pixel(
                adj_theta, adj_phi, d,
                image_width=self.WIDTH, 
                image_height=self.HEIGHT, 
                hfov_deg=self.HFOV
            )
            px, py = avoid_overlap(
                px, py,
                past_circle_coords[view_index],
                min_distance=2*self.circle_radius,
                image_width=self.WIDTH,
                image_height=self.HEIGHT,
                margin=20,
            )
            past_circle_coords[view_index].append((px, py))
            visualize_info[view_index][cand_vid] = {
                "index": i,
                "view_index": view_index,
                "pixel_coord": (px, py),
                "distance": d,
            }
        
        candidate_ord, __ = self.sortCandidate()
        for view_index in visualize_info.keys():
            direction = self.DIRECTIONS[view_index]
            image = pano_images[view_index]
            image = np.ascontiguousarray(image)
            radius_map = {}
            if self.adaptive_radius:
                num_cands = len(visualize_info[view_index].keys())
                cand_vids = sorted(
                    list(visualize_info[view_index].keys()),
                    key=lambda vid: visualize_info[view_index][vid]["distance"],
                )
                if num_cands == 1: 
                    radius_map[cand_vids[0]] = self.circle_radius
                else: 
                    for rank, vid in enumerate(cand_vids):
                        t = rank / (num_cands - 1)  # 0 (最近) -> 1 (最远)
                        radius = self.circle_radius - t * (self.circle_radius - int(self.circle_radius*0.7))
                        radius_map[vid] = int(round(radius))
            else: radius_map = {vid: self.circle_radius for vid in visualize_info[view_index].keys()}

            for cand_vid in visualize_info[view_index].keys():
                cand_idx = candidate_ord[cand_vid]
                px, py = visualize_info[view_index][cand_vid]["pixel_coord"]
                cur_radius = radius_map[cand_vid]
                cur_ratio = cur_radius / self.circle_radius
                cv2.circle(image, (px, py), cur_radius, lightblue, -1)
                cv2.circle(image, (px, py), cur_radius, (0, 0, 255), 1)
                if self.draw_viewpoint_id:
                    cv2.putText(image, f"{cand_idx+1}", (px - int(10*cur_ratio), py + int(10*cur_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1*cur_ratio, (0, 0, 0), 2)
                # pano_images[view_index] = image

        all_circle_coords = sum(past_circle_coords.values(), [])
        return pano_images, all_circle_coords

    def drawPanorama(self, panorama: np.ndarray, past_circles:List=None) -> Image.Image:
        """ Draw the panorama images into a single large image for visualization. """
        fig, axes, ax_ord = self.get_figure_plan()
        if past_circles is not None:
            all_past_ys = [(1 - y/self.HEIGHT) for (x, y) in past_circles]
            median_y = np.median(all_past_ys)
            y_coord = 0.92 if median_y <= 0.5 else 0.03
            y_coord = adjust_title_y(y_coord, all_past_ys, self.HEIGHT, min_distance_pix=self.circle_radius*2)
        else: y_coord = 0.92  # default at the top

        for ix, (direction, image) in enumerate(zip(self.DIRECTIONS, panorama)):
            ax: plt.Axes = axes[np.where(ax_ord == ix)][0]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title(direction)
            ax.text(0.5, y_coord, direction, transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'),)
        # hide unused axes
        unused = axes[np.where(ax_ord == -1)]
        if len(unused) > 0:
            for ax in unused: ax.axis('off')
        if hasattr(self, "draw_special_axis"): self.draw_special_axis(unused[0])

        fig.tight_layout()
        # Convert matplotlib figure to PIL Image
        buf = BytesIO()
        fig.savefig(buf, format='jpg', dpi=128)
        buf.seek(0)
        panorama_image = Image.open(buf).convert('RGB')  # 1024 x 1024
        plt.close(fig)
        buf.close()

        print("Panorama image size:", panorama_image.size)
        return panorama_image

    def make_pano(self) -> List[np.ndarray]:
        raise NotImplementedError("Subclasses must implement make_pano method.")
    
    def get_figure_plan(self) -> Tuple[plt.Figure, List[plt.Axes] , np.ndarray]:
        raise NotImplementedError("Subclasses must implement get_figure_plan method.")

    def assign_view(self, rel_heading: float, rel_elevation: float) -> Tuple[int, float, float]:
        raise NotImplementedError("Subclasses must implement assign_view method.")

    def getCandidate(self):
        """
        Get the agent's candidate list from pre-stored navigable dict.
        """
        unique_views = set()
        self.candidate = self.navigable_dict[self.viewpoint_ID]
        for cand_vid, cand_attr in self.candidate.items():
            cand_attr['viewpointID'] = cand_vid
            cand_attr['position'] = self.location_dict[cand_vid]
            # Obtain the image corresponding to each candidate viewpoint
            canx, cany, canz = cand_attr['position']
            agx, agy, agz = self.position.tolist()
            theta, phi, d = cartesian_to_spherical(canx - agx, cany - agy, canz - agz)
            view_index = self.assign_view(theta, phi, self.heading, self.elevation)
            cand_attr['view_index'] = view_index[0]  # the index of the view in the panorama
            unique_views.add(view_index[0])
        unique_views = sorted(list(unique_views))
        print(f"Unique view indices for candidates: {unique_views}")

        # Get the order of candidates based on heading and elevation
        # sort by view_index%12 first, then by distance
        sorted_cands = sorted(
            self.candidate.items(),
            key=lambda item: (item[1]['view_index'] % self.n_horizonal_views, 
                              item[1]['distance'])
        )
        for idx, (cand_vid, __) in enumerate(sorted_cands):
            self.candidate[cand_vid]['__ord'] = idx
            self.candidate[cand_vid]['image_index'] = unique_views.index(
                self.candidate[cand_vid]['view_index'])
            self.candidate[cand_vid]['total_images'] = unique_views

        self.updateGraph()


class DiscreteVisualSimulatorV1(DiscreteVisualBase):
    ''' Not Used - A simple simulator in Matterport3D environment with Panoramic visual observations'''
    WIDTH = 512
    HEIGHT = 512
    DIRECTIONS = [
        "Front-Up",    "Right-Up",    "Rear-Up",    "Left-Up",   
        "Front",       "Right",       "Rear",       "Left",      
        "Front-Down",  "Right-Down",  "Rear-Down",  "Left-Down", 
    ]
    VIEWPOINT_SIZE = 4
    VFOV = 90
    HFOV = vfov_to_hfov(VFOV, WIDTH, HEIGHT)

    @property
    def n_horizonal_views(self) -> int:
        """ Number of discrete horizontal views in the panorama """
        return self.VIEWPOINT_SIZE
    
    @property
    def n_side_views(self) -> int:
        """ Number of side views to the left/right of the front view """
        return self.n_horizonal_views // 3
    
    @property
    def delta_heading(self) -> float:
        """ Heading change between adjacent horizontal views in radians """
        return math.radians(self.VFOV)
    
    @property
    def n_vertical_views(self) -> int:
        """ Number of discrete vertical views in the panorama """
        return 3
    
    @property
    def delta_elevation(self) -> float:
        """ Elevation change between adjacent vertical views in radians """
        return math.radians(self.HFOV)

    def make_pano(self) -> np.ndarray:
        """
        Get the panoramic RGB images at current viewpoint.
        :return: np.ndarray: (num_views, H, W, 3), in RGB channel, where as num_views = n_horizonal_views * n_vertical_views
        """
        num_views = self.n_horizonal_views * self.n_vertical_views
        panorama = []
        for ix in range(num_views):
            if ix == 0: # look up
                self.sim.newEpisode([self.scan_ID], [self.viewpoint_ID], [self.heading], 
                                    [self.elevation + self.delta_elevation])
            elif ix % self.n_horizonal_views == 0:  # look down and turn right
                self.sim.makeAction([0], [self.delta_heading], [-self.delta_elevation]) 
            else: self.sim.makeAction([0], [self.delta_heading], [0])  # turn right
            state = self.sim.getState()[0]
            image = np.array(state.rgb, copy=True)[..., ::-1]   # image_direction = self.DIRECTIONS[ix]
            panorama.append(image)
        panorama = np.array(panorama)  # (num_views, H, W, 3)
        return panorama

    def assign_view(self, abs_heading: float, abs_elevation: float,
                          agent_heading: float, agent_elevation: float) -> Tuple[int, float, float]:
        theta = normalize_angle_rad(abs_heading - agent_heading)
        phi = normalize_angle_rad(abs_elevation - agent_elevation)
        num_views = self.n_horizonal_views
        sector_size = 360 / num_views

        theta = normalize_angle_rad(theta)
        theta_deg = np.rad2deg(theta) % 360
        view_index = int((theta_deg + sector_size / 2) // sector_size) % num_views
        adj_theta = theta - view_index * (2 * math.pi / num_views)

        phi = normalize_angle_rad(phi)
        phi_deg = np.rad2deg(phi) % 360

        # [-135, -45], [-45, 45], [45, 135]
        if self.VFOV/2 <= phi_deg < self.VFOV*3/2: # up
            view_index = view_index
            adj_phi = phi - np.deg2rad(90)
        elif 360 - 3/2*self.VFOV <= phi_deg <= 360 - self.VFOV/2: # down
            view_index = view_index + 2*num_views
            adj_phi = phi - np.deg2rad(270)
        else: # level
            view_index = view_index + num_views
            adj_phi = phi

        return view_index, adj_theta, adj_phi

    def get_figure_plan(self):
        n_rows, n_cols = self.n_vertical_views, self.n_horizonal_views
        ax_ord = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
        ax_ord = np.roll(ax_ord, shift=self.n_side_views, axis=1)  # center front view
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.5*n_rows))
        return fig, axes, ax_ord


class DiscreteVisualSimulatorV2(DiscreteVisualBase):
    ''' A simple simulator in Matterport3D environment with Compass-Like visual observations'''
    WIDTH = 512
    HEIGHT = 512
    DIRECTIONS = ["Front", "Front-Right", "Right", "Rear-Right", "Rear", "Rear-Left", "Left", "Front-Left"]
    VIEWPOINT_SIZE = 8
    VFOV = 90
    HFOV = vfov_to_hfov(VFOV, WIDTH, HEIGHT)
    DELTA_HOR = 45

    @property
    def n_horizonal_views(self) -> int:
        """ Number of discrete horizontal views in the panorama """
        return self.VIEWPOINT_SIZE
    
    @property
    def n_side_views(self) -> int:
        """ Number of side views to the left/right of the front view """
        return self.n_horizonal_views // 3
    
    @property
    def delta_heading(self) -> float:
        """ Heading change between adjacent horizontal views in radians """
        return math.radians(self.DELTA_HOR)
    
    @property
    def n_vertical_views(self) -> int:
        """ Number of discrete vertical views in the panorama """
        return 1
    
    @property
    def delta_elevation(self) -> float:
        """ Elevation change between adjacent vertical views in radians """
        return 0

    def make_pano(self) -> Union[np.ndarray, Image.Image]:
        """
        Get the panoramic RGB images at current viewpoint.
        :return: np.ndarray: (num_views, H, W, 3), in RGB channel, where as num_views = n_horizonal_views * n_vertical_views
        """
        num_views = self.n_horizonal_views * self.n_vertical_views
        panorama = []
        for ix in range(num_views):
            if ix == 0: # initial position, force the elevation to be level
                self.sim.newEpisode([self.scan_ID], [self.viewpoint_ID], [self.heading], [0])
            else: self.sim.makeAction([0], [self.delta_heading], [0])  # turn right
            state = self.sim.getState()[0]
            image = np.array(state.rgb, copy=True)[..., ::-1]   # image_direction = self.DIRECTIONS[ix]
            panorama.append(image)
        panorama = np.array(panorama)  # (num_views, H, W, 3)
        return panorama
    
    def assign_view(self, cur_heading: float, cur_elevation: float,
                          agent_heading: float, agent_elevation: float) -> Tuple[int, float, float]:
        theta = normalize_angle_rad(cur_heading - agent_heading)
        phi = normalize_angle_rad(cur_elevation - 0)
        num_views = self.n_horizonal_views
        sector_size = self.DELTA_HOR

        theta = normalize_angle_rad(theta)
        theta_deg = np.rad2deg(theta) % 360
        view_index = int((theta_deg + sector_size / 2) // sector_size) % num_views
        adj_theta = theta - view_index * (2 * math.pi / num_views)
        return view_index, adj_theta, phi

    def get_figure_plan(self) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
        ax_ord = np.array([
            [7, 0, 1], 
            [6, -1, 2],
            [5, 4, 3]
        ])
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        return fig, axes, ax_ord
    
    def draw_special_axis(self, ax: plt.Axes):
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        center_x, center_y = 0.5, 0.5
        arrow_len = 0.25

        ax.annotate('', xy=(center_x, center_y + arrow_len), xytext=(center_x, center_y),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))  # up
        ax.annotate('', xy=(center_x, center_y - arrow_len), xytext=(center_x, center_y),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'),) # down
        ax.annotate('', xy=(center_x - arrow_len, center_y), xytext=(center_x, center_y),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'),) # left
        ax.annotate('', xy=(center_x + arrow_len, center_y), xytext=(center_x, center_y),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'),) # right
        
        ax.text(center_x, center_y + arrow_len + 0.05, 'Front', ha='center', va='bottom', fontsize=12)
        ax.text(center_x, center_y - arrow_len - 0.05, 'Rear', ha='center', va='top', fontsize=12)
        ax.text(center_x - arrow_len - 0.05, center_y, 'Left', ha='right', va='center', fontsize=12)
        ax.text(center_x + arrow_len + 0.05, center_y, 'Right', ha='left', va='center', fontsize=12)

        ax.text(center_x, center_y, 'me', ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))


class DiscreteVisualSimulatorV3(DiscreteVisualBase):
    ''' A simple simulator in Matterport3D environment with Sequential visual observations'''
    WIDTH = 256
    HEIGHT = 256
    DIRECTIONS = ["Front", "Front Right", "Right", "Rear Right", "Rear", "Rear Left", "Left", "Front Left"]
    VIEWPOINT_SIZE = 8
    VFOV = 90
    HFOV = vfov_to_hfov(VFOV, WIDTH, HEIGHT)
    DELTA_HOR = 45

    @property
    def n_horizonal_views(self) -> int:
        """ Number of discrete horizontal views in the panorama """
        return self.VIEWPOINT_SIZE
    
    @property
    def n_side_views(self) -> int:
        """ Number of side views to the left/right of the front view """
        return self.n_horizonal_views // 3
    
    @property
    def delta_heading(self) -> float:
        """ Heading change between adjacent horizontal views in radians """
        return math.radians(self.DELTA_HOR)
    
    @property
    def n_vertical_views(self) -> int:
        """ Number of discrete vertical views in the panorama """
        return 1
    
    @property
    def delta_elevation(self) -> float:
        """ Elevation change between adjacent vertical views in radians """
        return 0

    def getPanorama(self) -> Union[np.ndarray, Image.Image]:
        """
        Get the panoramic RGB images at current viewpoint.
        :return: np.ndarray: (num_views, H, W, 3), in RGB channel, where as num_views = n_horizonal_views * n_vertical_views
        """
        panorama = self.make_pano() # (num_views, H, W, 3)
        panorama_image = dict()
        for direction, image in zip(self.DIRECTIONS, panorama):
            image = Image.fromarray(image).convert('RGB')
            panorama_image[direction] = image
        # finally, reset to original heading and elevation
        self.sim.newEpisode([self.scan_ID], [self.viewpoint_ID], [self.heading], [self.elevation])
        return panorama_image

    def make_pano(self) -> Union[np.ndarray, Image.Image]:
        """
        Get the panoramic RGB images at current viewpoint.
        :return: np.ndarray: (num_views, H, W, 3), in RGB channel, where as num_views = n_horizonal_views * n_vertical_views
        """
        num_views = self.n_horizonal_views * self.n_vertical_views
        panorama = []
        for ix in range(num_views):
            if ix == 0: # initial position, force the elevation to be level
                self.sim.newEpisode([self.scan_ID], [self.viewpoint_ID], [self.heading], [0])
            else: self.sim.makeAction([0], [self.delta_heading], [0])  # turn right
            state = self.sim.getState()[0]
            image = np.array(state.rgb, copy=True)[..., ::-1]   # image_direction = self.DIRECTIONS[ix]
            panorama.append(image)
        panorama = np.array(panorama)  # (num_views, H, W, 3)
        return panorama


class DiscreteVisualSimulatorV4(DiscreteVisualBase):
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60
    HFOV = vfov_to_hfov(VFOV, WIDTH, HEIGHT)
    DELTA_HOR = 30

    def __init__(self, navigable_dir, connectivity_dir, scan_dir, *args, **kwargs):
        super().__init__(navigable_dir, connectivity_dir, scan_dir, *args, **kwargs)

        self.use_panorama = False  # V4 does not support panorama

    @property
    def n_horizonal_views(self) -> int:
        """ Number of discrete horizontal views in the panorama """
        return 12
    
    @property
    def delta_heading(self) -> float:
        """ Heading change between adjacent horizontal views in radians """
        return math.radians(30)
    
    @property
    def n_vertical_views(self) -> int:
        """ Number of discrete vertical views in the panorama """
        return 3
    
    @property
    def delta_elevation(self) -> float:
        """ Elevation change between adjacent vertical views in radians """
        return math.radians(30)

    def make_candiate_views(self, view_indices: List[str]) -> List[Image.Image]:
        """ Make the candidate views images. """
        cand_view_images = []
        for view_index in view_indices:
            view_heading = self.heading + (view_index % self.n_horizonal_views) * self.delta_heading
            view_elevation = self.elevation + (1 - (view_index // self.n_horizonal_views)) * self.delta_elevation
            self.sim.newEpisode([self.scan_ID], [self.viewpoint_ID], [view_heading], [view_elevation])
            state = self.sim.getState()[0]
            image = np.array(state.rgb, copy=True)[..., ::-1]
            cand_view_images.append(Image.fromarray(image).convert('RGB'))
        assert len(cand_view_images) == len(view_indices)
    
        # set back to original heading and elevation
        self.sim.newEpisode([self.scan_ID], [self.viewpoint_ID], [self.heading], [self.elevation])
        return cand_view_images

    def make_pano(self) -> np.ndarray:
        """
        Get the panoramic RGB images at current viewpoint.
        :return: np.ndarray: (num_views, H, W, 3), in RGB channel, where as num_views = n_horizonal_views * n_vertical_views
        """
        num_views = self.n_horizonal_views * self.n_vertical_views
        panorama = []
        for ix in range(num_views):
            if ix == 0: # look up
                self.sim.newEpisode([self.scan_ID], [self.viewpoint_ID], [self.heading], 
                                    [self.elevation + self.delta_elevation])
            elif ix % self.n_horizonal_views == 0:  # look down and turn right
                self.sim.makeAction([0], [self.delta_heading], [-self.delta_elevation]) 
            else: self.sim.makeAction([0], [self.delta_heading], [0])  # turn right
            state = self.sim.getState()[0]
            image = np.array(state.rgb, copy=True)[..., ::-1]   # image_direction = self.DIRECTIONS[ix]
            panorama.append(image)
        panorama = np.array(panorama)  # (num_views, H, W, 3)
        return panorama

    def assign_view(self, abs_heading: float, abs_elevation: float,
                          agent_heading: float, agent_elevation: float) -> Tuple[int, float, float]:
        theta = normalize_angle_rad(abs_heading - agent_heading)
        phi = normalize_angle_rad(abs_elevation - agent_elevation)
        num_views = self.n_horizonal_views
        sector_size = 360 / num_views

        theta = normalize_angle_rad(theta)
        theta_deg = np.rad2deg(theta) % 360
        view_index = int((theta_deg + sector_size / 2) // sector_size) % num_views

        phi = normalize_angle_rad(phi)
        phi_deg = np.rad2deg(phi) % 360

        # [-90, -30], [-30, 30], [30, 90]
        if self.VFOV/2 <= phi_deg < self.VFOV*3/2: # up
            view_index = view_index
        elif 360 - 3/2*self.VFOV <= phi_deg <= 360 - self.VFOV/2: # down
            view_index = view_index + 2*num_views
        else: # level
            view_index = view_index + num_views

        return view_index, theta, phi
    

# ===============================================================================


class DiscreteEnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, 
                 simulator_class: str,
                 navigable_dir: str, 
                 location_dir:str, 
                 connectivity_dir: str=None, 
                 scan_dir: str=None, 
                 batch_size:int=100, 
                 feature_dir: str=None,
                 use_panorama: bool=False,
                 **kwargs
                 ):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.navigable_dir = navigable_dir
        self.connectivity_dir = connectivity_dir
        self.scan_dir = scan_dir
        self.feat_db = self.get_feature_db(feature_dir)
        
        self.sims: List[DiscreteSimulator] = []
        for i in range(batch_size):
            if simulator_class == "text": 
                sim = DiscreteTextSimulator(navigable_dir, location_dir)
            elif simulator_class == "visual_v1":
                sim = DiscreteVisualSimulatorV1(
                    navigable_dir, connectivity_dir, scan_dir, 
                    use_panorama=use_panorama, 
                    location_dir=location_dir
                )
            elif simulator_class == "visual_v2":
                sim = DiscreteVisualSimulatorV2(
                    navigable_dir, connectivity_dir, scan_dir, 
                    use_panorama=use_panorama, 
                    location_dir=location_dir
                )
            elif simulator_class == "visual_v3":
                sim = DiscreteVisualSimulatorV3(
                    navigable_dir, connectivity_dir, scan_dir, 
                    use_panorama=use_panorama, 
                    location_dir=location_dir
                )
            else: raise NotImplementedError(f"Simulator class {simulator_class} not implemented.")
            self.sims.append(sim)
        
        self.locations = self.load_locations(location_dir)
    
    @staticmethod
    def load_locations(location_dir):
        location_dict = defaultdict(dict)
        for filename in os.listdir(location_dir):
            if not filename.endswith('_vp_locations.json'):
                continue
            scan_id = filename.replace('_vp_locations.json', '')
            position_path = os.path.join(location_dir, filename)
            with open(position_path, 'r') as f: cur_location_dict = json.load(f)
            for k, v in cur_location_dict.items():
                if isinstance(v, str): 
                    location_dict[scan_id][k] = json.loads(v)
        return location_dict
    
    def _make_id(self, scanId: str, viewpointId: str) -> str:
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds: List[str], viewpointIds: List[str], headings: List[int]):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)    # default elevation=0

    def getStates(self):
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()
            feature = self.feat_db.get_image_observation(state["scanID"], state["viewpointID"])
            feature_states.append((feature, state))
        return feature_states

    def makeActions(self, next_viewpointIds: List[str]):
        ''' Take an action using the full state dependent action interface (with batched input)'''
        for i, next_viewpoint in enumerate(next_viewpointIds):
            if next_viewpoint is None: continue
            self.sims[i].makeAction(next_viewpoint)

    @staticmethod
    def get_feature_db(feature_dir: str) -> ImageObservationsDB:
        img_obs_dir = os.path.join(feature_dir, 'observations_list_summarized')
        img_obs_sum_dir = os.path.join(feature_dir, 'observations_summarized')
        img_obj_dir = os.path.join(feature_dir, 'objects_list')

        feat_db = ImageTextObservationsDB(
            img_obs_dir=img_obs_dir,
            img_obs_sum_dir=img_obs_sum_dir,
            img_obj_dir=img_obj_dir
        )
        return feat_db

    def get_vp_location(self, scan_id: str, viewpoint_ID: str) -> List[float]:
        """ Get the 3D location of a viewpoint by its ID. """
        if scan_id not in self.locations:
            raise ValueError(f"Scan ID {scan_id} not found in locations.")
        if viewpoint_ID not in self.locations[scan_id]:
            raise ValueError(f"Viewpoint ID {viewpoint_ID} not found in locations for scan {scan_id}.")
        return self.locations[scan_id][viewpoint_ID]

