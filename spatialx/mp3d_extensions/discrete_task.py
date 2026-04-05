""" Dataset and Task API for discrete navigation in Matterport3D environments. """
from typing import List, Dict, Any, Union
import json
import random
import numpy as np
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass, field

from .mp_utils import load_nav_graphs
from .discrete_env import DiscreteEnvBatch
from .measures import DiscreteEvaluator


@dataclass
class DiscreteDataConfig:
    """ Configuration for discrete navigation. """
    data_name: str = field(
        default="R2R",
        metadata={"help": "Name of the dataset/task, e.g., 'R2R', 'REVERIE', 'RxR'."}
    )
    data_dir: str = field(
        default="data/tasks/R2R",
        metadata={"help": "Directory containing the dataset files."}
    )
    splits: Union[str, List[str]] = field(
        default_factory=lambda: ["val_seen", "val_unseen"],
        metadata={"help": "Dataset splits to use (e.g., 'train', 'val', 'test')."}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for training or evaluation."}
    )
    simulator_name: str = field(
        default="text_sim",
        metadata={"help": "Name of the simulator to use, choose from [text_sim, visual_v1_sim, visual_v2_sim]."}
    )
    scan_dir: str = field(
        default="data/scene_datasets/mp3d/v1/scans",
        metadata={"help": "Directory containing 3D scan data."}
    )
    navigable_dir: str = field(
        default="data/tasks/R2R/navigable",
        metadata={"help": "Directory containing navigable location data."}
    )
    location_dir: str = field(
        default="data/tasks/R2R/locations",
        metadata={"help": "Directory containing location data."}
    )
    connectivity_dir: str = field(
        default="data/tasks/R2R/connectivity",
        metadata={"help": "Directory containing connectivity graph data."}
    )
    feature_dir: str = field(
        default="data/tasks/R2R",
        metadata={"help": "Directory containing precomputed feature data."}
    )
    use_panorama: bool = field(
        default=True,
        metadata={"help": "Whether to use panoramic visual features."}
    )



class VLNDataset(object):

    def __init__(self, *args, **kwargs):
        self.data = []
        pass

    @staticmethod
    def split_instrs(data: List[Dict[str, Any]]):
        new_data: List[dict] = []
        for item in data:
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr
                del new_item['instructions']
                new_data.append(new_item)
        return new_data

    @staticmethod
    def load_datasets(splits: Union[str, List[str]], dataset:str="R2R", data_dir="data/tasks/R2R"):
        """ should be changed based on your task and data """
        if isinstance(splits, str): splits = [splits]
        
        data = []
        for split in splits:
            with open('%s/%s_%s.json' % (data_dir, dataset, split)) as f:
                data += json.load(f)
        return data

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


class R2RDataset(VLNDataset):
    """ Dataset class for Room-to-Room navigation data. """
    
    def __init__(self, data_dir:str, splits:Union[str, List[str]], **kwargs):
        self.data_dir = data_dir
        self.splits = splits

        if "data" not in kwargs: 
            self.data = self.load_datasets(splits, dataset="R2R", data_dir=data_dir)
            self.data = self.split_instrs(self.data)
        else: self.data = kwargs["data"]
        self.id2data = dict()
    
    def __getitem__(self, index):
        """ Get a data item by index. 
        Returns: A dictionary containing the data for the specified index.
        {
            "scan": scan_id,
            "path_id": path_id,
            "instr_id": instruction_id,
            "path": [viewpoint_id1, viewpoint_id2, ...],
            "instruction": instr,
            "heading": heading_in_radians,
        }
        """
        it_or_its = self.data[index]
        if isinstance(it_or_its, list):
            for it in it_or_its:
                instr_id = it['instr_id']
                if instr_id not in self.id2data:
                    self.id2data[instr_id] = it
        else:  
            instr_id = it_or_its['instr_id']
            if instr_id not in self.id2data:
                self.id2data[instr_id] = it_or_its
        return it_or_its
    
    def get_data_by_id(self, instr_id: str) -> str:
        """ Get instruction text by instruction ID. """
        if instr_id in self.id2data:
            return self.id2data[instr_id]
        else:
            for item in self.data:
                if item['instr_id'] == instr_id:
                    self.id2data[instr_id] = item
                    return item
        raise KeyError(f"Instruction ID {instr_id} not found in dataset.")

    @property
    def name(self) -> str:
        return "R2R("+",".join(self.splits)+")"
    
    def __repr__(self) -> str:
        return f"R2R(splits={self.splits}, size={len(self)})"



class REVERIEDataset(R2RDataset):

    def __init__(self, data_dir:str, splits:Union[str, List[str]], **kwargs):
        self.data_dir = data_dir
        self.splits = splits

        if "data" not in kwargs: 
            self.data = self.load_datasets(splits, dataset="REVERIE", data_dir=data_dir)
            self.data = self.split_instrs(self.data)
        else: self.data = kwargs["data"]

        self.id2data = dict()

    @staticmethod
    def split_instrs(data: List[Dict[str, Any]]):
        new_data: List[dict] = []
        for item in data:
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['id'], j)
                new_item['instruction'] = instr
                del new_item['instructions']
                new_data.append(new_item)
        return new_data

    @property
    def name(self) -> str:
        return "REVERIE("+",".join(self.splits)+")"
    
    def __repr__(self) -> str:
        return f"REVERIE(splits={self.splits}, size={len(self)})"



class RxRDataset(R2RDataset):

    def __init__(self, data_dir:str, splits:Union[str, List[str]], **kwargs):
        self.data_dir = data_dir
        if isinstance(splits, str): splits = [splits]
        self.splits = splits

        if "data" not in kwargs: 
            self.data = []
            for split in splits:
                with open('%s/RxR_%s_english_guide.json' % (data_dir, split), "r") as f:
                    self.data += json.load(f)
        else: self.data = kwargs["data"]

        for item in self.data:
            item["instr_id"] = str(item["instruction_id"])
        self.id2data = dict()

    @property
    def name(self) -> str:
        return "RxR("+",".join(self.splits)+")"
    
    def __repr__(self) -> str:
        return f"RxR(splits={self.splits}, size={len(self)})"



class DiscreteNavBatch(object):
    """ A batch container for discrete navigation tasks. """
    env_type = "mp3d"
    
    def __init__(self, name:str, dataset: List[Dict[Any, Any]], batch_size:int, 
                 navigable_dir: str, location_dir:str, 
                 connectivity_dir:str=None, scan_dir: str=None, **kwargs):
        self.name = name
        self.data = dataset
        self.batch_size = batch_size

        self.navigable_dir = navigable_dir
        self.location_dir = location_dir
        self.connectivity_dir = connectivity_dir
        self.scan_dir = scan_dir

        self.env = DiscreteEnvBatch(
            simulator_class=self.parse_simulator_class(),
            navigable_dir=navigable_dir,
            location_dir=location_dir,
            connectivity_dir=connectivity_dir, 
            scan_dir=scan_dir,
            batch_size=batch_size, 
            **kwargs
        )

        self.ix = 0
        self.scans = list({item['scan'] for item in self.data})
        self._load_nav_graphs()

        self._load_gt_trajs()
        self.evaluator = DiscreteEvaluator(
            graphs=self.graphs,
            distances=self.shortest_distances
        )
        print(f"Initialized DiscreteNavBatch with {self.name}.")
    
    def parse_simulator_class(self) -> str:
        """ Parse the simulator class from the task name. """
        if "_sim" in self.name.lower():
            return self.name.lower().split("_sim")[0]
        else: raise NotImplementedError(f"Simulator class for task {self.name} not implemented.")

    def __len__(self):
        return len(self.data)
    
    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs: Dict[nx.Graph] = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _load_gt_trajs(self):
        """ Load ground-truth trajectories for each data item. """
        self.gt_repo = {}
        for item in self.data:
            instr_id = item['instr_id']
            scan = item['scan']
            self.gt_repo[(scan, instr_id)] = item['path']

    def _next_minibatch(self, batch_size=None, tile_one=False, sort=True, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if tile_one:
            batch = [self.data[self.ix]] * self.batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+self.batch_size]
            if len(batch) < self.batch_size:
                random.shuffle(self.data)
                self.ix = self.batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += self.batch_size
        
        if sort and 'instr_length' in batch[0]: # by instructions length, high -> low:
            batch = sorted(batch, key=lambda item: item['instr_length'], reverse=True)

        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self.observe()

    def observe(self) -> List[Dict[str, Any]]:
        """ Get observations from the environment. """
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]

            if 'rgb' in state: obs_image = state['rgb']
            else: obs_image = feature.get('image_rgb', None)

            if 'panorama' in state: obs_panorama = state['panorama']
            else: obs_panorama = feature.get('panorama_rgb', None)

            _ob = {
                'scan'     : state['scanID'],                       # scan id
                'id': item.get("id", item['path_id']),              # unique id
                'path_id' : item['path_id'],                        # path id
                'instr_id' : item['instr_id'],                      # instruction id
                'instruction' : item['instruction'],                # instruction
                'viewpoint' : state['viewpointID'],                 # current viewpoint id
                'position' : state['position'],                     # current position (x, y, z)
                'heading'  : state['heading'],                      # current heading
                'elevation': state['elevation'],                    # current elevation
                'candidate': state['candidate'],                    # candidate viewpoints
                # ---------- Textual Observations ----------
                'obs_detail' : feature["detail"],                   # NavGPT annotated details
                'obs_summary' : feature["summary"],                 # NavGPT annotated summary
                'obs_objects' : feature["objects"],                 # NavGPT annotated objects
                # ---------- Visual Observations  ----------
                'obs_image': obs_image,                             # front view
                'obs_panorama': obs_panorama,                       # panoramic view
            }
            _ob['distance'] = self.shortest_distances[_ob['scan']][_ob['viewpoint']][item['path'][-1]],
            obs.append(_ob)
        
        return obs

    def step(self, next_viewpointIds: List[Union[str, None]]) -> List[Dict[str, Any]]:
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(next_viewpointIds)
        return self.observe()

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """ Compute evaluation metrics for the given results. """
        metrics = defaultdict(list)

        for res in results:
            # if 'error' in res: continue
            if 'scan' not in res: continue
            scan = res['scan']
            instr_id = res['instr_id']
            gt_traj = self.gt_repo[(scan, instr_id)]
            res["ground_truth"] = gt_traj
            pred_traj = res['trajectory']
            screos = self.evaluator(pred_traj, gt_traj, scan)
            res["scores"] = screos
            for k, v in screos.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items() if k != 'instr_id'}
        return avg_metrics, metrics

    def get_vp_location(self, scan_id: str, viewpoint_ID: str) -> List[float]:
        """ Get the 3D location of a viewpoint by its ID. """
        return self.env.get_vp_location(scan_id, viewpoint_ID)

