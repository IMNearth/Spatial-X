import os
import math
import json
import networkx as nx
import numpy as np
from PIL import Image
from collections import defaultdict, deque


# -----------------------------
# ----  Simulator utilities  ----
# -----------------------------
def build_mp3d_simulator(connectivity_dir, scan_dir, width:int=448, height:int=448, vfov:int=45):
    import MatterSim

    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(width, height)
    sim.setCameraVFOV(math.radians(vfov))
    # sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim


# -----------------------------
# ----  Feature utilities  ----
# -----------------------------
class ImageObservationsDB(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_image_observation(self, scan: str, viewpoint: str) -> dict:
        raise NotImplementedError


class ImageTextObservationsDB(ImageObservationsDB):
    def __init__(self, img_obs_dir, img_obs_sum_dir, img_obj_dir):
        self.img_obs_dir = img_obs_dir
        self.img_obs_sum_dir = img_obs_sum_dir
        self.img_obj_dir = img_obj_dir
        assert os.path.exists(self.img_obs_dir), f"Image observation directory {self.img_obs_dir} does not exist."
        assert os.path.exists(self.img_obs_sum_dir), f"Image observation summary directory {self.img_obs_sum_dir} does not exist."
        assert os.path.exists(self.img_obj_dir), f"Image object directory {self.img_obj_dir} does not exist."
        self._obs_store = {}

    def get_image_observation(self, scan: str, viewpoint: str) -> dict:
        key = '%s_%s' % (scan, viewpoint)
        if key in self._obs_store:
            obs = self._obs_store[key]
        else:
            # Load image observation
            with open(os.path.join(self.img_obs_dir, f'{scan}.json'), 'r') as f:
                obs = json.load(f)[viewpoint]
                self._obs_store[key] = {}
                self._obs_store[key]['detail'] = obs
            # Load image observation summary for history
            with open(os.path.join(self.img_obs_sum_dir, f'{scan}_summarized.json'), 'r') as f:
                obs_sum = json.load(f)[viewpoint]
                self._obs_store[key]['summary'] = obs_sum
            # Load image objects
            with open(os.path.join(self.img_obj_dir, f'{scan}.json'), 'r') as f:
                obj = json.load(f)[viewpoint]
                self._obs_store[key]['objects'] = obj
            obs = self._obs_store[key]
        return obs


class ImageVisionObservationsDB(ImageObservationsDB):
    def __init__(self, img_pic_dir, is_lazy=True):
        self.img_pic_dir = img_pic_dir
        assert os.path.exists(self.img_pic_dir), f"Image picture directory {self.img_pic_dir} does not exist."
        self.is_lazy = is_lazy
        self._obs_store = {}

    def get_image_observation(self, scan: str, viewpoint: str) -> dict:
        key = '%s_%s' % (scan, viewpoint)
        if key in self._obs_store:
            obs = self._obs_store[key]
        else:
            # Load image picture
            img_path = os.path.join(self.img_pic_dir, scan, f'{viewpoint}.jpg')
            self._obs_store[key] = {}
            self._obs_store[key]['image'] = img_path if self.is_lazy \
                else Image.open(img_path).convert('RGB')

        return obs


class ImageMixedObservationsDB(ImageObservationsDB):
    def __init__(self, img_pic_dir, img_obs_dir, img_obs_sum_dir, img_obj_dir, is_lazy=True):
        self.img_pic_dir = img_pic_dir
        self.img_obs_dir = img_obs_dir
        self.img_obs_sum_dir = img_obs_sum_dir
        self.img_obj_dir = img_obj_dir
        assert os.path.exists(self.img_pic_dir), f"Image picture directory {self.img_pic_dir} does not exist."
        assert os.path.exists(self.img_obs_dir), f"Image observation directory {self.img_obs_dir} does not exist."
        assert os.path.exists(self.img_obs_sum_dir), f"Image observation summary directory {self.img_obs_sum_dir} does not exist."
        assert os.path.exists(self.img_obj_dir), f"Image object directory {self.img_obj_dir} does not exist."
        self.is_lazy = is_lazy
        self._obs_store = {}
    
    def get_image_observation(self, scan: str, viewpoint: str) -> dict:
        key = '%s_%s' % (scan, viewpoint)
        if key in self._obs_store:
            obs = self._obs_store[key]
        else:
            # Load image picture
            img_path = os.path.join(self.img_pic_dir, scan, f'{viewpoint}.jpg')
            self._obs_store[key] = {}
            self._obs_store[key]['image'] = img_path if self.is_lazy \
                else Image.open(img_path).convert('RGB')
            # Load image observation
            with open(os.path.join(self.img_obs_dir, f'{scan}.json'), 'r') as f:
                obs = json.load(f)[viewpoint]
                self._obs_store[key]['detail'] = obs
            # Load image observation summary for history
            with open(os.path.join(self.img_obs_sum_dir, f'{scan}_summarized.json'), 'r') as f:
                obs_sum = json.load(f)[viewpoint]
                self._obs_store[key]['summary'] = obs_sum
            # Load image objects
            with open(os.path.join(self.img_obj_dir, f'{scan}.json'), 'r') as f:
                obj = json.load(f)[viewpoint]
                self._obs_store[key]['objects'] = obj
            obs = self._obs_store[key]
        return obs


# ---------------------------
# ----  Graph utilities  ----
# ---------------------------

def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


class NavGraph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def update_connection(self, node1, node2):
        self.add_node(node1)
        self.add_node(node2)
        if node2 in self.graph[node1]:
            return None
        self.graph[node1].append(node2)
        self.graph[node2].append(node1)

    def bfs_shortest_path(self, start, end) -> list:
        if start not in self.graph or end not in self.graph:
            return []

        visited = {start: None}
        queue = deque([start])

        while queue:
            current_node = queue.popleft()

            if current_node == end:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = visited[current_node]
                return path[::-1]

            for neighbor in self.graph[current_node]:
                if neighbor not in visited:
                    visited[neighbor] = current_node
                    queue.append(neighbor)

        return []


class FloydGraph:
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    if self._dis[x][k] + self._dis[k][y] < self._dis[x][y]:
                        self._dis[x][y] = self._dis[x][k] + self._dis[k][y]
                        self._dis[y][x] = self._dis[x][y]
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)


# ------------------------------
# ----  Evaluation metrics  ----
# ------------------------------  

def cal_dtw(shortest_distances, prediction, reference, success=None, threshold=3.0):
    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction)+1):
        for j in range(1, len(reference)+1):
            best_previous_cost = min(
                dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
            cost = shortest_distances[prediction[i-1]][reference[j-1]]
            dtw_matrix[i][j] = cost + best_previous_cost

    dtw = dtw_matrix[len(prediction)][len(reference)]
    ndtw = np.exp(-dtw/(threshold * len(reference)))
    if success is None:
        success = float(shortest_distances[prediction[-1]][reference[-1]] < threshold)
    sdtw = success * ndtw

    return {
        'DTW': dtw,
        'nDTW': ndtw,
        'SDTW': sdtw
    }


def cal_cls(shortest_distances, prediction, reference, threshold=3.0):
    def length(nodes):
      return np.sum([
          shortest_distances[a][b]
          for a, b in zip(nodes[:-1], nodes[1:])
      ])

    coverage = np.mean([
        np.exp(-np.min([  # pylint: disable=g-complex-comprehension
            shortest_distances[u][v] for v in prediction
        ]) / threshold) for u in reference
    ])
    expected = coverage * length(reference)
    score = expected / (expected + np.abs(expected - length(prediction)))
    return coverage * score