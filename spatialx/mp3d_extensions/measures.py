import networkx as nx
import numpy as np


class DTW(object):
    """ Dynamic Time Warping (DTW) evaluation metrics.
    Python doctest:
        >>> graph = nx.grid_graph([3, 4])
        >>> prediction = [(0, 0), (1, 0), (2, 0), (3, 0)]
        >>> reference = [(0, 0), (1, 0), (2, 1), (3, 2)]
        >>> dtw = DTW(graph)
        >>> assert np.isclose(dtw(prediction, reference, 'dtw'), 3.0)
        >>> assert np.isclose(dtw(prediction, reference, 'ndtw'), 0.77880078307140488)
        >>> assert np.isclose(dtw(prediction, reference, 'sdtw'), 0.77880078307140488)
        >>> assert np.isclose(dtw(prediction[:2], reference, 'sdtw'), 0.0)
    """

    def __init__(self, graph=None, distance=None, weight='weight', threshold=3.0):
        """ Initializes a DTW object.
        Args:
            graph: networkx graph for the environment.
            distance: nx.all_pairs_dijkstra_path_length of a graph
            weight: networkx edge weight key (str).
            threshold: distance threshold dth (float).
        """
        assert graph is not None or distance is not None
        self.graph = graph
        self.weight = weight
        self.threshold = threshold
        if distance is None:
            self.distance = dict(nx.all_pairs_dijkstra_path_length(
                self.graph, weight=self.weight))
        else:
            self.distance = distance

    def __call__(self, prediction, reference, metric=['sdtw']):
        """ Computes DTW metrics.
        Args:
            prediction: list of nodes (str), path predicted by agent.
            reference: list of nodes (str), the ground truth path.
            metric: a list of ['ndtw', 'sdtw', 'dtw'].
            Returns:
            the DTW between the prediction and reference path (float).
        """
        assert set(metric) < {'ndtw', 'sdtw', 'dtw'}

        dtw_matrix = np.inf * \
            np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction)+1):
            for j in range(1, len(reference)+1):
                best_previous_cost = min(
                    dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
                cost = self.distance[prediction[i-1]][reference[j-1]]
                dtw_matrix[i][j] = cost + best_previous_cost
        dtw = dtw_matrix[len(prediction)][len(reference)]

        ndtw = np.exp(-dtw/(self.threshold * len(reference)))
        success = self.distance[prediction[-1]][reference[-1]] <= self.threshold
        sdtw = success * ndtw

        values = {"dtw": dtw, "ndtw": ndtw, "sdtw": sdtw}
        return [values[k] for k in metric]


class CLS(object):
    """Coverage weighted by length score (CLS).

    Python doctest:

    >>> cls = CLS(nx.grid_graph([3, 4]))
    >>> reference = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]
    >>> assert np.isclose(cls(reference, reference), 1.0)
    >>> prediction = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)]
    >>> assert np.isclose(cls(reference, prediction), 0.81994915125863865)
    >>> prediction = [(0, 1), (1, 1), (2, 1), (3, 1)]
    >>> assert np.isclose(cls(reference, prediction), 0.44197196102702557)

    Link to the original paper:
        https://arxiv.org/abs/1905.12255
    """

    def __init__(self, graph=None, distance=None, weight='weight', threshold=3.0):
        """Initializes a CLS object.

        Args:
        graph: networkx graph for the environment.
        weight: networkx edge weight key (str).
        threshold: distance threshold $d_{th}$ (float).
        """
        assert graph is not None or distance is not None
        self.graph = graph
        self.weight = weight
        self.threshold = threshold
        if distance is None:
            self.distance = dict(
                nx.all_pairs_dijkstra_path_length(
                    self.graph, weight=self.weight))
        else: self.distance = distance

    def __call__(self, prediction, reference):
        """Computes the CLS metric.

        Args:
        prediction: list of nodes (str), path predicted by agent.
        reference: list of nodes (str), the ground truth path.

        Returns:
        the CLS between the prediction and reference path (float).
        """

        def length(nodes):
            return float(np.sum([self.distance[edge[0]][edge[1]]
                            for edge in zip(nodes[:-1], nodes[1:])]))

        coverage = np.mean([
            np.exp(-np.min([  # pylint: disable=g-complex-comprehension
                self.distance[u][v] for v in prediction
            ]) / self.threshold) for u in reference
        ])
        expected = coverage * length(reference)
        score = expected / (expected + np.abs(expected - length(prediction)))
        return coverage * score


class DiscreteEvaluator(object):
    ERROR_MARGIN = 3.0

    def __init__(self, graphs=None, distances=None):
        """ Initializes an Evaluator object.
        Args:
            graphs: dict of networkx graphs for the environments.
            distances: dict of nx.all_pairs_dijkstra_path_length of graphs
        """
        self.graphs = graphs
        assert graphs is not None or distances is not None

        if distances is None:
            self.distances = {
                scan: dict(nx.all_pairs_dijkstra_path_length(graph))
                for scan, graph in self.graphs.items()
            }
        else: self.distances = distances
        
        self.scans = list(self.distances.keys())
    
    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            if not item: continue
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id
    
    def __call__(self, pred_path, gt_path, scan):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule).
            The path contains [vewpoint_id_1, vewpoint_id_2, .... ] or 
            [[vewpoint_id_1, vewpoint_id_2], [vewpoint_id_3], .... ]
        '''
        assert scan in self.scans, f'Scan {scan} not found in the dataset.'
        shortest_distances = self.distances[scan]  

        flat_path = sum(pred_path, [])      # flatten the path
        assert gt_path[0] == flat_path[0], 'Result trajectories should include the start position'
        action_steps = len(pred_path) - 1
        traj_steps = len(flat_path) - 1
        
        final_position = flat_path[-1]
        goal = gt_path[-1]
        nearest_position = self._get_nearest(scan, goal, pred_path)
        nav_error = shortest_distances[final_position][goal]
        oracle_error = shortest_distances[nearest_position][goal]
        nav_success = float(nav_error < self.ERROR_MARGIN)
        oracle_success = float(oracle_error < self.ERROR_MARGIN)

        dtw_worker = DTW(distance=shortest_distances, threshold=self.ERROR_MARGIN)
        ndtw, sdtw = dtw_worker(flat_path, gt_path, metric=['ndtw', 'sdtw'])

        cls_worker = CLS(distance=shortest_distances, threshold=self.ERROR_MARGIN)
        cls_score = cls_worker(flat_path, gt_path)

        pred_path_length = 0.0
        for a, b in zip(flat_path[:-1], flat_path[1:]):
            pred_path_length += shortest_distances[a][b]
        
        gt_path_length = 0.0
        for a, b in zip(gt_path[:-1], gt_path[1:]):
            gt_path_length += shortest_distances[a][b]
        spl = nav_success * (gt_path_length / max(pred_path_length, gt_path_length, 1e-5))

        scores = {
            "action_steps": action_steps,
            "trajectory_steps": traj_steps,
            "trajectory_lengths": pred_path_length,
            "nav_error": nav_error,
            "oracle_error": oracle_error,
            "nav_success": nav_success,
            "oracle_success": oracle_success,
            "spl": spl,
            "ndtw": ndtw,
            "sdtw": sdtw,
            "cls": cls_score,
        }
        return scores

