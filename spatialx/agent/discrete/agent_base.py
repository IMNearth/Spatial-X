import os
import json
import time
import traceback
from spatialx.mp3d_extensions import DiscreteNavBatch


class BaseAgent(object):
    ''' Base class for an REVERIE agent to generate and save trajectories. '''
    safe_max_iters = 1000

    def __init__(self, env:DiscreteNavBatch):
        self.env = env
        self.results = {}
        self.extra_output = []

        if len(self.env) < self.safe_max_iters: 
            self.safe_max_iters = len(self.env) + 1
        print(f"Safe max iters set to {self.safe_max_iters}.")

    def get_results(self, detailed_output=False):
        ''' Extract the results into a list. '''
        llm_model_name = self.config.llm_model_name if hasattr(self, 'config') \
                         and hasattr(self.config, 'llm_model_name') else "unknown"
        output = []
        for k, v in self.results.items():
            output.append({
                'scan': v['scan'], 
                'instr_id': k, 
                'trajectory': v['path'],
                'llm_model_name': llm_model_name
            })
            if 'error' in v: output[-1]['error'] = v['error']
            if detailed_output:
                if v.get('details', None):
                    output[-1]['details'] = v['details']
                if v.get('action_plan', None) is not None:
                    output[-1]['action_plan'] = v['action_plan']
                if v.get('llm_output', None) is not None:
                    output[-1]['llm_output'] = v['llm_output']
                if v.get('llm_thought', None) is not None:
                    output[-1]['llm_thought'] = v['llm_thought']
                if v.get('llm_observation', None) is not None:
                    output[-1]['llm_observation'] = v['llm_observation']
                if v.get('a_t', None) is not None:
                    output[-1]['a_t'] = v['a_t']
        
        if self.extra_output:
            for item in self.extra_output:
                output.append({
                    'scan': item['scan'], 
                    'instr_id': item['instr_id'], 
                    'trajectory': item['trajectory'],
                    'llm_model_name': item.get('llm_model_name', llm_model_name)
                })
                if detailed_output and "details" in item:
                    output[-1]['details'] = item['details']
                    if item.get('action_plan', None) is not None:
                        output[-1]['action_plan'] = item['action_plan']
                    if item.get('llm_output', None) is not None:
                        output[-1]['llm_output'] = item['llm_output']
                    if item.get('llm_thought', None) is not None:
                        output[-1]['llm_thought'] = item['llm_thought']
                    if item.get('llm_observation', None) is not None:
                        output[-1]['llm_observation'] = item['llm_observation']
                    if item.get('a_t', None) is not None:
                        output[-1]['a_t'] = item['a_t']
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing the results. '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, reset=False, **kwargs):
        if reset: # If iters is not none, shuffle the env batch
            self.env.reset_epoch(shuffle=(iters is not None))
        
        if "time_str" in kwargs:
            time_str = kwargs.pop("time_str")
            if not time_str: time_str = time.strftime("%m%d-%H%M")
        else: time_str = time.strftime("%m%d-%H%M")
        worker_idx = kwargs.pop("worker_idx", None)
        if worker_idx is not None: time_str += f"_mp{worker_idx}"
        test_save_path = os.path.join(self.config.save_dir, f'runtime_{time_str}.json')
        os.makedirs(os.path.dirname(test_save_path), exist_ok=True)

        self.results = {} 
        self.extra_output = [] if 'restore_results' not in kwargs else kwargs.pop('restore_results')
        # We rely on env showing the entire batch before repeating anything
        looped = False
        if iters is not None:
            # For each time, it will run the first 'iters' iterations.
            for i in range(iters):
                try:
                    for traj in self.rollout(**kwargs):
                        self.results[traj['instr_id']] = traj
                        preds_detail = self.get_results(detailed_output=True)
                        json.dump(
                            preds_detail, open(test_save_path, 'w'),
                            sort_keys=True, indent=4, separators=(',', ': ')
                        )
                except Exception as e:
                    error_trace = traceback.format_exc()
                    print(f"!! Exception during rollout at iteration {i}: {error_trace}")
                    for traj in self.traj:
                        if traj['instr_id'] not in self.results:
                            self.results[traj['instr_id']] = {"error": error_trace}
                        else: self.results[traj['instr_id']]["error"] = error_trace
                        preds_detail = self.get_results(detailed_output=True)
                        json.dump(
                            preds_detail, open(test_save_path, 'w'),
                            sort_keys=True, indent=4, separators=(',', ': ')
                        )
                    pass
        else: # Do a full round
            i = 0
            while not looped:
                try: 
                    for traj in self.rollout(**kwargs):
                        if traj['instr_id'] in self.results:
                            looped = True
                        else:
                            self.results[traj['instr_id']] = traj
                            preds_detail = self.get_results(detailed_output=True)
                            json.dump(
                                preds_detail, open(test_save_path, 'w'),
                                sort_keys=True, indent=4, separators=(',', ': ')
                        )
                except Exception as e:
                    error_trace = traceback.format_exc()
                    print(f"Exception during rollout at iteration {i}: {error_trace}")
                    for traj in self.traj:
                        traj["error"] = error_trace
                        self.results[traj['instr_id']] = traj
                        preds_detail = self.get_results(detailed_output=True)
                        json.dump(
                            preds_detail, open(test_save_path, 'w'),
                            sort_keys=True, indent=4, separators=(',', ': ')
                        )
                    pass
                i += 1
                if i > self.safe_max_iters: break # Safeguard against infinite loop
        
        return test_save_path

