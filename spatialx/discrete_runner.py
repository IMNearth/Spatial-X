from typing import List, Tuple, Dict, Any
import json
import os, sys
sys.path.append(os.path.abspath("./Matterport3DSimulator/build"))
import argparse
import time
from dataclasses import dataclass, field
from spatialx.utils import HfArgumentParser
from spatialx.utils import print_with_color as print
from spatialx.agent import DISCRETE_AGENT_CONFIG_REGISTRY as AGENT_CONFIG_REGISTRY


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(path: str, data: List[Dict[str, Any]], mode: str='w'):
    with open(path, mode, encoding='utf-8') as f:
        for item in data: f.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")


def read_data(path: str) -> List[Dict[str, Any]]:
    if path.endswith('.jsonl'):
        return read_jsonl(path)
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else: raise ValueError(f"Unsupported file format: {path}")


def save_result_data(path: str, data: List[Dict[str, Any]]):
    if path.endswith('.jsonl'): 
        save_jsonl(path, data, mode="a")
    elif path.endswith('.json'):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, sort_keys=True, indent=4, ensure_ascii=False)
    else: raise ValueError(f"Unsupported file format: {path}")


def build_discrete_tasks(config):
    from spatialx.mp3d_extensions import R2RDataset, REVERIEDataset, RxRDataset
    from spatialx.mp3d_extensions import DiscreteNavBatch

    if config.data_name == "R2R":
        dataset = R2RDataset(
            data_dir=config.data_dir,
            splits=config.splits,
        )
    elif config.data_name == "REVERIE":
        dataset = REVERIEDataset(
            data_dir=config.data_dir,
            splits=config.splits,
        )
    elif config.data_name == "RXR":
        dataset = RxRDataset(
            data_dir=config.data_dir,
            splits=config.splits,
        )
    else: raise ValueError(f"Unknown discrete task dataset: {config.data_name}")
    print(f"Loaded {len(dataset)} episodes from splits: {config.splits}", color="cyan")
    print(json.dumps(dataset[0], indent=4), color="cyan")

    task_env = DiscreteNavBatch(
        name=config.simulator_name,
        dataset=dataset,
        batch_size=config.batch_size,
        navigable_dir=config.navigable_dir,
        location_dir=config.location_dir,
        connectivity_dir=config.connectivity_dir,
        scan_dir=config.scan_dir,
        feature_dir=config.feature_dir,
        use_panorama=config.use_panorama,
    )
    return task_env, dataset


def build_continuous_tasks(config):
    raise NotImplementedError("Continuous tasks are not implemented yet.")


def build_task(args, task_config):
    if args.task_type == "discrete":
        return build_discrete_tasks(task_config)
    elif args.task_type == "continuous":
        return build_continuous_tasks(task_config)
    else: raise ValueError(f"Unknown task type: {args.task_type}")


def build_agent(args, env, agent_config):
    if args.task_type == "discrete":
        if args.agent.lower() == "navgpt":
            from spatialx.agent import NavGPTAgent
            return NavGPTAgent(env, agent_config)
        if args.agent.lower() == "navgpt_spatial":
            from spatialx.agent import NavGPTSpatialAgent
            return NavGPTSpatialAgent(env, agent_config)
        if args.agent.lower() == "mapgpt":
            from spatialx.agent import MapGPTAgent
            return MapGPTAgent(env, agent_config)
        if args.agent.lower() == "mapgpt_spatial":
            from spatialx.agent import MapGPTSpatialAgent
            return MapGPTSpatialAgent(env, agent_config)
        if args.agent.lower() == "spatialnav_base":
            from spatialx.agent import SpatialNavBaseAgent
            return SpatialNavBaseAgent(env, agent_config)
        if args.agent.lower() == "spatialnav":
            from spatialx.agent import SpatialNavAgent
            return SpatialNavAgent(env, agent_config)
    else: raise ValueError(f"Unknown agent: {args.agent}")


def post_process_config(task_config, agent_config, general_args):
    # agent_config.llm_model_name = "Qwen2.5-VL-7B-Instruct"
    # agent_config.api_base_url = "http://localhost:8182/v1"
    # agent_config.api_key = "testkey"
    agent_save_dir = os.path.join(general_args.save_dir, general_args.agent.lower())
    os.makedirs(agent_save_dir, exist_ok=True)
    agent_config.save_dir = agent_save_dir

    if hasattr(agent_config, "api_base_url"):
        agent_config.api_base_url = general_args.api_base_url
    if hasattr(agent_config, "api_key"):
        agent_config.api_key = general_args.api_key
    if hasattr(agent_config, "seed"):
        agent_config.seed = general_args.seed
    if hasattr(task_config, "seed"):
        task_config.seed = general_args.seed
    
    if hasattr(agent_config, "scene_graph"):
        agent_config.scene_graph.draw_history = general_args.map_draw_history
        agent_config.scene_graph.grid_size = general_args.map_grid_size
        agent_config.scene_graph.draw_room_bounds = general_args.map_room_bounds
        agent_config.scene_graph.crop_map = general_args.map_crop_local_map
        agent_config.scene_graph.use_pred_layouts = general_args.use_pred_layouts
        if general_args.map_agent_orient == "none":
            agent_config.scene_graph.agent_front_up = False
        elif general_args.map_agent_orient == "absolute":
            agent_config.scene_graph.agent_front_up = True
            agent_config.scene_graph.rotation_strategy = "absolute"
        elif general_args.map_agent_orient == "relative":
            agent_config.scene_graph.agent_front_up = True
            agent_config.scene_graph.rotation_strategy = "relative"
        agent_config.map_save_dir = agent_save_dir + "/maps"
        if general_args.time_stamp:
            agent_config.map_save_dir += "/" + general_args.time_stamp
        os.makedirs(agent_config.map_save_dir, exist_ok=True)
    
    if hasattr(task_config, "data_name"):
        task_config.data_name = general_args.task.upper()
    
    # Ensure only one sample per batch for evaluation
    task_config.batch_size = 1
    


def parse_hierarchical_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--task_type",
        type=str,
        default="discrete",
        choices=["discrete", "continuous"],
        help="Type of navigation task: 'discrete' or 'continuous'.",
    )
    parser.add_argument(
        "--task", 
        type=str,
        default="R2R",
        help="Specific navigation task to run.",
    )
    parser.add_argument(
        "--agent", 
        type=str,
        default="spatialnav",
        choices=list(AGENT_CONFIG_REGISTRY.keys()),
        help="The name of the agent.",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default="http://openai.com/v1",
        help="Base URL for the OpenAI API.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="your_api_key_here",
        help="API key for the OpenAI API.",
    )
    parser.add_argument(
        "--config_file", 
        type=str,
        default=None,
        help="Path to the task configuration file.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Number of iterations to run. If None, run through the entire dataset once.",
    )
    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Whether to save detailed output including step-by-step actions and observations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--map_crop_local_map",
        action="store_true",
        help="Whether to crop the local map around the agent.",
    )
    parser.add_argument(
        "--map_draw_history",
        action="store_true",
        help="Whether to draw the agent's trajectory history on the map.",
    )
    parser.add_argument(
        '--map_agent_orient',
        choices=["none", "absolute", "relative"],
        default="absolute",
        help='Whether the agent front is always facing up in the map.'
    )
    parser.add_argument(
        "--map_grid_size",
        type=float,
        default=0.015,
        help="Grid size (in meters) for the scene graph mapper.",
    )
    parser.add_argument(
        "--map_room_bounds", 
        action="store_true",
        help="Whether to use room bounds in the scene graph mapper.",
    )
    parser.add_argument(
        '--restore_file',
        type=str,
        default=None,
        help='Path to a checkpoint file to restore the generated data.',
    )
    parser.add_argument(
        '--compute_preds_file', 
        type=str, 
        default=None, 
        help='Path to a prediction file to compute metrics on.',
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/results",
        help="Directory to save results and logs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1, 
        help="Number of workers for distributed evaluation.",
    )
    parser.add_argument(
        "--worker_idx",
        type=int,
        default=None, 
        help="Worker index for distributed evaluation.",
    )
    parser.add_argument(
        "--time_stamp", 
        type=str,
        default=None, 
        help="Time stamp for the current run.",
    )
    parser.add_argument(
        "--use_pred_layouts",
        action="store_true",
        help="Whether to use predicted layouts from the layout predictor.",
    )
    general_args, remainings = parser.parse_known_args()
    # print(remainings, color="yellow")

    agent_cfg_class = AGENT_CONFIG_REGISTRY.get(general_args.agent.lower())
    if general_args.task_type == "discrete":
        from spatialx.mp3d_extensions import TASK_CONFIG_REGISTRY as DISCRETE_TASKS
        task_cfg_class = DISCRETE_TASKS.get(general_args.task.lower())
        hf_parser = HfArgumentParser((task_cfg_class, agent_cfg_class))
        task_config, agent_config = hf_parser.parse_args_into_dataclasses(args=remainings)
    elif general_args.task_type == "continuous":
        raise NotImplementedError("Continuous tasks are not implemented yet.")
    else: raise ValueError(f"Unknown task type: {general_args.task_type}")
    post_process_config(task_config, agent_config, general_args)

    return general_args, task_config, agent_config



if __name__ == "__main__":
    args, data_cfg, agent_cfg = parse_hierarchical_args()
    print("Parsed Arguments:", color="green")
    print(args)
    print("Data Config:", color="blue")
    print(data_cfg)
    print("Agent Config:", color="magenta")
    print(agent_cfg)

    task_env, dataset = build_task(args, data_cfg)
    agent = build_agent(args, task_env, agent_cfg)
    print(f"Agent built successfully. {len(dataset)} episodes to run ...", color="yellow")

    if args.compute_preds_file is not None:
        print(f"Computing metrics for predictions in {args.compute_preds_file}", color="cyan")
        preds = read_data(args.compute_preds_file)
        print(f"Loaded {len(preds)} predictions.", color="cyan")
        # preds = [p for p in preds[:5] if p["instr_id"] != "4041_0"]
        avg_metrics, metrics = task_env.compute_metrics(preds)
        print(len(metrics["instr_id"]), " episodes evaluated.", color="green")
        print("Average Metrics:", json.dumps(avg_metrics, indent=4), color="green")
        # json.dump(preds, open(args.compute_preds_file.replace('.json', '_with_scores.json'), 'w'),
        #           sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        sys.exit(0)
    
    if args.restore_file is not None and os.path.exists(args.restore_file):
        print(f"Restoring results from {args.restore_file}")
        preds = read_data(args.restore_file)
        preds = [p for p in preds if 'error' not in p]
        print(f"Loaded {len(preds)} bug-free predictions.")
        new_data = []
        for data in dataset.data:
            if any(pred['instr_id'] == data['instr_id'] for pred in preds):
                continue
            new_data.append(data)
        dataset.data = new_data
        task_env.data = dataset
        print(f"{len(task_env.data)} episodes remaining after restoration.", color="yellow")
    else: preds = None

    if args.num_workers > 1 and args.worker_idx is not None:
        num_samples = len(dataset)
        subset_size = num_samples // args.num_workers
        start_idx = args.worker_idx * subset_size
        end_idx = (args.worker_idx + 1) * subset_size \
            if args.worker_idx != args.num_workers - 1 else num_samples
        dataset.data = dataset.data[start_idx:end_idx]
        task_env.data = dataset
        print(f"[Worker {args.worker_idx}] Starting inference on {len(task_env.data)} episodes (index {start_idx} to {end_idx})", color="yellow")
        # deal with restored results
        if preds is not None: 
            preds_end_idx = (args.worker_idx + 1) * (len(preds) // args.num_workers) \
                if args.worker_idx != args.num_workers - 1 else len(preds)
            preds = preds[args.worker_idx * (len(preds) // args.num_workers): preds_end_idx]
    else: print(f"Starting inference on {len(task_env.data)} episodes.", color="yellow")

    llm_short_name = agent_cfg.llm_model_name.split('-2025')[0]
    if hasattr(agent_cfg, "prompt_version"): llm_short_name += f"-{agent_cfg.prompt_version}"
    result_save_file = f"{llm_short_name}_{task_env.name}_{dataset.name}.jsonl"
    if "spatial" in args.agent.lower():
        result_save_file = result_save_file.replace('.json', f'_ori:{args.map_agent_orient[:3]}.json')
    result_save_path = os.path.join(agent.config.save_dir, result_save_file)
    print(f"Results will be saved to {result_save_path}", color="cyan")
    
    if (not os.path.exists(result_save_path)) or (args.restore_file is not None):
        start_time = time.time()
        if args.restore_file is not None:
            agent.test(iters=args.iters, restore_results=preds, worker_idx=args.worker_idx, time_str=args.time_stamp)
        else: agent.test(iters=args.iters, worker_idx=args.worker_idx, time_str=args.time_stamp)
        print('Inference cost time: %.2fs' % (time.time() - start_time), color="yellow")
        
        if not (args.num_workers > 1 and args.worker_idx is not None):
            preds = agent.get_results(detailed_output=False)
            save_result_data(result_save_path, preds)
            if args.detailed_output:
                detailed_preds = agent.get_results(detailed_output=True)
                detailed_result_save_file = result_save_file.replace('.json', '_detailed.json')
                save_result_data(os.path.join(agent.config.save_dir, detailed_result_save_file), detailed_preds)
        else: 
            with open(f"{agent.config.save_dir}/save_path.txt", "w") as f: f.write(result_save_path)
    else: 
        print(f"Results already exist at {result_save_path}, skipping inference.", color="green")
        preds = read_data(result_save_path)

    if not (args.num_workers > 1 and args.worker_idx is not None):
        print(f"Loaded {len(preds)} predictions.", color="cyan")
        avg_metrics, metrics = task_env.compute_metrics(preds)
        print(len(metrics["instr_id"]), " episodes evaluated.", color="green")
        print("Average Metrics:", json.dumps(avg_metrics, indent=4), color="green")
    
    pass


