"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach.
   And with Multimodal Input (e.g., images, top-down spatial map)."""
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Dict, Union
import re
import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict
from functools import partial

# language chain dependencies
from langchain.agents.agent import (
    AgentExecutor,
    AgentAction, 
    AgentFinish, 
    AgentOutputParser, 
    OutputParserException, 
    Callbacks,
)
from langchain.agents.tools import Tool
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# custom dependencies
from spatialx.mp3d_extensions import DiscreteNavBatch
from spatialx.mp3d_extensions.pc_utils import cartesian_to_spherical
from spatialx.spatial.agraph import SceneObjectGraph, SceneObjectConfig, Object3D
from .config import SpatialNavConfig
from .prompts import (
    HISTORY_PROMPT,
    MAKE_ACTION_TOOL_NAME,
    MAKE_ACTION_TOOL_DESCRIPTION,
    VLN_SPATIAL_VISUAL_PROMPT
)
from spatialx.utils import MultimodalPromptTemplate, MultimodalOpenAI
from .agent_spatial import NavGPTSpatialAgent


FINAL_ANSWER_ACTION = "Final Answer:"
EXCEPTION_TOOL_NAME = "_Exception"

DEFAULT_PARSE_ERROR_MESSAGE = (
    "Invalid Format: The output format is invalid. Try again.")
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:")
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'")
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:")



class SpatialNavVLNAgent(ZeroShotAgent):
    """ An agent for Vision-and-Language Navigation with spatial reasoning. """
    # 限制 scratchpad 长度，避免超过 LLM 上下文窗口
    max_scratchpad_length: int = 4096
    # 随着导航进程不断更新的历史观察与场景图
    history: Optional[List[str]] = None 
    scene_graph: Optional[Image.Image] = None
    panorama_img: Optional[Image.Image] = None
    class Config: arbitrary_types_allowed = True

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought: "

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]], 
        init_observation: str=""
    ) -> str:
        """ Construct the scratchpad that lets the agent continue its thought process. """
        thoughts = ""
        nav_step = 1
        for i, (action, observation) in enumerate(intermediate_steps):
            # action_log = action.log.strip(self.llm_prefix).strip()
            if self.llm_prefix.strip() in action.log:
                action_log = action.log.split(self.llm_prefix.strip())[-1].strip()
            else: action_log = action.log.strip()
            action_log = "\n".join([x.strip() for x in action_log.split("\n") if x.strip() != ""])
            thoughts += action_log
            if (i == len(intermediate_steps) - 1) or (action.tool != MAKE_ACTION_TOOL_NAME):
                if action.tool != MAKE_ACTION_TOOL_NAME: # handle the parsing error case
                    cur_observation = observation + "\n" + init_observation
                    for j in range(i-1, -1, -1):
                        action_j, observation_j = intermediate_steps[j]
                        if action_j.tool == MAKE_ACTION_TOOL_NAME:
                            cur_observation = observation + "\n" + observation_j
                else: cur_observation = observation
                thoughts += f"\n{self.observation_prefix}{cur_observation}\n{self.llm_prefix}"
            else:
                thoughts += f"\n{self.observation_prefix}{self.history[nav_step]}\n{self.llm_prefix}"
                nav_step += 1
        return thoughts

    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps, kwargs["init_observation"])
        thoughts = thoughts[-self.max_scratchpad_length:]
        if len(thoughts) == self.max_scratchpad_length:
            thoughts = "... ..." + thoughts[-(self.max_scratchpad_length - 7):]
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        
        if len(intermediate_steps) == 0:
            full_inputs = {**kwargs, **new_inputs}
        else:
            kwargs["init_observation"] = self.history[0]
            if isinstance(self.panorama_img, Image.Image):
                kwargs["images"] = [self.scene_graph, self.panorama_img]
            elif isinstance(self.panorama_img, list):
                kwargs["images"] = [self.scene_graph] + self.panorama_img
            full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
        try: parsed_output = self.output_parser.parse(full_output)
        except OutputParserException as e:
            if len(intermediate_steps) == 0:
                prev_observation = kwargs["init_observation"]
            else: prev_observation = intermediate_steps[-1][1]
            e.observation += "\n" + prev_observation
            raise e
        return parsed_output



class SpatialNavBaseAgent(NavGPTSpatialAgent):
    name = "spatialnav_wo_obj"
    panorama_save_dir = "outputs/results/spatialnav_wo_obj/panorama_images"

    def __init__(self, env: DiscreteNavBatch, config: SpatialNavConfig):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The discrete Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env, config)
        self.vln_agent: SpatialNavVLNAgent = self.agent_executor.agent
    
    @property
    def agent_prompt_template(self) -> str:
        return VLN_SPATIAL_VISUAL_PROMPT

    @staticmethod
    def angle_to_left_right(angle):
        return f"left {-angle:.2f}°" if angle < 0 else f"right {angle:.2f}°"

    @staticmethod
    def compact_dict_str(d, indent=4):
        if len(d) == 0: return " {" + "}"
        output_str = [" "* indent + "{"]
        for i, (k, v) in enumerate(d.items()):
            cur_str = " "* (indent + 2) + f"'{k}': '{v}'"
            if i != len(d) - 1: cur_str += ", "
            output_str.append(cur_str)
        output_str.append(" "* indent + "}")
        return "\n".join(output_str)

    def modify_heading_angles(self, 
                              heading_angle:float, 
                              observation_list:List[str], 
                              candidate_dict: List[Dict[str, Dict[str, Any]]], 
                              object_list: List[Dict[str, Dict[str, Any]]]) -> str:
        """ 
        Generate structured text based on the current heading angle,
        dividing the field of view into eight directions and providing
        relative angles, navigable candidates, nearby objects, etc. 
        """
        
        directions = ['Front',          # index 0, range (-22.5, 22.5)
                      'Front Right',    # index 1, range (22.5, 67.5)
                      'Right',          # index 2, range (67.5, 112.5)
                      'Rear Right',     # index 3, range (112.5, 157.5)
                      'Rear',           # index 4, range (157.5, 202.5)
                      'Rear Left',      # index 5, range (202.5, 247.5)
                      'Left',           # index 6, range (247.5, 292.5)
                      'Front Left']     # index 7, range (292.5, 337.5)
        range_idx = int((heading_angle - 22.5) // 45) + 1
        obs_idx = [(i + range_idx) % 8 for i in range(8)]
        
        candidate_range = defaultdict(dict)
        for viewpoint_id, viewpoint_data in candidate_dict.items():
            viewpoint_heading = np.rad2deg(viewpoint_data['heading'])
            vp_range_idx = int((viewpoint_heading - 22.5) // 45) + 1
            rel_viewpoint_heading = viewpoint_heading - heading_angle
            rel_viewpoint_heading = self.normalize_angle(rel_viewpoint_heading)
            rel_viewpoint_heading = self.angle_to_left_right(rel_viewpoint_heading)
            vp_description = rel_viewpoint_heading + f', {viewpoint_data["distance"]:.2f}m away'
            candidate_range[vp_range_idx].update({viewpoint_id: vp_description})

        angle_ranges = [(angle - 22.5 - heading_angle, 
                         angle + 22.5 - heading_angle) for angle in range(0, 360, 45)]
        formatted_strings = []
        for direction, idx in zip(directions, obs_idx):
            rel_angle1 = self.normalize_angle(angle_ranges[idx][0])
            rel_angle2 = self.normalize_angle(angle_ranges[idx][1])
            left_right1 = self.angle_to_left_right(rel_angle1)
            left_right2 = self.angle_to_left_right(rel_angle2)
            formatted_string = f"- {direction}, range ({left_right1} to {left_right2}) --> "

            if candidate_range.get(idx): 
                formatted_string += f" {direction} Navigable Viewpoints: " + \
                    self.compact_dict_str(candidate_range[idx], indent=2)
            else: formatted_string += f" {direction} Navigable Viewpoints: None"
            formatted_strings.append(formatted_string)
        
        output_string = '\n'.join(formatted_strings)
        return output_string

    @property
    def panorama_observation(self) -> str:
        return "<ImageHere>"

    def _create_make_action_tool(self, llm: MultimodalOpenAI) -> Tool:
        """ Create a tool to make single action prediction in MP3D. """
        history_prompt = PromptTemplate(
            template=HISTORY_PROMPT,
            input_variables=["history", "previous_action", "observation"],
        )
        self.history_chain = LLMChain(llm=llm, prompt=history_prompt)

        def _make_action(*args, **kwargs) -> str:
            '''Make single step action in MatterSim.'''
            observation_str = "\n\t**Current Top-down Map**: <ImageHere>" + \
                              f"\n\t**Current Panorama**: {self.panorama_observation}"

            action_plan = self.cur_action_plan
            cur_ob = self.env.observe()[self.cur_env_index]
            feature, pos, orient, navigable = self.get_env_feature(cur_ob)

            # the next viewpointID is passed as the first argument
            action = args[0].strip(" ").strip('"').strip("'")

            if action not in self.env.env.sims[self.cur_env_index].navigable_dict.keys():
                history = f'ViewpointID "{action}" is not valid, no action taken for the agent.'
                self.vln_agent.history.append(history)
                observation_str = f"\nViewpointID '{action}' is not valid, agent not moved. " + \
                    "DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose " + \
                    f"from current viewpoints are: {[key for key in navigable.keys()]}." + \
                    observation_str + f"\n\t**Current Viewpoint**:\n{feature}"
                return observation_str

            batch_actions = [None] * self.env.batch_size
            batch_actions[self.cur_env_index] = action
            turned_angle_list, new_ob_list = self.make_equiv_action(batch_actions)
            turned_angle = turned_angle_list[self.cur_env_index]
            new_ob = new_ob_list[self.cur_env_index]
            
            new_feature, new_pos, new_orient, new_navigable = self.get_env_feature(new_ob)
            new_orientation = f'heading: {new_orient[0]:.2f}, elevation: {new_orient[1]:.2f}'
            new_scene_graph = self.get_visual_map(
                scan_id=new_ob['scan'],
                position=new_pos,
                orientation=new_orient,
                navigable_viewpoints=new_navigable
            )
            new_panorama = self.get_panorama_image(new_ob)  # new_ob["obs_panorama"]

            if self.config.use_history_chain:
                history = self.history_chain.run(
                    observation = new_ob['obs_summary'], 
                    history = self.vln_agent.history[-1], 
                    previous_action = turned_angle
                )
            else: history = self.get_text_history(new_ob, turned_angle)
            self.vln_agent.history.append(history)
            self.vln_agent.scene_graph = new_scene_graph
            self.vln_agent.panorama_img = new_panorama

            detail = {
                "viewpoint": action,
                "turned_angle": turned_angle,
                "feature": new_feature,
                "history": self.vln_agent.history[-1],
            }
            self.traj[self.cur_env_index]['details'].append(detail)

            if self.config.use_relative_angle:
                observation_str += f'\n\t**Current Viewpoint "{action}"**:\n{new_feature}'
                return observation_str
            else:
                observation_str += f"\n\t**Current Orientation**:\n{new_orientation}" + \
                                   f"\n\t**Current Viewpoint**:\n{new_feature}" + \
                                    "\n\t**Navigable Viewpoints**:" + self.compact_dict_str(new_navigable, indent=2)
                return observation_str
        
        return Tool(
            name=MAKE_ACTION_TOOL_NAME,
            func=_make_action,
            description=MAKE_ACTION_TOOL_DESCRIPTION,
        )

    def create_vln_agent(self) -> AgentExecutor:
        self.action_maker = self._create_make_action_tool(self.llm)
        self.back_tracer = self._create_back_trace_tool(self.llm)

        tools = [x for x in [self.action_maker, self.back_tracer] if x is not None]
        agent_prompt = MultimodalPromptTemplate(
            template=self.agent_prompt_template,
            image_key="images",
            input_variables=["action_plan", "init_observation", "agent_scratchpad", "images"],
            partial_variables={
                "tool_names": ", ".join([tool.name for tool in tools]),
                "tool_descriptions": "\n".join(
                    [f"{tool.name}: {tool.description}" for tool in tools]
                ),
            },
        )
        # insetantiate the VLNAgent with the LLM chain and tools
        agent = SpatialNavVLNAgent(
            llm_chain=LLMChain(llm=self.llm, prompt=agent_prompt),
            allowed_tools=[tool.name for tool in tools],
            output_parser=self.output_parser, 
            max_scratchpad_length=self.config.max_scratchpad_length,
        )

        # Construct the AgentExecutor
        return AgentExecutor.from_agent_and_tools(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            max_iterations=self.config.max_iterations,
        )

    def get_text_observation(self, env_feature, orientation, navigable):
        heading, elevation = orientation
        orientation = f'heading: {heading:.2f}, elevation: {elevation:.2f}'

        if self.config.use_relative_angle:
            observation = "\n\t**Current Top-down Map**: <ImageHere>" + \
                        f"\n\t**Current Panorama**: {self.panorama_observation}" + \
                        f"\n\t**Current Viewpoint**:\n{env_feature}"
        
        else: observation =  "\n\t**Current Top-down Map**: <ImageHere>" + \
                            f"\n\t**Current Panorama**: {self.panorama_observation}" + \
                            f"\n\t**Current Orientation**:\n{orientation}" + \
                            f"\n\t**Current Viewpoint**:\n{env_feature}" + \
                            f"\n\t**Navigable Viewpoints**:" + self.compact_dict_str(navigable, indent=2)
        return observation

    def rollout(self, reset=True):
        if reset: obs = self.env.reset()
        else: obs = self.env.observe()

        self.init_trajecotry(obs)
        
        instructions = [ob['instruction'] for ob in obs]
        if self.config.load_instruction:
            action_plans = instructions
        else:
            action_plans = []
            for instruction in instructions:
                action_plan = self.plan_chain.run(instruction = instruction)
                action_plans.append(action_plan)

        for i, init_ob in enumerate(obs):
            if hasattr(self, "mapper"): self.mapper.prev_level_id = None
            if hasattr(self, "object_tracker"): self.object_tracker.prev_level_id = None
            self.cur_env_index = i          
            self.cur_action_plan = action_plans[i]
            self.traj[i]['action_plan'] = self.cur_action_plan
            
            cur_feature, pos, orient, navigable = self.get_env_feature(init_ob)
            scene_image = self.get_visual_map(
                scan_id=init_ob['scan'],
                position=pos,
                orientation=orient,
                navigable_viewpoints=navigable
            )
            panorama_image = self.get_panorama_image(init_ob)
            init_observation = self.get_text_observation(cur_feature, orient, navigable)
            print("=== Initial Observation ===")
            print(init_observation)
            mllm_inputs = {
                "action_plan": self.cur_action_plan, 
                "init_observation": init_observation,
                "images": [scene_image, panorama_image],
            }

            # === LangChain 核心执行 ===
            # 调用 AgentExecutor：内部会驱动 MRKL 循环（最多 max_iterations），并返回：
            # - 'output'：最终字符串（通常包含 Final Answer）
            # - 'intermediate_steps'：[(AgentAction, observation_str), ...]
            output = self.agent_executor(mllm_inputs)

            self.traj[i]['llm_output'] = output['output']
            intermediate_steps = output['intermediate_steps']
            self.traj[i]['llm_thought'] = []
            self.traj[i]['llm_observation'] = [init_observation]
            for action, observation in intermediate_steps:
                thought = action.log
                self.traj[i]['llm_thought'].append(thought)
                self.traj[i]['llm_observation'].append(observation)
            self.traj[i]['llm_thought'].append(output.get('finish_thought', "MAX STEP REACHED."))

        return self.traj

    def get_panorama_image(self, ob) -> Image.Image:
        """ Get the panorama image from the observation. """
        pano_img = ob["obs_panorama"]
        scan_id = ob['scan']
        
        instr_id = self.traj[self.cur_env_index]['instr_id']
        num_steps = len(self.traj[self.cur_env_index]['path']) - 1
        cur_save_path = os.path.join(self.panorama_save_dir, scan_id, f"{instr_id}_step{num_steps}.jpg")
        os.makedirs(os.path.dirname(cur_save_path), exist_ok=True)
        pano_img.save(cur_save_path)
        
        return pano_img



class SpatialNavAgent(SpatialNavBaseAgent):
    name = "spatialnav"
    panorama_save_dir = "outputs/results/spatialnav/panorama_images"
    DIRECTIONS = ['Front',          # index 0, range (-22.5, 22.5)
                  'Front Right',    # index 1, range (22.5, 67.5)
                  'Right',          # index 2, range (67.5, 112.5)
                  'Rear Right',     # index 3, range (112.5, 157.5)
                  'Rear',           # index 4, range (157.5, 202.5)
                  'Rear Left',      # index 5, range (202.5, 247.5)
                  'Left',           # index 6, range (247.5, 292.5)
                  'Front Left']     # index 7, range (292.5, 337.5)
    use_relevant_local_objects = False
    
    def __init__(self, env: DiscreteNavBatch, config: SpatialNavConfig):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The discrete Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env, config)

        data_name = self.env.data.name.split('(')[0]
        object_cfg = SceneObjectConfig(
            category_col=config.object_category_set,
            visibility_radius=self.config.object_visibility_radius,
        )
        self.object_tracker = SceneObjectGraph(object_cfg, 
                                               scans=self.env.scans, 
                                               data_name=data_name)

    @staticmethod
    def smallest_angle_diff(a, b):
        return a-b

    def get_relevant_objects(self, scan_id, viewpoint_id, position, object_proposals):
        pos_info = self.object_tracker.get_local_objects(
            scan_id=scan_id,
            viewpoint_id=viewpoint_id,
            viewpoint_pos=position,
        )
        possible_objects: List[Object3D] = pos_info["local_objects"]
        pos_categories = [obj.category for obj in possible_objects]
        pos_categories = list(set(pos_categories))

        valid_objects = dict()
        counter = dict()
        if len(possible_objects) == 0: return valid_objects
        
        room_type = pos_info["room_type"]
        relevant_cats = self.object_tracker.get_relevant_categories(
            object_proposals=object_proposals,
            candidate_categories=pos_categories
        )
        for obj in possible_objects:
            obj_label = obj.category
            if obj_label not in relevant_cats: continue
            obj_position = obj.obb.center
            obj_cnt = counter.get(obj_label, 0) + 1
            valid_objects[f"{obj_label}_{obj_cnt}"] = {
                "id": obj.id,
                "category": obj_label,
                "room_type": room_type,
                "position": obj_position,
                "distance": np.linalg.norm(np.array(obj_position) - np.array(position)),
            }
            counter[obj_label] = obj_cnt
        return valid_objects

    def get_env_feature(self, cur_ob: dict):
        """ Return the environment feature, position, orientation, and navigable locations. """
        heading = np.rad2deg(cur_ob['heading'])
        elevation = np.rad2deg(cur_ob['elevation'])
        orientation = (heading, elevation)
        position = cur_ob['position']

        # print(cur_ob['instruction'])
        object_proposals = self.object_tracker.get_interest_objects(
            path_id=cur_ob['id'],
            instruction=cur_ob['instruction'],
        )

        if self.use_relevant_local_objects:
            # Get local objects around the current viewpoint
            local_objects = self.get_relevant_objects(
                scan_id=cur_ob['scan'],
                viewpoint_id=cur_ob['viewpoint'],
                position=cur_ob['position'],
                object_proposals=object_proposals
            )
            obj_by_direction = {dir: dict() for dir in self.DIRECTIONS}
            for obj in local_objects:
                obj_center = local_objects[obj]['position']
                theta, phi, distance = cartesian_to_spherical(
                    obj_center[0] - position[0],
                    obj_center[1] - position[1],
                    obj_center[2] - position[2],
                )
                rel_heading = np.rad2deg(theta - heading)
                rel_elevation = np.rad2deg(phi - elevation)
                range_idx = int((rel_heading - 22.5) // 45) + 1
                direction = self.DIRECTIONS[range_idx % 8]
                obj_by_direction[direction][obj] = {
                    "category": local_objects[obj]['category'],
                    "rel_heading": self.normalize_angle(rel_heading),
                    "rel_elevation": rel_elevation,
                    "distance": distance,
                }
            
            def sort_key(info, sector_center_deg):
                dist = info["distance"]
                # 与扇区中心方向的夹角越小越好
                heading_offset = self.smallest_angle_diff(info["rel_heading"], sector_center_deg)
                # 与水平线的夹角越小越好（更接近视线高度）
                elev_offset = self.smallest_angle_diff(info["rel_elevation"], 0.0)
                return (heading_offset, elev_offset, dist)
            
            for direction in self.DIRECTIONS:
                objs_info = obj_by_direction[direction]
                if not objs_info: continue
                dir_idx = self.DIRECTIONS.index(direction)
                sector_center_deg = dir_idx * 45.0
                objs_info = {x['category']: x for (_, x) in sorted(
                    objs_info.items(),
                    key=lambda item: sort_key(item[1], sector_center_deg)
                )}
                obj_by_direction[direction] = objs_info
        else: obj_by_direction = cur_ob['obs_objects']

        # Get navigable viewpoints with nearby objects
        navigable = cur_ob['candidate']
        for cand_vid, cand_info in navigable.items():
            cand_objects = self.get_relevant_objects(
                scan_id=cur_ob['scan'], 
                viewpoint_id=cand_vid, 
                position=cand_info['position'],
                object_proposals=object_proposals
            )
            cand_info["objects"] = cand_objects

        # Modify the observation feature with relative angles, navigable candidates, and nearby objects
        feature = cur_ob['obs_detail']
        feature = self.modify_heading_angles(heading, feature, navigable, obj_by_direction)
        return feature, position, orientation, navigable

    @staticmethod
    def _build_cluster_record(category, cluster_items, index):
        distances = [info["distance"] for info in cluster_items]
        rep_distance = float(np.round(np.mean(distances), 1))

        count = len(cluster_items)
        if count > 1: display_category = category + "s"
        else: display_category = category

        return {
            "display_category": display_category, 
            "distance": rep_distance,
            "count": count,
            "raw_category": category,
            "index": index,
            # "objects": cluster_items
        }

    def merge_obejcts(self, cand_objects: Dict[str, Dict], d_tol=0.2) -> List[str]:
        # 1. 先按照类别分组
        grouped_items = defaultdict(list)
        for obj, info in cand_objects.items():
            base_name = info['category']
            grouped_items[base_name].append(info)

        grouped_results = defaultdict(list)
        cat_groups = dict()
        # 2. 对每个类别内部做距离聚类
        for cat, items in grouped_items.items():
            items.sort(key=lambda x: x["distance"])
            current_cluster = [items[0]]
            cluster_min = items[0]["distance"]
            cat_num = 0
            for item in items[1:]:
                d = item["distance"]
                if d - cluster_min <= 2 * d_tol: 
                    current_cluster.append(item)
                else:
                    grouped_results[cat].append(self._build_cluster_record(cat, current_cluster, cat_num))
                    current_cluster = [item]
                    cluster_min = d
                    cat_num += 1
            if current_cluster: grouped_results[cat].append(self._build_cluster_record(cat, current_cluster, cat_num))
            cat_num += 1
            cat_groups[cat] = cat_num

        # 3. 生成描述文本
        merged_list = []
        # for record in grouped_results:
        #     merged_list.append(f"{record['display_category']} ({record['distance']:.1f}m)")
        for cat, grouped_records in grouped_results.items():
            if len(grouped_records) == 1:
                record = grouped_records[0]
                merged_list.append(f"{record['display_category']} ({record['distance']:.1f}m)")
            else:
                distances = [round(rec['distance'], 1) for rec in grouped_records]
                display_category = cat + "s"
                merged_list.append(f"{display_category} (" + ", ".join([f"{d}m" for d in distances]) + ")")
        return merged_list

    def modify_heading_angles(self, 
                              heading_angle:float, 
                              observation_list:List[str], 
                              candidate_dict: List[Dict[str, Dict[str, Any]]], 
                              object_list: List[Dict[str, Dict[str, Any]]]) -> str:
        """ 
        Generate structured text based on the current heading angle,
        dividing the field of view into eight directions and providing
        relative angles, navigable candidates, nearby objects, etc. 
        """
        directions = self.DIRECTIONS
        range_idx = int((heading_angle - 22.5) // 45) + 1
        obs_idx = [(i + range_idx) % 8 for i in range(8)]
        
        candidate_range = defaultdict(dict)
        for viewpoint_id, viewpoint_data in candidate_dict.items():
            viewpoint_heading = np.rad2deg(viewpoint_data['heading'])
            vp_range_idx = int((viewpoint_heading - 22.5) // 45) + 1
            rel_viewpoint_heading = viewpoint_heading - heading_angle
            rel_viewpoint_heading = self.normalize_angle(rel_viewpoint_heading)
            rel_viewpoint_heading = self.angle_to_left_right(rel_viewpoint_heading)
            vp_description = rel_viewpoint_heading + f', {viewpoint_data["distance"]:.2f}m away'
            # add nearby objects info
            cand_objects = viewpoint_data.get("objects", {})
            if len(cand_objects) > 0:
                # objects_str = sorted(list(cand_objects.keys()))
                objects_str = self.merge_obejcts(cand_objects)
                objects_str = f", objects near {viewpoint_id} (with distances): [" + ", ".join(objects_str) + "]"
            else: objects_str = f", objects near {viewpoint_id} (with distances): None"
            vp_description += objects_str
            candidate_range[vp_range_idx].update({viewpoint_id: vp_description})

            candidate_range[vp_range_idx].update({viewpoint_id: vp_description})

        angle_ranges = [(angle - 22.5 - heading_angle, 
                         angle + 22.5 - heading_angle) for angle in range(0, 360, 45)]
        formatted_strings = []
        for direction, idx in zip(directions, obs_idx):
            rel_angle1 = self.normalize_angle(angle_ranges[idx][0])
            rel_angle2 = self.normalize_angle(angle_ranges[idx][1])
            left_right1 = self.angle_to_left_right(rel_angle1)
            left_right2 = self.angle_to_left_right(rel_angle2)
            formatted_string = f"- {direction}, range ({left_right1} to {left_right2}) --> "

            if candidate_range.get(idx): 
                formatted_string += f" {direction} Navigable Viewpoints: " + \
                    self.compact_dict_str(candidate_range[idx], indent=2)
            else: formatted_string += f" {direction} Navigable Viewpoints: None"
            formatted_strings.append(formatted_string)
        
        output_string = '\n'.join(formatted_strings)
        return output_string


