from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Dict, Union
import sys, os
import time
import json
import numpy as np
from collections import defaultdict
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from PIL import Image

# custom dependencies
from spatialx.utils.model_utils import MultimodalOpenAI, MLLMChain, MultimodalPromptTemplate
from spatialx.mp3d_extensions import DiscreteNavBatch
from spatialx.mp3d_extensions.discrete_env import DiscreteVisualSimulatorV4
from spatialx.mp3d_extensions.pc_utils import cartesian_to_spherical
from spatialx.spatial.mapper import SceneGraphMapper
from ..agent_base import BaseAgent
from .config import MapGPTConfig, MapGPTSpatialConfig
from .prompts import (
    MapGPTPromptManager, 
    TextHistorySpatialMapGPTPromptManager, 
    VisualHistorySpatialMapGPTPromptManager
)
from ..agent_base import BaseAgent


class BaseMapGPTAgent(BaseAgent):
    name = "base_mapgpt"
    image_save_dir = "outputs/results/base_mapgpt/images"

    def __init__(self, env: DiscreteNavBatch):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The discrete Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env)

        self.prompt_manager: MapGPTPromptManager = None
        self.llm: MultimodalOpenAI = None
        self.chain: LLMChain = None

    def build_prompter(self):
        raise NotImplementedError("build_prompter method not implemented.")

    def init_trajecotry(self, obs: List[dict]):
        """Initialize the trajectory with the given observation."""
        batch_size = len(obs)

        self.traj = [{
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],
            'action_plan': ob['instruction'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'a_t': {},
        } for ob in obs]

        self.prompt_manager.history = ['' for _ in range(batch_size)]
        self.prompt_manager.nodes_list = [[] for _ in range(batch_size)]
        self.prompt_manager.node_imgs = [[] for _ in range(batch_size)]
        self.prompt_manager.graph = [{} for _ in range(batch_size)]
        self.prompt_manager.trajectory = [[] for _ in range(batch_size)]
        self.prompt_manager.planning = [["Navigation has just started, with no planning yet."] for _ in range(batch_size)]

    def make_equiv_action(self, actions: List[str]) -> str:
        """ Execute the navigation action and return new observation. """
        cur_obs = self.env.observe()
        new_obs = self.env.step(actions)

        self.traj[0]['path'].append(
            self.env.env.sims[0].gmap.bfs_shortest_path(
                start=cur_obs[0]['viewpoint'], 
                end=actions[0])[1:]
        )
        return new_obs

    def observe_candidate(self, ob: Dict[str, Any], step: int) -> List[Image.Image]:
        r"""Observe candidate views and return images."""
        navigable = ob['candidate']
        total_images = list(navigable.values())[0]['total_images']
        cur_sim: DiscreteVisualSimulatorV4 = self.env.env.sims[0]
        image_list = cur_sim.make_candiate_views(view_indices=total_images)

        # update images in candidate
        for cand_vid, cc in navigable.items():
            cc['image'] = image_list[cc['image_index']]

        for i, image in enumerate(image_list):
            if image is not None: 
                image_save_path = os.path.join(
                    self.image_save_dir, ob['instr_id'], f"step_{step}_view_{total_images[i]}.jpg")
                os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
                image.save(image_save_path)
        
        return image_list

    def rollout(self, reset=True):
        if reset: obs = self.env.reset() # Reset env
        else: obs = self.env.observe()
        
        self.init_trajecotry(obs)

        if self.traj[0]['instr_id'] in self.results:
            self.traj = [self.results[self.traj[0]['instr_id']]] # Skip already completed
            return self.traj
        
        previous_angle = [{'heading': ob['heading'], 
                           'elevation': ob['elevation']} for ob in obs]
        for t in range(self.config.max_iterations):
            start_time = time.time()

            all_cand_images = self.observe_candidate(obs[0], step=t)
            # 构建 prompt 输入
            cand_inputs = self.prompt_manager.make_action_prompt(
                obs=obs, previous_angle=previous_angle)
            nav_input = self.prompt_manager.make_r2r_json_prompts(
                cand_inputs=cand_inputs, obs=obs, t=t)
            # 所有的节点图像
            all_node_images = self.prompt_manager.node_imgs[0]

            system_prompt = nav_input["task_description"]
            environment_prompt = nav_input["prompts"][0]
            user_prompt, user_images = self.construct_user_input(
                text=environment_prompt, 
                ob=obs[0], step=t,
                all_cand_images=all_cand_images,
                all_node_images=all_node_images,
            )
            
            # 调用 GPT：输入任务描述 + 环境信息 + 图像
            self.llm.prefix_messages = [{"role": "system", "content": system_prompt}]
            nav_output = self.chain.run(user_input=user_prompt, images=user_images)
            end_time = time.time()
            
            # 记录每一步的 prompt 和 LLM 输出到 traj
            self.traj[0]['system_prompt'] = system_prompt
            self.traj[0]['details'][f"step_{t}"] = {"user_prompt": user_prompt, "model_response": nav_output}
            print(f'---------- [EP {self.traj[0]["instr_id"]} -- STEP {t}] LLM Time: {end_time - start_time:.2f}s ----------' + \
                  f'\n[System Prompt]\n{system_prompt}\n\n[User Prompt]\n{user_prompt}\n\n[Agent Response]\n{nav_output}\n')
            sys.stdout.flush()

            # 解码动作
            a_t = self.prompt_manager.parse_json_action(
                json_output=json.loads(nav_output),
                only_options_batch=nav_input["only_options"],
                t=t)
            
            # 判断是否是停止动作
            a_t_stop = [a_t_i == 0 for a_t_i in a_t]
            a_t_cpu, a_t_nav = [], []
            for i in range(len(obs)):
                if a_t_stop[i]: 
                    a_t_cpu.append(-1)
                    a_t_nav.append(None)
                    cur_action_viewpoint = "stop"
                else: 
                    cur_action = a_t[i]-1
                    cur_action_viewpoint = cand_inputs['cand_vpids'][i][cur_action]
                    a_t_cpu.append(cur_action)
                    a_t_nav.append(cur_action_viewpoint)
                # 记录动作到 traj
                self.traj[i]['a_t'][f"step_{t}"] = (a_t[i], cur_action_viewpoint)
            print(f"---------- [EP {self.traj[0]['instr_id']} -- STEP {t}] Action taken: {[a_t_cpu, cur_action_viewpoint]} ----------\n")

            # 如果所有停止，则结束导航
            if all(a_t_stop): 
                print(f"---------- [EP {self.traj[0]['instr_id']} -- STEP {t}] Ending rollout. ----------\n")
                break

            # 执行动作
            obs = self.make_equiv_action(a_t_nav)
            previous_angle = [{'heading': ob['heading'], 
                               'elevation': ob['elevation']} for ob in obs]
            
            # 更新 prompt 历史
            self.prompt_manager.make_history(a_t, nav_input, t)
        
        return self.traj

    def construct_user_input(self, text, ob: Dict[str, Any], step:int=0, 
                             all_cand_images: List[Image.Image]=None, 
                             all_node_images: List[Image.Image]=None,
                             **kwargs) -> Tuple[str, List[Image.Image]]:
        r"""Construct user input for LLM."""
        raise NotImplementedError("construct_user_input method not implemented.")


class MapGPTAgent(BaseMapGPTAgent):
    # ~25 dollors per 158 episodes -- "high"
    name = "mapgpt"
    image_save_dir = "outputs/results/mapgpt/images"

    def __init__(self, env: DiscreteNavBatch, config: MapGPTConfig):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The discrete Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env)

        self.config = config
        self.build_prompter()
    
        self.llm = MultimodalOpenAI(
            model_name=config.llm_model_name,
            openai_api_base=config.api_base_url,
            openai_api_key=config.api_key,
            temperature=getattr(config, "llm_temperature", 0.0),
            response_format={"type": "json_object"},
        )
        self.chain = LLMChain(
            llm=self.llm, prompt=MultimodalPromptTemplate(
            template="{user_input}", image_key="images",
            input_variables=["user_input", "images"],
            image_detail=self.config.image_detail,
        ))
    
    def build_prompter(self):
        self.prompt_manager = MapGPTPromptManager(allow_stop_after=self.config.allow_stop_after)
        print(f"Initialized MapGPTAgent with LLM: {self.config.llm_model_name}")

    def construct_user_input(self, text, ob: Dict[str, Any], step:int=0, 
                             all_node_images: List[Image.Image]=None, **kwargs) -> Tuple[str, List[Image.Image]]:
        r"""Construct user input for LLM."""
        input_txt = []
        input_images = []
        for i, image in enumerate(all_node_images):
            if image is not None: 
                input_txt.append(f"Image {i}: <ImageHere>")
                input_images.append((image, "low"))
        input_txt.append(text)
        input_txt = "\n".join(input_txt)
        return input_txt, input_images


class MapGPTSpatialAgent(BaseMapGPTAgent):
    name = "mapgpt_spatial"
    image_save_dir = "outputs/results/mapgpt_spatial/images"
    history_type = "visual"  # "text" or "visual"

    def __init__(self, env: DiscreteNavBatch, config: MapGPTSpatialConfig):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The discrete Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env)

        self.config = config
        self.history_type = config.agent_history_type
        self.build_prompter()
    
        self.llm = MultimodalOpenAI(
            model_name=config.llm_model_name,
            openai_api_base=config.api_base_url,
            openai_api_key=config.api_key,
            temperature=getattr(config, "llm_temperature", 0.0),
            response_format={"type": "json_object"},
        )
        self.chain = LLMChain(
            llm=self.llm, prompt=MultimodalPromptTemplate(
            template="{user_input}", image_key="images",
            input_variables=["user_input", "images"],
            image_detail=self.config.image_detail,
        ))

        self.mapper = SceneGraphMapper(
            config=self.config.scene_graph, 
            scans=self.env.scans,
            env_type=self.env.env_type,
        )
        # for mapgpt-spatial agent, we explicitly allow to draw the node index on the map
        self.mapper.config.draw_navigable_index = True
        self.mapper.config.draw_history_index = True

    def build_prompter(self):
        if self.history_type == "text":
            self.prompt_manager = TextHistorySpatialMapGPTPromptManager(
                allow_stop_after=self.config.allow_stop_after)
        elif self.history_type == "visual":
            self.prompt_manager = VisualHistorySpatialMapGPTPromptManager(
                allow_stop_after=self.config.allow_stop_after)
        else: raise ValueError(f"Unknown history type: {self.history_type}")
        print(f"Initialized MapGPTSpatialAgent with LLM: {self.config.llm_model_name} | History: {self.history_type}")

    def construct_user_input(self, text, ob: Dict[str, Any], step:int=0, 
                             all_cand_images: List[Image.Image]=None, 
                             all_node_images: List[Image.Image]=None,
                             **kwargs) -> Tuple[str, List[Image.Image]]:
        r"""Construct user input for LLM."""
        heading = np.rad2deg(ob['heading'])
        elevation = np.rad2deg(ob['elevation'])
        cur_scene_graph = self.get_visual_map(
            scan_id=ob['scan'],
            position=ob['position'],
            orientation=(heading, elevation),
            navigable_viewpoints=ob['candidate']
        )

        if self.history_type == "text": 
            valid_images = all_cand_images
        elif self.history_type == "visual":
            valid_images = all_node_images
        else: raise ValueError(f"Unknown history type: {self.history_type}")

        input_txt = []
        input_images = []
        for i, image in enumerate(valid_images):
            if image is not None: 
                input_txt.append(f"Image {i}: <ImageHere>")
                input_images.append((image, "low"))
        input_images.append((cur_scene_graph, "high"))

        input_txt.append(text)
        input_txt = "\n".join(input_txt)
        return input_txt, input_images

    def get_visual_map(self, scan_id: str, position: Tuple[float, float, float], 
                       orientation: Tuple[float, float], navigable_viewpoints: List[dict], 
                       vertical_threshold: float=2.0) -> Image.Image:
        """ Get the visual top-down spatial map for the given position and orientation. """
        history_path = self.traj[0]['path']
        history_trajectory = sum(history_path, [])
        history_dict = dict()
        for i, vp_id in enumerate(history_trajectory):
            vp_pos = self.env.get_vp_location(scan_id, vp_id)
            within_same_level = bool(abs(vp_pos[2] - position[2]) <= vertical_threshold) # m
            history_dict[vp_id] = {
                "position": vp_pos, 
                "history_order": self.prompt_manager.nodes_list[0].index(vp_id), 
                "within_same_level": within_same_level
            }
        # all viewpoints before the last different level point are marked as False
        for i, vp_id in enumerate(history_trajectory[::-1]):
            if not history_dict[vp_id]["within_same_level"]:
                for j, vp_j in enumerate(history_trajectory[:len(history_trajectory)-i]):
                    history_dict[vp_j]["within_same_level"] = False
                break
        
        for cand_vid, cc in navigable_viewpoints.items():
            cc['local_order'] = self.prompt_manager.nodes_list[0].index(cand_vid)
        
        scene_graph = self.mapper.get_visual_map(
            scan_id=scan_id,
            position=position,
            orientation=orientation,
            navigable_viewpoints=navigable_viewpoints, 
            history_viewpoints=history_dict
        )
        
        instr_id = self.traj[0]['instr_id']
        num_steps = len(self.traj[0]['path']) - 1
        cur_save_path = os.path.join(self.config.map_save_dir, instr_id, f"{instr_id}_step{num_steps}.jpg")
        os.makedirs(os.path.dirname(cur_save_path), exist_ok=True)
        scene_graph.save(cur_save_path)
        
        return scene_graph
