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
from spatialx.spatial.mapper import SceneGraphMapper
from ..agent_base import BaseAgent
from .config import SpatialNavConfig
from .prompts import (
    PLANNER_PROMPT, HISTORY_PROMPT, 
    VLN_SPATIAL_GPT4_PROMPT, VLN_SPATIAL_GPT5_PROMPT, 
    VLN_SPATIAL_GPT4_PROMPT_V2, VLN_SPATIAL_GPT5_PROMPT_V2,
    BACK_TRACE_PROMPT, BACK_TRACE_TOOL_NAME, BACK_TRACE_TOOL_DESCRIPTION,
    MAKE_ACTION_TOOL_NAME, MAKE_ACTION_TOOL_DESCRIPTION,
)
from spatialx.utils import MultimodalPromptTemplate, MultimodalOpenAI


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


def get_agent_prompt_template(prompt_version: str, llm_model_name: str) -> str:
    """ Return the appropriate prompt template based on version and model name. """
    if prompt_version == "v1":
        if "gpt-4" in llm_model_name.lower():
            return VLN_SPATIAL_GPT4_PROMPT
        elif "gpt-5" in llm_model_name.lower():
            return VLN_SPATIAL_GPT5_PROMPT
        elif "gemini" in llm_model_name.lower():
            return VLN_SPATIAL_GPT5_PROMPT
        elif "qwen3-vl" in  llm_model_name.lower():
            return VLN_SPATIAL_GPT5_PROMPT
        else: raise ValueError(f"Unknown LLM model name: {llm_model_name}.")
    elif prompt_version == "v2":
        if "gpt-4" in llm_model_name.lower():
            return VLN_SPATIAL_GPT4_PROMPT_V2
        elif "gpt-5" in llm_model_name.lower():
            return VLN_SPATIAL_GPT5_PROMPT_V2
        elif "gemini" in llm_model_name.lower():
            return VLN_SPATIAL_GPT5_PROMPT_V2   # Gemini 使用与 GPT-5 相同的模板
        elif "qwen3-vl" in  llm_model_name.lower():
            return VLN_SPATIAL_GPT5_PROMPT_V2   # 使用与 GPT-5 相同的模板
    
    raise ValueError(f"Unknown prompt version: {prompt_version} for model {llm_model_name}.")


class SpatialAgentOutputParser(AgentOutputParser):

    def get_format_instructions(self) -> str:
        # return MRKL format instructions
        # not used in this code but kept for completeness.
        return (
            "You should strictly follow the below output format: \n"
            "Thought: you should always think about what to do \n" 
            "Action: the action to take\n"
            "Action Input: the input to the action\n\n"
        )

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        # Action: <tool-name>
        # Action Input: "<32位hex的viewpointID>"
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*\"?([a-fA-F0-9]{32})\"?"
        )
        action_match = re.search(regex, text, re.DOTALL)

        # ---- old logic: raise error if both final answer and action exist ----
        # if action_match:
        #     if includes_answer:
        #         raise OutputParserException(
        #             f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}", 
        #             observation=DEFAULT_PARSE_ERROR_MESSAGE, 
        #             llm_output=text,
        #             send_to_llm=True,
        #         )
        #     action = action_match.group(1).strip()
        #     action_input = action_match.group(2)
        #     tool_input = action_input.strip(" ")
        #     return AgentAction(action, tool_input, text)
        # elif includes_answer:
        #     thought = text.strip()
        #     valid_reasion = text.split(FINAL_ANSWER_ACTION)[-1].strip()
        #     return AgentFinish({"finish_thought": thought, "output": valid_reasion}, text)

        if includes_answer:  # if final answer exists, prioritize it
            thought = text.strip()
            valid_reasion = text.split(FINAL_ANSWER_ACTION)[-1].strip()
            return AgentFinish({"finish_thought": thought, "output": valid_reasion}, text)
        elif action_match:  # else parse action normally
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            return AgentAction(action, tool_input, text)

        # Error handling for missing formats
        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else: raise OutputParserException(
            f"Could not parse LLM output: `{text}`", 
            observation=DEFAULT_PARSE_ERROR_MESSAGE
        )

    @property
    def _type(self) -> str:
        return "mrkl-SpatialGPT"


class SpatialVLNAgent(ZeroShotAgent):
    """ An agent for Vision-and-Language Navigation with spatial reasoning. """
    # 限制 scratchpad 长度，避免超过 LLM 上下文窗口
    max_scratchpad_length: int = 4096
    # 随着导航进程不断更新的历史观察与场景图
    history: Optional[List[str]] = None 
    scene_graph: Optional[Image.Image] = None
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
            kwargs["scene_graph"] = self.scene_graph
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


class NavGPTSpatialAgent(BaseAgent):
    """ An agent that uses SpatialGPT for Vision-and-Language Navigation in Matterport3D. """
    name = "navgpt_spatial"

    def __init__(self, env: DiscreteNavBatch, config: SpatialNavConfig):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The discrete Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env)

        self.config = config
        self.mapper = SceneGraphMapper(
            config=config.scene_graph, 
            scans=self.env.scans,
            env_type=self.env.env_type,
        )

        self.llm = MultimodalOpenAI(
            temperature=config.temperature,
            model_name=config.llm_model_name,
            openai_api_base=config.api_base_url,
            openai_api_key=config.api_key,
        )
        self.output_parser = SpatialAgentOutputParser()
        self.agent_executor: AgentExecutor = self.create_vln_agent()
        self.vln_agent: SpatialVLNAgent = self.agent_executor.agent

        plan_prompt = PromptTemplate(
            template=PLANNER_PROMPT,
            input_variables=["instruction"],
        )
        self.plan_chain = LLMChain(llm=self.llm, prompt=plan_prompt)

        # Recording variables used when batch-size > 1
        self.cur_action_plan = None         # current action plan
        self.cur_env_index = None           # current environment index in the batch
        print("SpatialGPT Agent initialized.")
    
    def init_trajecotry(self, obs: List[dict]):
        """Initialize the trajectory with the given observation."""
        self.traj = [{
            'scan': obs[0]['scan'],
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': [],
        } for ob in obs ]

        self.vln_agent.history = [
            f'Navigation start, no actions taken yet.\n' + \
            f'Current viewpoint "{ob["viewpoint"]}": ' + \
            f'Scene from the viewpoint is a {ob["obs_summary"]}' for ob in obs]

    def make_equiv_action(self, actions: List[str]) -> str:
        """
        Interface between Panoramic view and Egocentric view
        Take in the next viewpoint ID and move the agent to that viewpoint
        return the turned angle and new observation
        """
        cur_obs = self.env.observe()
        new_obs = self.env.step(actions)

        action_descriptions = []
        for idx, (cur_ob, next_ob, action_viewpoint) in enumerate(zip(cur_obs, new_obs, actions)):
            cur_scan = cur_ob['scan']
            cur_heading = np.rad2deg(cur_ob['heading'])
            next_heading = np.rad2deg(next_ob['heading'])
            turned_angle = next_heading - cur_heading
            cur_heading_str = self.angle_to_left_right(self.normalize_angle(cur_heading))
            next_heading_str = self.angle_to_left_right(self.normalize_angle(next_heading))
            action_desc = f'Turn heading direction {turned_angle:.2f} degrees ' + \
                          f'from {cur_heading_str} to {next_heading_str}.'
            action_descriptions.append(action_desc)
            self.traj[idx]['path'].append(
                self.env.shortest_paths[cur_scan]\
                    [cur_ob['viewpoint']][action_viewpoint][1:]
            )
        return action_descriptions, new_obs

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
            vp_description = rel_viewpoint_heading + f', {viewpoint_data["distance"]:.2f}m'
            candidate_range[vp_range_idx].update({viewpoint_id: vp_description})

        angle_ranges = [(angle - 22.5 - heading_angle, 
                         angle + 22.5 - heading_angle) for angle in range(0, 360, 45)]
        formatted_strings = []
        for direction, idx in zip(directions, obs_idx):
            rel_angle1 = self.normalize_angle(angle_ranges[idx][0])
            rel_angle2 = self.normalize_angle(angle_ranges[idx][1])
            left_right1 = self.angle_to_left_right(rel_angle1)
            left_right2 = self.angle_to_left_right(rel_angle2)
            formatted_string = f"- {direction}, range ({left_right1} to {left_right2}): \n " + \
                               f"  '{observation_list[idx]}'"

            if self.config.use_surround_objects:
                object_dict = {}
                if len(object_list[idx]) > 0:
                    object = object_list[idx]
                    for obj, obj_data in object.items():
                        rel_obj_heading = obj_data['heading'] - heading_angle
                        rel_obj_heading = self.normalize_angle(rel_obj_heading)
                        rel_obj_heading = self.angle_to_left_right(rel_obj_heading)
                        object_dict[obj] = f'{rel_obj_heading}, {obj_data["distance"]:.2f}m'
                    formatted_string += f"\n  {direction} Objects in 3m: {object_dict}"
                                        # self.compact_dict_str(object_dict, indent=2)
                else: formatted_string += f"\n  {direction} Objects in 3m: None"    

            if candidate_range.get(idx): 
                formatted_string += f"\n  {direction} Navigable Viewpoints: {candidate_range[idx]}"
                                    # self.compact_dict_str(candidate_range[idx], indent=2)
            else: formatted_string += f"\n  {direction} Navigable Viewpoints: None"
            formatted_strings.append(formatted_string)
        
        output_string = '\n'.join(formatted_strings)
        return output_string
    
    def get_env_feature(self, cur_ob: dict):
        """ Return the environment feature, position, orientation, and navigable locations. """
        heading = np.rad2deg(cur_ob['heading'])
        elevation = np.rad2deg(cur_ob['elevation'])
        orientation = (heading, elevation)
        position = cur_ob['position']

        feature = cur_ob['obs_detail']
        objects = cur_ob['obs_objects']
        navigable = cur_ob['candidate']
        if self.config.use_relative_angle:
            feature = self.modify_heading_angles(heading, feature, navigable, objects)
        
        return feature, position, orientation, navigable
    
    def get_text_history(self, ob: dict, angle: str) -> str:
        '''Return the history of actions taken.'''
        history = f'{angle}\nCurrent viewpoint "{ob["viewpoint"]}": ' + \
                  f'Scene from the viewpoint is a {ob["obs_summary"]}'
        return history

    def get_his_viewpoints(self) -> str:
        '''Return the history of visited viewpoints for back tracing.'''
        his_viewpoints = ''
        # 将历史路径中的每个 viewpoint 的观测特征整理为文本，供“回溯工具”LLM 判断回退点。
        # 注意：最后一个节点不参与回溯候选。
        for i, detail in enumerate(self.traj[self.cur_env_index]['details'][:-1]):
            viewpointID = detail['viewpoint']
            viewpoint_ob = detail['feature']
            his_viewpoints += f"Step {i+1}. Viewpoint ID '{viewpointID}':\n {viewpoint_ob}\n\n"
        return his_viewpoints

    def get_text_observation(self, 
                             env_feature:str, 
                             orientation:Tuple[float, float], 
                             navigable:List[dict]
    ) -> str:
        ''' Return the text observation for the agent. '''
        heading, elevation = orientation
        orientation = f'heading: {heading:.2f}, elevation: {elevation:.2f}'

        if self.config.use_relative_angle:
            observation = "\n\t**Current Top-down Map**: <ImageHere>" + \
                         f"\n\t**Current Viewpoint**:\n{env_feature}"
        
        else: observation =  "\n\t**Current Top-down Map**: <ImageHere>" + \
                            f"\n\t**Current Orientation**:\n{orientation}" + \
                            f"\n\t**Current Viewpoint**:\n{env_feature}" + \
                            f"\n\t**Navigable Viewpoints**:" + self.compact_dict_str(navigable, indent=2)
        return observation

    def _create_make_action_tool(self, llm: MultimodalOpenAI) -> Tool:
        """ Create a tool to make single action prediction in MP3D. """
        history_prompt = PromptTemplate(
            template=HISTORY_PROMPT,
            input_variables=["history", "previous_action", "observation"],
        )
        self.history_chain = LLMChain(llm=llm, prompt=history_prompt)

        def _make_action(*args, **kwargs) -> str:
            '''Make single step action in MatterSim.'''
            observation_str = "\n\t**Current Top-down Map**: <ImageHere>"

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

            if self.config.use_history_chain:
                history = self.history_chain.run(
                    observation = new_ob['obs_summary'], 
                    history = self.vln_agent.history[-1], 
                    previous_action = turned_angle
                )
            else: history = self.get_text_history(new_ob, turned_angle)
            self.vln_agent.history.append(history)
            self.vln_agent.scene_graph = new_scene_graph

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

    def _create_back_trace_tool(self, llm: MultimodalOpenAI) -> Tool:
        """ Create a tool to back trace during navigation. """
        if not self.config.use_backtrace: return None

        prompt = PromptTemplate(
            template=BACK_TRACE_PROMPT,
            input_variables=["action_plan", "history", "observation"],
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        def _parse_action(llm_output: str) -> Tuple[str, str]:
            regex = r"(.*?)Final Answer:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")

            thought = match.group(1).strip()
            action = match.group(2).strip(" ").strip('"').strip("'")
            return thought, action

        def _back_trace(*args, **kwargs) -> str:
            '''Back trace the action plan.'''
            action_plan = self.cur_action_plan
            cur_ob = self.env.observe()[self.cur_env_index]
            feature, pos, orient, navigable = self.get_env_feature(cur_ob)
            orientation = f'heading: {orient[0]:.2f}, elevation: {orient[1]:.2f}'
            
            history_str = self.get_his_viewpoints()
            LLM_output = chain.run(
                action_plan=action_plan, 
                observation=history_str, 
                history=self.vln_agent.history[-1]
            )
            thought, action = _parse_action(LLM_output)

            if action not in self.env.env.sims[self.cur_env_index].navigable_dict.keys():
                return f"\nViewpointID '{action}' is not valid. " + \
                       f"DO NOT fabricate nonexistent IDs.\n" + \
                       f"\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"
            
            batch_actions = [None] * self.env.batch_size
            batch_actions[self.cur_env_index] = action
            new_ob = self.make_equiv_action(batch_actions)[1][self.cur_env_index]

            new_feature, new_pos, new_orient, new_navigable = self.get_env_feature(new_ob)
            new_orientation = f'heading: {new_orient[0]:.2f}, elevation: {new_orient[1]:.2f}'
            new_scene_graph = self.get_visual_map(
                scan_id=new_ob['scan'],
                position=new_pos,
                orientation=new_orient,
                navigable_viewpoints=new_navigable
            )
            self.vln_agent.scene_graph = new_scene_graph

            history = self.get_text_history(
                new_ob, 
                'Seems going in a wrong way, back trace to a previous point.')
            self.vln_agent.history.append(history)

            if self.config.use_relative_angle:
                return f"\n\t**Current Viewpoint**:{action}\n{new_feature}"
            else:
                return f"\n\t**Current Orientation**:\n{new_orientation}" + \
                       f"\n\t**Current Viewpoint**:\n{new_feature}"

        return Tool(
            name=BACK_TRACE_TOOL_NAME,
            func=_back_trace,
            description=BACK_TRACE_TOOL_DESCRIPTION,
        )

    @property
    def agent_prompt_template(self) -> str:
        return get_agent_prompt_template(self.config.prompt_version, self.config.llm_model_name)

    def create_vln_agent(self) -> AgentExecutor:
        self.action_maker = self._create_make_action_tool(self.llm)
        self.back_tracer = self._create_back_trace_tool(self.llm)

        tools = [x for x in [self.action_maker, self.back_tracer] if x is not None]
        agent_prompt = MultimodalPromptTemplate(
            template=self.agent_prompt_template,
            image_key="scene_graph",
            input_variables=["action_plan", "init_observation", "agent_scratchpad", "scene_graph"],
            partial_variables={
                "tool_names": ", ".join([tool.name for tool in tools]),
                "tool_descriptions": "\n".join(
                    [f"{tool.name}: {tool.description}" for tool in tools]
                ),
            },
        )
        # insetantiate the VLNAgent with the LLM chain and tools
        agent = SpatialVLNAgent(
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

    def rollout(self, reset=True, **kwargs) -> List[dict]:
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
            
            cur_feature, pos, orient, navigable = self.get_env_feature(init_ob)
            scene_graph = self.get_visual_map(
                scan_id=init_ob['scan'],
                position=pos,
                orientation=orient,
                navigable_viewpoints=navigable
            )
            init_observation = self.get_text_observation(cur_feature, orient, navigable)
            print("=== Initial Observation ===")
            print(init_observation)
            mllm_inputs = {
                "action_plan": self.cur_action_plan, 
                "init_observation": init_observation,
                "scene_graph": scene_graph,
            }

            # === LangChain Core ===
            output = self.agent_executor(mllm_inputs)

            self.traj[i]['llm_output'] = output['output']
            self.traj[i]['action_plan'] = output['action_plan']

            intermediate_steps = output['intermediate_steps']
            self.traj[i]['llm_thought'] = []
            self.traj[i]['llm_observation'] = [init_observation]
            for action, observation in intermediate_steps:
                thought = action.log
                self.traj[i]['llm_thought'].append(thought)
                self.traj[i]['llm_observation'].append(observation)
            self.traj[i]['llm_thought'].append(output.get('finish_thought', "MAX STEP REACHED."))

        return self.traj

    @staticmethod
    def normalize_angle(angle):
        while angle > 180: angle -= 360
        while angle <= -180: angle += 360
        return angle

    @staticmethod
    def angle_to_left_right(angle):
        return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"

    @staticmethod
    def compact_dict_str(d, indent=4):
        if len(d) == 0: return " {" + "}"
        if len(d) == 1: return f" {d}"

        output_str = [" "* indent + "{"]
        for i, (k, v) in enumerate(d.items()):
            cur_str = " "* (indent + 2) + f"'{k}': '{v}'"
            if i != len(d) - 1: cur_str += ", "
            output_str.append(cur_str)
        output_str.append(" "* indent + "}")
        return "\n".join(output_str)

    def get_visual_map(self, scan_id: str, position: Tuple[float, float, float], 
                       orientation: Tuple[float, float], navigable_viewpoints: List[dict], 
                       vertical_threshold: float=2.0) -> Image.Image:
        """ Get the visual top-down spatial map for the given position and orientation. """
        # add random noise to the position and heading to simulate 
        # real-world uncertainty (optional, can be commented out)
        if os.environ.get("AGENT_ADD_NOISE", "False").lower() == "true":
            t_step = len(self.traj[self.cur_env_index]['path'])-1
            p_noise = np.linspace(0.1, 0.25, num=15)[min(t_step, 14)]
            delta = np.random.uniform(-p_noise, p_noise+1e-5)
            add_to = np.random.choice([0, 1])
            if add_to == 0: position = [position[0] + delta, position[1], position[2]]
            else: position = [position[0], position[1] + delta, position[2]]
            h_noise = np.linspace(5.0, 15.0, num=15)[min(t_step, 14)]
            heading, elevation = orientation
            heading += np.random.uniform(-h_noise, h_noise+1e5)
            orientation = (heading, elevation)

        history_path = self.traj[self.cur_env_index]['path']
        history_trajectory = sum(history_path, [])
        history_dict = dict()
        for i, vp_id in enumerate(history_trajectory):
            vp_pos = self.env.get_vp_location(scan_id, vp_id)
            within_same_level = bool(abs(vp_pos[2] - position[2]) <= vertical_threshold) # m
            history_dict[vp_id] = {"position": vp_pos, "history_order": i, "within_same_level": within_same_level}
        # all viewpoints before the last different level point are marked as False
        for i, vp_id in enumerate(history_trajectory[::-1]):
            if not history_dict[vp_id]["within_same_level"]:
                for j, vp_j in enumerate(history_trajectory[:len(history_trajectory)-i]):
                    history_dict[vp_j]["within_same_level"] = False
                break
        
        scene_graph = self.mapper.get_visual_map(
            scan_id=scan_id,
            position=position,
            orientation=orientation,
            navigable_viewpoints=navigable_viewpoints, 
            history_viewpoints=history_dict
        )
        
        instr_id = self.traj[self.cur_env_index]['instr_id']
        num_steps = len(self.traj[self.cur_env_index]['path']) - 1
        cur_save_path = os.path.join(self.config.map_save_dir, scan_id, f"{instr_id}_step{num_steps}.jpg")
        os.makedirs(os.path.dirname(cur_save_path), exist_ok=True)
        scene_graph.save(cur_save_path)
        
        return scene_graph

