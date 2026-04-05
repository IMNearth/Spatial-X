"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Dict, Union
import os
import re
import json
import numpy as np

from langchain.agents.agent import AgentExecutor, AgentAction, AgentOutputParser
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    OutputParserException
)
from langchain.base_language import BaseLanguageModel

from spatialx.mp3d_extensions import DiscreteNavBatch

from spatialx.utils import MultimodalOpenAI
from ..agent_base import BaseAgent
from .config import NavGPTConfig
from .prompts import (
    ACTION_PROMPT,
    HISTORY_PROMPT,
    PLANNER_PROMPT,
    BACK_TRACE_PROMPT,
    MAKE_ACTION_TOOL_NAME,
    MAKE_ACTION_TOOL_DESCRIPTION,
    BACK_TRACE_TOOL_NAME,
    BACK_TRACE_TOOL_DESCRIPTION,
    VLN_ORCHESTRATOR_PROMPT,
    VLN_GPT4_PROMPT,
    VLN_GPT5_PROMPT
)

from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
FINAL_ANSWER_ACTION = "Final Answer:"
EXCEPTION_TOOL_NAME = "_Exception"
MAX_SCRATCHPAD_LENGTH = 7000

MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)


class NavGPTOutputParser(AgentOutputParser):
    """MRKL Output parser for the chat agent."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        # Action: <tool-name>
        # Action Input: "<32-hex viewpointID>"
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*\"?([a-fA-F0-9]{32})\"?"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

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
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "mrkl-NavGPT"



class VLNAgent(ZeroShotAgent):
    history: Optional[List[str]] = None 

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        nav_step = 1
        for i, (action, observation) in enumerate(intermediate_steps):
            if self.llm_prefix.strip() in action.log:
                action_log = action.log.split(self.llm_prefix.strip())[-1].strip()
            else: action_log = action.log.strip()
            action_log = "\n".join([x.strip() for x in action_log.split("\n") if x.strip() != ""])
            thoughts += action_log
            if (i == len(intermediate_steps) - 1) or (action.tool != MAKE_ACTION_TOOL_NAME):
                thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
            else:
                thoughts += f"\n{self.observation_prefix}{self.history[nav_step]}\n{self.llm_prefix}"
                nav_step += 1
        return thoughts

    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)[-MAX_SCRATCHPAD_LENGTH:]
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        if len(intermediate_steps) == 0:
            full_inputs = {**kwargs, **new_inputs}
        else:
            kwargs["init_observation"] = self.history[0]
            full_inputs = {**kwargs, **new_inputs}
        return full_inputs



class NavGPTAgent(BaseAgent):
    
    name = "navgpt"

    def __init__(self, env: DiscreteNavBatch, config: NavGPTConfig):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The discrete Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env)
        self.config = config

        self.llm = MultimodalOpenAI(
            temperature=config.temperature,
            model_name=config.llm_model_name,
            openai_api_base=config.api_base_url,
            openai_api_key=config.api_key,
        )
        
        self.output_parser = NavGPTOutputParser()
        self.agent_executor: AgentExecutor = self.create_vln_agent()
        self.vln_agent: VLNAgent = self.agent_executor.agent

        plan_prompt = PromptTemplate(
            template=PLANNER_PROMPT,
            input_variables=["instruction"],
        )
        self.plan_chain = LLMChain(llm=self.llm, prompt=plan_prompt)
    
    def parse_action(self, llm_output: str) -> Tuple[str, str]:
        regex = r"(.*?)Final Answer:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        thought = match.group(1).strip()
        action = match.group(2).strip(" ").strip('"').strip("'")

        return thought, action

    def get_his_viewpoints(self) -> str:
        '''Return the history of visited viewpoints for back tracing.'''
        his_viewpoints = ''
        for i, detail in enumerate(self.traj[0]['details'][:-1]):
            viewpointID = detail['viewpointID']
            viewpoint_ob = detail['feature']
            his_viewpoints += f"Step {i+1}. Viewpoint ID '{viewpointID}':\n {viewpoint_ob}\n\n"
        return his_viewpoints
    
    def get_history(self, obs: dict, angle: str) -> str:
        '''Return the history of actions taken.'''
        history = f'{angle}\nCurrent viewpoint "{obs["viewpoint"]}": Scene from the viewpoint is a {obs["obs_summary"]}'
        return history

    def get_navigable_str(self, cur_heading: float, cur_elevation: float, navigable: dict) -> str:
        '''Return the navigable viewpoints as a string.'''
        navigable_str = ''

        for vp, items in navigable.items():
            heading = np.rad2deg(items['heading'])
            elevation = np.rad2deg(items['elevation'])
            distance = items['distance']
            rel_heading = heading - cur_heading
            rel_elevation = elevation - cur_elevation

            if self.config.use_relative_angle:
                navigable_str += f"'{vp}':\nheading: {rel_heading:.2f}, elevation: {rel_elevation:.2f}, distance: {distance:.2f}\n"
            else:
                navigable_str += f"'{vp}':\nheading: {heading:.2f}, elevation: {elevation:.2f}, distance: {distance:.2f}\n"

        return navigable_str

    def modify_heading_angles(self, heading_angle, observation_list, candidate_dict, object_list):
        def normalize_angle(angle):
            while angle > 180: angle -= 360
            while angle <= -180: angle += 360
            return angle
        
        def angle_to_left_right(angle):
            return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"
        
        directions = ['Front', 'Front Right', 'Right', 'Rear Right', 'Rear', 'Rear Left', 'Left', 'Front Left']
        range_idx = int((heading_angle - 22.5) // 45) + 1
        obs_idx = [(i + range_idx) % 8 for i in range(8)]
        
        candidate_range = {}
        if not self.config.use_navigable:
            for viewpoint_id, viewpoint_data in candidate_dict.items():
                viewpoint_heading = np.rad2deg(viewpoint_data['heading'])
                vp_range_idx = int((viewpoint_heading - 22.5) // 45) + 1
                rel_viewpoint_heading = viewpoint_heading - heading_angle
                rel_viewpoint_heading = normalize_angle(rel_viewpoint_heading)
                rel_viewpoint_heading = angle_to_left_right(rel_viewpoint_heading)
                vp_description = rel_viewpoint_heading + f', {viewpoint_data["distance"]:.2f}m'
                candidate_range.setdefault(vp_range_idx, {}).update({viewpoint_id: vp_description})

        angle_ranges = [(angle - 22.5 - heading_angle, angle + 22.5 - heading_angle) for angle in range(0, 360, 45)]
        formatted_strings = []
        
        for direction, idx in zip(directions, obs_idx):
            rel_angle1 = normalize_angle(angle_ranges[idx][0])
            rel_angle2 = normalize_angle(angle_ranges[idx][1])
            left_right1 = angle_to_left_right(rel_angle1)
            left_right2 = angle_to_left_right(rel_angle2)
            formatted_string = f"{direction}, range ({left_right1} to {left_right2}): \n'{observation_list[idx]}'"

            object_dict = {}
            if len(object_list[idx]) > 0:
                object = object_list[idx]
                for obj, obj_data in object.items():
                    rel_obj_heading = obj_data['heading'] - heading_angle
                    rel_obj_heading = normalize_angle(rel_obj_heading)
                    rel_obj_heading = angle_to_left_right(rel_obj_heading)
                    object_dict[obj] = f'{rel_obj_heading}, {obj_data["distance"]:.2f}m'
                formatted_string += f'\n{direction} Objects in 3m: {object_dict}'
            else:
                formatted_string += f'\n{direction} Objects in 3m: None'

            if candidate_range.get(idx):
                formatted_string += f'\n{direction} Navigable Viewpoints:{candidate_range[idx]}'
            else:
                formatted_string += f'\n{direction} Navigable Viewpoints: None'
            formatted_strings.append(formatted_string)
        
        output_string = '\n'.join(formatted_strings)
        return output_string

    def init_trajecotry(self, obs: List[dict]):
        """Initialize the trajectory with the given observation."""
        self.traj = [{
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': [],
        } for ob in obs]
        
        self.vln_agent.history = [
            f'Navigation start, no actions taken yet.\n' + \
            f'Current viewpoint "{obs[0]["viewpoint"]}": ' + \
            f'Scene from the viewpoint is a {obs[0]["obs_summary"]}']

    def _create_make_action_tool(
            self,
            llm: BaseLanguageModel,
    ) -> Tool:
        """Create a tool to make single action prediction in MP3D."""

        action_prompt = PromptTemplate(
            template=ACTION_PROMPT,
            input_variables=["action_plan", "observation", "history", "navigable_viewpoints"],
        )
        history_prompt = PromptTemplate(
            template=HISTORY_PROMPT,
            input_variables=["history", "previous_action", "observation"],
        )
        self.action_chain = LLMChain(llm=llm, prompt=action_prompt)
        self.history_chain = LLMChain(llm=llm, prompt=history_prompt)

        def _make_action(*args, **kwargs) -> str:
            '''Make single step action in MatterSim.'''
            cur_obs = self.env.observe()[0]

            feature = cur_obs['obs_detail']
            heading = np.rad2deg(cur_obs['heading'])
            elevation = np.rad2deg(cur_obs['elevation'])
            objects = cur_obs['obs_objects']
            orientation = f'\nheading: {heading:.2f}, elevation: {elevation:.2f}'
            navigable = cur_obs['candidate']
            if self.config.use_relative_angle:
                feature = self.modify_heading_angles(heading, feature, navigable, objects)
            if self.config.use_navigable:
                navigable = self.get_navigable_str(heading, elevation, navigable)

            if self.config.use_tool_chain:
                action_plan = self.cur_action_plan
                LLM_action_output = self.action_chain.run(
                    action_plan = action_plan, 
                    observation = feature, 
                    history = self.vln_agent.history[-1], 
                    navigable_viewpoints = navigable
                )
                thought, action = self.parse_action(LLM_action_output)
            else:
                # the next viewpointID is passed as the first argument
                action = args[0].strip(" ").strip('"').strip("'")

            if action not in self.env.env.sims[0].navigable_dict.keys():
                history = f'ViewpointID "{action}" is not valid, no action taken for the agent.'
                self.vln_agent.history.append(history)
                if self.config.use_navigable:
                    return f"\nViewpointID '{action}' is not valid, agent not moved. DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose from current viewpoints are: {[key for key in navigable.keys()]}.\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                else:
                    return f"\nViewpointID '{action}' is not valid, agent not moved. DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose from current viewpoints are: {[key for key in navigable.keys()]}.\n\tCurrent Viewpoint:\n{feature}"
            else:
                turned_angle, new_obs = self.make_equiv_action([action])

            # update the obervation, history and navigable for the next step
            new_feature = new_obs['obs_detail']
            new_feature_sum = new_obs['obs_summary']
            new_navigable = new_obs['candidate']
            new_objects = new_obs['obs_objects']
            new_heading = np.rad2deg(new_obs['heading'])
            new_elevation = np.rad2deg(new_obs['elevation'])
            if self.config.use_relative_angle:
                new_feature = self.modify_heading_angles(new_heading, new_feature, new_navigable, new_objects)
            new_orientation = f'\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}'
            if self.config.use_navigable:
                new_navigable = self.get_navigable_str(new_heading, new_elevation, new_navigable)

            if self.config.use_history_chain:
                history = self.history_chain.run(
                    observation = new_feature_sum, 
                    history = self.vln_agent.history[-1], 
                    previous_action = turned_angle
                )
            else:
                history = self.get_history(new_obs, turned_angle)
            
            self.vln_agent.history.append(history)

            # 5) record the trajectory details for analysis and debugging
            if self.config.use_tool_chain:
                detail = {
                    "viewpointID": action,
                    "turned_angle": turned_angle,
                    "acion_maker_thought": thought,
                    "feature": new_feature,
                    "history": self.vln_agent.history[-1],
                }
            else:
                detail = {
                    "viewpointID": action,
                    "turned_angle": turned_angle,
                    "feature": new_feature,
                    "history": self.vln_agent.history[-1],
                }
            self.traj[0]['details'].append(detail)

            # return the observation for the next loop
            if self.config.use_tool_chain:
                return f"\n\tAction_maker Thought:\n{thought}\n\tAction_maker Action:\n{turned_angle}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
            elif self.config.use_relative_angle:
                if self.config.use_navigable:
                    return f"\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f'\nCurrent Viewpoint "{action}":\n{new_feature}'
            else:
                if self.config.use_navigable:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}"
        
        return Tool(
            name=MAKE_ACTION_TOOL_NAME,
            func=_make_action,
            description=MAKE_ACTION_TOOL_DESCRIPTION,
        )

    def _create_back_trace_tool(
            self,
            llm: BaseLanguageModel,
    ) -> Tool:
        """Create a tool to back trace during navigation."""
        prompt = PromptTemplate(
            template=BACK_TRACE_PROMPT,
            input_variables=["action_plan", "history", "observation"],
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        def _back_trace(*args, **kwargs) -> str:
            '''Back trace the action plan.'''
            cur_obs = self.env.observe()[0]

            feature = cur_obs['obs_detail']
            navigable = cur_obs['candidate']
            objects = cur_obs['obs_objects']
            heading = np.rad2deg(cur_obs['heading'])
            elevation = np.rad2deg(cur_obs['elevation'])
            orientation = f'\nheading: {heading:.2f}, elevation: {elevation:.2f}'
            if self.config.use_relative_angle:
                feature = self.modify_heading_angles(heading, feature, navigable, objects)
            if self.config.use_navigable:
                navigable = self.get_navigable_str(heading, elevation, navigable)

            if self.config.use_tool_chain:
                action_plan = self.cur_action_plan
                previous_vp = self.get_his_viewpoints()
                LLM_output = chain.run(
                    action_plan = action_plan, 
                    observation = previous_vp, 
                    history = self.vln_agent.history[-1]
                )
                thought, action = self.parse_action(LLM_output)
            else:
                action = args[0].strip(" ").strip('"').strip("'")

            if action not in self.env.env.sims[0].navigable_dict.keys():
                if self.config.use_navigable:
                    return f"\nViewpointID '{action}' is not valid. DO NOT fabricate nonexistent IDs.\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                else:
                    return f"\nViewpointID '{action}' is not valid. DO NOT fabricate nonexistent IDs.\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"
            else:
                _, new_obs = self.make_equiv_action([action])
            
            new_feature = new_obs['obs_detail']
            new_navigable = new_obs['candidate']
            new_objects = new_obs['obs_objects']
            new_heading = np.rad2deg(new_obs['heading'])
            new_elevation = np.rad2deg(new_obs['elevation'])
            new_orientation = f'\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}'
            if self.config.use_relative_angle:
                new_feature = self.modify_heading_angles(new_heading, new_feature, new_navigable, new_objects)
            if self.config.use_navigable:
                new_navigable = self.get_navigable_str(new_heading, new_elevation, new_navigable)

            history = self.get_history(new_obs, 'Seems going in a wrong way, back trace to a previous point.')
            self.vln_agent.history.append(history)
            if self.config.use_tool_chain:
                return f"\tBack_tracer Thought:\n{thought}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
            elif self.config.use_relative_angle:
                if self.config.use_navigable:
                    return f"\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\nCurrent Viewpoint:{action}\n{new_feature}"
            else:
                if self.config.use_navigable:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}"

        return Tool(
            name=BACK_TRACE_TOOL_NAME,
            func=_back_trace,
            description=BACK_TRACE_TOOL_DESCRIPTION,
        )

    def create_vln_agent(
        self,
    ) -> AgentExecutor:
        """Instantiate API planner and controller for a given trajectory."""

        self.action_maker = self._create_make_action_tool(self.llm)
        self.back_tracer = self._create_back_trace_tool(self.llm)

        if self.config.use_tool_chain:
            tools = [self.action_maker, self.back_tracer]
            prompt = PromptTemplate(
                template=VLN_ORCHESTRATOR_PROMPT,
                input_variables=["action_plan", "init_observation", "observation", "agent_scratchpad"],
                partial_variables={
                    "tool_names": ", ".join([tool.name for tool in tools]),
                    "tool_descriptions": "\n".join(
                        [f"{tool.name}: {tool.description}" for tool in tools]
                    ),
                },
            )
        elif self.config.use_single_action:
            tools = [self.action_maker]
            prompt = PromptTemplate(
                template=VLN_GPT4_PROMPT if "gpt-5" not in self.config.llm_model_name.lower() else VLN_GPT5_PROMPT,
                input_variables=["action_plan", "init_observation", "agent_scratchpad"],
                partial_variables={
                    "tool_names": ", ".join([tool.name for tool in tools]),
                    "tool_descriptions": "\n".join(
                        [f"{tool.name}: {tool.description}" for tool in tools]
                    ),
                },
            )
        else:
            tools = [self.action_maker, self.back_tracer]
            prompt = PromptTemplate(
                template=VLN_ORCHESTRATOR_PROMPT,
                input_variables=["action_plan", "init_observation", "agent_scratchpad"],
                partial_variables={
                    "tool_names": ", ".join([tool.name for tool in tools]),
                    "tool_descriptions": "\n".join(
                        [f"{tool.name}: {tool.description}" for tool in tools]
                    ),
                },
            )

        # insetantiate the VLNAgent with the LLM chain and tools
        agent = VLNAgent(
            llm_chain=LLMChain(llm=self.llm, prompt=prompt),
            allowed_tools=[tool.name for tool in tools],
            output_parser = self.output_parser
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

    def make_equiv_action(self, actions: List[str]) -> str:
        """
        Interface between Panoramic view and Egocentric view
        Take in the next viewpoint ID and move the agent to that viewpoint
        return the turned angle and new observation
        """
        def normalize_angle(angle):
            """ Normalize angle to [-180, 180] degrees. """
            while angle > 180:
                angle -= 360
            while angle <= -180:
                angle += 360
            return angle

        def angle_to_left_right(angle):
            """ Convert angle to left/right string. """
            return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"
        
        cur_obs = self.env.observe()[0]
        cur_heading = np.rad2deg(cur_obs['heading'])
        
        new_obs = self.env.step(actions)[0]
        new_heading = np.rad2deg(new_obs['heading'])
        self.traj[0]['path'].append(
            self.env.env.sims[0].gmap.bfs_shortest_path(cur_obs['viewpoint'], actions[0])[1:]
        )
        
        turned_angle = new_heading - cur_heading
        cur_heading = angle_to_left_right(normalize_angle(cur_heading))
        new_heading = angle_to_left_right(normalize_angle(new_heading))
        action_description = f'Turn heading direction {turned_angle:.2f} degrees from {cur_heading} to {new_heading}.'
        return action_description, new_obs

    def rollout(self, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env.observe()

        self.init_trajecotry(obs)

        instructions = [ob['instruction'] for ob in obs]
        if self.config.load_instruction:
            action_plans = instructions
        elif self.config.load_action_plan:
            action_plans = [ob['action_plan'] for ob in obs]
        else:
            action_plans = []
            for instruction in instructions:
                action_plan = self.plan_chain.run(instruction = instruction)
                action_plans.append(action_plan)

        for i, init_ob in enumerate(obs):
            self.cur_action_plan = action_plans[i]
            if self.config.use_tool_chain:
                first_obs = self.action_maker('')
                input = {
                    'action_plan': self.cur_action_plan,
                    'init_observation': init_ob['obs_summary'],
                    'observation': first_obs,
                }
            else:
                feature = init_ob['obs_detail']
                navigable = init_ob['candidate']
                objects = init_ob['obs_objects']
                heading = np.rad2deg(init_ob['heading'])
                elevation = np.rad2deg(init_ob['elevation'])
                orientation = f'\nheading: {heading:.2f}, elevation: {elevation:.2f}'
                if self.config.use_relative_angle:
                    feature = self.modify_heading_angles(heading, feature, navigable, objects)
                if self.config.use_navigable:
                    navigable = self.get_navigable_str(heading, elevation, navigable)

                if self.config.use_relative_angle:
                    if self.config.use_navigable:
                        init_observation = f"\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                    else:
                        init_observation = f"\n\tCurrent Viewpoint:\n{feature}"
                else:
                    if self.config.use_navigable:
                        init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                    else:
                        init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"

                input = {
                    'action_plan': self.cur_action_plan,
                    'init_observation': init_observation,
                }

            output = self.agent_executor(input)

            self.traj[i]['llm_output'] = output['output']
            self.traj[i]['action_plan'] = output['action_plan']

            intermediate_steps = output['intermediate_steps']
            self.traj[i]['llm_thought'] = []
            self.traj[i]['llm_observation'] = []
            for action, observation in intermediate_steps:
                thought = action.log
                self.traj[i]['llm_thought'].append(thought)
                self.traj[i]['llm_observation'].append(observation)

        return self.traj


