""" https://github.com/chen-judge/MapGPT/blob/main/GPT/one_stage_prompt_manager.py """
import re
import math


class BasePromptManager(object):
    def __init__(self, allow_stop_after:int=3, batch_size:int=1, **kwargs):

        self.batch_size = batch_size
        self.allow_stop_after = allow_stop_after

        self.history  = ['' for _ in range(batch_size)]
        self.nodes_list = [[] for _ in range(batch_size)]
        self.node_imgs = [[] for _ in range(batch_size)]
        self.graph  = [{} for _ in range(batch_size)]
        self.trajectory = [[] for _ in range(batch_size)]
        self.planning = [["Navigation has just started, with no planning yet."] for _ in range(batch_size)]

    def get_action_concept(self, rel_heading, rel_elevation):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def make_action_prompt(self, obs, previous_angle):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def make_history(self, a_t, nav_input, t):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def make_r2r_json_prompts(self, obs, cand_inputs, t):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def make_map_prompt(self, i):
        # graph-related text
        trajectory = self.trajectory[i]
        nodes_list = self.nodes_list[i]
        graph = self.graph[i]

        no_dup_nodes = []
        trajectory_text = 'Place'
        graph_text = ''

        candidate_nodes = graph[trajectory[-1]]

        # trajectory and map connectivity
        for node in trajectory:
            node_index = nodes_list.index(node)
            trajectory_text += f""" {node_index}"""

            if node not in no_dup_nodes:
                no_dup_nodes.append(node)

                adj_text = ''
                adjacent_nodes = graph[node]
                for adj_node in adjacent_nodes:
                    adj_index = nodes_list.index(adj_node)
                    adj_text += f""" {adj_index},"""

                graph_text += f"""\nPlace {node_index} is connected with Places{adj_text}"""[:-1]

        # ghost nodes info
        graph_supp_text = ''
        supp_exist = None
        for node_index, node in enumerate(nodes_list):

            if node in trajectory or node in candidate_nodes:
                continue
            supp_exist = True
            graph_supp_text += f"""\nPlace {node_index}, which is corresponding to Image {node_index}"""

        if supp_exist is None:
            graph_supp_text = """Nothing yet."""

        return trajectory_text, graph_text, graph_supp_text

    def make_action_options(self, cand_inputs, t):
        action_options_batch = []  # complete action options
        only_options_batch = []  # only option labels
        batch_action_prompts = cand_inputs["action_prompts"]
        batch_size = len(batch_action_prompts)

        for i in range(batch_size):
            action_prompts = batch_action_prompts[i]
            if bool(self.allow_stop_after):
                if t >= self.allow_stop_after:
                    action_prompts = ['stop'] + action_prompts

            full_action_options = [chr(j + 65)+'. '+action_prompts[j] for j in range(len(action_prompts))]
            only_options = [chr(j + 65) for j in range(len(action_prompts))]
            action_options_batch.append(full_action_options)
            only_options_batch.append(only_options)

        return action_options_batch, only_options_batch

    def parse_planning(self, nav_output):
        """
        Only supports parsing outputs in the style of GPT-4v.
        Please modify the parsers if the output style is inconsistent.
        """
        batch_size = len(nav_output)
        keyword1 = '\nNew Planning:'
        keyword2 = '\nAction:'
        for i in range(batch_size):
            output = nav_output[i].strip()
            start_index = output.find(keyword1) + len(keyword1)
            end_index = output.find(keyword2)

            if output.find(keyword1) < 0 or start_index < 0 or end_index < 0 or start_index >= end_index:
                planning = "No plans currently."
            else:
                planning = output[start_index:end_index].strip()

            planning = planning.replace('new', 'previous').replace('New', 'Previous')

            self.planning[i].append(planning)

        return planning

    def parse_json_planning(self, json_output):
        try:
            planning = json_output["New Planning"]
        except:
            planning = "No plans currently."

        self.planning[0].append(planning)
        return planning

    def parse_action(self, nav_output, only_options_batch, t):
        """
        Only supports parsing outputs in the style of GPT-4v.
        Please modify the parsers if the output style is inconsistent.
        """
        batch_size = len(nav_output)
        output_batch = []
        output_index_batch = []

        for i in range(batch_size):
            output = nav_output[i].strip()

            pattern = re.compile("Action")  # keyword
            matches = pattern.finditer(output)
            indices = [match.start() for match in matches]
            output = output[indices[-1]:]

            search_result = re.findall(r"Action:\s*([A-M])", output)
            if search_result:
                output = search_result[-1]

                if output in only_options_batch[i]:
                    output_batch.append(output)
                    output_index = only_options_batch[i].index(output)
                    output_index_batch.append(output_index)
                else:
                    output_index = 0
                    output_index_batch.append(output_index)
            else:
                output_index = 0
                output_index_batch.append(output_index)

        if bool(self.allow_stop_after):
            if t < self.allow_stop_after:
                for i in range(batch_size):
                    output_index_batch[i] = output_index_batch[i] + 1  # add 1 to index (avoid stop within 3 steps)
        return output_index_batch

    def parse_json_action(self, json_output, only_options_batch, t):
        try:
            output = str(json_output["Action"])
            if output in only_options_batch[0]:
                output_index = only_options_batch[0].index(output)
            else:
                output_index = 0

        except:
            output_index = 0

        if bool(self.allow_stop_after):
            if t < self.allow_stop_after:
                output_index += 1  # add 1 to index (avoid stop within 3 steps)

        output_index_batch = [output_index]
        return output_index_batch



class MapGPTPromptManager(BasePromptManager):

    def get_action_concept(self, rel_heading, rel_elevation):
        heading_deg = math.degrees(rel_heading)         # rad to deg
        while heading_deg < -180: heading_deg += 360
        while heading_deg >= 180: heading_deg -= 360    # normalize to [-180, 180)
        elevation_deg = math.degrees(rel_elevation)     # rad to deg
        while elevation_deg < -180: elevation_deg += 360
        while elevation_deg >= 180: elevation_deg -= 360   # normalize to [-180, 180)

        if elevation_deg > 30:
            action_text = 'go up'
        elif rel_elevation < -30:
            action_text = 'go down'
        else:
            if -15 <= heading_deg <= 15:
                action_text = "go forward"
            elif 15 < heading_deg <= 90:
                action_text = "turn right"
            elif 90 < heading_deg <= 180:
                action_text = "turn around"
            elif -180 <= heading_deg < -90:
                action_text = "turn around"
            elif -90 <= heading_deg < -15:
                action_text = "turn left"

        return action_text

    def make_action_prompt(self, obs, previous_angle):

        nodes_list, graph, trajectory, node_imgs = self.nodes_list, self.graph, self.trajectory, self.node_imgs

        batch_cand_vpids = []
        batch_action_prompts = []
        for i, ob in enumerate(obs):
            cand_vpids = []
            action_prompts = []

            if ob['viewpoint'] not in nodes_list[i]:
                # update nodes list (place 0)
                nodes_list[i].append(ob['viewpoint'])
                node_imgs[i].append(None)

            # update trajectory
            trajectory[i].append(ob['viewpoint'])

            # cand views
            for cand_vid, cc in sorted(ob['candidate'].items(), key=lambda x: x[1]['__ord']):
                cand_vpids.append(cand_vid)
                
                # obatin the action concept
                direction = self.get_action_concept(
                    cc['heading'] - previous_angle[i]['heading'],
                    cc['elevation'] - previous_angle[i]['elevation']
                )

                if cand_vid not in nodes_list[i]:
                    nodes_list[i].append(cand_vid)
                    node_imgs[i].append(cc['image'])
                    node_index = nodes_list[i].index(cand_vid)
                else: # update image
                    node_index = nodes_list[i].index(cand_vid)
                    node_imgs[i][node_index] = cc['image']

                action_text = direction + f" to Place {node_index} which is corresponding to Image {node_index}"
                action_prompts.append(action_text)

            batch_cand_vpids.append(cand_vpids)
            batch_action_prompts.append(action_prompts)

            # update graph
            if ob['viewpoint'] not in graph[i].keys():
                graph[i][ob['viewpoint']] = cand_vpids

        return {
            'cand_vpids': batch_cand_vpids,
            'action_prompts': batch_action_prompts,
        }

    def make_history(self, a_t, nav_input, t):
        batch_size = len(a_t)
        for i in range(batch_size):
            nav_input["only_actions"][i] = ['stop'] + nav_input["only_actions"][i]
            last_action = nav_input["only_actions"][i][a_t[i]]
            if t == 0:
                self.history[i] += f"""step {str(t)}: {last_action}"""
            else:
                self.history[i] += f""", step {str(t)}: {last_action}"""

    def make_r2r_json_prompts(self, obs, cand_inputs, t):

        background = """You are an embodied robot that navigates in the real world."""
        background_supp = """You need to explore between some places marked with IDs and ultimately find the destination to stop.""" \
        + """ At each step, a series of images corresponding to the places you have explored and have observed will be provided to you."""

        instr_des = """'Instruction' is a global, step-by-step detailed guidance, but you might have already executed some of the commands. You need to carefully discern the commands that have not been executed yet."""
        traj_info = """'Trajectory' represents the ID info of the places you have explored. You start navigating from Place 0."""
        map_info = """'Map' refers to the connectivity between the places you have explored and other places you have observed."""
        map_supp = """'Supplementary Info' records some places and their corresponding images you have ever seen but have not yet visited. These places are only considered when there is a navigation error, and you decide to backtrack for further exploration."""
        history = """'History' represents the places you have explored in previous steps along with their corresponding images. It may include the correct landmarks mentioned in the 'Instruction' as well as some past erroneous explorations."""
        option = """'Action options' are some actions that you can take at this step."""
        pre_planning = """'Previous Planning' records previous long-term multi-step planning info that you can refer to now."""

        requirement = """For each provided image of the places, you should combine the 'Instruction' and carefully examine the relevant information, such as scene descriptions, landmarks, and objects. You need to align 'Instruction' with 'History' (including corresponding images) to estimate your instruction execution progress and refer to 'Map' for path planning. Check the Place IDs in the 'History' and 'Trajectory', avoiding repeated exploration that leads to getting stuck in a loop, unless it is necessary to backtrack to a specific place."""
        dist_require = """If you can already see the destination, estimate the distance between you and it. If the distance is far, continue moving and try to stop within 1 meter of the destination."""
        thought = """Your answer should be JSON format and must include three fields: 'Thought', 'New Planning', and 'Action'. You need to combine 'Instruction', 'Trajectory', 'Map', 'Supplementary Info', your past 'History', 'Previous Planning', 'Action options', and the provided images to think about what to do next and why, and complete your thinking into 'Thought'."""
        new_planning = """Based on your 'Map', 'Previous Planning' and current 'Thought', you also need to update your new multi-step path planning to 'New Planning'."""
        action = """At the end of your output, you must provide a single capital letter in the 'Action options' that corresponds to the action you have decided to take, and place only the letter into 'Action', such as "Action: A"."""

        task_description = f"""{background} {background_supp}\n{instr_des}\n{history}\n{traj_info}\n{map_info}\n{map_supp}\n{pre_planning}\n{option}\n{requirement}\n{dist_require}\n{thought}\n{new_planning}\n{action}"""

        init_history = 'The navigation has just begun, with no history.'

        batch_size = len(obs)
        action_options_batch, only_options_batch = self.make_action_options(cand_inputs, t=t)
        prompt_batch = []
        for i in range(batch_size):
            instruction = obs[i]["instruction"]

            trajectory_text, graph_text, graph_supp_text = self.make_map_prompt(i)

            if t == 0:
                prompt = f"""Instruction: {instruction}\nHistory: {init_history}\nTrajectory: {trajectory_text}\nMap:{graph_text}\nSupplementary Info: {graph_supp_text}\nPrevious Planning:\n{self.planning[i][-1]}\nAction options (step {str(t)}): {action_options_batch[i]}"""
            else:
                prompt = f"""Instruction: {instruction}\nHistory: {self.history[i]}\nTrajectory: {trajectory_text}\nMap:{graph_text}\nSupplementary Info: {graph_supp_text}\nPrevious Planning:\n{self.planning[i][-1]}\nAction options (step {str(t)}): {action_options_batch[i]}"""

            prompt_batch.append(prompt)

        nav_input = {
            "task_description": task_description,
            "prompts" : prompt_batch,
            "only_options": only_options_batch,
            "action_options": action_options_batch,
            "only_actions": cand_inputs["action_prompts"]
        }

        return nav_input



class TextHistorySpatialMapGPTPromptManager(BasePromptManager):

    def get_action_concept(self, rel_heading, rel_elevation):
        heading_deg = math.degrees(rel_heading)         # rad to deg
        if heading_deg < 0: heading_deg += 360
        if heading_deg >= 360: heading_deg -= 360       # normalize to [0, 360)
        elevation_deg = math.degrees(rel_elevation)     # rad to deg
        if elevation_deg < -180: elevation_deg += 360
        if elevation_deg >= 180: elevation_deg -= 360   # normalize to [-180, 180)

        if 0 <= heading_deg < 30 or heading_deg >= 330:
            action_text = 'go forward'
        elif 30 <= heading_deg < 90:
            action_text =  'turn slight left'
        elif 90 <= heading_deg < 150:
            action_text =  'turn sharp left'
        elif 150 <= heading_deg <= 210:
            action_text =  'turn around'
        elif 210 < heading_deg < 270:
            action_text =  'turn sharp right'
        elif 270 <= heading_deg < 330:
            action_text =  'turn slight right'
        else: raise ValueError("Heading degree out of range [0, 360)")

        if elevation_deg > 30:
            action_text += ' and go up'
        elif elevation_deg < -30:
            action_text += ' and go down'

        return action_text

    def make_action_prompt(self, obs, previous_angle):
        nodes_list, graph, trajectory = self.nodes_list, self.graph, self.trajectory

        batch_cand_vpids, batch_action_prompts = [], []
        for i, ob in enumerate(obs):
            cand_vpids = []
            action_prompts = []

            # update viewpoints observed
            if ob['viewpoint'] not in nodes_list[i]:
                nodes_list[i].append(ob['viewpoint'])
            
            # update trajectory
            trajectory[i].append(ob['viewpoint'])

            for cand_vid, cc in sorted(ob['candidate'].items(), key=lambda x: x[1]['__ord']):
                cand_vpids.append(cand_vid)

                # obtain the index of the candidate viewpoint in the nodes list
                if cand_vid not in nodes_list[i]:
                    nodes_list[i].append(cand_vid)
                node_index = nodes_list[i].index(cand_vid)

                # obatin the action concept
                direction = self.get_action_concept(
                    cc['heading'] - previous_angle[i]['heading'],
                    cc['elevation'] - previous_angle[i]['elevation']
                )
                action_text = direction + f" to Place {node_index} which is corresponding to Image {cc['image_index']}"
                action_prompts.append(action_text)

            batch_cand_vpids.append(cand_vpids)
            batch_action_prompts.append(action_prompts)

            # update graph
            if ob['viewpoint'] not in graph[i].keys():
                graph[i][ob['viewpoint']] = cand_vpids
        
        return {
            'cand_vpids': batch_cand_vpids,
            'action_prompts': batch_action_prompts,
        }

    def make_history(self, a_t, nav_input, t):
        batch_size = len(a_t)
        for i in range(batch_size):
            nav_input["only_actions"][i] = ['stop'] + nav_input["only_actions"][i]
            last_action = nav_input["only_actions"][i][a_t[i]]
            # added by jiwen to fix history format
            last_action = last_action.split("which is corresponding to Image")[0].strip()
            if t == 0: self.history[i] += f"""step {str(t)}: {last_action}"""
            else: self.history[i] += f""", step {str(t)}: {last_action}"""

    def make_r2r_json_prompts(self, obs, cand_inputs, t):

        background = """You are an embodied robot that navigates in the real world."""
        background_supp = """You need to explore between some places marked with IDs and ultimately find the destination to stop.""" \
        + """ At each step, a series of images corresponding to the places you have explored and have observed will be provided to you."""

        instr_des = """'Instruction' is a global, step-by-step detailed guidance, but you might have already executed some of the commands. You need to carefully discern the commands that have not been executed yet."""
        traj_info = """'Trajectory' represents the ID info of the places you have explored. You start navigating from Place 0."""
        map_info = """'Map' refers to the connectivity between the places you have explored and other places you have observed."""
        spatial_map_info = """'Current Top-Down View' provides the top-down view at this step with your location highlighted as a solid red circle. Your facing orientation is indicated by a red arow extending from the center of the red circle, representing the front direction. The navigable places around your are indicated as hollow blue circles. Your history trajectory (if any) are drawn as orange lines in the map connecting the solid orange circles that mark the history places you have previously visited. Since this map is a 2D projection, you should refer to the images when you encounter staircases to determine the corect direction."""
        history = """'History' represents the places you have explored in previous steps, and these places are marked as oragnce circles in the top-down view. It may include the correct paths mentioned in the 'Instruction' as well as some past erroneous explorations."""
        option = """'Action options' are some actions that you can take at this step."""
        pre_planning = """'Previous Planning' records previous long-term multi-step planning info that you can refer to now."""

        requirement = """For each provided image of the places, you should combine the 'Instruction' and carefully examine the relevant information, such as scene descriptions, landmarks, and objects. You need to align 'Instruction' with 'History' to estimate your instruction execution progress and refer to 'Map' for path planning. Check the Place IDs in the 'History', 'Trajectory' and current 'Top-Down View', avoiding repeated exploration that leads to getting stuck in a loop, unless it is necessary to backtrack to a specific place."""
        dist_require = """If you can already see the destination, estimate the distance between you and it. If the distance is far, continue moving and try to stop within 1 meter of the destination."""
        thought = """Your answer should be JSON format and must include three fields: 'Thought', 'New Planning', and 'Action'. You need to combine 'Instruction', 'Trajectory', 'Map', 'Current Top-Down View', your past 'History', 'Previous Planning', 'Action options', and the provided images to think about what to do next and why, and complete your thinking into 'Thought'."""
        new_planning = """Based on your 'Map', 'Current Top-Down View', 'Previous Planning' and current 'Thought', you also need to update your new multi-step path planning to 'New Planning'."""
        action = """At the end of your output, you must provide a single capital letter in the 'Action options' that corresponds to the action you have decided to take, and place only the letter into 'Action', such as "Action: A"."""

        task_description = f"""{background} {background_supp}\n{instr_des}\n{history}\n{traj_info}\n{map_info}\n{spatial_map_info}\n{pre_planning}\n{option}\n{requirement}\n{dist_require}\n{thought}\n{new_planning}\n{action}"""

        init_history = 'The navigation has just begun, with no history.'

        batch_size = len(obs)
        action_options_batch, only_options_batch = self.make_action_options(cand_inputs, t=t)
        prompt_batch = []
        for i in range(batch_size):
            instruction = obs[i]["instruction"]

            trajectory_text, graph_text, graph_supp_text = self.make_map_prompt(i)

            if t == 0:
                prompt = f"""Instruction: {instruction}\nHistory: {init_history}\nTrajectory: {trajectory_text}\nMap:{graph_text}\nCurrent Top-Down View: <ImageHere>\nPrevious Planning:\n{self.planning[i][-1]}\nAction options (step {str(t)}): {action_options_batch[i]}"""
            else:
                prompt = f"""Instruction: {instruction}\nHistory: {self.history[i]}\nTrajectory: {trajectory_text}\nMap:{graph_text}\nCurrent Top-Down View: <ImageHere>\nPrevious Planning:\n{self.planning[i][-1]}\nAction options (step {str(t)}): {action_options_batch[i]}"""

            prompt_batch.append(prompt)

        nav_input = {
            "task_description": task_description,
            "prompts" : prompt_batch,
            "only_options": only_options_batch,
            "action_options": action_options_batch,
            "only_actions": cand_inputs["action_prompts"]
        }

        return nav_input




class VisualHistorySpatialMapGPTPromptManager(MapGPTPromptManager, TextHistorySpatialMapGPTPromptManager):
    """ Use Visual history and spatial map """
    
    def make_action_prompt(self, obs, previous_angle):

        nodes_list, graph, trajectory, node_imgs = self.nodes_list, self.graph, self.trajectory, self.node_imgs

        batch_cand_vpids = []
        batch_action_prompts = []
        for i, ob in enumerate(obs):
            cand_vpids = []
            action_prompts = []

            if ob['viewpoint'] not in nodes_list[i]:
                # update nodes list (place 0)
                nodes_list[i].append(ob['viewpoint'])
                node_imgs[i].append(None)

            # update trajectory
            trajectory[i].append(ob['viewpoint'])

            # cand views
            for cand_vid, cc in sorted(ob['candidate'].items(), key=lambda x: x[1]['__ord']):
                cand_vpids.append(cand_vid)
                
                # obatin the action concept
                direction = self.get_action_concept(
                    cc['heading'] - previous_angle[i]['heading'],
                    cc['elevation'] - previous_angle[i]['elevation']
                )

                if cand_vid not in nodes_list[i]:
                    nodes_list[i].append(cand_vid)
                    node_imgs[i].append(cc['image'])
                    node_index = nodes_list[i].index(cand_vid)
                else: # update image
                    node_index = nodes_list[i].index(cand_vid)
                    node_imgs[i][node_index] = cc['image']

                action_text = direction + f" to Place {node_index} which is corresponding to Image {node_index}"
                action_prompts.append(action_text)

            batch_cand_vpids.append(cand_vpids)
            batch_action_prompts.append(action_prompts)

            # update graph
            if ob['viewpoint'] not in graph[i].keys():
                graph[i][ob['viewpoint']] = cand_vpids

        return {
            'cand_vpids': batch_cand_vpids,
            'action_prompts': batch_action_prompts,
        }
    
    def make_history(self, a_t, nav_input, t):
        batch_size = len(a_t)
        for i in range(batch_size):
            nav_input["only_actions"][i] = ['stop'] + nav_input["only_actions"][i]
            last_action = nav_input["only_actions"][i][a_t[i]]
            if t == 0:
                self.history[i] += f"""step {str(t)}: {last_action}"""
            else:
                self.history[i] += f""", step {str(t)}: {last_action}"""
    
    def make_r2r_json_prompts(self, obs, cand_inputs, t):

        background = """You are an embodied robot that navigates in the real world."""
        background_supp = """You need to explore between some places marked with IDs and ultimately find the destination to stop.""" \
            + """ At each step, a series of images corresponding to the places you have explored and have observed will be provided to you."""

        instr_des = """'Instruction' is a global, step-by-step detailed guidance, but you might have already executed some of the commands. """ \
            + """You need to carefully discern the commands that have not been executed yet."""
        traj_info = """'Trajectory' represents the ID info of the places you have explored. You start navigating from Place 0."""
        map_info = """'Map' refers to the connectivity between the places you have explored and other places you have observed. """ + \
            """Besides the textual descriptions, you are also provided with an image of the top-down view of the map at this step, """ + \
            """which you should use together for navigation. In this visual map, your location is highlighted as a solid red circle. """ + \
            """Your facing orientation is indicated by a red arow extending from the center of the red circle, representing the front direction. """ + \
            """The other places you have observed around your are indicated as hollow blue circles. """ + \
            """Your trajectory are drawn as orange lines in the map connecting the solid orange circles that mark the explored places. """ + \
            """Since this map is a 2D projection, when you encounter staircases, """ + \
            """you should refer to the series of images correlated to places so as to determine the corect direction."""
        map_supp = """'Supplementary Info' records some places and their corresponding images you have ever seen but have not yet visited. """ \
            + """Therefore, this places are not visible in the visual top-down map. """ \
            + """These places are only considered when there is a navigation error, and you decide to backtrack for further exploration."""
        history = """'History' represents the places you have explored in previous steps along with their corresponding images. """ \
            + """It may include the correct landmarks mentioned in the 'Instruction' as well as some past erroneous explorations."""
        option = """'Action options' are some actions that you can take at this step."""
        pre_planning = """'Previous Planning' records previous long-term multi-step planning info that you can refer to now."""

        requirement = """For each provided image of the places, you should combine the 'Instruction' and carefully examine the relevant information, """ + \
            """such as scene descriptions, landmarks, and objects. You need to align 'Instruction' with 'History' (including corresponding images) """ + \
            """to estimate your instruction execution progress and refer to 'Map' for path planning. Check the Place IDs in the 'History' and 'Trajectory', avoiding repeated exploration that leads to getting stuck in a loop, unless it is necessary to backtrack to a specific place."""
        dist_require = """If you can already see the destination, estimate the distance between you and it. If the distance is far, continue moving and try to stop within 1 meter of the destination."""
        thought = """Your answer should be JSON format and must include three fields: 'Thought', 'New Planning', and 'Action'. You need to combine 'Instruction', 'Trajectory', 'Map', 'Supplementary Info', your past 'History', 'Previous Planning', 'Action options', and the provided images to think about what to do next and why, and complete your thinking into 'Thought'."""
        new_planning = """Based on your 'Map', 'Previous Planning' and current 'Thought', you also need to update your new multi-step path planning to 'New Planning'."""
        action = """At the end of your output, you must provide a single capital letter in the 'Action options' that corresponds to the action you have decided to take, and place only the letter into 'Action', such as "Action: A"."""

        task_description = f"""{background} {background_supp}\n{instr_des}\n{history}\n{traj_info}\n{map_info}\n{map_supp}\n{pre_planning}\n{option}\n{requirement}\n{dist_require}\n{thought}\n{new_planning}\n{action}"""

        init_history = 'The navigation has just begun, with no history.'

        batch_size = len(obs)
        action_options_batch, only_options_batch = self.make_action_options(cand_inputs, t=t)
        prompt_batch = []
        for i in range(batch_size):
            instruction = obs[i]["instruction"]

            trajectory_text, graph_text, graph_supp_text = self.make_map_prompt(i)

            if t == 0:
                prompt = f"""Instruction: {instruction}\nHistory: {init_history}\nTrajectory: {trajectory_text}\nMap: <ImageHere>{graph_text}\nSupplementary Info: {graph_supp_text}\nPrevious Planning:\n{self.planning[i][-1]}\nAction options (step {str(t)}): {action_options_batch[i]}"""
            else:
                prompt = f"""Instruction: {instruction}\nHistory: {self.history[i]}\nTrajectory: {trajectory_text}\nMap: <ImageHere>{graph_text}\nSupplementary Info: {graph_supp_text}\nPrevious Planning:\n{self.planning[i][-1]}\nAction options (step {str(t)}): {action_options_batch[i]}"""

            prompt_batch.append(prompt)

        nav_input = {
            "task_description": task_description,
            "prompts" : prompt_batch,
            "only_options": only_options_batch,
            "action_options": action_options_batch,
            "only_actions": cand_inputs["action_prompts"]
        }

        return nav_input
