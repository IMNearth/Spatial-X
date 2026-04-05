import os
from dataclasses import dataclass, field
from spatialx.spatial.mapper import SceneGraphConfig


@dataclass
class SpatialNavConfig:
    """配置 SpatialNav 行为的参数"""

    # Agent
    llm_model_name: str = field(default="gpt-4")                    # LLM 模型名称，例如 'gpt-4'
    api_base_url: str = field(default="https://api.openai.com/v1")  # API 基础 URL
    api_key: str = field(default="")                                # (OpenAI) API Key
    temperature: float = field(default=0.0)                         # 生成温度（控制随机性）
    prompt_version: str = field(default="v1")                       # Prompt 版本（v1 / v2）
    
    # Environment
    use_relative_angle: bool = field(default=True)      # 是否在 prompt 中使用相对角度
    use_surround_objects: bool = field(default=True)    # 是否在 prompt 中包含周围物体信息
    load_instruction: bool = field(default=True)        # 是否加载任务指令当作 action_plan（跳过 plan_chain）
    load_action_plan: bool = field(default=False)       # 是否直接加载已有 action_plan（跳过 plan_chain）
    
    # General Config
    max_iterations: int = field(default=10)            # AgentExecutor 的最大推理步数（防止死循环）
    max_scratchpad_length: int = field(default=7000)   # AgentExecutor scratchpad 的最大长度（防止过长截断）

    # Prompting
    use_tool_chain: bool = field(default=False)         # 是否使用多工具链（orchestrator 调用多个 tool）
    use_history_chain: bool = field(default=False)      # 是否用 LLMChain 压缩历史（替代 get_history）
    use_single_action: bool = field(default=True)       # 是否只使用模型一步进行动作用选择
    use_backtrace: bool = field(default=False)          # 是否启用 backtrace 功能（允许回溯之前的步骤）

    # Scene Graph Config
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)

    # Scene Object Config
    object_category_set: str = field(default="raw_category")  # 物体类别集
    object_visibility_radius: float = field(default=3.0)      # 物体可见半径（米）

    # Saving and Logging
    save_dir: str = field(default="outputs/results/spatialgpt")   # 结果保存目录
    map_save_dir: str = field(default="outputs/results/maps")     # 地图保存目录
