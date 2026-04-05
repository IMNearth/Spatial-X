import os
from dataclasses import dataclass, field
from spatialx.spatial.mapper import SceneGraphConfig



@dataclass
class MapGPTConfig:
    """配置 MapGPT / VLNAgent 行为的参数"""

    # Agent
    llm_model_name: str = field(default="gpt-4")                    # LLM 模型名称，例如 'gpt-4'
    api_base_url: str = field(default="https://api.openai.com/v1")  # API 基础 URL
    api_key: str = field(default="")                                # (OpenAI) API Key
    temperature: float = field(default=0.0)                         # 生成温度（控制随机性）
    image_detail: str = field(default="low")                        # 图像细节级别（'low', 'high'）
    
    # Prompting
    max_iterations: int = field(default=10)                         # 最大推理步数（防止死循环）
    allow_stop_after: int = field(default=3)                        # 允许在多少步后停止

    # Saving and Logging
    save_dir: str = field(default="outputs/results/mapgpt")         # 结果保存目录



@dataclass
class MapGPTSpatialConfig:
    """配置 MapGPT / VLNAgent 行为的参数"""

    # Agent
    llm_model_name: str = field(default="gpt-4")                    # LLM 模型名称，例如 'gpt-4'
    api_base_url: str = field(default="https://api.openai.com/v1")  # API 基础 URL
    api_key: str = field(default="")                                # (OpenAI) API Key
    temperature: float = field(default=0.0)                         # 生成温度（控制随机性）
    image_detail: str = field(default="high")                       # 图像细节级别（'low', 'high'）
    
    # Prompting
    max_iterations: int = field(default=10)                         # 最大推理步数（防止死循环）
    allow_stop_after: int = field(default=3)                        # 允许在多少步后停止
    agent_history_type: str = field(default="visual")               # 智能体历史类型（'visual'或'text'）

    # Scene Graph Config
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)

    # Saving and Logging
    save_dir: str = field(default="outputs/results/mapgpt_spatial")         # 结果保存目录
    map_save_dir: str = field(default="outputs/results/maps")       # 地图保存目录
