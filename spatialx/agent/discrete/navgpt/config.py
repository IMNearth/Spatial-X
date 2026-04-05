import os
from dataclasses import dataclass, field


@dataclass
class NavGPTConfig:
    """配置 NavGPT / VLNAgent 行为的参数"""
    
    # Agent
    llm_model_name: str = field(default="gpt-4")                    # LLM 模型名称，例如 'gpt-4'
    api_base_url: str = field(default="https://api.openai.com/v1")  # API 基础 URL
    api_key: str = field(default="")                                # (OpenAI) API Key
    temperature: float = field(default=0.0)                         # 生成温度（控制随机性）
    
    # Environment
    use_relative_angle: bool = field(default=True)      # 是否在 prompt 中使用相对角度
    use_navigable: bool = field(default=False)          # 是否在 prompt 中展示可导航视点信息
    load_instruction: bool = field(default=True)        # 是否加载任务指令当作 action_plan（跳过 plan_chain）
    load_action_plan: bool = field(default=False)       # 是否直接加载已有 action_plan（跳过 plan_chain）
    
    # General Config
    max_iterations: int = field(default=10)            # AgentExecutor 的最大推理步数（防止死循环）
    max_scratchpad_length: int = field(default=7000)   # AgentExecutor scratchpad 的最大长度（防止过长截断）

    # Prompting
    use_tool_chain: bool = field(default=False)         # 是否使用多工具链（orchestrator 调用多个 tool）
    use_history_chain: bool = field(default=False)      # 是否用 LLMChain 压缩历史（替代 get_history）
    use_single_action: bool = field(default=True)       # 是否只使用模型一步进行动作用选择

    # Saving and Logging
    save_dir: str = field(default="outputs/results/navgpt")         # 结果保存目录

