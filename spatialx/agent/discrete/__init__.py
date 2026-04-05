from .navgpt import (
    NavGPTAgent, 
    NavGPTConfig,
)
from .spatialnav import (
    NavGPTSpatialAgent,
    SpatialNavAgent,
    SpatialNavBaseAgent,
    SpatialNavConfig,
)
from .mapgpt import (
    MapGPTAgent,
    MapGPTSpatialAgent,
    MapGPTConfig,
    MapGPTSpatialConfig,
)

DISCRETE_AGENT_CONFIG_REGISTRY = {
    "navgpt": NavGPTConfig,
    "navgpt_spatial": SpatialNavConfig,
    "mapgpt": MapGPTConfig,
    "mapgpt_spatial": MapGPTSpatialConfig,
    "spatialnav_base": SpatialNavConfig,
    "spatialnav": SpatialNavConfig,
}

__all__ = [
    # agents
    "NavGPTAgent",
    "NavGPTSpatialAgent",
    "SpatialNavAgent",
    "SpatialNavBaseAgent",
    "MapGPTAgent",
    "MapGPTSpatialAgent",
    # configs
    "NavGPTConfig",
    "SpatialNavConfig",
    "MapGPTConfig",
    "MapGPTSpatialConfig",
    # registry
    "DISCRETE_AGENT_CONFIG_REGISTRY"
]