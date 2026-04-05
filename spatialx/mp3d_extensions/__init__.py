from .discrete_task import (
    DiscreteDataConfig, 
    R2RDataset, 
    REVERIEDataset,
    RxRDataset,
    DiscreteNavBatch,
)


TASK_CONFIG_REGISTRY = {
    "r2r": DiscreteDataConfig,
    "reverie": DiscreteDataConfig,
    "rxr": DiscreteDataConfig,
}
