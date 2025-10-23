import yaml
from typing import List, Literal

from utils.utils import instantiate_from_config

def get_realestate10k(machine: Literal['cvg28', 'jureca', 'marvin'], 
                      split: Literal['train', 'validation'] = 'validation',
                      depth_dataset: bool = False,
                      **kwargs) -> List[str]:
    
    data_config_file = f"configs/data/realestate10k_{machine}.yaml"
    if depth_dataset == True:
        data_config_file = f"configs/data/realestate10k_depth_{machine}.yaml"
        
    with open(data_config_file, 'r') as f:
        config = yaml.safe_load(f)

    dataset_config = config['params'][split]
    dataset_config['params'].update(kwargs)
    dataset = instantiate_from_config(dataset_config)

    return dataset