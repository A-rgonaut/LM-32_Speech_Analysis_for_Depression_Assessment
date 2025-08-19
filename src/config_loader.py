import yaml
from pathlib import Path

class Config:
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

def load_config(config_path: str = './config.yaml') -> Config:
    path = Path(config_path)

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    common_config = config_dict['common']
    model_type = common_config['active_model']
    model_config = config_dict[model_type]
    
    final_config = common_config.copy()
    final_config.update(model_config)
    final_config["model_save_dir"] += model_type
    final_config["result_dir"] += model_type

    return Config(final_config)