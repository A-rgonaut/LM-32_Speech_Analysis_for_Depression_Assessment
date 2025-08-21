from transformers import AutoConfig
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
    if model_type == 'ssl' and final_config.get('use_preextracted_features', False):
        ssl_conf = AutoConfig.from_pretrained(final_config['ssl_model_name'])
        final_config['num_ssl_layers'] = ssl_conf.num_hidden_layers + 1

    ssl_model_name = final_config.get('ssl_model_name', '').replace('/', '-')
    if ssl_model_name:
        final_config['feature_path'] += ssl_model_name
    if model_type == 'ssl' and not final_config.get('use_all_layers', False):
        layer_to_use = final_config.get('layer_to_use', None)
        if layer_to_use is not None:
            final_config["model_save_dir"] += f"ssl/{ssl_model_name}/layer{layer_to_use}"
            final_config["result_dir"] += f"ssl/{ssl_model_name}/layer{layer_to_use}"
        else:
            final_config["model_save_dir"] += model_type
            final_config["result_dir"] += model_type
    else:
        final_config["model_save_dir"] += model_type
        final_config["result_dir"] += model_type

    return Config(final_config)