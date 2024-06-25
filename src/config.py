import yaml
import os


class Config:
    def __init__(self):
        self.WORKDIR = os.path.dirname(os.path.abspath(__file__))
        self.load_config()
    
    def load_config(self):
        self.HUGGINGFACE_ACCESS_TOKEN = None
        with open(f'{self.WORKDIR}/config.yaml') as f:
            for key, value in yaml.load(f, Loader=yaml.FullLoader).items():
                setattr(self, key, value)


CONFIG = Config()