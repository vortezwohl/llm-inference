import os

__ROOT__ = os.path.dirname(__file__)
os.environ['HF_HOME'] = os.path.join(__ROOT__, '.cache')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
