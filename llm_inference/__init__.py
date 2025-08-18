import os

__ROOT__ = os.path.dirname(__file__)
os.environ['TRANSFORMERS_CACHE'] = os.path.join(__ROOT__, '.cache')
