from .blender import BlenderDataset
from .llff import LLFFDataset
from .dataset import KlevrDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'klevr': KlevrDataset}