from typing import Dict, List
from app.bean.bean_collection import ToplevelPathway
from app.bean.dataset_collection import Dataset
from app.bean.dataset_factory import ToplevelPathwayFactory
from app.bean.dataset_loader import DatasetLoader
from app.bean.model_path import ModelSelector

dataset_dict: Dict[str, Dataset] = DatasetLoader().load_dataset()

all_toplevel_pathways: List[ToplevelPathway] = ToplevelPathwayFactory().initialize_toplevel_pathways()

model_selector: ModelSelector = ModelSelector()
