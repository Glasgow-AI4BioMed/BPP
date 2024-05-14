from app.bean.bean_collection import ToplevelPathway
from app.bean.dataset_collection import Dataset
from app.bean.dataset_factory import DatasetFactory, ToplevelPathwayFactory


class DatasetLoader:
    def __init__(self):
        self.__dataset_dict: dict[str, Dataset] = dict()

    def load_dataset(self):
        toplevel_pathway_list: list[ToplevelPathway] = ToplevelPathwayFactory().initialize_toplevel_pathways()
        for toplevel_pathway in toplevel_pathway_list:
            toplevel_pathway_name = toplevel_pathway.name
            dataset = DatasetFactory(toplevel_pathway_name).get_dataset()
            self.__dataset_dict[toplevel_pathway_name] = dataset

        return self.__dataset_dict

