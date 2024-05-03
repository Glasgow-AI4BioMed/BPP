from case_study.bean import Dataset, ToplevelPathway, ToplevelPathwayFactory
from case_study.bean.dataset_factory import DatasetFactory


class DataWithVersion:
    def __init__(self, data_version_name: str):
        self.data_version_name: str = data_version_name
        self.dataset_dict: dict[str, Dataset] = dict()
        DatasetLoader(self).load_dataset()

'''
    def load_dataset(self):
        toplevel_pathway_list: list[ToplevelPathway] = ToplevelPathwayFactory().initialize_toplevel_pathways()
        for toplevel_pathway in toplevel_pathway_list:
            toplevel_pathway_name = toplevel_pathway.name
            dataset = DatasetFactory(self.data_version_name, toplevel_pathway_name).get_dataset()
            self.dataset_dict[toplevel_pathway_name] = dataset
'''


class DatasetLoader:
    def __init__(self, data_version: DataWithVersion):
        self.__data_version = data_version

    def load_dataset(self) -> DataWithVersion:
        toplevel_pathway_list: list[ToplevelPathway] = ToplevelPathwayFactory().initialize_toplevel_pathways()
        for toplevel_pathway in toplevel_pathway_list:
            toplevel_pathway_name = toplevel_pathway.name
            dataset = DatasetFactory(self.__data_version.data_version_name, toplevel_pathway_name).get_dataset()
            self.__data_version.dataset_dict[toplevel_pathway_name] = dataset

        return self.__data_version

