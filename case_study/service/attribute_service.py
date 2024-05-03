from case_study.bean import Dataset
from case_study.bean.bean_collection import Attribute
from case_study.bean.data_version import DataWithVersion


class AttributeService:
    def __init__(self):
        pass

    def get_attribute_list_from_dataset(self, toplevel_pathway_name: str, data_version: DataWithVersion) -> list[Attribute]:
        dataset: Dataset = data_version.dataset_dict[toplevel_pathway_name]
        attribute_list: list[Attribute] = dataset.get_attribute_list()
        return attribute_list

    def get_attribute_from_dataset_based_on_index(self, toplevel_pathway_name: str, index: int, data_version: DataWithVersion) -> Attribute:
        dataset: Dataset = data_version.dataset_dict[toplevel_pathway_name]
        attribute = dataset.select_attribute_based_on_index(index)
        return attribute


attribute_service = AttributeService()
