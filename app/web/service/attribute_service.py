from typing import List
from app.bean import dataset_dict, Dataset
from app.bean.bean_collection import Attribute


# todo
"""
try to use singleton pattern
"""
class AttributeService:
    def __init__(self):
        pass

    def get_attribute_list_from_dataset(self, toplevel_pathway_name: str) -> List[Attribute]:
        dataset: Dataset = dataset_dict[toplevel_pathway_name]
        attribute_list: list[Attribute] = dataset.get_attribute_list()
        return attribute_list

    def get_attribute_from_dataset_based_on_index(self, toplevel_pathway_name: str, index: int) -> Attribute:
        dataset: Dataset = dataset_dict[toplevel_pathway_name]
        attribute = dataset.select_attribute_based_on_index(index)
        return attribute


attribute_service_obj = AttributeService()
