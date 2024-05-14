from typing import List
from app.bean import Dataset, dataset_dict
from app.bean.bean_collection import Node


class NodeService:
    def __init__(self):
        pass

    def get_node_list_from_dataset(self, toplevel_pathway_name: str) -> List[Node]:
        dataset: Dataset = dataset_dict[toplevel_pathway_name]
        node_list: list[Node] = dataset.get_node_list()
        return node_list

    def get_node_from_dataset_based_on_index(self, toplevel_pathway_name: str, index: int) -> Node:
        dataset: Dataset = dataset_dict[toplevel_pathway_name]
        node: Node = dataset.select_node_based_on_index(index)
        return node


node_service_obj = NodeService()
