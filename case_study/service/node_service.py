from typing import Dict, List
from case_study.bean import Dataset
from case_study.bean.bean_collection import Node
from case_study.bean.data_version import DataWithVersion


class NodeService:
    def __init__(self):
        pass

    def get_node_list_from_dataset(self, toplevel_pathway_name: str, data_version: DataWithVersion) -> List[Node]:
        dataset: Dataset = data_version.dataset_dict[toplevel_pathway_name]
        node_list: list[Node] = dataset.get_node_list()
        return node_list

    def get_node_from_dataset_based_on_index(self, toplevel_pathway_name: str, index: int, data_version: DataWithVersion) -> Node:
        dataset: Dataset = data_version.dataset_dict[toplevel_pathway_name]
        node: Node = dataset.select_node_based_on_index(index)
        return node

    def get_degree2node_dict(self, toplevel_pathway_name: str, data_version: DataWithVersion) -> Dict[int, List[int]]:
        dataset: Dataset = data_version.dataset_dict[toplevel_pathway_name]
        degree2node_dict: dict[int, list[int]] = dataset.get_degree2nodes_dict()
        return degree2node_dict


node_service = NodeService()
