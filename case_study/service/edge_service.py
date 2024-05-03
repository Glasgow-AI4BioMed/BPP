from case_study.bean import Dataset
from case_study.bean.bean_collection import Edge
from case_study.bean.data_version import DataWithVersion


class EdgeService:
    def __init__(self):
        pass

    def get_edge_list_from_dataset(self, toplevel_pathway_name: str, data_version: DataWithVersion) -> list[Edge]:
        dataset: Dataset = data_version.dataset_dict[toplevel_pathway_name]
        edge_list: list[Edge] = dataset.get_edge_list()
        return edge_list

    def get_edge_from_dataset_based_on_index(self, toplevel_pathway_name: str, index: int,
                                             data_version: DataWithVersion) -> Edge:
        dataset: Dataset = data_version.dataset_dict[toplevel_pathway_name]
        edge: Edge = dataset.select_edge_based_on_index(index)
        return edge

    def get_edge_from_dataset_based_on_name(self, toplevel_pathway_name: str, content: str,
                                            data_version: DataWithVersion) -> list[Edge]:
        dataset: Dataset = data_version.dataset_dict[toplevel_pathway_name]
        search_edge_list: list[Edge] = list()
        for edge in dataset.edges_list:
            if edge.name in content or content in edge.name:
                search_edge_list.append(edge)

        return search_edge_list

    def get_edge_from_dataset_based_on_stId(self, toplevel_pathway_name: str, content: str,
                                            data_version: DataWithVersion) -> list[Edge]:
        dataset: Dataset = data_version.dataset_dict[toplevel_pathway_name]
        search_edge_list: list[Edge] = list()
        for edge in dataset.edges_list:
            if edge.stId in content or content in edge.stId:
                search_edge_list.append(edge)

        return search_edge_list

    def generate_new_edge(self, toplevel_pathway_name: str, stId: str, name: str):
        return Edge(index=-1, pathway_name=toplevel_pathway_name, stId=stId, name=name)


edge_service = EdgeService()
