from case_study.bean import Dataset
from case_study.bean.bean_collection import Relationship
from case_study.bean.data_version import DataWithVersion


class DataSetComparator:
    def __init__(self, old_dataset: Dataset, new_dataset: Dataset):
        self.__old_dataset = old_dataset
        self.__new_dataset = new_dataset

    def get_relationships_newly_added(self) -> list[Relationship]:
        """
           This method compares the list of relationships in the new dataset
           with the list of relationships in the old dataset and returns a list of newly added relationships.

           :return: A list of newly added relationships
           :rtype: list[Relationship]
           """
        relationships_newly_added = list(
            set(self.__new_dataset.relationships_list).difference(set(self.__old_dataset.relationships_list)))
        return relationships_newly_added

    def format_relationship_index_as_old_version_data(self, relationship_new_version: Relationship):
        if relationship_new_version.node in self.__old_dataset.nodes_list and relationship_new_version.edge in self.__old_dataset.edges_list:
            node = set([node for node in self.__old_dataset.nodes_list if relationship_new_version.node == node]).pop()
            edge = set([edge for edge in self.__old_dataset.edges_list if relationship_new_version.edge == edge]).pop()
            relationship = Relationship(node_index=node.index, edge_index=edge.index, node=node, edge=edge,
                                        direction=relationship_new_version.direction)
            return relationship
        else:
            raise Exception("This relationship doesn't have the node or edge in old version data")

    def get_relationships_newly_added_with_node_and_edge_in_old_data(self):
        """
            Returns a list of newly added relationships with both their node and edge already present in the old dataset.

            :return: A list of newly added relationships with both their node and edge already present in the old dataset
            :rtype: list[Relationship]
            """
        relationships_newly_added: list[Relationship] = self.get_relationships_newly_added()
        relationships_newly_added_with_node_and_edge_in_old_data: list[Relationship] = list()
        for relationship in relationships_newly_added:
            if relationship.node in self.__old_dataset.nodes_list and relationship.edge in self.__old_dataset.edges_list:
                relationship = self.format_relationship_index_as_old_version_data(relationship)
                relationships_newly_added_with_node_and_edge_in_old_data.append(relationship)

        return relationships_newly_added_with_node_and_edge_in_old_data


class DataComparator:
    def __init__(self, old_data: DataWithVersion, new_data: DataWithVersion):
        self.__old_data = old_data
        self.__new_data = new_data

    def choose_dataset(self, dataset_name):
        old_dataset = self.__old_data.dataset_dict[dataset_name]
        new_dataset = self.__new_data.dataset_dict[dataset_name]
        return DataSetComparator(old_dataset, new_dataset)
