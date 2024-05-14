import copy
import os
from typing import Dict, List, Tuple
import pandas as pd
import sys
import torch

from utils.utils import read_file_via_lines, encode_node_features, encode_edges_features

sys.path.append("../src/")


class Database:
    """
    This is a dataloader for link prediction dataset
    Args:
            name (string): Name of the dataset e.g. Disease.
            task (string): Name of the task e.g. input link prediction dataset
    Return:
            self.train/test/valid (df): Dataframe of train/test/valid sets.

    """

    def __init__(self, name, task):
        self.dataset = name
        self.task = task
        self.load_dataset()

    def load_dataset(self):
        self.train = self.load_train_to_graph(self.dataset, self.task, "train")
        self.test = self.load_other_to_graph(self.dataset, self.task, "test")
        self.valid = self.load_other_to_graph(self.dataset, self.task, "validation")

    def load_train_to_graph(self, name, task, subset):
        data_path = os.path.join("../app/static/data", name)
        relation_path = os.path.join(data_path, task, subset, "relationship.txt")
        mapping_path = os.path.join(data_path, task, subset, "components-mapping.txt")
        mat = pd.read_csv(
            relation_path, names=["entity", "reaction", "type"], header=None
        )
        my_file = open(mapping_path, "r")
        mapping = my_file.read()
        mapping_list = mapping.split("\n")
        new_list = [i.split(",") for i in mapping_list]
        final_list = []
        for i in new_list:
            final_list.append([int(j) for j in i])
        # mapping = pd.read_csv(train_mapping_path)
        feature_dimension = max(sum(final_list, [])) + 1
        num_nodes = max(mat["entity"])
        # print(
        #     subset,
        #     "Num of interactions: %2d.\n Number of nodes: %2d.\n Number of features: %2d"
        #     % (len(mat), num_nodes, feature_dimension),
        # )
        return mat

    def load_other_to_graph(self, name, task, subset):
        data_path = os.path.join("../app/static/data", name)
        relation_path = os.path.join(data_path, task, subset, "relationship.txt")
        mat = pd.read_csv(
            relation_path, names=["entity", "reaction", "type"], header=None
        )
        # print("Load %s set" % subset)
        return mat

# E:\Python_Project\reactome_visual\utils\data_loader.py
# E:\Python_Project\reactome_visual\app\static\data\Disease\relationship.txt
class DataLoaderBase:
    def __init__(self, sub_dataset_name, task_name):
        self.sub_dataset_name = sub_dataset_name
        self.task_name = task_name

        # define path of file
        # self.__raw_data_file_path = os.path.join("data", sub_dataset_name)
        # todo
        # print(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
        # print(os.path.abspath(os.path.dirname(os.getcwd())))
        # print(os.path.abspath(os.path.join(os.getcwd(), "..")))
        self.raw_data_file_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'app', 'static', 'data', sub_dataset_name)
        self.task_file_path = os.path.join(self.raw_data_file_path, task_name)

    def get_num_of_nodes_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)
        node_line_message_list: list[str] = read_file_via_lines(path, "nodes.txt")
        num_of_nodes = len(node_line_message_list)
        return num_of_nodes

    def get_num_of_features_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)
        feature_line_message_list: list[str] = read_file_via_lines(
            path, "components-all.txt"
        )
        num_of_features = len(feature_line_message_list)
        return num_of_features

    def get_num_of_edges_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)
        edge_line_message_list: list[str] = read_file_via_lines(path, "edges.txt")
        num_of_edges = len(edge_line_message_list)
        return num_of_edges

    def get_nodes_features_assist(self, type_name: str):
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)

        num_of_nodes = self.get_num_of_nodes_based_on_type_name(type_name)
        num_of_edges = self.get_num_of_edges_based_on_type_name(type_name)
        num_of_feature_dimension = self.get_num_of_features_based_on_type_name()

        relationship_path = os.path.join(path, "relationship.txt")
        # mat = pd.read_csv(os.path.join(self.__project_root_path, relationship_path), names=['entity', 'reaction', 'type'], header=None)
        mat = pd.read_csv(
            relationship_path, names=["entity", "reaction", "type"], header=None
        )

        components_mapping_line_message_list: list[str] = read_file_via_lines(
            path, "components-mapping.txt"
        )
        components_mapping_list_with_str_style = [
            components_mapping_line_message.split(",")
            for components_mapping_line_message in components_mapping_line_message_list
        ]

        components_mapping_list = []

        for components_mapping_str in components_mapping_list_with_str_style:
            components_mapping_line_int_style = [
                int(component) for component in components_mapping_str
            ]
            components_mapping_list.append(components_mapping_line_int_style)

        num_of_pair_of_entity_and_component: int = 0
        for components_of_single_entity in components_mapping_list:
            num_of_pair_of_entity_and_component = (
                    num_of_pair_of_entity_and_component + len(components_of_single_entity)
            )

        nodes_features = encode_node_features(
            components_mapping_list, num_of_nodes, num_of_feature_dimension
        )

        # print(
        #     type_name + " dataset\n",
        #     "Number of interactions: %2d.\n Number of nodes: %2d.\n Number of features: %2d.\n Number of pair of node and feature: %2d.\n Number of edges: %2d."
        #     % (
        #         len(mat),
        #         num_of_nodes,
        #         num_of_feature_dimension,
        #         num_of_pair_of_entity_and_component,
        #         num_of_edges,
        #     ),
        # )

        return nodes_features

    def get_edge_of_nodes_list_regardless_direction(self, param) -> List[List[int]]:
        """
        Get the nodes of all the hyper edges
        :return: [[1,2,3], [3,7,9], [4,6,7,8,10,11]...] while [1,2,3], [3,7,9], .. represent the hyper edges
        """
        pass

    def get_edge_to_list_of_nodes_dict_based_on_relationship(self, type_name: str):
        """
        :param type_name: "raw" for raw dataset, "test" for test dataset, "train" for train dataset, "validation" for validation dataset
        :return:
        """
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)

        relationship_line_message_list: list[str] = read_file_via_lines(
            path, "relationship.txt"
        )

        (
            edge_to_list_of_nodes_dict,
            edge_to_list_of_input_nodes_dict,
            edge_to_list_of_output_nodes_dict,
        ) = self.get_edge_to_list_of_nodes_dict_assist(relationship_line_message_list)

        return (
            edge_to_list_of_nodes_dict,
            edge_to_list_of_input_nodes_dict,
            edge_to_list_of_output_nodes_dict,
        )

    def get_edge_to_list_of_masked_nodes_dict(self, type_name: str):
        path: str = os.path.join(self.task_file_path, type_name)
        relationship_line_message_list: list[str] = read_file_via_lines(
            path, "relationship-mask.txt"
        )
        (
            edge_to_list_of_nodes_dict,
            edge_to_list_of_input_nodes_dict,
            edge_to_list_of_output_nodes_dict,
        ) = self.get_edge_to_list_of_nodes_dict_assist(relationship_line_message_list)

        return (
            edge_to_list_of_nodes_dict,
            edge_to_list_of_input_nodes_dict,
            edge_to_list_of_output_nodes_dict,
        )

    def get_edge_to_list_of_nodes_dict_assist(
            self, relationship_line_message_list: List[str]
    ):
        edge_to_list_of_nodes_dict: dict[int, list[int]] = dict()
        edge_to_list_of_input_nodes_dict: dict[int, list[int]] = dict()
        edge_to_list_of_output_nodes_dict: dict[int, list[int]] = dict()

        for relationship_line_message in relationship_line_message_list:
            elements: list[str] = relationship_line_message.split(",")
            node_index: int = int(elements[0])
            edge_index: int = int(elements[1])
            direction: int = int(elements[2])

            if edge_index not in edge_to_list_of_nodes_dict.keys():
                edge_to_list_of_nodes_dict[edge_index] = list()
            edge_to_list_of_nodes_dict[edge_index].append(node_index)

            if direction < 0:
                if edge_index not in edge_to_list_of_input_nodes_dict.keys():
                    edge_to_list_of_input_nodes_dict[edge_index] = list()
                edge_to_list_of_input_nodes_dict[edge_index].append(node_index)

            elif direction > 0:
                if edge_index not in edge_to_list_of_output_nodes_dict.keys():
                    edge_to_list_of_output_nodes_dict[edge_index] = list()
                edge_to_list_of_output_nodes_dict[edge_index].append(node_index)

        return (
            edge_to_list_of_nodes_dict,
            edge_to_list_of_input_nodes_dict,
            edge_to_list_of_output_nodes_dict,
        )

    def get_labels(self):
        pass

    def get_nodes_mask_assist(self, type_name: str) -> List[int]:
        nodes_mask: list[int] = list()

        path: str = os.path.join(self.task_file_path, type_name)
        if "train" != type_name:
            file_name = "nodes.txt"
        else:
            file_name = "nodes-mask.txt"

        node_line_message_list: list[str] = read_file_via_lines(path, file_name)

        for node_line_message in node_line_message_list:
            elements = node_line_message.split(",")
            node_index = int(elements[0])
            nodes_mask.append(node_index)

        return nodes_mask

    def get_edges_mask_assist(self, type_name: str) -> List[int]:
        edges_mask: list[int] = list()

        path: str = os.path.join(self.task_file_path, type_name)
        edges_line_message_list: list[str] = read_file_via_lines(
            path, "edges.txt"
        )

        for node_line_message in edges_line_message_list:
            elements = node_line_message.split(",")
            edge_index = int(elements[0])
            edges_mask.append(edge_index)

        return edges_mask


class DataLoaderAttribute(DataLoaderBase):
    def __init__(self, sub_dataset_name, task_name):
        super().__init__(sub_dataset_name, task_name)

        # node mask
        (
            self.__train_nodes_mask,
            self.__validation_nodes_mask,
            self.__test_nodes_mask,
        ) = self.__get_nodes_mask()

        # node features
        self.__raw_nodes_features = super().get_nodes_features_assist("raw")
        self.__train_nodes_features = super().get_nodes_features_assist("train")

        # print the information
        super().get_nodes_features_assist("validation")
        super().get_nodes_features_assist("test")

        # get labels
        self.train_labels, self.validation_labels, self.test_labels = self.get_labels()

        self.__function_dict = {
            "num_nodes": self.get_num_of_nodes_based_on_type_name(),
            "num_features": self.get_num_of_features_based_on_type_name(),
            "num_edges": self.get_num_of_edges_based_on_type_name(),
            "edge_list": self.get_edge_of_nodes_list_regardless_direction("train"),
            "raw_nodes_features": self.__raw_nodes_features,
            # train, validation, and test dataset just use train dataset
            "train_nodes_features": self.__train_nodes_features,
            "validation_nodes_features": self.__train_nodes_features,
            "test_nodes_features": self.__train_nodes_features,
            "train_labels": self.train_labels,
            "validation_labels": self.validation_labels,
            "test_labels": self.test_labels,
            "train_node_mask": self.__train_nodes_mask,
            "val_node_mask": self.__validation_nodes_mask,
            "test_node_mask": self.__test_nodes_mask,
        }

    def __getitem__(self, key):
        return self.__function_dict[key]

    def get_labels(self):
        train_all_nodes_features = self.__train_nodes_features
        num_of_nodes_validation: int = self.get_num_of_nodes_based_on_type_name(
            "validation"
        )
        num_of_nodes_test: int = self.get_num_of_nodes_based_on_type_name("test")
        num_of_features: int = self.get_num_of_features_based_on_type_name("train")
        train_labels = torch.FloatTensor(train_all_nodes_features)[
            self.__train_nodes_mask
        ]

        validation_masked_nodes_features = (
            self.__get_nodes_to_list_of_masked_components_assist("validation")
        )

        validation_labels = torch.FloatTensor(
            encode_node_features(
                validation_masked_nodes_features,
                num_of_nodes_validation,
                num_of_features,
            )
        )

        test_masked_nodes_features = (
            self.__get_nodes_to_list_of_masked_components_assist("test")
        )

        test_labels = torch.FloatTensor(
            encode_node_features(
                test_masked_nodes_features, num_of_nodes_test, num_of_features
            )
        )

        return train_labels, validation_labels, test_labels

    def __get_nodes_to_list_of_masked_components_assist(self, type_name: str):
        path: str = os.path.join(self.task_file_path, type_name)
        file_name: str = "components-mapping-mask.txt"
        node_line_message_list: list[str] = read_file_via_lines(path, file_name)

        nodes_to_list_of_masked_components: list[list[int]] = list()
        for node_line_message in node_line_message_list:
            masked_components_list = [
                int(component_str)
                for component_str in node_line_message.split(":")[1].split(",")
            ]
            nodes_to_list_of_masked_components.append(masked_components_list)

        return nodes_to_list_of_masked_components

    def __get_complete_nodes_features_mix_negative_for_attribute_prediction(
            self, node_mask: List[int], type_name: str
    ):
        nodes_features_mix_negative: list[
            list[int]
        ] = self.__get_nodes_features_mix_negative_assist(type_name)

        path: str = self.raw_data_file_path
        components_mapping_line_message_list: list[str] = read_file_via_lines(
            path, "components-mapping.txt"
        )
        components_mapping_list_with_str_style = [
            components_mapping_line_message.split(",")
            for components_mapping_line_message in components_mapping_line_message_list
        ]

        raw_nodes_components_mapping_list = []

        for components_mapping_str in components_mapping_list_with_str_style:
            components_mapping_line_int_style = [
                int(component) for component in components_mapping_str
            ]
            raw_nodes_components_mapping_list.append(components_mapping_line_int_style)

        for i, node_mask_index in enumerate(node_mask):
            raw_nodes_components_mapping_list[
                node_mask_index
            ] = nodes_features_mix_negative[i]

        num_of_nodes = self.get_num_of_nodes_based_on_type_name()
        num_of_feature_dimension = self.get_num_of_features_based_on_type_name()

        nodes_features = encode_node_features(
            raw_nodes_components_mapping_list, num_of_nodes, num_of_feature_dimension
        )

        return nodes_features

    def __get_nodes_features_mix_negative_assist(
            self, type_name: str
    ) -> List[List[int]]:
        if "test" != type_name and "validation" != type_name:
            raise Exception('The type should be "test" or "validation"')
        if "attribute prediction dataset" != self.task_name:
            raise Exception(
                'The method "self.__get_nodes_features_mix_negative_assist" is only for attribute prediction task'
            )
        path: str = os.path.join(self.task_file_path, type_name)
        components_mapping_line_message_mix_negative_list: list[
            str
        ] = read_file_via_lines(path, "components-mapping-mix-negative.txt")

        nodes_features_mix_negative: list[list[int]] = list()

        for (
                components_mapping_line_message_mix_negative
        ) in components_mapping_line_message_mix_negative_list:
            elements: list[str] = components_mapping_line_message_mix_negative.split(
                "||"
            )
            positive_components_list_str_message: str = elements[0]
            negative_components_list_str_style: list[str] = elements[1:-1]

            components_list: list[int] = list()

            positive_components_list: list[int] = [
                int(positive_component_str_style)
                for positive_component_str_style in positive_components_list_str_message.split(
                    ","
                )
            ]
            negative_components_list: list[int] = [
                int(negative_components_str_style)
                for negative_components_str_style in negative_components_list_str_style
            ]

            components_list.extend(positive_components_list)

            components_list.extend(negative_components_list)

            nodes_features_mix_negative.append(copy.deepcopy(components_list))

        return nodes_features_mix_negative

    def __get_nodes_mask(self) -> Tuple[List[int], List[int], List[int]]:
        train_nodes_mask = super().get_nodes_mask_assist("train")
        validation_nodes_mask = super().get_nodes_mask_assist("validation")
        test_nodes_mask = super().get_nodes_mask_assist("test")

        return train_nodes_mask, validation_nodes_mask, test_nodes_mask

    def get_edge_of_nodes_list_regardless_direction(
            self, type_name: str
    ) -> List[List[int]]:
        """
        :return: [[1,2,3], [3,7,9], [4,6,7,8,10,11]...] while [1,2,3], [3,7,9], .. represent the hyper edges
        """
        (
            edge_to_list_of_nodes_dict,
            _,
            _,
        ) = self.get_edge_to_list_of_nodes_dict_based_on_relationship(type_name)

        edge_of_nodes_list_without_direction: list[list[int]] = [
            list_of_nodes
            for edge_index, list_of_nodes in edge_to_list_of_nodes_dict.items()
        ]

        return edge_of_nodes_list_without_direction


class DataLoaderLink(DataLoaderBase):
    def __init__(self, sub_dataset_name, task_name):
        super().__init__(sub_dataset_name, task_name)

        (
            self.__raw_edge_to_nodes_dict,
            self.__train_edge_to_nodes_dict,
            self.__validation_edge_to_nodes_dict,
            self.__test_edge_to_nodes_dict,
        ) = self.__get_edge_to_list_of_nodes_dict()
        (
            self.__train_edge_mask,
            self.__validation_edge_mask,
            self.__test_edge_mask,
        ) = self.get_edges_mask()

        (
            self.__raw_nodes_features,
            self.__train_nodes_features,
            self.__validation_nodes_features,
            self.__test_nodes_features,
        ) = (
            super().get_nodes_features_assist("raw"),
            super().get_nodes_features_assist("train"),
            super().get_nodes_features_assist("validation"),
            super().get_nodes_features_assist("test"),
        )

        (
            self.__list_of_edge_of_nodes_train,
            self.__list_of_edge_of_input_nodes_train,
            self.__list_of_edge_of_output_nodes_train,
        ) = self.__get_list_of_edges_of_nodes_based_on_train_dataset()
        (
            self.__train_labels,
            self.__test_labels,
            self.__validation_labels,
        ) = self.get_labels()

        self.__function_dict = {
            "num_nodes": self.get_num_of_nodes_based_on_type_name(),
            "num_features": self.get_num_of_features_based_on_type_name(),
            "num_edges": self.get_num_of_edges_based_on_type_name(),
            "raw_edge_list": self.get_edge_of_nodes_list_regardless_direction("raw"),
            "train_edge_list": self.get_edge_of_nodes_list_regardless_direction(
                "train"
            ),
            "train_masked_edge_list": self.get_masked_train_edge_of_nodes_list_regardless_direction(
                self.__train_edge_mask),
            "train_edge_list_with_input_nodes": self.__list_of_edge_of_input_nodes_train,
            "train_edge_list_with_output_nodes": self.__list_of_edge_of_output_nodes_train,
            "validation_edge_list": self.get_edge_of_nodes_list_regardless_direction(
                "validation"
            ),
            "test_edge_list": self.get_edge_of_nodes_list_regardless_direction("test"),
            "raw_nodes_features": self.__raw_nodes_features,
            "train_nodes_features": self.__train_nodes_features,
            "validation_nodes_features": self.__validation_nodes_features,
            "test_nodes_features": self.__test_nodes_features,
            "train_labels": self.__train_labels,
            "test_labels": self.__test_labels,
            "validation_labels": self.__validation_labels,
            "train_edge_mask": self.__train_edge_mask,
            "val_edge_mask": self.__validation_edge_mask,
            "test_edge_mask": self.__test_edge_mask,
        }

    def __getitem__(self, key):
        return self.__function_dict[key]

    def __get_edge_to_list_of_nodes_dict(
            self,
    ) -> Tuple[
        Dict[int, List[int]],
        Dict[int, List[int]],
        Dict[int, List[int]],
        Dict[int, List[int]],
    ]:
        (
            raw_edge_to_list_of_nodes_dict,
            _,
            _,
        ) = self.get_edge_to_list_of_nodes_dict_based_on_relationship("raw")
        (
            train_edge_to_list_of_nodes_dict,
            _,
            _,
        ) = super().get_edge_to_list_of_nodes_dict_based_on_relationship("train")
        (
            validation_edge_to_list_of_nodes_dict,
            _,
            _,
        ) = super().get_edge_to_list_of_nodes_dict_based_on_relationship("validation")
        (
            test_edge_to_list_of_nodes_dict,
            _,
            _,
        ) = super().get_edge_to_list_of_nodes_dict_based_on_relationship("test")

        return (
            raw_edge_to_list_of_nodes_dict,
            train_edge_to_list_of_nodes_dict,
            validation_edge_to_list_of_nodes_dict,
            test_edge_to_list_of_nodes_dict,
        )

    def get_edge_of_nodes_list_regardless_direction(self, type_name) -> List[List[int]]:
        """
        :return: [[1,2,3], [3,7,9], [4,6,7,8,10,11]...] while [1,2,3], [3,7,9], .. represent the hyper edges
        """

        type_dict = {
            "raw": self.__raw_edge_to_nodes_dict,
            "train": self.__train_edge_to_nodes_dict,
            "validation": self.__validation_edge_to_nodes_dict,
            "test": self.__test_edge_to_nodes_dict,
        }

        if type_name not in type_dict.keys():
            raise Exception('Please input "train", "validation" or "test" ')

        edge_of_nodes_list_without_direction: list[list[int]] = [
            list_of_nodes for edge_index, list_of_nodes in type_dict[type_name].items()
        ]

        return edge_of_nodes_list_without_direction

    def __get_list_of_edges_of_nodes_based_on_train_dataset(
            self,
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        num_of_edges = self.get_num_of_edges_based_on_type_name("train")
        (
            edge_to_list_of_nodes_dict,
            edge_to_list_of_input_nodes_dict,
            edge_to_list_of_output_nodes_dict,
        ) = super().get_edge_to_list_of_nodes_dict_based_on_relationship("train")

        # edges_all = edge_to_list_of_nodes_dict.keys()
        # edges_input = edge_to_list_of_input_nodes_dict.keys()
        #
        # print(set(edges_all) - set(edges_input))

        list_of_edge_of_nodes: list[list[int]] = list()
        list_of_edge_of_input_nodes: list[list[int]] = list()
        list_of_edge_of_output_nodes: list[list[int]] = list()
        for i in range(num_of_edges):
            if i in edge_to_list_of_nodes_dict.keys():
                list_of_edge_of_nodes.append(edge_to_list_of_nodes_dict.get(i))
            else:
                list_of_edge_of_nodes.append(list())
            if i in edge_to_list_of_input_nodes_dict.keys():
                list_of_edge_of_input_nodes.append(
                    edge_to_list_of_input_nodes_dict.get(i)
                )
            else:
                list_of_edge_of_input_nodes.append(list())
            if i in edge_to_list_of_output_nodes_dict.keys():
                list_of_edge_of_output_nodes.append(
                    edge_to_list_of_output_nodes_dict.get(i)
                )
            else:
                list_of_edge_of_output_nodes.append(list())

        return (
            list_of_edge_of_nodes,
            list_of_edge_of_input_nodes,
            list_of_edge_of_output_nodes,
        )

    def __get_list_of_edges_of_masked_nodes(self, type_name: str):
        type_dict = {
            "raw": self.__raw_edge_to_nodes_dict,
            "train": self.__train_edge_to_nodes_dict,
            "validation": self.__validation_edge_to_nodes_dict,
            "test": self.__test_edge_to_nodes_dict,
        }
        if type_name not in type_dict.keys():
            raise Exception('Please input "train", "validation" or "test" ')
        (
            edge_to_list_of_nodes_dict,
            _,
            _,
        ) = super().get_edge_to_list_of_masked_nodes_dict(type_name)
        list_of_edges_of_masked_nodes: list[list[int]] = list()
        for edge, list_of_nodes in edge_to_list_of_nodes_dict.items():
            list_of_edges_of_masked_nodes.append(list_of_nodes)

        return list_of_edges_of_masked_nodes

    def get_labels(self):
        if "input link prediction dataset" == self.task_name:
            list_of_edge_of_nodes_train = self.__list_of_edge_of_input_nodes_train

        elif "output link prediction dataset" == self.task_name:
            list_of_edge_of_nodes_train = self.__list_of_edge_of_output_nodes_train

        else:
            raise Exception(
                'The task name should be "input link prediction dataset" or "output link prediction dataset"'
            )

        list_of_edge_of_masked_nodes_test = self.__get_list_of_edges_of_masked_nodes(
            "test"
        )

        list_of_edge_of_masked_nodes_validation = (
            self.__get_list_of_edges_of_masked_nodes("validation")
        )

        num_of_nodes_of_raw_dataset = self.get_num_of_nodes_based_on_type_name("train")

        train_labels_for_link_prediction = torch.FloatTensor(
            encode_edges_features(
                list_of_edge_of_nodes_train,
                len(list_of_edge_of_nodes_train),
                num_of_nodes_of_raw_dataset,
            )
        )
        train_labels_for_link_prediction = train_labels_for_link_prediction[
            self.__train_edge_mask
        ]

        test_labels_for_link_prediction = torch.FloatTensor(
            encode_edges_features(
                list_of_edge_of_masked_nodes_test,
                len(list_of_edge_of_masked_nodes_test),
                num_of_nodes_of_raw_dataset,
            )
        )

        validation_labels_for_link_prediction = torch.FloatTensor(
            encode_edges_features(
                list_of_edge_of_masked_nodes_validation,
                len(list_of_edge_of_masked_nodes_validation),
                num_of_nodes_of_raw_dataset,
            )
        )

        return (
            train_labels_for_link_prediction,
            test_labels_for_link_prediction,
            validation_labels_for_link_prediction,
        )

    def get_edges_mask(self):
        validation_edges_mask = super().get_edges_mask_assist("validation")
        test_edges_mask = super().get_edges_mask_assist("test")

        train_edges_set = set()
        for validation_edges_idx in validation_edges_mask:
            train_edges_set.add(validation_edges_idx)
        for test_edges_idx in test_edges_mask:
            train_edges_set.add(test_edges_idx)

        train_edges_mask = list(train_edges_set)

        train_edges_mask.sort()

        return train_edges_mask, validation_edges_mask, test_edges_mask

    def get_masked_train_edge_of_nodes_list_regardless_direction(self, train_edge_mask: List[int]):
        train_edge_of_nodes_list_regardless_direction: list[
            list[int]] = self.get_edge_of_nodes_list_regardless_direction("train")

        train_edge_of_nodes_list_regardless_direction: list[list[int]] = [
            train_edge_of_nodes_list_regardless_direction[train_edge_mask_index] for train_edge_mask_index in
            train_edge_mask]

        return copy.deepcopy(train_edge_of_nodes_list_regardless_direction)


if __name__ == "__main__":
    # name = 'Disease'
    # task = 'attribute prediction dataset'
    # data_base = Database(name, task)
    # train, train_fea = data_base.train
    # print(train['entity'].tolist())
    data_loader = DataLoaderAttribute("Disease", "attribute prediction dataset")

    num = data_loader["num_nodes"]
    # validation_nodes_features

    validation_nodes_features = data_loader["validation_nodes_features"]

    print("validation_nodes_features: ", len(validation_nodes_features))

    # feature_test = [[1, 0, 1], [1, 1, 1], [1, 0, 0]]
    # feature_test = utils.get_normalized_features_in_tensor(feature_test)
    # print(feature_test)
