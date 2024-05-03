import os
from copy import copy, deepcopy
from functools import wraps
from typing import cast

import numpy as np

from case_study.bean.bean_collection import ToplevelPathway, Attribute, Node, Edge, Relationship, PairOfNodeAndAttribute
from case_study.bean.dataset_collection import Dataset
from case_study.utils.constant_definition import ToplevelPathwayNameEnum, FileNameEnum, ReactionDirectionEnum


class ToplevelPathwayFactory:
    def initialize_toplevel_pathways(self) -> list[ToplevelPathway]:
        disease_description: str = "Donec id elit non mi porta gravida at eget metus. Fusce dapibus, tellus ac cursus commodo, tortormauris condimentum nibh, ut fermentum massa justo sit amet risus. Etiam porta sem malesuada magnamollis euismod. Donec sed odio dui. "
        immune_system_description: str = "Donec id elit non mi porta gravida at eget metus. Fusce dapibus, tellus ac cursus commodo, tortormauris condimentum nibh, ut fermentum massa justo sit amet risus. Etiam porta sem malesuada magnamollis euismod. Donec sed odio dui. "
        metabolism_description: str = "Donec id elit non mi porta gravida at eget metus. Fusce dapibus, tellus ac cursus commodo, tortormauris condimentum nibh, ut fermentum massa justo sit amet risus. Etiam porta sem malesuada magnamollis euismod. Donec sed odio dui. "
        signal_transduction_description: str = "Donec id elit non mi porta gravida at eget metus. Fusce dapibus, tellus ac cursus commodo, tortormauris condimentum nibh, ut fermentum massa justo sit amet risus. Etiam porta sem malesuada magnamollis euismod. Donec sed odio dui. "

        list_of_toplevel_pathways: list[ToplevelPathway] = list()

        list_of_toplevel_pathways.append(ToplevelPathway(0, ToplevelPathwayNameEnum.DISEASE.value, disease_description))
        list_of_toplevel_pathways.append(
            ToplevelPathway(1, ToplevelPathwayNameEnum.IMMUNE_SYSTEM.value, immune_system_description))
        list_of_toplevel_pathways.append(
            ToplevelPathway(2, ToplevelPathwayNameEnum.METABOLISM.value, metabolism_description))
        list_of_toplevel_pathways.append(
            ToplevelPathway(3, ToplevelPathwayNameEnum.SIGNAL_TRANSDUCTION.value, signal_transduction_description))

        return list_of_toplevel_pathways


class DatasetProcessor:

    @staticmethod
    def merge_dataset(*dataset_group):
        """

        :param dataset_group: Tuple of Dataset, ex. merge_dataset(dataset_A, dataset_B, dataset_C......)
        :return: a merged dataset
        """
        merged_dataset: Dataset = Dataset()

        merged_attribute_list: list[Attribute] = list()
        merged_node_list: list[Node] = list()
        merged_edge_list: list[Edge] = list()

        merged_pair_of_node_and_attribute_list: list[PairOfNodeAndAttribute] = list()
        merged_relationship_list: list[Relationship] = list()

        for dataset in dataset_group:
            for attribute in dataset.attributes_list:
                if attribute not in merged_attribute_list:
                    attribute_copy = DataFactoryUtils.copy_member_variables(attribute, Attribute,
                                                                            ["index", "pathway_name", "stId", "name",
                                                                             "is_masked"])
                    merged_attribute_list.append(attribute_copy)

            for node in dataset.nodes_list:
                if node not in merged_node_list:
                    node_copy = DataFactoryUtils.copy_member_variables(node, Node,
                                                                       ["index", "pathway_name", "stId", "name",
                                                                        "is_masked"])
                    merged_attribute_list.append(node_copy)

            for edge in dataset.edges_list:
                if edge not in merged_edge_list:
                    edge_copy = DataFactoryUtils.copy_member_variables(edge, Edge,
                                                                       ["index", "pathway_name", "stId", "name",
                                                                        "is_masked"])
                    merged_edge_list.append(edge_copy)

        merged_attribute_list.sort(key=lambda attribute: attribute.name)
        merged_node_list.sort(key=lambda node: node.name)
        merged_edge_list.sort(key=lambda edge: edge.name)

        for i, attribute in enumerate(merged_attribute_list):
            attribute.index = i
        for i, node in enumerate(merged_node_list):
            node.index = i
        for i, edge in enumerate(merged_edge_list):
            edge.index = i

        for dataset in dataset_group:
            for original_pair_of_node_and_attribute in dataset.pair_of_node_and_attribute_list:
                assert isinstance(original_pair_of_node_and_attribute, PairOfNodeAndAttribute)
                original_node = original_pair_of_node_and_attribute.node
                original_attribute = original_pair_of_node_and_attribute.attribute

                attribute = next((attribute for attribute in merged_attribute_list if original_attribute == attribute),
                                 None)
                node = next((node for node in merged_node_list if original_node == node),
                            None)
                pair_of_node_and_attribute = PairOfNodeAndAttribute(node.index, attribute.index, node, attribute)
                merged_pair_of_node_and_attribute_list.append(pair_of_node_and_attribute)

            for original_relationship in dataset.relationships_list:
                assert isinstance(original_relationship, Relationship)
                original_node = original_relationship.node
                original_edge = original_relationship.edge

                node = next((node for node in merged_node_list if original_node == node),
                            None)
                edge = next((edge for edge in merged_edge_list if original_edge == edge),
                            None)
                relationship = Relationship(node.index, edge.index, node, edge, original_relationship.direction)
                merged_relationship_list.append(relationship)

        for attribute in merged_attribute_list:
            merged_dataset.add_attribute(attribute)
        for node in merged_node_list:
            merged_dataset.add_node(node)
        for edge in merged_edge_list:
            merged_dataset.add_edge(edge)
        for pair_of_node_and_attribute in merged_pair_of_node_and_attribute_list:
            if pair_of_node_and_attribute not in merged_dataset.pair_of_node_and_attribute_list:
                merged_dataset.add_pair_of_node_and_attribute(pair_of_node_and_attribute)
        for relationship in merged_relationship_list:
            if relationship not in merged_dataset.relationships_list:
                merged_dataset.add_relationship(relationship)

        return merged_dataset

    @staticmethod
    def save_dataset_to_static_file(dataset_name: str, dataset: Dataset):
        print("to do")


class DatasetFactory:
    def __init__(self, dataset_version_name: str, toplevel_pathway_name: str):
        self.__toplevel_pathway_name = toplevel_pathway_name
        self.__file_name_args: dict[str, str] = {
            'attribute_stId_file_name': FileNameEnum.ATTRIBUTE_STID_FILE_NAME.value,
            'attribute_name_file_name': FileNameEnum.ATTRIBUTE_NAME_FILE_NAME.value,
            'node_stId_file_name': FileNameEnum.NODE_STID_FILE_NAME.value,
            'node_name_file_name': FileNameEnum.NODE_NAME_FILE_NAME.value,
            'edge_stId_file_name': FileNameEnum.EDGE_STID_FILE_NAME.value,
            'edge_name_file_name': FileNameEnum.EDGE_NAME_FILE_NAME.value,
            'relationship_file_name': FileNameEnum.RELATIONSHIP_FILE_NAME.value,
            'pair_of_node_and_attribute_file_name': FileNameEnum.PAIR_OF_NODE_AND_ATTRIBUTE_FILE_NAME.value}

        DataFactoryUtils.initialize_config(dataset_version_name)

        self.__dataset = self.__create_dataset(self.__toplevel_pathway_name, self.__file_name_args)

    def get_dataset(self):
        return self.__dataset

    def __create_dataset(self, toplevel_pathway_name: str, file_name_args: dict[str, str]) -> Dataset:
        dataset = Dataset()
        attributes_dict: dict[int, Attribute] = DataFactoryUtils.ReadFileUtils.read_attributes_dict_from_file(
            toplevel_pathway_name,
            file_name_args[
                'attribute_stId_file_name'],
            file_name_args[
                'attribute_name_file_name'])
        nodes_dict: dict[int, Node] = DataFactoryUtils.ReadFileUtils.read_nodes_dict_from_file(toplevel_pathway_name,
                                                                                               file_name_args[
                                                                                                   'node_stId_file_name'],
                                                                                               file_name_args[
                                                                                                   'node_name_file_name'])

        edges_dict: dict[int, Edge] = DataFactoryUtils.ReadFileUtils.read_edges_dict_from_file(toplevel_pathway_name,
                                                                                               file_name_args[
                                                                                                   'edge_stId_file_name'],
                                                                                               file_name_args[
                                                                                                   'edge_name_file_name'])

        # todo
        relationships_list: list[Relationship] = DataFactoryUtils.ReadFileUtils.read_relationships_list(
            toplevel_pathway_name,
            file_name_args[
                'relationship_file_name'],
            nodes_dict, edges_dict)

        pair_of_node_and_attribute_list: list[
            PairOfNodeAndAttribute] = DataFactoryUtils.ReadFileUtils.read_pair_of_node_and_component_list(
            toplevel_pathway_name,
            file_name_args[
                'pair_of_node_and_attribute_file_name'],
            nodes_dict, attributes_dict)

        DataFactoryUtils.fill_nodes_inner_attributes_list_(pair_of_node_and_attribute_list, nodes_dict, attributes_dict)

        DataFactoryUtils.fill_edges_inner_nodes_list_(relationships_list, edges_dict, nodes_dict)

        for index, attribute in attributes_dict.items():
            dataset.add_attribute(attribute)

        for index, node in nodes_dict.items():
            dataset.add_node(node)

        for index, edge in edges_dict.items():
            dataset.add_edge(edge)

        for relationship in relationships_list:
            dataset.add_relationship(relationship)

        for pair_of_node_and_attribute in pair_of_node_and_attribute_list:
            dataset.add_pair_of_node_and_attribute(pair_of_node_and_attribute)

        return dataset


class DataFactoryUtils:
    DATASET_NAME: str = ""

    DATASET_ROOTPATH: str = ""

    DIRS_BEFORE_DATA_DIR: str = "case_study/multi_version_datasets"

    # case_study/multi_version_datasets/data

    @staticmethod
    def initialize_config(dataset_version_name: str):
        # case_study\multi_version_datasets
        DataFactoryUtils.__dataset_version_name = dataset_version_name
        # print(DataFactoryUtils.get_project_root_path())

        # dirs = ["case_study", "multi_version_datasets"]
        dirs: list[str] = DataFactoryUtils.DIRS_BEFORE_DATA_DIR.strip('/').split('/')

        DataFactoryUtils.DATASET_ROOTPATH = os.path.join(DataFactoryUtils.get_project_root_path(), *dirs,
                                                         DataFactoryUtils.__dataset_version_name)

    @staticmethod
    def get_project_root_path():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    @staticmethod
    def __check_initialize(func):
        @wraps(func)
        def check(*args, **kwargs):
            print("Decorator: Begin to check the initialization parameters")

            if DataFactoryUtils.DATASET_NAME == "" or DataFactoryUtils.DATASET_ROOTPATH == "":
                raise Exception("Please initialize dataset name and dataset root path")

            # the original code
            res = func(*args, **kwargs)

            return res

        return check

    @staticmethod
    def generate_path(pathway_name: str, file_name: str) -> str:
        raw_data_path: str = os.path.join(DataFactoryUtils.DATASET_ROOTPATH, pathway_name, file_name)
        # raw_data_path: str = os.path.join('static', 'data', pathway_name, file_name)
        return raw_data_path

    @staticmethod
    def convert_relationship_line_message_to_relationship(relationship_line_message: str, nodes_dict,
                                                          edges_dict) -> Relationship:
        elements = relationship_line_message.split(",")
        node_index: int = int(elements[0])
        edge_index: int = int(elements[1])

        node = nodes_dict[node_index]
        edge = edges_dict[edge_index]

        direction: int = int(elements[2])

        return Relationship(node_index=node_index, edge_index=edge_index, node=node, edge=edge, direction=direction)

    @staticmethod
    def convert_node_and_attributes_mapping_line_message_to_list_of_attributes_indexes(
            node_and_attributes_mapping_line_message: str) -> list[int]:
        list_of_attributes_string_style = node_and_attributes_mapping_line_message.split(",")
        list_of_attributes_index = [int(attributes_string_style) for attributes_string_style in
                                    list_of_attributes_string_style]
        return list_of_attributes_index

    @staticmethod
    def fill_nodes_inner_attributes_list_(pair_of_node_and_component_list: list[PairOfNodeAndAttribute],
                                          nodes_dict: dict[int, Node], attributes_dict: dict[int, Attribute]):

        for pair_of_node_and_component in pair_of_node_and_component_list:
            try:
                node = nodes_dict[pair_of_node_and_component.node_index]
                attribute = attributes_dict[pair_of_node_and_component.attribute_index]
                node.add_attribute_to_inner_list(attribute)
            except:
                # todo
                print(
                    'Error: The association[node index={} and attribute index={}] is not in the nodes & attributes dict'.format(
                        pair_of_node_and_component.node_index, pair_of_node_and_component.attribute_index))

    @staticmethod
    def fill_edges_inner_nodes_list_(relationships_list: list[Relationship], edges_dict: dict[int, Edge],
                                     nodes_dict: dict[int, Node]):
        for relationship in relationships_list:
            try:
                edge = edges_dict[relationship.edge_index]
                node = nodes_dict[relationship.node_index]

                direction = relationship.direction

                if ReactionDirectionEnum.INPUT_FLAG.value == direction:
                    edge.add_node_to_inner_input_list(node)
                elif ReactionDirectionEnum.OUTPUT_FLAG.value == direction:
                    edge.add_node_to_inner_output_list(node)
                elif ReactionDirectionEnum.REGULATOR_FLAG.value == direction:
                    edge.add_node_to_inner_regulator_list(node)
            except:
                # todo
                print(
                    'Error: The relationship[edge index={} and node index={}] is not in the edges & nodes dict'.format(
                        relationship.edge_index, relationship.node_index))

    @staticmethod
    def copy_member_variables(source, target_class, variables_for_copy):
        kwargs = {attr: getattr(source, attr, None) for attr in variables_for_copy}
        return target_class(**kwargs)

    class ReadFileUtils:
        @staticmethod
        def read_file_via_lines(path: str, file_name: str = '') -> list[str]:
            # root_path: str = get_root_path_of_project("PathwayGNN")
            if '' == file_name or None is file_name:
                url: str = path
            else:
                url: str = os.path.join(path, file_name)
            # url: str = os.path.join(path, file_name)
            res_list: list[str] = []

            try:
                file_handler = open(url, "r", encoding='utf-8')
                while True:
                    # Get next line from file
                    line = file_handler.readline()
                    line = line.replace("\r", "").replace("\n", "").replace("\t", "")

                    # If the line is empty then the end of file reached
                    if not line:
                        break
                    res_list.append(line)
            except Exception as e:
                print(e)
                print("we can't find the " + url + ", please make sure that the file exists")
            finally:
                return res_list

        @staticmethod
        def read_attributes_dict_from_file(pathway_name: str, attribute_stId_file_name: str,
                                           attribute_name_file_name: str) -> dict[int, Attribute]:
            attribute_stId_file_path = DataFactoryUtils.generate_path(pathway_name, attribute_stId_file_name)
            attribute_name_file_path = DataFactoryUtils.generate_path(pathway_name, attribute_name_file_name)

            attribute_stId_list = DataFactoryUtils.ReadFileUtils.read_file_via_lines(attribute_stId_file_path)
            attribute_name_list = DataFactoryUtils.ReadFileUtils.read_file_via_lines(attribute_name_file_path)

            if len(attribute_stId_list) != len(attribute_name_list):
                raise Exception("The length of files can't match, please check your data in files")

            attributes_dict: dict[int, Attribute] = dict()

            for index in range(len(attribute_stId_list)):
                attribute = Attribute(index=index, pathway_name=pathway_name, stId=attribute_stId_list[index],
                                      name=attribute_name_list[index])
                attributes_dict[index] = attribute

            return attributes_dict

        @staticmethod
        def read_nodes_dict_from_file(pathway_name: str, node_stId_file_name: str, node_name_file_name: str):
            node_stId_file_path = DataFactoryUtils.generate_path(pathway_name, node_stId_file_name)
            node_name_file_path = DataFactoryUtils.generate_path(pathway_name, node_name_file_name)

            node_stId_list = DataFactoryUtils.ReadFileUtils.read_file_via_lines(node_stId_file_path)
            node_name_list = DataFactoryUtils.ReadFileUtils.read_file_via_lines(node_name_file_path)

            if len(node_stId_list) != len(node_name_list):
                raise Exception("The length of files can't match, please check your data in files")

            nodes_dict: dict[int, Node] = dict()

            for index in range(len(node_stId_list)):
                node = Node(index=index, pathway_name=pathway_name, stId=node_stId_list[index],
                            name=node_name_list[index])
                nodes_dict[index] = node

            return nodes_dict

        @staticmethod
        def read_edges_dict_from_file(pathway_name: str, edge_stId_file_name: str, edge_name_file_name: str):
            edge_stId_file_path = DataFactoryUtils.generate_path(pathway_name, edge_stId_file_name)
            edge_name_file_path = DataFactoryUtils.generate_path(pathway_name, edge_name_file_name)

            edge_stId_list = DataFactoryUtils.ReadFileUtils.read_file_via_lines(edge_stId_file_path)
            edge_name_list = DataFactoryUtils.ReadFileUtils.read_file_via_lines(edge_name_file_path)

            if len(edge_stId_list) != len(edge_name_list):
                raise Exception("The length of files can't match, please check your data in files")

            edges_dict: dict[int, Edge] = dict()

            for index in range(len(edge_stId_list)):
                edge = Edge(index=index, pathway_name=pathway_name, stId=edge_stId_list[index],
                            name=edge_name_list[index])
                edges_dict[index] = edge

            return edges_dict

        @staticmethod
        def read_relationships_list(pathway_name: str, relationship_file_name: str, nodes_dict, edges_dict) -> list[
            Relationship]:
            relationship_file_path = DataFactoryUtils.generate_path(pathway_name, relationship_file_name)
            relationship_line_message_list = DataFactoryUtils.ReadFileUtils.read_file_via_lines(relationship_file_path)

            relationships_list: list[Relationship] = list()

            for relationship_line_message in relationship_line_message_list:
                relationship = DataFactoryUtils.convert_relationship_line_message_to_relationship(
                    relationship_line_message, nodes_dict, edges_dict)
                relationships_list.append(relationship)

            return relationships_list

        @staticmethod
        def read_pair_of_node_and_component_list(pathway_name: str, pair_of_node_and_attribute_file_name: str,
                                                 nodes_dict,
                                                 attributes_dict) -> list[
            PairOfNodeAndAttribute]:

            pair_of_node_and_attribute_list: list[PairOfNodeAndAttribute] = list()

            pair_of_node_and_attribute_file_path = DataFactoryUtils.generate_path(pathway_name,
                                                                                  pair_of_node_and_attribute_file_name)

            node_and_attributes_mapping_line_message_list = DataFactoryUtils.ReadFileUtils.read_file_via_lines(
                pair_of_node_and_attribute_file_path)

            for node_index, node_and_attributes_mapping_line_message in enumerate(
                    node_and_attributes_mapping_line_message_list):
                list_of_attribute_indexes = DataFactoryUtils.convert_node_and_attributes_mapping_line_message_to_list_of_attributes_indexes(
                    node_and_attributes_mapping_line_message)

                node = nodes_dict[node_index]

                for attribute_index in list_of_attribute_indexes:
                    attribute = attributes_dict[attribute_index]
                    pair_of_node_and_attribute = PairOfNodeAndAttribute(node_index=node_index,
                                                                        attribute_index=attribute_index, node=node,
                                                                        attribute=attribute)
                    pair_of_node_and_attribute_list.append(pair_of_node_and_attribute)

            return pair_of_node_and_attribute_list

    class WriteToFileUtils:
        @staticmethod
        def createFile(path, file_name):
            if '' == file_name or None is file_name:
                url: str = path
            else:
                url: str = os.path.join(path, file_name)
            if not os.path.exists(os.path.join(path)):
                os.makedirs(os.path.join(path))
            if os.path.exists(url):
                print("file exists, we'll delete the original file \"" + file_name + "\", then create a new one")
                os.remove(url)
            file = open(url, 'w', encoding='utf-8')

        @staticmethod
        def delete_file(path, file_name) -> None:
            url = os.path.join(path, file_name)
            if os.path.exists(url):
                os.remove(url)

        @staticmethod
        def write_message_to_file(message: list[str], path: str, file_name: str = ''):
            if '' == file_name or None is file_name:
                url: str = path
            else:
                url: str = os.path.join(path, file_name)
            if not os.path.exists(url):
                print("error! the file \"" + file_name + "\" doesn't exist!")

            message = np.array(message)
            # np.savetxt(url, message, delimiter=',', fmt='%s', encoding='utf-8')

            file = open(url, "w", encoding="UTF-8")
            for index, line in enumerate(message):
                if index == (len(message) - 1):
                    file.write(line)
                else:
                    file.write(line + "\n")
            file.close()

        @staticmethod
        def write_to_attribute_file(dataset: Dataset, pathway_name: str, attribute_stId_file_name: str,
                                    attribute_name_file_name: str):

            attribute_stId_file_path = DataFactoryUtils.generate_path(pathway_name, attribute_stId_file_name)
            attribute_name_file_path = DataFactoryUtils.generate_path(pathway_name, attribute_name_file_name)

            DataFactoryUtils.WriteToFileUtils.createFile(attribute_stId_file_path, attribute_stId_file_name)
            DataFactoryUtils.WriteToFileUtils.createFile(attribute_name_file_path, attribute_name_file_name)

            attribute_list: list[Attribute] = dataset.attributes_list
            attribute_stId_list: list[str] = list()
            attribute_name_list: list[str] = list()

            for attribute in attribute_list:
                attribute_stId_list.append(attribute.stId)
                attribute_name_list.append(attribute.name)

            DataFactoryUtils.WriteToFileUtils.write_message_to_file(message=attribute_stId_list,
                                                                    path=attribute_stId_file_path,
                                                                    file_name=attribute_stId_file_name)
            DataFactoryUtils.WriteToFileUtils.write_message_to_file(message=attribute_stId_list,
                                                                    path=attribute_stId_file_path,
                                                                    file_name=attribute_name_file_name)





class FileProcessor:
    def __init__(self):
        self.filename_reactions = "edges.txt"
        self.filename_reactions_names = "edges-names.txt"
        self.filename_physical_entities = "nodes.txt"
        self.filename_physical_entities_names = "nodes-names.txt"
        self.filename_relationships = "relationship.txt"
        self.filename_components_mapping = "components-mapping.txt"
        self.filename_components_all = "components-all.txt"
        self.filename_components_all_names = "components-all-names.txt"
        # PathwayGNN
        # self.root_path = cur_path[:cur_path.find("PathwayGNN\\") + len("PathwayGNN\\")]

    # data/All_data_in_Reactome/components-all.txt
    # create the txt file to store the data
    def createFile(self, path, file_name):
        url = os.path.join("..", "..", path, file_name)
        if not os.path.exists(os.path.join("..", "..", path)):
            os.makedirs(os.path.join("..", "..", path))
        if os.path.exists(url):
            print("file exists, we'll delete the original file \"" + file_name + "\", then create a new one")
            os.remove(url)
        file = open(url, 'w', encoding='utf-8')

    def delete_file(self, path, file_name) -> None:
        url = os.path.join(path, file_name)
        if os.path.exists(url):
            os.remove(url)

    # write message to txt file
    def writeMessageToFile(self, path, file_name, message: list[str]):
        url = os.path.join("..", "..", path, file_name)
        if not os.path.exists(url):
            print("error! the file \"" + file_name + "\" doesn't exist!")

        message = np.array(message)
        # np.savetxt(url, message, delimiter=',', fmt='%s', encoding='utf-8')

        file = open(url, "w", encoding="UTF-8")
        for index, line in enumerate(message):
            if index == (len(message) - 1):
                file.write(line)
            else:
                file.write(line + "\n")
        file.close()

    def create_and_write_message_to_file(self, path, file_name, message: list):
        self.createFile(path, file_name)
        self.writeMessageToFile(path, file_name, message)

    def execute_for_single_pathway(self, pathway_name, reaction_ids, physical_entity_ids,
                                   relationships_between_nodes_edges, component_ids, entity_component_mapping_list):

        path = os.path.join("data", pathway_name)

        # write message to the file
        file_professor = FileProcessor()

        file_professor.createFile(path, self.filename_reactions)
        file_professor.createFile(path, self.filename_physical_entities)
        file_professor.createFile(path, self.filename_relationships)
        file_professor.createFile(path, self.filename_components_all)
        file_professor.createFile(path, self.filename_components_mapping)

        file_professor.writeMessageToFile(path, self.filename_reactions, reaction_ids)
        file_professor.writeMessageToFile(path, self.filename_physical_entities, physical_entity_ids)
        file_professor.writeMessageToFile(path, self.filename_relationships, relationships_between_nodes_edges)
        file_professor.writeMessageToFile(path, self.filename_components_all, component_ids)
        file_professor.writeMessageToFile(path, self.filename_components_mapping, entity_component_mapping_list)

    def execute_for_single_pathway_with_name_files(self, pathway_name, reaction_ids, reaction_names,
                                                   physical_entity_ids,
                                                   physical_entity_names,
                                                   relationships_between_nodes_edges, component_ids, component_names,
                                                   entity_component_mapping_list):

        path = os.path.join("data", pathway_name)

        # write message to the file
        file_professor = FileProcessor()

        file_professor.createFile(path, self.filename_reactions)
        file_professor.createFile(path, self.filename_reactions_names)
        file_professor.createFile(path, self.filename_physical_entities)
        file_professor.createFile(path, self.filename_physical_entities_names)
        file_professor.createFile(path, self.filename_relationships)
        file_professor.createFile(path, self.filename_components_all)
        file_professor.createFile(path, self.filename_components_all_names)
        file_professor.createFile(path, self.filename_components_mapping)

        file_professor.writeMessageToFile(path, self.filename_reactions, reaction_ids)
        file_professor.writeMessageToFile(path, self.filename_reactions_names, reaction_names)
        file_professor.writeMessageToFile(path, self.filename_physical_entities, physical_entity_ids)
        file_professor.writeMessageToFile(path, self.filename_physical_entities_names, physical_entity_names)

        file_professor.writeMessageToFile(path, self.filename_relationships, relationships_between_nodes_edges)
        file_professor.writeMessageToFile(path, self.filename_components_all, component_ids)
        file_professor.writeMessageToFile(path, self.filename_components_all_names, component_names)

        file_professor.writeMessageToFile(path, self.filename_components_mapping, entity_component_mapping_list)

    def read_file_via_lines(self, path, file_name):
        url = os.path.join("..", "..", path, file_name)
        res_list = []

        try:
            file_handler = open(url, "r")
            while True:
                # Get next line from file
                line = file_handler.readline()
                line = line.replace('\r', '').replace('\n', '').replace('\t', '')

                # If the line is empty then the end of file reached
                if not line:
                    break
                res_list.append(line)
        except Exception as e:
            print(e)
            print("we can't find the " + url + ", please make sure that the file exists")
        finally:
            return res_list
