import os
from typing import Dict, List

from app.bean.bean_collection import ToplevelPathway, Attribute, Node, Edge, Relationship, PairOfNodeAndAttribute
from app.bean.dataset_collection import Dataset
from utils.constant_definition import ReactionDirectionEnum, FileNameEnum, ToplevelPathwayNameEnum


class ToplevelPathwayFactory:
    def initialize_toplevel_pathways(self) -> List[ToplevelPathway]:
        disease_description: str = "The DISEASE dataset is a comprehensive collection of pathway data related to a disease, detailing the biological processes, and molecular interactions involved in its progression. Specifically, the dataset contains 1,853 biochemical entities and 1,080 pathway biochemical reactions. You can utilize the BPP platform to predict missing links within the dataset, which can significantly enhance disease modeling and support the discovery of novel therapeutic targets."
        immune_system_description: str = "The Immune System dataset is an extensive resource of pathway data focused on the immune system, elucidating the biological processes and molecular interactions that govern immune responses. This dataset specifically contains 2,097 biochemical entities and 2,265 pathway biochemical reactions. Researchers can leverage the BPP platform to predict missing links within the dataset, enhancing their understanding of immune mechanisms."
        metabolism_description: str = "The Metabolism dataset is a detailed collection of pathway data concerning metabolic processes, highlighting the biological activities and molecular interactions that underpin metabolism. This dataset includes 2,643 biochemical entities and 1,623 pathway biochemical reactions. Using the BPP platform, you can predict missing links within the dataset, providing deeper insights into metabolic pathways and their regulation."
        signal_transduction_description: str = "The Signal Transduction dataset is a comprehensive resource of pathway data centered on signal transduction mechanisms, describing the biological processes and molecular interactions involved in cellular signaling. This dataset features 3,894 biochemical entities and 2,507 pathway biochemical reactions. By utilizing the BPP platform, you can predict missing links within the dataset, advancing the understanding of cellular communication and signaling networks."

        list_of_toplevel_pathways: list[ToplevelPathway] = list()

        list_of_toplevel_pathways.append(ToplevelPathway(0, ToplevelPathwayNameEnum.DISEASE.value, disease_description))
        list_of_toplevel_pathways.append(
            ToplevelPathway(1, ToplevelPathwayNameEnum.IMMUNE_SYSTEM.value, immune_system_description))
        list_of_toplevel_pathways.append(
            ToplevelPathway(2, ToplevelPathwayNameEnum.METABOLISM.value, metabolism_description))
        list_of_toplevel_pathways.append(
            ToplevelPathway(3, ToplevelPathwayNameEnum.SIGNAL_TRANSDUCTION.value, signal_transduction_description))

        return list_of_toplevel_pathways


class DatasetFactory:
    def __init__(self, toplevel_pathway_name: str):
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

        self.__dataset = self.__create_dataset(self.__toplevel_pathway_name, self.__file_name_args)

    def get_dataset(self):
        return self.__dataset

    def __create_dataset(self, toplevel_pathway_name: str, file_name_args: Dict[str, str]) -> Dataset:
        dataset = Dataset()
        attributes_dict: dict[int, Attribute] = DataFactoryUtils.read_attributes_dict_from_file(toplevel_pathway_name,
                                                                                                file_name_args[
                                                                                                    'attribute_stId_file_name'],
                                                                                                file_name_args[
                                                                                                    'attribute_name_file_name'])
        nodes_dict: dict[int, Node] = DataFactoryUtils.read_nodes_dict_from_file(toplevel_pathway_name,
                                                                                 file_name_args['node_stId_file_name'],
                                                                                 file_name_args['node_name_file_name'])

        edges_dict: dict[int, Edge] = DataFactoryUtils.read_edges_dict_from_file(toplevel_pathway_name,
                                                                                 file_name_args['edge_stId_file_name'],
                                                                                 file_name_args['edge_name_file_name'])

        # todo
        relationships_list: list[Relationship] = DataFactoryUtils.read_relationships_list(toplevel_pathway_name,
                                                                                          file_name_args[
                                                                                              'relationship_file_name'])

        pair_of_node_and_attribute_list: list[
            PairOfNodeAndAttribute] = DataFactoryUtils.read_pair_of_node_and_component_list(toplevel_pathway_name,
                                                                                            file_name_args[
                                                                                                'pair_of_node_and_attribute_file_name'])

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
    @staticmethod
    def read_file_via_lines(path: str, file_name: str = '') -> List[str]:
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
    def generate_path(pathway_name: str, file_name: str) -> str:
        raw_data_path: str = os.path.join('app', 'static', 'data', pathway_name, file_name)
        # raw_data_path: str = os.path.join('static', 'data', pathway_name, file_name)
        return raw_data_path

    @staticmethod
    def read_attributes_dict_from_file(pathway_name: str, attribute_stId_file_name: str,
                                       attribute_name_file_name: str) -> Dict[int, Attribute]:
        attribute_stId_file_path = DataFactoryUtils.generate_path(pathway_name, attribute_stId_file_name)
        attribute_name_file_path = DataFactoryUtils.generate_path(pathway_name, attribute_name_file_name)

        attribute_stId_list = DataFactoryUtils.read_file_via_lines(attribute_stId_file_path)
        attribute_name_list = DataFactoryUtils.read_file_via_lines(attribute_name_file_path)

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

        node_stId_list = DataFactoryUtils.read_file_via_lines(node_stId_file_path)
        node_name_list = DataFactoryUtils.read_file_via_lines(node_name_file_path)

        if len(node_stId_list) != len(node_name_list):
            raise Exception("The length of files can't match, please check your data in files")

        nodes_dict: dict[int, Node] = dict()

        for index in range(len(node_stId_list)):
            node = Node(index=index, pathway_name=pathway_name, stId=node_stId_list[index], name=node_name_list[index])
            nodes_dict[index] = node

        return nodes_dict

    @staticmethod
    def read_edges_dict_from_file(pathway_name: str, edge_stId_file_name: str, edge_name_file_name: str):
        edge_stId_file_path = DataFactoryUtils.generate_path(pathway_name, edge_stId_file_name)
        edge_name_file_path = DataFactoryUtils.generate_path(pathway_name, edge_name_file_name)

        edge_stId_list = DataFactoryUtils.read_file_via_lines(edge_stId_file_path)
        edge_name_list = DataFactoryUtils.read_file_via_lines(edge_name_file_path)

        if len(edge_stId_list) != len(edge_name_list):
            raise Exception("The length of files can't match, please check your data in files")

        edges_dict: dict[int, Edge] = dict()

        for index in range(len(edge_stId_list)):
            edge = Edge(index=index, pathway_name=pathway_name, stId=edge_stId_list[index], name=edge_name_list[index])
            edges_dict[index] = edge

        return edges_dict

    @staticmethod
    def read_relationships_list(pathway_name: str, relationship_file_name: str) -> List[Relationship]:
        relationship_file_path = DataFactoryUtils.generate_path(pathway_name, relationship_file_name)
        relationship_line_message_list = DataFactoryUtils.read_file_via_lines(relationship_file_path)

        relationships_list: list[Relationship] = list()

        for relationship_line_message in relationship_line_message_list:
            relationship = DataFactoryUtils.convert_relationship_line_message_to_relationship(
                relationship_line_message)
            relationships_list.append(relationship)

        return relationships_list

    @staticmethod
    def convert_relationship_line_message_to_relationship(relationship_line_message: str) -> Relationship:
        elements = relationship_line_message.split(",")
        node_index: int = int(elements[0])
        edge_index: int = int(elements[1])
        direction: int = int(elements[2])

        return Relationship(node_index=node_index, edge_index=edge_index, direction=direction)

    @staticmethod
    def read_pair_of_node_and_component_list(pathway_name: str, pair_of_node_and_attribute_file_name: str) -> List[
        PairOfNodeAndAttribute]:

        pair_of_node_and_attribute_list: list[PairOfNodeAndAttribute] = list()

        pair_of_node_and_attribute_file_path = DataFactoryUtils.generate_path(pathway_name,
                                                                              pair_of_node_and_attribute_file_name)

        node_and_attributes_mapping_line_message_list = DataFactoryUtils.read_file_via_lines(
            pair_of_node_and_attribute_file_path)

        for node_index, node_and_attributes_mapping_line_message in enumerate(
                node_and_attributes_mapping_line_message_list):
            list_of_attribute_indexes = DataFactoryUtils.convert_node_and_attributes_mapping_line_message_to_list_of_attributes(
                node_and_attributes_mapping_line_message)

            for attribute_index in list_of_attribute_indexes:
                pair_of_node_and_attribute = PairOfNodeAndAttribute(node_index=node_index,
                                                                    attribute_index=attribute_index)
                pair_of_node_and_attribute_list.append(pair_of_node_and_attribute)

        return pair_of_node_and_attribute_list

    @staticmethod
    def convert_node_and_attributes_mapping_line_message_to_list_of_attributes(
            node_and_attributes_mapping_line_message: str) -> List[int]:
        list_of_attributes_string_style = node_and_attributes_mapping_line_message.split(",")
        list_of_attributes = [int(attributes_string_style) for attributes_string_style in
                              list_of_attributes_string_style]
        return list_of_attributes

    @staticmethod
    def fill_nodes_inner_attributes_list_(pair_of_node_and_component_list: List[PairOfNodeAndAttribute],
                                          nodes_dict: Dict[int, Node], attributes_dict: Dict[int, Attribute]):

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
    def fill_edges_inner_nodes_list_(relationships_list: List[Relationship], edges_dict: Dict[int, Edge],
                                     nodes_dict: Dict[int, Node]):
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
