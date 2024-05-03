from __future__ import annotations

import copy
import os
import random
import re
from enum import Enum

import numpy as np
from extract_data_form_reactome import FileProcessor
from property import Properties


class ReactomeDataDivider:
    def __init__(self, pathway_name):
        random.seed(1121)
        self.__file_processor = FileProcessor()
        self.__edges_file_name = "edges.txt"
        self.__nodes_file_name = "nodes.txt"
        self.__relationship_file_name = "relationship.txt"
        self.__all_components_file_name = "components-all.txt"
        self.__entities_components_mapping_file_name = "components-mapping.txt"

        self.__attribute_prediction_task = "attribute prediction dataset"
        self.__input_link_prediction_task = "input link prediction dataset"
        self.__output_link_prediction_task = "output link prediction dataset"
        self.__regulation_link_prediction_task = "regulation link prediction dataset"

        self.__train_type_divided_dataset = "train"
        self.__validation_type_divided_dataset = "validation"
        self.__test_type_divided_dataset = "test"

        self.__entity_id_index_of_relationship = 0
        self.__reaction_id_index_of_relationship = 1
        self.__direction_index_of_relationship = 2

        self.__entity_index_of_pair_of_entity_and_component = 0
        self.__component_index_of_pair_of_entity_and_component = 1

        self.__pathway_name = pathway_name

        self.__ultimate_initialisation()

        # self.__input_link_prediction_reaction_ids, self.__output_link_prediction_reactions_ids, self.__regulation_link_prediction_reactions_ids = self.__get_three_divided_reaction_ids()

        # self.__input_link_prediction_reaction_ids, self.__output_link_prediction_reactions_ids, self.__regulation_link_prediction_reactions_ids = self.__get_two_divided_reaction_ids()

        self.__input_link_prediction_reaction_ids = copy.deepcopy(self.__reactions_ids)

        self.__output_link_prediction_reactions_ids = copy.deepcopy(self.__reactions_ids)

        self.__regulation_link_prediction_reactions_ids = copy.deepcopy(self.__reactions_ids)

        self.__input_link_prediction_initialise_reactions_and_entities_components_and_relationships_and_list_of_pair_of_entity_and_component()

        self.__output_link_prediction_initialise_reactions_and_entities_components_and_relationships_and_list_of_pair_of_entity_and_component()

        self.__regulation_link_prediction_initialise_reactions_and_entities_components_and_relationships_and_list_of_pair_of_entity_and_component()

    def __ultimate_initialisation(self):
        self.__reactions_ids = self.__read_reactions_of_one_pathway_from_file()
        self.__entities_ids = self.__read_entities_of_one_pathway_from_file()
        self.__components_ids = self.__read_all_components_of_one_pathway_from_file()

        self.__relationships = self.__read_all_relationships_of_one_pathway_from_file()
        self.__list_of_pair_of_entity_and_component = self.__read_all_pair_of_entity_and_component_of_one_pathway_from_file()

        self.__reaction_to_list_of_entities_dict: dict[str, list[str]] = {}
        self.__reaction_to_list_of_input_entities_dict: dict[str, list[str]] = {}
        self.__reaction_to_list_of_output_entities_dict: dict[str, list[str]] = {}
        self.__reaction_to_list_of_regulation_entities_dict: dict[str, list[str]] = {}

        self.__entity_to_list_of_reactions_dict: dict[str, list[str]] = {}
        self.__entity_to_list_of_input_reactions_dict: dict[str, list[str]] = {}
        self.__entity_to_list_of_output_reactions_dict: dict[str, list[str]] = {}
        self.__entity_to_list_of_regulation_reactions_dict: dict[str, list[str]] = {}

        self.__entity_to_list_of_components_dict: dict[str, list[str]] = {}
        self.__component_to_list_of_entities_dict: dict[str, list[str]] = {}

        # initialise all the above dictionaries
        self.__initialisation_all_reactions_entities_and_components_dict()

    def get_raw_reaction_ids(self) -> list[str]:
        return self.__reactions_ids

    def get_raw_entities_ids(self) -> list[str]:
        return self.__entities_ids

    def get_raw_component_ids(self) -> list[str]:
        return self.__components_ids

    def get_entity_to_list_of_components_dict(self) -> dict[str, list[str]]:
        return self.__entity_to_list_of_components_dict

    def get_reaction_to_list_of_entities_dict(self) -> dict[str, list[str]]:
        return self.__reaction_to_list_of_entities_dict

    def __read_reactions_of_one_pathway_from_file(self) -> list[str]:
        reactions_ids = self.__file_processor.read_file_via_lines(os.path.join("data", self.__pathway_name),
                                                                  self.__edges_file_name)
        return reactions_ids

    def __read_entities_of_one_pathway_from_file(self) -> list[str]:
        entities_ids = self.__file_processor.read_file_via_lines(os.path.join("data", self.__pathway_name),
                                                                 self.__nodes_file_name)
        return entities_ids

    def __read_all_relationships_of_one_pathway_from_file(self) -> list[list[str]]:
        relationships_string_style = self.__file_processor.read_file_via_lines(
            os.path.join("data", self.__pathway_name),
            self.__relationship_file_name)

        relationships: list[list[str]] = list()

        for relationship in relationships_string_style:
            # 13,192,-1.0
            # entity_id_index, reaction_id_index, direction
            # line_of_reaction_id_and_entity_id_and_direction
            elements = relationship.split(",")

            entity_index = elements[self.__entity_id_index_of_relationship]
            entity_id = self.__entities_ids[int(entity_index)]

            reaction_index = elements[self.__reaction_id_index_of_relationship]
            reaction_id = self.__reactions_ids[int(reaction_index)]

            direction = elements[self.__direction_index_of_relationship]

            line_of_reaction_id_and_entity_id_and_direction: list[str] = list()

            line_of_reaction_id_and_entity_id_and_direction.append(entity_id)
            line_of_reaction_id_and_entity_id_and_direction.append(reaction_id)
            line_of_reaction_id_and_entity_id_and_direction.append(direction)

            relationships.append(line_of_reaction_id_and_entity_id_and_direction)

        # entity_id_index, reaction_id_index, direction
        return relationships

    def __read_all_components_of_one_pathway_from_file(self) -> list[str]:
        component_ids = self.__file_processor.read_file_via_lines(os.path.join("data", self.__pathway_name),
                                                                  self.__all_components_file_name)
        return component_ids

    # list of [entity_id, component_id]
    def __read_all_pair_of_entity_and_component_of_one_pathway_from_file(self) -> list[list[str]]:
        # 355,1190,1209
        list_of_entity_components_mappings_with_index_style = self.__file_processor.read_file_via_lines(
            os.path.join("data", self.__pathway_name),
            self.__entities_components_mapping_file_name)
        list_of_pair_of_entity_and_component: list[list[str]] = list()
        for i in range(len(list_of_entity_components_mappings_with_index_style)):
            entity_id = self.__entities_ids[i]
            components_str = list_of_entity_components_mappings_with_index_style[i]
            list_of_component_index_str_style = components_str.split(",")

            for component_str in list_of_component_index_str_style:
                line_list_of_entity_id_and_component_id: list[str] = list()
                component_index = int(component_str)
                component_id = self.__components_ids[component_index]
                line_list_of_entity_id_and_component_id.append(entity_id)
                line_list_of_entity_id_and_component_id.append(component_id)
                list_of_pair_of_entity_and_component.append(line_list_of_entity_id_and_component_id)

        return list_of_pair_of_entity_and_component

    def __initialisation_all_reactions_entities_and_components_dict(self):
        self.__initialisation_inner_reaction_to_list_of_entities_and_entity_to_list_of_reactions_dict()
        self.__initialisation_inner_entity_and_component_dict()

    def __initialisation_inner_reaction_to_list_of_entities_and_entity_to_list_of_reactions_dict(self):
        """ initialise the inner dictionary of reaction to entities and entity to reactions based on different direction
        This method initialise the following inner dictionaries, and will be called by self.__initialisation_set_reactions_entities_and_components_dict(self)
        self.__all_reaction_to_list_of_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_input_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_output_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_regulation_entities_dict: dict[str, list[str]] = {}

        self.__all_entity_to_list_of_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_input_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_output_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_regulation_reactions_dict: dict[str, list[str]] = {}
        :return:
        """

        for relationship in self.__relationships:
            entity_id = relationship[self.__entity_id_index_of_relationship]
            reaction_id = relationship[self.__reaction_id_index_of_relationship]
            direction = relationship[self.__direction_index_of_relationship]

            # general reaction to list of entities
            if reaction_id in self.__reaction_to_list_of_entities_dict.keys():
                entities_list = self.__reaction_to_list_of_entities_dict[reaction_id]
                entities_list.append(entity_id)
            else:
                entities_list = list()
                entities_list.append(entity_id)
                self.__reaction_to_list_of_entities_dict[reaction_id] = entities_list

            # general entity to list of reactions dict
            if entity_id in self.__entity_to_list_of_reactions_dict.keys():
                reactions_list = self.__entity_to_list_of_reactions_dict[entity_id]
                reactions_list.append(reaction_id)
            else:
                reactions_list = list()
                reactions_list.append(reaction_id)
                self.__entity_to_list_of_reactions_dict[entity_id] = reactions_list

            # direction = -1, input
            if int(eval(direction)) < 0:
                # reaction to list of input entities dict
                if reaction_id in self.__reaction_to_list_of_input_entities_dict.keys():
                    entities_list = self.__reaction_to_list_of_input_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    self.__reaction_to_list_of_input_entities_dict[reaction_id] = entities_list

                # entity to list of input reactions dict
                if entity_id in self.__entity_to_list_of_input_reactions_dict.keys():
                    reactions_list = self.__entity_to_list_of_input_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    self.__entity_to_list_of_input_reactions_dict[entity_id] = reactions_list

            # direction = 1, output
            elif int(eval(direction)) > 0:
                # reaction to list of output entities dict
                if reaction_id in self.__reaction_to_list_of_output_entities_dict.keys():
                    entities_list = self.__reaction_to_list_of_output_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    self.__reaction_to_list_of_output_entities_dict[reaction_id] = entities_list

                # entity to list of output reactions dict
                if entity_id in self.__entity_to_list_of_output_reactions_dict.keys():
                    reactions_list = self.__entity_to_list_of_output_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    self.__entity_to_list_of_output_reactions_dict[entity_id] = reactions_list

            # direction = 0, regulation
            else:
                # reaction to list of regulation entities dict
                if reaction_id in self.__reaction_to_list_of_regulation_entities_dict.keys():
                    entities_list = self.__reaction_to_list_of_regulation_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    self.__reaction_to_list_of_regulation_entities_dict[reaction_id] = entities_list

                # entity to list of regulation reactions dict
                if entity_id in self.__entity_to_list_of_regulation_reactions_dict.keys():
                    reactions_list = self.__entity_to_list_of_regulation_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    self.__entity_to_list_of_regulation_reactions_dict[entity_id] = reactions_list

    def __initialisation_inner_entity_and_component_dict(self):
        """
        self.__all_entity_to_list_of_components_dict: dict[str, list[str]] = {}
        self.__all_component_to_list_of_entities_dict: dict[str, list[str]] = {}
        :return:
        """

        for pair_of_entity_and_component in self.__list_of_pair_of_entity_and_component:
            entity_id = pair_of_entity_and_component[self.__entity_index_of_pair_of_entity_and_component]
            component_id = pair_of_entity_and_component[self.__component_index_of_pair_of_entity_and_component]

            # initialise self.__all_entity_to_list_of_components_dict
            if entity_id in self.__entity_to_list_of_components_dict.keys():
                components_list = self.__entity_to_list_of_components_dict[entity_id]
                components_list.append(component_id)
            else:
                components_list = list()
                components_list.append(component_id)
                self.__entity_to_list_of_components_dict[entity_id] = components_list

            # initialise self.__all_component_to_list_of_entities_dict
            if component_id in self.__component_to_list_of_entities_dict.keys():
                entities_list = self.__component_to_list_of_entities_dict[component_id]
                entities_list.append(entity_id)
            else:
                entities_list = list()
                entities_list.append(entity_id)
                self.__component_to_list_of_entities_dict[component_id] = entities_list

    def __get_inner_reaction_to_list_of_entities_and_entity_to_list_of_reactions_dict(self, relationships):
        """ initialise the inner dictionary of reaction to entities and entity to reactions based on different direction
        This method initialise the following inner dictionaries, and will be called by self.__initialisation_set_reactions_entities_and_components_dict(self)
        self.__all_reaction_to_list_of_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_input_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_output_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_regulation_entities_dict: dict[str, list[str]] = {}

        self.__all_entity_to_list_of_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_input_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_output_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_regulation_reactions_dict: dict[str, list[str]] = {}
        :return:
        """

        reaction_to_list_of_entities_dict: dict[str, list[str]] = dict()
        entity_to_list_of_reactions_dict: dict[str, list[str]] = dict()

        reaction_to_list_of_input_entities_dict: dict[str, list[str]] = dict()
        entity_to_list_of_input_reactions_dict: dict[str, list[str]] = dict()

        reaction_to_list_of_output_entities_dict: dict[str, list[str]] = dict()
        entity_to_list_of_output_reactions_dict: dict[str, list[str]] = dict()

        reaction_to_list_of_regulation_entities_dict: dict[str, list[str]] = dict()
        entity_to_list_of_regulation_reactions_dict: dict[str, list[str]] = dict()

        for relationship in relationships:
            entity_id = relationship[self.__entity_id_index_of_relationship]
            reaction_id = relationship[self.__reaction_id_index_of_relationship]
            direction = relationship[self.__direction_index_of_relationship]

            # general reaction to list of entities
            if reaction_id in reaction_to_list_of_entities_dict.keys():
                entities_list = reaction_to_list_of_entities_dict[reaction_id]
                entities_list.append(entity_id)
            else:
                entities_list = list()
                entities_list.append(entity_id)
                reaction_to_list_of_entities_dict[reaction_id] = entities_list

            # general entity to list of reactions dict
            if entity_id in entity_to_list_of_reactions_dict.keys():
                reactions_list = entity_to_list_of_reactions_dict[entity_id]
                reactions_list.append(reaction_id)
            else:
                reactions_list = list()
                reactions_list.append(reaction_id)
                entity_to_list_of_reactions_dict[entity_id] = reactions_list

            # direction = -1, input
            if int(eval(direction)) < 0:
                # reaction to list of input entities dict
                if reaction_id in reaction_to_list_of_input_entities_dict.keys():
                    entities_list = reaction_to_list_of_input_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    reaction_to_list_of_input_entities_dict[reaction_id] = entities_list

                # entity to list of input reactions dict
                if entity_id in entity_to_list_of_input_reactions_dict.keys():
                    reactions_list = entity_to_list_of_input_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    entity_to_list_of_input_reactions_dict[entity_id] = reactions_list

            # direction = 1, output
            elif int(eval(direction)) > 0:
                # reaction to list of output entities dict
                if reaction_id in reaction_to_list_of_output_entities_dict.keys():
                    entities_list = reaction_to_list_of_output_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    reaction_to_list_of_output_entities_dict[reaction_id] = entities_list

                # entity to list of output reactions dict
                if entity_id in entity_to_list_of_output_reactions_dict.keys():
                    reactions_list = entity_to_list_of_output_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    entity_to_list_of_output_reactions_dict[entity_id] = reactions_list

            # direction = 0, regulation
            else:
                # reaction to list of regulation entities dict
                if reaction_id in reaction_to_list_of_regulation_entities_dict.keys():
                    entities_list = reaction_to_list_of_regulation_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    reaction_to_list_of_regulation_entities_dict[reaction_id] = entities_list

                # entity to list of regulation reactions dict
                if entity_id in entity_to_list_of_regulation_reactions_dict.keys():
                    reactions_list = entity_to_list_of_regulation_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    entity_to_list_of_regulation_reactions_dict[entity_id] = reactions_list

        return reaction_to_list_of_entities_dict, entity_to_list_of_reactions_dict, \
               reaction_to_list_of_input_entities_dict, entity_to_list_of_input_reactions_dict, \
               reaction_to_list_of_output_entities_dict, entity_to_list_of_output_reactions_dict, \
               reaction_to_list_of_regulation_entities_dict, entity_to_list_of_regulation_reactions_dict

    def __get_inner_entity_and_component_dict(self, list_of_pair_of_entity_and_component: list[list[str]]):
        """
        self.__all_entity_to_list_of_components_dict: dict[str, list[str]] = {}
        self.__all_component_to_list_of_entities_dict: dict[str, list[str]] = {}
        :return:
        """
        entity_to_list_of_components_dict: dict[str, list[str]] = dict()
        component_to_list_of_entities_dict: dict[str, list[str]] = dict()

        for pair_of_entity_and_component in list_of_pair_of_entity_and_component:
            entity_id = pair_of_entity_and_component[self.__entity_index_of_pair_of_entity_and_component]
            component_id = pair_of_entity_and_component[self.__component_index_of_pair_of_entity_and_component]

            # initialise self.__all_entity_to_list_of_components_dict
            if entity_id in entity_to_list_of_components_dict.keys():
                components_list = entity_to_list_of_components_dict[entity_id]
                components_list.append(component_id)
            else:
                components_list = list()
                components_list.append(component_id)
                entity_to_list_of_components_dict[entity_id] = components_list

            # initialise self.__all_component_to_list_of_entities_dict
            if component_id in component_to_list_of_entities_dict.keys():
                entities_list = component_to_list_of_entities_dict[component_id]
                entities_list.append(entity_id)
            else:
                entities_list = list()
                entities_list.append(entity_id)
                component_to_list_of_entities_dict[component_id] = entities_list

        return entity_to_list_of_components_dict, component_to_list_of_entities_dict

    def test_attributes(self):
        # " ( {:.2%}".format(float(reaction_num_with_one_rela) / float(total_num))
        num_0: int = 0
        num_1: int = 0
        num_2: int = 0
        num_3: int = 0
        num_4: int = 0
        num_5: int = 0
        num_6: int = 0
        num_7: int = 0
        num_8: int = 0
        num_9: int = 0

        for component, list_of_entities in self.__component_to_list_of_entities_dict.items():
            if 0 == len(list_of_entities):
                num_0 = num_0 + 1
            elif 1 == len(list_of_entities):
                num_1 = num_1 + 1
            elif 2 == len(list_of_entities):
                num_2 = num_2 + 1
            elif 3 == len(list_of_entities):
                num_3 = num_3 + 1
            elif 4 == len(list_of_entities):
                num_4 = num_4 + 1
            elif 5 == len(list_of_entities):
                num_5 = num_5 + 1
            elif 6 == len(list_of_entities):
                num_6 = num_6 + 1
            elif 7 == len(list_of_entities):
                num_7 = num_7 + 1
            elif 8 == len(list_of_entities):
                num_8 = num_8 + 1
            else:
                num_9 = num_9 + 1

        totol_num = num_0 + num_1 + num_2 + num_3 + num_4 + num_5 + num_6 + num_7 + num_8 + num_9
        print("total num: " + str(totol_num))
        print("1: ", str(num_0) + "  " + "{:.2%}".format(float(num_0 / totol_num)))
        print("1: ", str(num_1) + "  " + "{:.2%}".format(float(num_1 / totol_num)))
        print("2: ", str(num_2) + "  " + "{:.2%}".format(float(num_2 / totol_num)))
        print("3: ", str(num_3) + "  " + "{:.2%}".format(float(num_3 / totol_num)))
        print("4: ", str(num_4) + "  " + "{:.2%}".format(float(num_4 / totol_num)))
        print("5: ", str(num_5) + "  " + "{:.2%}".format(float(num_5 / totol_num)))
        print("6: ", str(num_6) + "  " + "{:.2%}".format(float(num_6 / totol_num)))
        print("7: ", str(num_7) + "  " + "{:.2%}".format(float(num_7 / totol_num)))
        print("8: ", str(num_8) + "  " + "{:.2%}".format(float(num_8 / totol_num)))
        print("more than 8: ", str(num_9) + "  " + "{:.2%}".format(float(num_9 / totol_num)))

    def __delete_pair_of_entity_and_component(self, pair_of_entity_and_component: list[str]):
        # self.__list_of_pair_of_entity_and_component
        # self.__entity_to_list_of_components_dict
        # self.__component_to_list_of_entities_dict

        entity_id = pair_of_entity_and_component[self.__entity_index_of_pair_of_entity_and_component]
        component_id = pair_of_entity_and_component[self.__component_index_of_pair_of_entity_and_component]
        self.__list_of_pair_of_entity_and_component.remove(pair_of_entity_and_component)

        list_of_components = self.__entity_to_list_of_components_dict[entity_id]
        list_of_components.remove(component_id)

        list_of_entities = self.__component_to_list_of_entities_dict[component_id]
        list_of_entities.remove(entity_id)

    def __delete_relationship(self, relationship: list[str]):
        # self.__relationships
        # self.__reaction_to_list_of_entities_dict
        # self.__reaction_to_list_of_input_entities_dict
        # self.__reaction_to_list_of_output_entities_dict
        # self.__reaction_to_list_of_regulation_entities_dict

        # self.__entity_to_list_of_reactions_dict
        # self.__entity_to_list_of_input_reactions_dict
        # self.__entity_to_list_of_output_reactions_dict
        # self.__entity_to_list_of_regulation_reactions_dict

        entity_id = relationship[self.__entity_id_index_of_relationship]
        reaction_id = relationship[self.__reaction_id_index_of_relationship]
        direction = relationship[self.__direction_index_of_relationship]

        self.__relationships.remove(relationship)

        self.__reaction_to_list_of_entities_dict[reaction_id].remove(entity_id)
        self.__entity_to_list_of_reactions_dict[entity_id].remove(reaction_id)

        if int(eval(direction)) < 0:
            self.__reaction_to_list_of_input_entities_dict[reaction_id].remove(entity_id)
            self.__entity_to_list_of_input_reactions_dict[entity_id].remove(reaction_id)
        elif int(eval(direction)) > 0:
            self.__reaction_to_list_of_output_entities_dict[reaction_id].remove(entity_id)
            self.__entity_to_list_of_output_reactions_dict[entity_id].remove(reaction_id)
        else:
            self.__reaction_to_list_of_regulation_entities_dict[reaction_id].remove(entity_id)
            self.__entity_to_list_of_regulation_reactions_dict[entity_id].remove(reaction_id)

    def __input_link_prediction_delete_relationship(self, relationship: list[str]):
        entity_id = relationship[self.__entity_id_index_of_relationship]
        reaction_id = relationship[self.__reaction_id_index_of_relationship]
        direction = relationship[self.__direction_index_of_relationship]

        # print("The relationship to be deleted " + str(relationship))

        self.__relationships.remove(relationship)
        self.__input_link_prediction_relationships.remove(relationship)

        self.__input_link_prediction_reaction_to_list_of_entities_dict[reaction_id].remove(entity_id)
        self.__input_link_prediction_entity_to_list_of_reactions_dict[entity_id].remove(reaction_id)

        if int(eval(direction)) < 0:
            self.__input_link_prediction_reaction_to_list_of_input_entities_dict[reaction_id].remove(entity_id)
            self.__input_link_prediction_entity_to_list_of_input_reactions_dict[entity_id].remove(reaction_id)
        elif int(eval(direction)) > 0:
            self.__input_link_prediction_reaction_to_list_of_output_entities_dict[reaction_id].remove(entity_id)
            self.__input_link_prediction_entity_to_list_of_output_reactions_dict[entity_id].remove(reaction_id)
        else:
            self.__input_link_prediction_reaction_to_list_of_regulation_entities_dict[reaction_id].remove(entity_id)
            self.__input_link_prediction_entity_to_list_of_regulation_reactions_dict[entity_id].remove(reaction_id)

    def __output_link_prediction_delete_relationship(self, relationship: list[str]):
        entity_id = relationship[self.__entity_id_index_of_relationship]
        reaction_id = relationship[self.__reaction_id_index_of_relationship]
        direction = relationship[self.__direction_index_of_relationship]

        self.__relationships.remove(relationship)
        self.__output_link_prediction_relationships.remove(relationship)

        self.__output_link_prediction_reaction_to_list_of_entities_dict[reaction_id].remove(entity_id)
        self.__output_link_prediction_entity_to_list_of_reactions_dict[entity_id].remove(reaction_id)

        if int(eval(direction)) < 0:
            self.__output_link_prediction_reaction_to_list_of_input_entities_dict[reaction_id].remove(entity_id)
            self.__output_link_prediction_entity_to_list_of_input_reactions_dict[entity_id].remove(reaction_id)
        elif int(eval(direction)) > 0:
            self.__output_link_prediction_reaction_to_list_of_output_entities_dict[reaction_id].remove(entity_id)
            self.__output_link_prediction_entity_to_list_of_output_reactions_dict[entity_id].remove(reaction_id)
        else:
            self.__output_link_prediction_reaction_to_list_of_regulation_entities_dict[reaction_id].remove(entity_id)
            self.__output_link_prediction_entity_to_list_of_regulation_reactions_dict[entity_id].remove(reaction_id)

    def __regulation_link_prediction_delete_relationship(self, relationship: list[str]):
        entity_id = relationship[self.__entity_id_index_of_relationship]
        reaction_id = relationship[self.__reaction_id_index_of_relationship]
        direction = relationship[self.__direction_index_of_relationship]

        self.__regulation_link_prediction_relationships.remove(relationship)

        self.__regulation_link_prediction_reaction_to_list_of_entities_dict[reaction_id].remove(entity_id)
        self.__regulation_link_prediction_entity_to_list_of_reactions_dict[entity_id].remove(reaction_id)

        if int(eval(direction)) < 0:
            self.__regulation_link_prediction_reaction_to_list_of_input_entities_dict[reaction_id].remove(entity_id)
            self.__regulation_link_prediction_entity_to_list_of_input_reactions_dict[entity_id].remove(reaction_id)
        elif int(eval(direction)) > 0:
            self.__regulation_link_prediction_reaction_to_list_of_output_entities_dict[reaction_id].remove(entity_id)
            self.__regulation_link_prediction_entity_to_list_of_output_reactions_dict[entity_id].remove(reaction_id)
        else:
            self.__regulation_link_prediction_reaction_to_list_of_regulation_entities_dict[reaction_id].remove(
                entity_id)
            self.__regulation_link_prediction_entity_to_list_of_regulation_reactions_dict[entity_id].remove(reaction_id)

    def __get_list_of_relationships_based_on_entity_id(self, entity_id: str, original_relationships: list[list[str]]):
        list_of_relationships: list[list[str]] = list()
        for relationship in original_relationships:
            if relationship[self.__entity_id_index_of_relationship] == entity_id:
                list_of_relationships.append(copy.deepcopy(relationship))

        return list_of_relationships

    def __get_list_of_relationships_based_on_list_of_entity_ids(self, list_of_entity_ids: list[str],
                                                                original_relationships: list[list[str]]):
        list_of_relationships: list[list[str]] = list()
        for entity_id in list_of_entity_ids:
            list_of_relationships_for_single_entity = self.__get_list_of_relationships_based_on_entity_id(entity_id,
                                                                                                          original_relationships)
            list_of_relationships.extend(list_of_relationships_for_single_entity)

        list_of_relationships_tmp = list(set(tuple(relationship) for relationship in list_of_relationships))

        list_of_relationships: list[list[str]] = [list(relationships) for relationships in list_of_relationships_tmp]

        return list_of_relationships

    def __get_list_of_relationships_based_on_reaction_id(self, reaction_id):
        list_of_relationships: list[list[str]] = list()
        for relationship in self.__relationships:
            if relationship[self.__reaction_id_index_of_relationship] == reaction_id:
                list_of_relationships.append(copy.deepcopy(relationship))

        return list_of_relationships

    def __get_list_of_relationships_based_on_list_of_reaction_ids(self, list_of_reaction_ids):
        list_of_relationships: list[list[str]] = list()
        for reaction_id in list_of_reaction_ids:
            list_of_relationships_for_single_reaction_id = self.__get_list_of_relationships_based_on_reaction_id(
                reaction_id)
            list_of_relationships.extend(list_of_relationships_for_single_reaction_id)

        list_of_relationships = list(set(tuple(relationship) for relationship in list_of_relationships))

        # every edge from tuple to list
        list_of_relationships = list(list(relationship) for relationship in list_of_relationships)

        return list_of_relationships

    def __get_list_of_entities_based_on_list_of_relationships(self, list_of_relationships: list[list[str]]) -> list[
        str]:
        set_of_entities = set()
        for relationship in list_of_relationships:
            entity_id = relationship[self.__entity_id_index_of_relationship]
            set_of_entities.add(entity_id)

        return list(set_of_entities)

    def __get_list_of_components_based_on_list_of_entity_ids(self, list_of_entity_ids):
        list_of_components: list[str] = list()
        for entity_id in list_of_entity_ids:
            list_of_components_for_single_entity = self.__entity_to_list_of_components_dict[entity_id]
            list_of_components.extend(list_of_components_for_single_entity)

        return list(set(list_of_components))

    def __get_list_of_pair_of_entity_and_component_based_on_entity_id(self, entity_id: str):
        list_of_pair_of_entity_and_component_for_return: list[list[str]] = list()

        for pair_of_entity_and_component in self.__list_of_pair_of_entity_and_component:
            if pair_of_entity_and_component[self.__entity_index_of_pair_of_entity_and_component] == entity_id:
                list_of_pair_of_entity_and_component_for_return.append(copy.deepcopy(pair_of_entity_and_component))

        return list_of_pair_of_entity_and_component_for_return

    def __get_list_of_pair_of_entity_and_component_based_on_list_of_entity_ids(self, list_of_entity_ids: list[str]):
        list_of_pair_of_entity_and_component = list()
        for entity_id in list_of_entity_ids:
            list_of_pair_of_entity_and_component_for_single_entity = self.__get_list_of_pair_of_entity_and_component_based_on_entity_id(
                entity_id)
            list_of_pair_of_entity_and_component.extend(list_of_pair_of_entity_and_component_for_single_entity)

        list_of_pair_of_entity_and_component = list(set(
            tuple(pair_of_entity_and_component) for pair_of_entity_and_component in
            list_of_pair_of_entity_and_component))

        # every edge from tuple to list
        list_of_pair_of_entity_and_component = list(
            list(pair_of_entity_and_component) for pair_of_entity_and_component in list_of_pair_of_entity_and_component)

        return list_of_pair_of_entity_and_component

    # 针对attributes任务进行划分
    # 参与两个以上entity的attributes 占比在 20%(Metabolism),30%(Disease)到40%(Immune System, Signal Transduction)之间
    # 划分原则:
    # 只 mask 参与两个以上entity的 attribute，保证最后attributes之间的比例是 8 : 1 : 1
    # 算法：
    # 我们计算参与多个entity的attribute出现的次数，比如一个 attribute 参与三个entity，那么它就出现了 3 次
    # 把它们和只参与一个entity的attribute合并在一起，就得到一个总数total，我们要做的是把这个总数按8：1：1来划分
    # 我们根据这个计算出validation和test的attribute数量的大小 validation_size 和 test_size
    # 我们随机找到一个entity，检查它的components长度是否大于1(==1 直接quit），然后在它的component中随机找到一个attribute，检查这个attribute是不是
    # 出现过2次及以上，如果是，我们检查validation_size 和 test_size装满了没有，
    # 没装满的话，我们选择mask掉它，将这个attribute和其对应entity拿出来，做成entity-component list
    # 将entity对应的所有reactions找出来，做成多个relationships，这个在原数据集中不删除
    def divide_data_for_attribute_prediction_task(self):
        random.seed(1121)
        train_data_bean = DataBeanForReactome(self.__pathway_name, self.__attribute_prediction_task,
                                              self.__train_type_divided_dataset, self)
        validation_data_bean = DataBeanForReactome(self.__pathway_name, self.__attribute_prediction_task,
                                                   self.__validation_type_divided_dataset, self)
        test_data_bean = DataBeanForReactome(self.__pathway_name, self.__attribute_prediction_task,
                                             self.__test_type_divided_dataset, self)

        total_of_attributes: int = 0
        for component_id, list_of_entity_ids in self.__component_to_list_of_entities_dict.items():
            occur_times = len(list_of_entity_ids)
            total_of_attributes = total_of_attributes + occur_times


        adjust_rate: float = 1.0
        if "Disease" == self.__pathway_name:
            adjust_rate = 0.8
        if "Metabolism" == self.__pathway_name:
            adjust_rate = 0.6
        if "Immune System" == self.__pathway_name or "Signal Transduction" == self.__pathway_name:
            adjust_rate = 1.0

        validation_size = test_size = int(total_of_attributes * adjust_rate / 10)

        length_of_entities = len(self.__entities_ids)
        end_index_of_entities = length_of_entities - 1
        validation_counter: int = 0
        test_counter: int = 0

        list_of_validation_mask_entity_id: list[str] = list()
        list_of_test_mask_entity_id: list[str] = list()

        random_entity_index_memory: dict[int, int] = dict()

        print("validation_size: " + str(validation_size))
        print("test_size: " + str(test_size))


        while validation_counter < validation_size or test_counter < test_size:
            random_entity_index = random.randint(0, end_index_of_entities)


            if random_entity_index in random_entity_index_memory.keys():
                continue

            random_entity_id = self.__entities_ids[random_entity_index]
            list_of_components = self.__entity_to_list_of_components_dict.get(random_entity_id)

            if len(list_of_components) >= 2:
                random_component_index = random.randint(0, len(list_of_components) - 1)
                random_component_id = list_of_components[random_component_index]
                list_of_entities = self.__component_to_list_of_entities_dict[random_component_id]
                if len(list_of_entities) >= 2:


                    random_entity_index_memory[random_entity_index] = 1

                    pair_of_entity_and_component: list[str] = list()
                    pair_of_entity_and_component.append(random_entity_id)
                    pair_of_entity_and_component.append(random_component_id)

                    train_data_bean.add_train_entity_id_mask_to_inner_train_entity_mask_list(random_entity_id)
                    train_data_bean.add_pair_of_entity_and_component_masked_to_inner_pair_of_entity_and_component_masked_list(
                        pair_of_entity_and_component)

                    if validation_counter >= validation_size:
                        list_of_test_mask_entity_id.append(random_entity_id)
                        # test_data_bean.add_list_of_relationships(list_of_relationships)
                        test_data_bean.add_pair_of_entity_and_component_masked_to_inner_pair_of_entity_and_component_masked_list(
                            pair_of_entity_and_component)
                        test_counter = test_counter + 1
                    elif test_counter >= test_size:
                        list_of_validation_mask_entity_id.append(random_entity_id)
                        # validation_data_bean.add_list_of_relationships(list_of_relationships)
                        validation_data_bean.add_pair_of_entity_and_component_masked_to_inner_pair_of_entity_and_component_masked_list(
                            pair_of_entity_and_component)
                        validation_counter = validation_counter + 1
                    else:
                        flag = random.randint(0, 1)
                        if flag == 0:
                            list_of_test_mask_entity_id.append(random_entity_id)
                            # test_data_bean.add_list_of_relationships(list_of_relationships)
                            test_data_bean.add_pair_of_entity_and_component_masked_to_inner_pair_of_entity_and_component_masked_list(
                                pair_of_entity_and_component)
                            test_counter = test_counter + 1
                        elif flag == 1:
                            list_of_validation_mask_entity_id.append(random_entity_id)
                            # validation_data_bean.add_list_of_relationships(list_of_relationships)
                            validation_data_bean.add_pair_of_entity_and_component_masked_to_inner_pair_of_entity_and_component_masked_list(
                                pair_of_entity_and_component)
                            validation_counter = validation_counter + 1

                    self.__delete_pair_of_entity_and_component(pair_of_entity_and_component)

        list_of_validation_mask_entity_id = list(set(list_of_validation_mask_entity_id))
        list_of_test_mask_entity_id = list(set(list_of_test_mask_entity_id))

        relationships_for_validation_data_bean = self.__get_list_of_relationships_based_on_list_of_entity_ids(
            list_of_validation_mask_entity_id, self.__relationships)

        list_of_pair_of_entity_and_component_for_validation_data_bean = self.__get_list_of_pair_of_entity_and_component_based_on_list_of_entity_ids(
            list_of_validation_mask_entity_id)

        relationships_for_test_data_bean = self.__get_list_of_relationships_based_on_list_of_entity_ids(
            list_of_test_mask_entity_id, self.__relationships)

        list_of_pair_of_entity_and_component_for_test_data_bean = self.__get_list_of_pair_of_entity_and_component_based_on_list_of_entity_ids(
            list_of_test_mask_entity_id)

        validation_data_bean.add_list_of_relationships(relationships_for_validation_data_bean)

        validation_data_bean.add_list_of_pair_of_entity_and_component(
            list_of_pair_of_entity_and_component_for_validation_data_bean)

        test_data_bean.add_list_of_relationships(relationships_for_test_data_bean)

        test_data_bean.add_list_of_pair_of_entity_and_component(list_of_pair_of_entity_and_component_for_test_data_bean)

        train_data_bean.add_list_of_relationships(self.__relationships)
        train_data_bean.add_list_of_pair_of_entity_and_component(self.__list_of_pair_of_entity_and_component)

        train_data_bean.print_sub_data_bean_to_files()
        validation_data_bean.print_sub_data_bean_to_files()
        test_data_bean.print_sub_data_bean_to_files()

        validation_data_bean.generate_and_print_components_mapping_mix_negative_to_file(10)
        test_data_bean.generate_and_print_components_mapping_mix_negative_to_file(10)

        train_data_bean.information()
        validation_data_bean.information()
        test_data_bean.information()

        self.__ultimate_initialisation()

    # 将reactions 均分为3份
    def __get_three_divided_reaction_ids(self):
        random.seed(1121)
        reactions_ids = copy.deepcopy(self.__reactions_ids)
        reactions_ids.sort()
        np.random.shuffle(reactions_ids)
        # 1 : 1 : 1
        total_num = len(reactions_ids)
        num_of_input_link_prediction_reaction_ids = int(total_num * 0.33)
        num_of_output_link_prediction_reaction_ids = int(total_num * 0.33)
        num_of_regulation_link_prediction_reaction_ids = total_num - num_of_input_link_prediction_reaction_ids - num_of_output_link_prediction_reaction_ids

        output_link_prediction_reaction_ids_start_index = input_link_prediction_reaction_ids_end_index = num_of_input_link_prediction_reaction_ids

        regulation_link_prediction_reaction_ids_start_index = output_link_prediction_reaction_ids_end_index = num_of_input_link_prediction_reaction_ids + num_of_output_link_prediction_reaction_ids

        input_link_prediction_reaction_ids = reactions_ids[0:input_link_prediction_reaction_ids_end_index]
        output_link_prediction_reactions_ids = reactions_ids[
                                               output_link_prediction_reaction_ids_start_index:output_link_prediction_reaction_ids_end_index]
        regulation_link_prediction_reactions_ids = reactions_ids[
                                                   regulation_link_prediction_reaction_ids_start_index:]

        return input_link_prediction_reaction_ids, output_link_prediction_reactions_ids, regulation_link_prediction_reactions_ids

    # 将reaction均分为2份
    def __get_two_divided_reaction_ids(self):
        random.seed(1121)
        reactions_ids = copy.deepcopy(self.__reactions_ids)
        reactions_ids.sort()
        random.shuffle(reactions_ids)
        # 1 : 1
        total_num = len(reactions_ids)
        num_of_input_link_prediction_reaction_ids = int(total_num * 0.5)
        num_of_output_link_prediction_reaction_ids = total_num - num_of_input_link_prediction_reaction_ids

        output_link_prediction_reaction_ids_start_index = input_link_prediction_reaction_ids_end_index = num_of_input_link_prediction_reaction_ids

        input_link_prediction_reaction_ids = reactions_ids[0:input_link_prediction_reaction_ids_end_index]
        output_link_prediction_reactions_ids = reactions_ids[
                                               output_link_prediction_reaction_ids_start_index:]

        input_link_prediction_reaction_ids.sort()
        output_link_prediction_reactions_ids.sort()

        # print(input_link_prediction_reaction_ids[:3])
        # print(output_link_prediction_reactions_ids[:3])
        return input_link_prediction_reaction_ids, output_link_prediction_reactions_ids, list()

    def __input_link_prediction_initialise_reactions_and_entities_components_and_relationships_and_list_of_pair_of_entity_and_component(
            self):
        reaction_ids = self.__input_link_prediction_reaction_ids
        self.__input_link_prediction_relationships = self.__get_list_of_relationships_based_on_list_of_reaction_ids(
            reaction_ids)
        self.__input_link_prediction_entity_ids = self.__get_list_of_entities_based_on_list_of_relationships(
            self.__input_link_prediction_relationships)

        self.__input_link_prediction_list_of_pair_of_entity_and_component = self.__get_list_of_pair_of_entity_and_component_based_on_list_of_entity_ids(
            self.__input_link_prediction_entity_ids)

        self.__input_link_prediction_component_ids = self.__get_list_of_components_based_on_list_of_entity_ids(
            self.__input_link_prediction_entity_ids)

        self.__input_link_prediction_reaction_to_list_of_entities_dict, self.__input_link_prediction_entity_to_list_of_reactions_dict, \
        self.__input_link_prediction_reaction_to_list_of_input_entities_dict, self.__input_link_prediction_entity_to_list_of_input_reactions_dict, \
        self.__input_link_prediction_reaction_to_list_of_output_entities_dict, self.__input_link_prediction_entity_to_list_of_output_reactions_dict, \
        self.__input_link_prediction_reaction_to_list_of_regulation_entities_dict, self.__input_link_prediction_entity_to_list_of_regulation_reactions_dict = self.__get_inner_reaction_to_list_of_entities_and_entity_to_list_of_reactions_dict(
            self.__input_link_prediction_relationships)

        self.__input_link_prediction_entity_to_list_of_components_dict, self.__input_link_prediction_component_to_list_of_entities_dict = self.__get_inner_entity_and_component_dict(
            self.__input_link_prediction_list_of_pair_of_entity_and_component)

    def __output_link_prediction_initialise_reactions_and_entities_components_and_relationships_and_list_of_pair_of_entity_and_component(
            self):
        reaction_ids = self.__output_link_prediction_reactions_ids
        self.__output_link_prediction_relationships = self.__get_list_of_relationships_based_on_list_of_reaction_ids(
            reaction_ids)
        self.__output_link_prediction_entity_ids = self.__get_list_of_entities_based_on_list_of_relationships(
            self.__output_link_prediction_relationships)

        self.__output_link_prediction_list_of_pair_of_entity_and_component = self.__get_list_of_pair_of_entity_and_component_based_on_list_of_entity_ids(
            self.__output_link_prediction_entity_ids)

        self.__output_link_prediction_component_ids = self.__get_list_of_components_based_on_list_of_entity_ids(
            self.__output_link_prediction_entity_ids)

        self.__output_link_prediction_reaction_to_list_of_entities_dict, self.__output_link_prediction_entity_to_list_of_reactions_dict, \
        self.__output_link_prediction_reaction_to_list_of_input_entities_dict, self.__output_link_prediction_entity_to_list_of_input_reactions_dict, \
        self.__output_link_prediction_reaction_to_list_of_output_entities_dict, self.__output_link_prediction_entity_to_list_of_output_reactions_dict, \
        self.__output_link_prediction_reaction_to_list_of_regulation_entities_dict, self.__output_link_prediction_entity_to_list_of_regulation_reactions_dict = self.__get_inner_reaction_to_list_of_entities_and_entity_to_list_of_reactions_dict(
            self.__output_link_prediction_relationships)

        self.__output_link_prediction_entity_to_list_of_components_dict, self.__output_link_prediction_component_to_list_of_entities_dict = self.__get_inner_entity_and_component_dict(
            self.__output_link_prediction_list_of_pair_of_entity_and_component)

    def __regulation_link_prediction_initialise_reactions_and_entities_components_and_relationships_and_list_of_pair_of_entity_and_component(
            self):
        reaction_ids = self.__regulation_link_prediction_reactions_ids
        self.__regulation_link_prediction_relationships = self.__get_list_of_relationships_based_on_list_of_reaction_ids(
            reaction_ids)
        self.__regulation_link_prediction_entity_ids = self.__get_list_of_entities_based_on_list_of_relationships(
            self.__regulation_link_prediction_relationships)

        self.__regulation_link_prediction_list_of_pair_of_entity_and_component = self.__get_list_of_pair_of_entity_and_component_based_on_list_of_entity_ids(
            self.__regulation_link_prediction_entity_ids)

        self.__regulation_link_prediction_component_ids = self.__get_list_of_components_based_on_list_of_entity_ids(
            self.__regulation_link_prediction_entity_ids)

        self.__regulation_link_prediction_reaction_to_list_of_entities_dict, self.__regulation_link_prediction_entity_to_list_of_reactions_dict, \
        self.__regulation_link_prediction_reaction_to_list_of_input_entities_dict, self.__regulation_link_prediction_entity_to_list_of_input_reactions_dict, \
        self.__regulation_link_prediction_reaction_to_list_of_output_entities_dict, self.__regulation_link_prediction_entity_to_list_of_output_reactions_dict, \
        self.__regulation_link_prediction_reaction_to_list_of_regulation_entities_dict, self.__regulation_link_prediction_entity_to_list_of_regulation_reactions_dict = self.__get_inner_reaction_to_list_of_entities_and_entity_to_list_of_reactions_dict(
            self.__regulation_link_prediction_relationships)

        self.__output_link_prediction_entity_to_list_of_components_dict, self.__regulation_link_prediction_component_to_list_of_entities_dict = self.__get_inner_entity_and_component_dict(
            self.__regulation_link_prediction_list_of_pair_of_entity_and_component)

    def divide_data_for_input_link_prediction_task(self):
        random.seed(1121)
        train_data_bean = DataBeanForReactome(self.__pathway_name, self.__input_link_prediction_task,
                                              self.__train_type_divided_dataset, self)
        validation_data_bean = DataBeanForReactome(self.__pathway_name, self.__input_link_prediction_task,
                                                   self.__validation_type_divided_dataset, self)
        test_data_bean = DataBeanForReactome(self.__pathway_name, self.__input_link_prediction_task,
                                             self.__test_type_divided_dataset, self)
        total_num: int = 0
        for entity_id in self.__input_link_prediction_entity_ids:
            list_of_reactions = self.__input_link_prediction_entity_to_list_of_input_reactions_dict.get(entity_id)
            if list_of_reactions is None:
                list_of_reactions = list()
            total_num = total_num + len(list_of_reactions)

        print("entity id num = ", len(self.__input_link_prediction_entity_ids))
        print("total num = ", total_num)

        adjust_rate: float = 1.0

        validation_size = test_size = int(total_num * adjust_rate / 10)

        length_of_reactions = len(self.__input_link_prediction_reaction_ids)

        end_index_of_reactions = length_of_reactions - 1

        validation_counter: int = 0
        test_counter: int = 0

        reaction_id_memory_dict: dict[str, int] = dict()
        entity_id_memory_dict: dict[str, int] = dict()

        # sort the dict to make sure we get the same data after division
        for reaction_id, list_of_input_entities_ids in self.__input_link_prediction_reaction_to_list_of_input_entities_dict.items():
            list_of_input_entities_ids.sort()

        list_of_validation_mask_entity_id: list[str] = list()
        list_of_test_mask_entity_id: list[str] = list()

        list_of_validation_mask_reaction_id: list[str] = list()
        list_of_test_mask_reaction_id: list[str] = list()

        while validation_counter < validation_size or test_counter < test_size:
            random_reaction_index = random.randint(0, end_index_of_reactions)
            random_reaction_id = self.__input_link_prediction_reaction_ids[random_reaction_index]
            list_of_input_entities = self.__input_link_prediction_reaction_to_list_of_input_entities_dict.get(
                random_reaction_id)
            if list_of_input_entities is None:
                list_of_input_entities = list()

            if len(list_of_input_entities) >= 2 and random_reaction_id not in reaction_id_memory_dict.keys():

                reaction_id_memory_dict[random_reaction_id] = 1

                random_entity_index = random.randint(0, len(list_of_input_entities) - 1)
                random_entity_id = list_of_input_entities[random_entity_index]
                list_of_reactions = self.__input_link_prediction_entity_to_list_of_reactions_dict.get(random_entity_id)
                if list_of_reactions is None:
                    list_of_reactions = list()

                # if len(list_of_reactions) >= 2 and (random_entity_id not in entity_id_memory_dict or entity_id_memory_dict[random_entity_id] < 5):

                if len(list_of_reactions) >= 2 and (random_entity_id not in entity_id_memory_dict.keys() or entity_id_memory_dict[random_entity_id] < 6):
                    if random_entity_id not in entity_id_memory_dict.keys():
                        entity_id_memory_dict[random_entity_id] = 1
                    else:
                        entity_id_memory_dict[random_entity_id] = entity_id_memory_dict.get(random_entity_id) + 1

                    relationship: list[str] = list()
                    relationship.append(random_entity_id)
                    relationship.append(random_reaction_id)
                    relationship.append(str(-1))
                    list_of_pair_of_entity_and_component = self.__get_list_of_pair_of_entity_and_component_based_on_entity_id(
                        random_entity_id)

                    # todo
                    list_of_pair_of_entity_and_component.sort()
                    train_data_bean.add_relationship_masked_to_inner_relationships_masked_list(relationship)

                    if validation_counter >= validation_size:
                        test_data_bean.add_relationship_masked_to_inner_relationships_masked_list(relationship)
                        list_of_test_mask_entity_id.append(random_entity_id)
                        list_of_test_mask_reaction_id.append(random_reaction_id)
                        test_counter = test_counter + 1
                    elif test_counter >= test_size:
                        validation_data_bean.add_relationship_masked_to_inner_relationships_masked_list(relationship)
                        list_of_validation_mask_entity_id.append(random_entity_id)
                        list_of_validation_mask_reaction_id.append(random_reaction_id)
                        validation_counter = validation_counter + 1
                    else:
                        flag = random.randint(0, 1)
                        if flag == 0:
                            test_data_bean.add_relationship_masked_to_inner_relationships_masked_list(relationship)
                            list_of_test_mask_entity_id.append(random_entity_id)
                            list_of_test_mask_reaction_id.append(random_reaction_id)
                            test_counter = test_counter + 1
                        elif flag == 1:
                            validation_data_bean.add_relationship_masked_to_inner_relationships_masked_list(
                                relationship)
                            list_of_validation_mask_entity_id.append(random_entity_id)
                            list_of_validation_mask_reaction_id.append(random_reaction_id)
                            validation_counter = validation_counter + 1
                    self.__input_link_prediction_delete_relationship(relationship)

        list_of_validation_mask_reaction_id = list(set(list_of_validation_mask_reaction_id))
        list_of_test_mask_reaction_id = list(set(list_of_test_mask_reaction_id))

        relationships_for_validation_data_bean = self.__get_list_of_relationships_based_on_list_of_reaction_ids(
            list_of_validation_mask_reaction_id)

        list_of_entities_for_validation_tmp = self.__get_list_of_entities_based_on_list_of_relationships(
            relationships_for_validation_data_bean)

        list_of_pair_of_entity_and_component_for_validation_data_bean = self.__get_list_of_pair_of_entity_and_component_based_on_list_of_entity_ids(
            list_of_entities_for_validation_tmp)

        relationships_for_test_data_bean = self.__get_list_of_relationships_based_on_list_of_reaction_ids(
            list_of_test_mask_reaction_id)

        list_of_entities_for_test_tmp = self.__get_list_of_entities_based_on_list_of_relationships(
            relationships_for_test_data_bean)

        list_of_pair_of_entity_and_component_for_test_data_bean = self.__get_list_of_pair_of_entity_and_component_based_on_list_of_entity_ids(
            list_of_entities_for_test_tmp)

        validation_data_bean.add_list_of_relationships(relationships_for_validation_data_bean)

        validation_data_bean.add_list_of_pair_of_entity_and_component(
            list_of_pair_of_entity_and_component_for_validation_data_bean)

        test_data_bean.add_list_of_relationships(relationships_for_test_data_bean)

        test_data_bean.add_list_of_pair_of_entity_and_component(list_of_pair_of_entity_and_component_for_test_data_bean)

        train_data_bean.add_list_of_relationships(self.__input_link_prediction_relationships)
        train_data_bean.add_list_of_pair_of_entity_and_component(
            self.__input_link_prediction_list_of_pair_of_entity_and_component)

        train_data_bean.print_sub_data_bean_to_files()
        validation_data_bean.print_sub_data_bean_to_files()
        test_data_bean.print_sub_data_bean_to_files()

        validation_data_bean.generate_and_print_relationships_mix_negative_to_file(10)
        test_data_bean.generate_and_print_relationships_mix_negative_to_file(10)

        train_data_bean.information()
        validation_data_bean.information()
        test_data_bean.information()

        self.__ultimate_initialisation()

    def divide_data_for_output_link_prediction_task(self):
        random.seed(1121)
        train_data_bean = DataBeanForReactome(self.__pathway_name, self.__output_link_prediction_task,
                                              self.__train_type_divided_dataset, self)
        validation_data_bean = DataBeanForReactome(self.__pathway_name, self.__output_link_prediction_task,
                                                   self.__validation_type_divided_dataset, self)
        test_data_bean = DataBeanForReactome(self.__pathway_name, self.__output_link_prediction_task,
                                             self.__test_type_divided_dataset, self)
        total_num: int = 0

        for entity_id in self.__output_link_prediction_entity_ids:
            list_of_reactions = self.__output_link_prediction_entity_to_list_of_output_reactions_dict.get(entity_id)
            if list_of_reactions is None:
                list_of_reactions = list()
            total_num = total_num + len(list_of_reactions)

        adjust_rate: float = 1.0

        if "Disease" == self.__pathway_name:
            adjust_rate = 0.55
        if "Metabolism" == self.__pathway_name:
            adjust_rate = 0.75
        if "Immune System" == self.__pathway_name:
            adjust_rate = 0.6
        if "Signal Transduction" == self.__pathway_name:
            adjust_rate = 0.51

        validation_size = test_size = int(total_num * adjust_rate / 10)

        length_of_reactions = len(self.__output_link_prediction_reactions_ids)

        end_index_of_reactions = length_of_reactions - 1

        validation_counter: int = 0
        test_counter: int = 0

        reaction_id_memory_dict: dict[str, int] = dict()
        entity_id_memory_dict: dict[str, int] = dict()

        print("entity id num = ", len(self.__output_link_prediction_entity_ids))
        print("total num = ", total_num)

        # sort the dict to make sure we get the same data after division
        for reaction_id, list_of_output_entities_ids in self.__output_link_prediction_reaction_to_list_of_output_entities_dict.items():
            list_of_output_entities_ids.sort()

        list_of_validation_mask_entity_id: list[str] = list()
        list_of_test_mask_entity_id: list[str] = list()

        list_of_validation_mask_reaction_id: list[str] = list()
        list_of_test_mask_reaction_id: list[str] = list()

        while validation_counter < validation_size or test_counter < test_size:
            random_reaction_index = random.randint(0, end_index_of_reactions)
            random_reaction_id = self.__output_link_prediction_reactions_ids[random_reaction_index]
            list_of_output_entities = self.__output_link_prediction_reaction_to_list_of_output_entities_dict.get(
                random_reaction_id)
            if list_of_output_entities is None:
                list_of_output_entities = list()

            if len(list_of_output_entities) >= 2 and random_reaction_id not in reaction_id_memory_dict.keys():

                reaction_id_memory_dict[random_reaction_id] = 1

                random_entity_index = random.randint(0, len(list_of_output_entities) - 1)
                random_entity_id = list_of_output_entities[random_entity_index]
                list_of_reactions = self.__output_link_prediction_entity_to_list_of_reactions_dict.get(random_entity_id)
                if list_of_reactions is None:
                    list_of_reactions = list()

                # if len(list_of_reactions) >= 2 and (random_entity_id not in entity_id_memory_dict or entity_id_memory_dict[random_entity_id] < 8):
                if len(list_of_reactions) >= 2 and (random_entity_id not in entity_id_memory_dict.keys() or entity_id_memory_dict[random_entity_id] < 3):
                    if random_entity_id not in entity_id_memory_dict.keys():
                        entity_id_memory_dict[random_entity_id] = 1
                    else:
                        entity_id_memory_dict[random_entity_id] = entity_id_memory_dict.get(random_entity_id) + 1

                    relationship: list[str] = list()
                    relationship.append(random_entity_id)
                    relationship.append(random_reaction_id)
                    relationship.append(str(1))
                    list_of_pair_of_entity_and_component = self.__get_list_of_pair_of_entity_and_component_based_on_entity_id(
                        random_entity_id)

                    # todo
                    list_of_pair_of_entity_and_component.sort()
                    train_data_bean.add_relationship_masked_to_inner_relationships_masked_list(relationship)

                    if validation_counter >= validation_size:
                        test_data_bean.add_relationship_masked_to_inner_relationships_masked_list(relationship)
                        list_of_test_mask_entity_id.append(random_entity_id)
                        list_of_test_mask_reaction_id.append(random_reaction_id)
                        test_counter = test_counter + 1
                    elif test_counter >= test_size:
                        validation_data_bean.add_relationship_masked_to_inner_relationships_masked_list(relationship)
                        list_of_validation_mask_entity_id.append(random_entity_id)
                        list_of_validation_mask_reaction_id.append(random_reaction_id)
                        validation_counter = validation_counter + 1
                    else:
                        flag = random.randint(0, 1)
                        if flag == 0:
                            test_data_bean.add_relationship_masked_to_inner_relationships_masked_list(relationship)
                            list_of_test_mask_entity_id.append(random_entity_id)
                            list_of_test_mask_reaction_id.append(random_reaction_id)
                            test_counter = test_counter + 1
                        elif flag == 1:
                            validation_data_bean.add_relationship_masked_to_inner_relationships_masked_list(
                                relationship)
                            list_of_validation_mask_entity_id.append(random_entity_id)
                            list_of_validation_mask_reaction_id.append(random_reaction_id)
                            validation_counter = validation_counter + 1
                    self.__output_link_prediction_delete_relationship(relationship)

        list_of_validation_mask_reaction_id = list(set(list_of_validation_mask_reaction_id))
        list_of_test_mask_reaction_id = list(set(list_of_test_mask_reaction_id))

        relationships_for_validation_data_bean = self.__get_list_of_relationships_based_on_list_of_reaction_ids(
            list_of_validation_mask_reaction_id)

        list_of_entities_for_validation_tmp = self.__get_list_of_entities_based_on_list_of_relationships(
            relationships_for_validation_data_bean)

        list_of_pair_of_entity_and_component_for_validation_data_bean = self.__get_list_of_pair_of_entity_and_component_based_on_list_of_entity_ids(
            list_of_entities_for_validation_tmp)

        relationships_for_test_data_bean = self.__get_list_of_relationships_based_on_list_of_reaction_ids(
            list_of_test_mask_reaction_id)

        list_of_entities_for_test_tmp = self.__get_list_of_entities_based_on_list_of_relationships(
            relationships_for_test_data_bean)

        list_of_pair_of_entity_and_component_for_test_data_bean = self.__get_list_of_pair_of_entity_and_component_based_on_list_of_entity_ids(
            list_of_entities_for_test_tmp)

        validation_data_bean.add_list_of_relationships(relationships_for_validation_data_bean)

        validation_data_bean.add_list_of_pair_of_entity_and_component(
            list_of_pair_of_entity_and_component_for_validation_data_bean)

        test_data_bean.add_list_of_relationships(relationships_for_test_data_bean)

        test_data_bean.add_list_of_pair_of_entity_and_component(list_of_pair_of_entity_and_component_for_test_data_bean)

        train_data_bean.add_list_of_relationships(self.__output_link_prediction_relationships)
        train_data_bean.add_list_of_pair_of_entity_and_component(
            self.__output_link_prediction_list_of_pair_of_entity_and_component)

        train_data_bean.print_sub_data_bean_to_files()
        validation_data_bean.print_sub_data_bean_to_files()
        test_data_bean.print_sub_data_bean_to_files()

        validation_data_bean.generate_and_print_relationships_mix_negative_to_file(10)
        test_data_bean.generate_and_print_relationships_mix_negative_to_file(10)

        train_data_bean.information()
        validation_data_bean.information()
        test_data_bean.information()

        self.__ultimate_initialisation()

    def divide_data_for_regulation_link_prediction_task(self):
        random.seed(1121)
        train_data_bean = DataBeanForReactome(self.__pathway_name, self.__regulation_link_prediction_task,
                                              self.__train_type_divided_dataset, self)
        validation_data_bean = DataBeanForReactome(self.__pathway_name, self.__regulation_link_prediction_task,
                                                   self.__validation_type_divided_dataset, self)
        test_data_bean = DataBeanForReactome(self.__pathway_name, self.__regulation_link_prediction_task,
                                             self.__test_type_divided_dataset, self)
        total_num: int = 0

        for entity_id in self.__regulation_link_prediction_entity_ids:
            list_of_reactions = self.__regulation_link_prediction_entity_to_list_of_regulation_reactions_dict.get(
                entity_id)
            if list_of_reactions is None:
                list_of_reactions = list()
            total_num = total_num + len(list_of_reactions)

        validation_size = test_size = int(total_num / 10)

        length_of_reactions = len(self.__regulation_link_prediction_reactions_ids)

        end_index_of_reactions = length_of_reactions - 1

        validation_counter: int = 0
        test_counter: int = 0

        reaction_id_memory: set[str] = set()
        entity_id_memory_dict: dict[str, int] = dict()

        while validation_counter < validation_size or test_counter < test_size:
            random_reaction_index = random.randint(0, end_index_of_reactions)
            random_reaction_id = self.__regulation_link_prediction_reactions_ids[random_reaction_index]
            list_of_regulation_entities = self.__regulation_link_prediction_reaction_to_list_of_regulation_entities_dict.get(
                random_reaction_id)
            if list_of_regulation_entities is None:
                list_of_regulation_entities = list()

            # if len(list_of_regulation_entities) >= 2 and random_reaction_id not in reaction_id_memory:
            if len(list_of_regulation_entities) >= 2:
                random_entity_index = random.randint(0, len(list_of_regulation_entities) - 1)
                random_entity_id = list_of_regulation_entities[random_entity_index]
                list_of_reactions = self.__regulation_link_prediction_entity_to_list_of_reactions_dict.get(
                    random_entity_id)
                if list_of_reactions is None:
                    list_of_reactions = list()

                # if len(list_of_reactions) >= 2 and (random_entity_id not in entity_id_memory_dict or entity_id_memory_dict[random_entity_id] < 8):
                if len(list_of_reactions) >= 2:
                    if random_entity_id not in entity_id_memory_dict.keys():
                        entity_id_memory_dict[random_entity_id] = 1
                    else:
                        entity_id_memory_dict[random_entity_id] = entity_id_memory_dict.get(random_entity_id) + 1

                    relationship: list[str] = list()
                    relationship.append(random_entity_id)
                    relationship.append(random_reaction_id)
                    relationship.append(str(0))
                    list_of_pair_of_entity_and_component = self.__get_list_of_pair_of_entity_and_component_based_on_entity_id(
                        random_entity_id)

                    if validation_counter >= validation_size:
                        test_data_bean.add_relationship(relationship)
                        test_data_bean.add_list_of_pair_of_entity_and_component(list_of_pair_of_entity_and_component)
                        test_counter = test_counter + 1
                    elif test_counter >= test_size:
                        validation_data_bean.add_relationship(relationship)
                        validation_data_bean.add_list_of_pair_of_entity_and_component(
                            list_of_pair_of_entity_and_component)
                        validation_counter = validation_counter + 1
                    else:
                        flag = random.randint(0, 1)
                        if flag == 0:
                            test_data_bean.add_relationship(relationship)
                            test_data_bean.add_list_of_pair_of_entity_and_component(
                                list_of_pair_of_entity_and_component)
                            test_counter = test_counter + 1
                        elif flag == 1:
                            validation_data_bean.add_relationship(relationship)
                            validation_data_bean.add_list_of_pair_of_entity_and_component(
                                list_of_pair_of_entity_and_component)
                            validation_counter = validation_counter + 1
                    self.__regulation_link_prediction_delete_relationship(relationship)

        train_data_bean.add_list_of_relationships(self.__regulation_link_prediction_relationships)
        train_data_bean.add_list_of_pair_of_entity_and_component(
            self.__regulation_link_prediction_list_of_pair_of_entity_and_component)

        train_data_bean.print_sub_data_bean_to_files()
        validation_data_bean.print_sub_data_bean_to_files()
        test_data_bean.print_sub_data_bean_to_files()

        train_data_bean.information()
        validation_data_bean.information()
        test_data_bean.information()

        self.__ultimate_initialisation()


class DataBeanForReactome:
    def __init__(self, pathway_name: str, task_of_sub_data_set: str, type_of_sub_data_set,
                 raw_data: ReactomeDataDivider):

        self.raw_data = raw_data

        self.__initialise()

        self.__entity_id_index_of_relationship = 0
        self.__reaction_id_index_of_relationship = 1
        self.__direction_index_of_relationship = 2

        self.__entity_index_of_pair_of_entity_and_component = 0
        self.__component_index_of_pair_of_entity_and_component = 1

        self.__edges_file_name = "edges.txt"
        self.__nodes_file_name = "nodes.txt"
        self.__nodes_mask_file_name = "nodes-mask.txt"
        self.__relationship_file_name = "relationship.txt"
        self.__all_components_file_name = "components-all.txt"
        self.__entities_components_mapping_file_name = "components-mapping.txt"
        self.__entities_components_mapping_mix_negative_file_name = "components-mapping-mix-negative.txt"
        self.__relationship_mix_negative_file_name = "relationship-mix-negative.txt"

        self.__entities_component_mapping_masked_file_name = "components-mapping-mask.txt"
        self.__relationships_masked_file_name = "relationship-mask.txt"

        self.__pathway_name = pathway_name
        self.__task_of_sub_data_set = task_of_sub_data_set
        self.__type_of_sub_data_set = type_of_sub_data_set

        self.__file_processor = FileProcessor()

        self.__reaction_to_list_of_entities_dict: dict[str, list[str]] = {}
        self.__reaction_to_list_of_input_entities_dict: dict[str, list[str]] = {}
        self.__reaction_to_list_of_output_entities_dict: dict[str, list[str]] = {}
        self.__reaction_to_list_of_regulation_entities_dict: dict[str, list[str]] = {}

        self.__entity_to_list_of_reactions_dict: dict[str, list[str]] = {}
        self.__entity_to_list_of_input_reactions_dict: dict[str, list[str]] = {}
        self.__entity_to_list_of_output_reactions_dict: dict[str, list[str]] = {}
        self.__entity_to_list_of_regulation_reactions_dict: dict[str, list[str]] = {}

        self.__entity_to_list_of_components_dict: dict[str, list[str]] = {}
        self.__component_to_list_of_entities_dict: dict[str, list[str]] = {}

    def __initialise(self):
        # list of reactions ids
        self.__reactions_ids: list[str] = list()

        # list of entities ids
        self.__entities_ids: list[str] = list()

        # list of components ids
        self.__components_ids: list[str] = list()

        # entity_id, reaction_id, direction
        self.__relationships: list[list[str]] = list()

        # entity_id, component_id
        self.__list_of_pair_of_entity_and_component: list[list[str]] = list()

        # list of (component_id, component_id,.......)
        self.__entities_component_ids_mapping_list: list[list[str]] = list()

        # list of (index_in_raw_data, entity_id)
        self.__train_entity_mask_list: list[str] = list()

        # todo
        self.__relationships_masked_list: list[list[str]] = list()

        self.__pair_of_entity_and_component_masked_list: list[list[str]] = list()

    def add_relationship(self, relationship: list[str]):
        self.__relationships.append(relationship)

    def add_list_of_relationships(self, list_of_relationships: list[list[str]]):
        for relationship in list_of_relationships:
            self.__relationships.append(relationship)

    def add_pair_of_entity_and_component(self, pair_of_entity_and_component: list[str]):
        self.__list_of_pair_of_entity_and_component.append(pair_of_entity_and_component)

    def add_list_of_pair_of_entity_and_component(self, list_of_pair_of_entity_and_component: list[list[str]]):
        for pair_of_entity_and_component in list_of_pair_of_entity_and_component:
            self.add_pair_of_entity_and_component(pair_of_entity_and_component)

    # add the nodes in train that have been masked
    def add_train_entity_id_mask_to_inner_train_entity_mask_list(self, entity_id_mask: str):
        train_entity_mask_list = self.__train_entity_mask_list
        train_entity_mask_set = set(train_entity_mask_list)
        train_entity_mask_set.add(entity_id_mask)

        self.__train_entity_mask_list = list(train_entity_mask_set)

    def add_relationship_masked_to_inner_relationships_masked_list(self, relationship_masked: list[str]):
        self.__relationships_masked_list.append(relationship_masked)

    def add_relationships_masked_list_to_inner_relationships_masked_list(self,
                                                                         relationships_masked_list: list[list[str]]):
        self.__relationships_masked_list.extend(relationships_masked_list)

    def add_pair_of_entity_and_component_masked_list_to_inner_pair_of_entity_and_component_masked_list(self,
                                                                                                       pair_of_entity_and_component_masked_list:
                                                                                                       list[list[str]]):
        self.__pair_of_entity_and_component_masked_list.extend(pair_of_entity_and_component_masked_list)

    def add_pair_of_entity_and_component_masked_to_inner_pair_of_entity_and_component_masked_list(self,
                                                                                                  pair_of_entity_and_component_masked:
                                                                                                  list[str]):
        self.__pair_of_entity_and_component_masked_list.append(pair_of_entity_and_component_masked)

    def __remove_duplicate_relationships(self):
        # PhysicalEntity_id, Reaction_id, 0/1    -a list
        self.__relationships = list(set(tuple(relationship) for relationship in self.__relationships))

        # every edge from tuple to list
        self.__relationships = list(list(relationship) for relationship in self.__relationships)

    def __remove_duplicate_list_of_pair_of_entity_and_component(self):
        # PhysicalEntity_id, Reaction_id, 0/1    -a list
        # You can't allow de-duplication here,
        # because if you turn a list into a set and then into a list, its order must have changed,
        # and if you de-duplicate it, for example, the attributes corresponding to the first node may go somewhere else
        # which is definitely not allowed
        self.__list_of_pair_of_entity_and_component = list(set(
            tuple(pair_of_entity_and_component) for pair_of_entity_and_component in
            self.__list_of_pair_of_entity_and_component))

        # every edge from tuple to list
        self.__list_of_pair_of_entity_and_component = list(
            list(pair_of_entity_and_component) for pair_of_entity_and_component in
            self.__list_of_pair_of_entity_and_component)

    def __build_reactions_ids_and_entities_ids_based_on_relationships(self):
        entities_ids_set = set()
        reactions_ids_set = set()
        for relationship in self.__relationships:
            entity_id = relationship[self.__entity_id_index_of_relationship]
            reaction_id = relationship[self.__reaction_id_index_of_relationship]
            entities_ids_set.add(entity_id)
            reactions_ids_set.add(reaction_id)
        self.__entities_ids = list(entities_ids_set)
        # sort the entities with the sequence of index in raw dataset
        self.sort_entities()
        self.__reactions_ids = list(reactions_ids_set)
        self.sort_reactions()

    def __build_components_ids_based_on_list_of_pair_of_entity_and_component(self):
        components_ids_set = set()
        for pair_of_entity_and_component in self.__list_of_pair_of_entity_and_component:
            component_id = pair_of_entity_and_component[self.__component_index_of_pair_of_entity_and_component]
            components_ids_set.add(component_id)
        self.__components_ids = list(components_ids_set)
        self.sort_components()

    def __build_entities_component_ids_mapping_list_for_print(self):
        # entities_components_mapping_list: list[list[str]] = list()
        entity_to_list_of_components_dict: dict[str, list[str]] = dict()

        for pair_of_entity_and_component in self.__list_of_pair_of_entity_and_component:
            entity_id = pair_of_entity_and_component[self.__entity_index_of_pair_of_entity_and_component]
            component_id = pair_of_entity_and_component[self.__component_index_of_pair_of_entity_and_component]

            # initialise self.__all_entity_to_list_of_components_dict
            if entity_id in entity_to_list_of_components_dict.keys():
                components_list = entity_to_list_of_components_dict[entity_id]
                components_list.append(component_id)
            else:
                components_list = list()
                components_list.append(component_id)
                entity_to_list_of_components_dict[entity_id] = components_list

        for entity_id in self.__entities_ids:
            components = entity_to_list_of_components_dict[entity_id]
            self.__entities_component_ids_mapping_list.append(components)

    # We want to take the entities we have and sort them from smallest to largest according to their index size in the original dataset
    def sort_entities(self):
        raw_entities_ids = copy.deepcopy(self.raw_data.get_raw_entities_ids())
        raw_entity_id_to_entity_index_dict = {entity_id: index for index, entity_id in enumerate(raw_entities_ids)}
        entities_to_be_sorted = copy.deepcopy(self.__entities_ids)
        entity_index_list = [raw_entity_id_to_entity_index_dict.get(entity_id) for entity_id in entities_to_be_sorted]
        entity_index_list.sort()

        self.__entities_ids = [raw_entities_ids[entity_index] for entity_index in entity_index_list]

    def sort_reactions(self):
        raw_reaction_ids = copy.deepcopy(self.raw_data.get_raw_reaction_ids())
        raw_reaction_id_to_reaction_index_dict = {reaction_id: index for index, reaction_id in
                                                  enumerate(raw_reaction_ids)}
        reactions_to_be_sorted = copy.deepcopy(self.__reactions_ids)
        reaction_index_list = [raw_reaction_id_to_reaction_index_dict[reaction_id] for reaction_id in
                               reactions_to_be_sorted]

        reaction_index_list.sort()

        self.__reactions_ids = [raw_reaction_ids[reaction_index] for reaction_index in reaction_index_list]

    def sort_components(self):
        raw_components_ids = copy.deepcopy(self.raw_data.get_raw_component_ids())
        raw_component_id_to_component_index_dict = {component_id: index for index, component_id in
                                                    enumerate(raw_components_ids)}
        components_to_be_sorted = copy.deepcopy(self.__components_ids)

        component_index_list = [raw_component_id_to_component_index_dict[component_id] for component_id in
                                components_to_be_sorted]

        component_index_list.sort()

        self.__components_ids = [raw_components_ids[component_index] for component_index in component_index_list]

    def __complete_the_data_bean(self):
        # self.__remove_duplicate_relationships()
        # self.__remove_duplicate_list_of_pair_of_entity_and_component()
        self.__build_reactions_ids_and_entities_ids_based_on_relationships()
        self.__build_components_ids_based_on_list_of_pair_of_entity_and_component()
        self.__build_entities_component_ids_mapping_list_for_print()

        # example:
        # pathway_name : Metabolism    sub_directory_name : divided_dataset_methodA or dataset_for_attribute_prediction   type_of_sub_data_set : test

    def print_sub_data_bean_to_files(self):
        self.__complete_the_data_bean()

        # reaction_id_to_reaction_index_dict = {reaction_id: index for index, reaction_id in
        #                                       enumerate(self.__reactions_ids)}

        # entity_id_to_entity_index_dict = {entity_id: index for index, entity_id in enumerate(self.__entities_ids)}

        # component_id_to_component_index_dict = {component_id: index for index, component_id in
        #                                         enumerate(self.__components_ids)}

        raw_reaction_id_to_reaction_index_dict = {reaction_id: index for index, reaction_id in
                                                  enumerate(self.raw_data.get_raw_reaction_ids())}

        raw_entity_id_to_entity_index_dict = {entity_id: index for index, entity_id in
                                              enumerate(self.raw_data.get_raw_entities_ids())}

        raw_component_id_to_component_index_dict = {component_id: index for index, component_id in
                                                    enumerate(self.raw_data.get_raw_component_ids())}

        entities_component_indexes_mapping_list_for_print: list[str] = list()

        for component_ids in self.__entities_component_ids_mapping_list:
            component_ids.sort()
            line_component_index_list = ""
            for component_id in component_ids:
                component_index = raw_component_id_to_component_index_dict[component_id]
                component_index = str(component_index)
                line_component_index_list = line_component_index_list + component_index + ","
            # remove the comma in the end
            line_component_index_list = line_component_index_list[:-1]
            entities_component_indexes_mapping_list_for_print.append(line_component_index_list)

        relationships_index_style_for_print: list[str] = list()
        for relationship in self.__relationships:
            # node_index,reaction_index,direction(-1 or 1)
            line_message = ""
            entity_id = relationship[self.__entity_id_index_of_relationship]
            reaction_id = relationship[self.__reaction_id_index_of_relationship]
            direction = relationship[self.__direction_index_of_relationship]

            entity_index = raw_entity_id_to_entity_index_dict[entity_id]
            reaction_index = raw_reaction_id_to_reaction_index_dict[reaction_id]

            line_message = line_message + str(entity_index) + "," + str(reaction_index) + "," + str(direction)

            relationships_index_style_for_print.append(line_message)

        relationships_index_style_for_print.sort(
            key=lambda l: (int(re.findall('\d+', l)[1]), int(re.findall('\d+', l)[0]), int(re.findall('-?\d+', l)[2])))

        index_and_component_id_for_print: list[str] = list()
        index_and_reaction_id_for_print: list[str] = list()
        index_and_entity_id_for_print: list[str] = list()

        for component_id in self.__components_ids:
            index = self.raw_data.get_raw_component_ids().index(component_id)
            index_and_component_id_line: str = str(index) + "," + component_id
            index_and_component_id_for_print.append(index_and_component_id_line)

        # sort the index_and_component_id_for_print via index sequence
        index_and_component_id_for_print.sort(key=lambda l: int(re.findall('\d+', l)[0]))

        for reaction_id in self.__reactions_ids:
            index = self.raw_data.get_raw_reaction_ids().index(reaction_id)
            index_and_reaction_line: str = str(index) + "," + reaction_id
            index_and_reaction_id_for_print.append(index_and_reaction_line)

        # sort the index_and_reaction_id_for_print via index sequence
        index_and_reaction_id_for_print.sort(key=lambda l: int(re.findall('\d+', l)[0]))

        for entity_id in self.__entities_ids:
            index = self.raw_data.get_raw_entities_ids().index(entity_id)
            index_and_entity_line: str = str(index) + "," + entity_id
            index_and_entity_id_for_print.append(index_and_entity_line)

        # sort the index_and_entity_id_for_print via index sequence
        index_and_entity_id_for_print.sort(key=lambda l: int(re.findall('\d+', l)[0]))

        # index and entity_id mask for print

        index_and_entity_id_mask_for_print: list[str] = list()
        for entity_id in self.__train_entity_mask_list:
            index = self.raw_data.get_raw_entities_ids().index(entity_id)
            index_and_entity_mask_line: str = str(index) + "," + entity_id
            index_and_entity_id_mask_for_print.append(index_and_entity_mask_line)

        # sort the index_and_entity_id_for_print via index sequence
        index_and_entity_id_mask_for_print.sort(key=lambda l: int(re.findall('\d+', l)[0]))

        # path = "data/" + self.__pathway_name + "/" + self.__task_of_sub_data_set + "/"
        path = os.path.join("data", self.__pathway_name, self.__task_of_sub_data_set, self.__type_of_sub_data_set)

        self.__file_processor.create_and_write_message_to_file(path, self.__all_components_file_name,
                                                               index_and_component_id_for_print)
        self.__file_processor.create_and_write_message_to_file(path,
                                                               self.__entities_components_mapping_file_name,
                                                               entities_component_indexes_mapping_list_for_print)
        self.__file_processor.create_and_write_message_to_file(path, self.__edges_file_name,
                                                               index_and_reaction_id_for_print)
        self.__file_processor.create_and_write_message_to_file(path, self.__nodes_file_name,
                                                               index_and_entity_id_for_print)
        self.__file_processor.create_and_write_message_to_file(path, self.__relationship_file_name,
                                                               relationships_index_style_for_print)

        if len(self.__train_entity_mask_list) > 0:
            self.__file_processor.create_and_write_message_to_file(path,
                                                                   self.__nodes_mask_file_name,
                                                                   index_and_entity_id_mask_for_print)

        if len(self.__pair_of_entity_and_component_masked_list) > 0:
            entities_component_indexes_mapping_masked_list_for_print = self.generate_pair_of_entity_and_component_masked_to_component_mapping_masked_list(
                raw_component_id_to_component_index_dict, raw_entity_id_to_entity_index_dict)
            entities_component_indexes_mapping_masked_list_for_print.sort(key=lambda l: int(re.findall('\d+', l)[0]))

            self.__file_processor.create_and_write_message_to_file(path,
                                                                   self.__entities_component_mapping_masked_file_name,
                                                                   entities_component_indexes_mapping_masked_list_for_print)

        if len(self.__relationships_masked_list) > 0:
            relationships_masked_index_style_for_print = self.generate_relationships_masked_index_style(
                raw_entity_id_to_entity_index_dict, raw_reaction_id_to_reaction_index_dict)

            self.__file_processor.create_and_write_message_to_file(path, self.__relationships_masked_file_name,
                                                                   relationships_masked_index_style_for_print)

    def generate_relationships_masked_index_style(self, raw_entity_id_to_entity_index_dict,
                                                  raw_reaction_id_to_reaction_index_dict):
        relationships_index_style_for_print: list[str] = list()
        for relationship_mask in self.__relationships_masked_list:
            # node_index,reaction_index,direction(-1 or 1)
            line_message = ""
            entity_id = relationship_mask[self.__entity_id_index_of_relationship]
            reaction_id = relationship_mask[self.__reaction_id_index_of_relationship]
            direction = relationship_mask[self.__direction_index_of_relationship]

            entity_index = raw_entity_id_to_entity_index_dict[entity_id]
            reaction_index = raw_reaction_id_to_reaction_index_dict[reaction_id]

            line_message = line_message + str(entity_index) + "," + str(reaction_index) + "," + str(direction)

            relationships_index_style_for_print.append(line_message)

        relationships_index_style_for_print.sort(
            key=lambda l: (int(re.findall('\d+', l)[1]), int(re.findall('\d+', l)[0]), int(re.findall('-?\d+', l)[2])))

        return relationships_index_style_for_print

    def generate_pair_of_entity_and_component_masked_to_component_mapping_masked_list(self,
                                                                                      raw_component_id_to_component_index_dict,
                                                                                      raw_entity_id_to_entity_index_dict):
        entity_to_list_of_components_masked_dict: dict[str, list[str]] = dict()

        for pair_of_entity_and_component_masked in self.__pair_of_entity_and_component_masked_list:
            entity_id = pair_of_entity_and_component_masked[self.__entity_index_of_pair_of_entity_and_component]
            component_id = pair_of_entity_and_component_masked[self.__component_index_of_pair_of_entity_and_component]

            # initialise self.__all_entity_to_list_of_components_dict
            if entity_id in entity_to_list_of_components_masked_dict.keys():
                components_list = entity_to_list_of_components_masked_dict[entity_id]
                components_list.append(component_id)
            else:
                components_list = list()
                components_list.append(component_id)
                entity_to_list_of_components_masked_dict[entity_id] = components_list

        self.entity_to_list_of_components_masked_dict = entity_to_list_of_components_masked_dict

        entities_component_indexes_mapping_masked_list_for_print: list[str] = list()
        for entity_id, list_of_component_ids_masked in entity_to_list_of_components_masked_dict.items():
            entity_index = raw_entity_id_to_entity_index_dict[entity_id]
            list_of_component_ids_masked.sort()
            line_component_index_list = str(entity_index) + ":"
            for component_id_masked in list_of_component_ids_masked:
                component_index_masked = raw_component_id_to_component_index_dict[component_id_masked]
                component_index_masked = str(component_index_masked)
                line_component_index_list = line_component_index_list + component_index_masked + ","
            # remove the comma in the end
            line_component_index_list = line_component_index_list[:-1]
            entities_component_indexes_mapping_masked_list_for_print.append(line_component_index_list)

        return entities_component_indexes_mapping_masked_list_for_print

    def generate_and_print_components_mapping_mix_negative_to_file(self, num_of_negative_elements: int):
        index_and_entity_id_list: list[str] = list()
        for entity_id in self.__entities_ids:
            index = self.raw_data.get_raw_entities_ids().index(entity_id)
            index_and_entity_line: str = str(index) + "," + entity_id
            index_and_entity_id_list.append(index_and_entity_line)

        # sort the index_and_entity_id_for_print via index sequence
        index_and_entity_id_list.sort(key=lambda l: int(re.findall('\d+', l)[0]))

        entity_id_list = [index_and_entity_id.split(',')[1] for index_and_entity_id in index_and_entity_id_list]

        list_of_components_masked = [self.entity_to_list_of_components_masked_dict[entity_id] for entity_id in entity_id_list]

        entity_to_list_of_components_dict = copy.deepcopy(self.raw_data.get_entity_to_list_of_components_dict())
        raw_component_ids = copy.deepcopy(self.raw_data.get_raw_component_ids())
        raw_component_id_to_component_index_dict = {component_id: index for index, component_id in
                                                    enumerate(raw_component_ids)}

        entities_component_indexes_mapping_list_for_print: list[str] = list()

        for component_ids in list_of_components_masked:
            line_component_index_list = ""
            for component_id in component_ids:
                component_index = raw_component_id_to_component_index_dict[component_id]
                component_index = str(component_index)
                line_component_index_list = line_component_index_list + component_index + ","
            # remove the comma in the end
            line_component_index_list = line_component_index_list[:-1]
            entities_component_indexes_mapping_list_for_print.append(line_component_index_list)

        for index, entity_id in enumerate(self.__entities_ids):
            list_of_components = entity_to_list_of_components_dict[entity_id]
            # Find the difference set
            ret = list(set(raw_component_ids) - set(list_of_components))
            ret.sort()
            # shuffle the ret list, then select n elements
            random.shuffle(ret)
            negative_components_ids = ret[0: num_of_negative_elements]

            negative_components_indexes: list[str] = [raw_component_id_to_component_index_dict[negative_components_id]
                                                      for negative_components_id in negative_components_ids]

            for negative_component_index in negative_components_indexes:
                entities_component_indexes_mapping_list_for_print[index] = \
                    entities_component_indexes_mapping_list_for_print[index] + "||" + str(negative_component_index)

        path = os.path.join("data", self.__pathway_name, self.__task_of_sub_data_set, self.__type_of_sub_data_set)

        self.__file_processor.createFile(path,
                                         self.__entities_components_mapping_mix_negative_file_name)

        self.__file_processor.writeMessageToFile(path,
                                                 self.__entities_components_mapping_mix_negative_file_name,
                                                 entities_component_indexes_mapping_list_for_print)

    def generate_and_print_relationships_mix_negative_to_file(self, num_of_negative_elements: int):
        raw_reaction_to_list_of_entities_dict: dict[str, list[str]] = copy.deepcopy(
            self.raw_data.get_reaction_to_list_of_entities_dict())

        raw_entities_ids: list[str] = copy.deepcopy(self.raw_data.get_raw_entities_ids())

        raw_reactions_ids: list[str] = copy.deepcopy(self.raw_data.get_raw_reaction_ids())

        raw_entity_id_to_entity_index_dict = {entity_id: index for index, entity_id in enumerate(raw_entities_ids)}

        raw_reaction_id_to_reaction_index_dict = {reaction_id: index for index, reaction_id in
                                                  enumerate(raw_reactions_ids)}

        relationships_index_style_for_print: list[str] = list()

        # generate the relationships_index_style, the index is the index position of entities and reactions in raw graph data
        for relationship in self.__relationships_masked_list:
            # node_index,reaction_index,direction(-1 or 1)
            line_message = ""
            entity_id = relationship[self.__entity_id_index_of_relationship]
            reaction_id = relationship[self.__reaction_id_index_of_relationship]
            direction = relationship[self.__direction_index_of_relationship]

            entity_index = raw_entity_id_to_entity_index_dict[entity_id]
            reaction_index = raw_reaction_id_to_reaction_index_dict[reaction_id]

            line_message = line_message + str(entity_index) + "," + str(reaction_index) + "," + str(direction)

            relationships_index_style_for_print.append(line_message)

        relationships_index_style_for_print.sort(
            key=lambda l: (int(re.findall('\d+', l)[1]), int(re.findall('\d+', l)[0]), int(re.findall('-?\d+', l)[2])))

        # generate the relationship list mixed negative
        relationship_index_style_list_mix_negative: list[str] = list()

        for relationship_index_style in relationships_index_style_for_print:
            elements = relationship_index_style.split(",")
            entity_index = int(elements[self.__entity_id_index_of_relationship])
            reaction_index = int(elements[self.__reaction_id_index_of_relationship])

            entity_id = self.raw_data.get_raw_entities_ids()[entity_index]
            reaction_id = self.raw_data.get_raw_reaction_ids()[reaction_index]
            direction = int(elements[self.__direction_index_of_relationship])

            list_of_entities = raw_reaction_to_list_of_entities_dict[reaction_id]

            # Find the difference set
            ret = list(set(raw_entities_ids) - set(list_of_entities))

            ret.sort()

            random.shuffle(ret)

            negative_entities_list: list[str] = ret[0: num_of_negative_elements]

            # generate the negative relationship list
            negative_relationship_list_index_style: list[str] = list()
            for negative_entity_id in negative_entities_list:
                negative_entity_index = raw_entity_id_to_entity_index_dict[negative_entity_id]

                negative_relationship_direction: int = 1 if random.random() < 0.5 else -1

                negative_relationship: str = str(negative_entity_index) + "," + str(reaction_index) + "," + str(
                    negative_relationship_direction)
                negative_relationship_list_index_style.append(negative_relationship)

            relationship_index_style_mix_negative = relationship_index_style

            for negative_relationship_index_style in negative_relationship_list_index_style:
                relationship_index_style_mix_negative = relationship_index_style_mix_negative + "||" + negative_relationship_index_style

            relationship_index_style_list_mix_negative.append(relationship_index_style_mix_negative)

        pre_path = "data/" + self.__pathway_name + "/" + self.__task_of_sub_data_set + "/"

        self.__file_processor.createFile(pre_path + self.__type_of_sub_data_set,
                                         self.__relationship_mix_negative_file_name)

        self.__file_processor.writeMessageToFile(pre_path + self.__type_of_sub_data_set,
                                                 self.__relationship_mix_negative_file_name,
                                                 relationship_index_style_list_mix_negative)

    def __initialisation_inner_reaction_to_list_of_entities_and_entity_to_list_of_reactions_dict(self):
        """ initialise the inner dictionary of reaction to entities and entity to reactions based on different direction
        This method initialise the following inner dictionaries, and will be called by self.__initialisation_set_reactions_entities_and_components_dict(self)
        self.__all_reaction_to_list_of_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_input_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_output_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_regulation_entities_dict: dict[str, list[str]] = {}

        self.__all_entity_to_list_of_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_input_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_output_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_regulation_reactions_dict: dict[str, list[str]] = {}
        :return:
        """

        for relationship in self.__relationships:
            entity_id = relationship[self.__entity_id_index_of_relationship]
            reaction_id = relationship[self.__reaction_id_index_of_relationship]
            direction = relationship[self.__direction_index_of_relationship]

            # general reaction to list of entities
            if reaction_id in self.__reaction_to_list_of_entities_dict.keys():
                entities_list = self.__reaction_to_list_of_entities_dict[reaction_id]
                entities_list.append(entity_id)
            else:
                entities_list = list()
                entities_list.append(entity_id)
                self.__reaction_to_list_of_entities_dict[reaction_id] = entities_list

            # general entity to list of reactions dict
            if entity_id in self.__entity_to_list_of_reactions_dict.keys():
                reactions_list = self.__entity_to_list_of_reactions_dict[entity_id]
                reactions_list.append(reaction_id)
            else:
                reactions_list = list()
                reactions_list.append(reaction_id)
                self.__entity_to_list_of_reactions_dict[entity_id] = reactions_list

            # direction = -1, input
            if int(eval(direction)) < 0:
                # reaction to list of input entities dict
                if reaction_id in self.__reaction_to_list_of_input_entities_dict.keys():
                    entities_list = self.__reaction_to_list_of_input_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    self.__reaction_to_list_of_input_entities_dict[reaction_id] = entities_list

                # entity to list of input reactions dict
                if entity_id in self.__entity_to_list_of_input_reactions_dict.keys():
                    reactions_list = self.__entity_to_list_of_input_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    self.__entity_to_list_of_input_reactions_dict[entity_id] = reactions_list

            # direction = 1, output
            elif int(eval(direction)) > 0:
                # reaction to list of output entities dict
                if reaction_id in self.__reaction_to_list_of_output_entities_dict.keys():
                    entities_list = self.__reaction_to_list_of_output_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    self.__reaction_to_list_of_output_entities_dict[reaction_id] = entities_list

                # entity to list of output reactions dict
                if entity_id in self.__entity_to_list_of_output_reactions_dict.keys():
                    reactions_list = self.__entity_to_list_of_output_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    self.__entity_to_list_of_output_reactions_dict[entity_id] = reactions_list

            # direction = 0, regulation
            else:
                # reaction to list of regulation entities dict
                if reaction_id in self.__reaction_to_list_of_regulation_entities_dict.keys():
                    entities_list = self.__reaction_to_list_of_regulation_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    self.__reaction_to_list_of_regulation_entities_dict[reaction_id] = entities_list

                # entity to list of regulation reactions dict
                if entity_id in self.__entity_to_list_of_regulation_reactions_dict.keys():
                    reactions_list = self.__entity_to_list_of_regulation_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    self.__entity_to_list_of_regulation_reactions_dict[entity_id] = reactions_list

    def __initialisation_inner_entity_and_component_dict(self):
        """
        self.__all_entity_to_list_of_components_dict: dict[str, list[str]] = {}
        self.__all_component_to_list_of_entities_dict: dict[str, list[str]] = {}
        :return:
        """

        for pair_of_entity_and_component in self.__list_of_pair_of_entity_and_component:
            entity_id = pair_of_entity_and_component[self.__entity_index_of_pair_of_entity_and_component]
            component_id = pair_of_entity_and_component[self.__component_index_of_pair_of_entity_and_component]

            # initialise self.__all_entity_to_list_of_components_dict
            if entity_id in self.__entity_to_list_of_components_dict.keys():
                components_list = self.__entity_to_list_of_components_dict[entity_id]
                components_list.append(component_id)
            else:
                components_list = list()
                components_list.append(component_id)
                self.__entity_to_list_of_components_dict[entity_id] = components_list

            # initialise self.__all_component_to_list_of_entities_dict
            if component_id in self.__component_to_list_of_entities_dict.keys():
                entities_list = self.__component_to_list_of_entities_dict[component_id]
                entities_list.append(entity_id)
            else:
                entities_list = list()
                entities_list.append(entity_id)
                self.__component_to_list_of_entities_dict[component_id] = entities_list

    def information(self):
        self.__initialisation_inner_reaction_to_list_of_entities_and_entity_to_list_of_reactions_dict()
        self.__initialisation_inner_entity_and_component_dict()
        print(self.__pathway_name)
        print(self.__task_of_sub_data_set)
        print(self.__type_of_sub_data_set)
        print("num of nodes: " + str(len(self.__entities_ids)))
        print("num of edges: " + str(len(self.__reactions_ids)))
        print("num of components: " + str(len(self.__components_ids)))
        print("num of relationships: " + str(len(self.__relationships)))

        reaction_with_relationship_information = [0, 0, 0, 0, 0, 0]
        reaction_with_input_relationship_information = [0, 0, 0, 0, 0, 0]
        reaction_with_output_relationship_information = [0, 0, 0, 0, 0, 0]
        total_num_of_reactions = len(self.__reactions_ids)

        for list_of_entities in self.__reaction_to_list_of_entities_dict.values():
            length = len(list_of_entities)
            if length > 4:
                reaction_with_relationship_information[5] = reaction_with_relationship_information[5] + 1
            else:
                reaction_with_relationship_information[length] = reaction_with_relationship_information[length] + 1

        print(
            "reaction with one relationship: " + str(reaction_with_relationship_information[1]) + " " + "{:.2%}".format(
                reaction_with_relationship_information[1] / total_num_of_reactions))
        print("reaction with two relationships: " + str(
            reaction_with_relationship_information[2]) + " " + "{:.2%}".format(
            reaction_with_relationship_information[2] / total_num_of_reactions))
        print("reaction with three relationships: " + str(
            reaction_with_relationship_information[3]) + " " + "{:.2%}".format(
            reaction_with_relationship_information[3] / total_num_of_reactions))
        print("reaction with four relationships: " + str(
            reaction_with_relationship_information[4]) + " " + "{:.2%}".format(
            reaction_with_relationship_information[4] / total_num_of_reactions))
        print("reaction with more than four relationships: " + str(
            reaction_with_relationship_information[5]) + " " + "{:.2%}".format(
            reaction_with_relationship_information[5] / total_num_of_reactions))

        for list_of_entities in self.__reaction_to_list_of_input_entities_dict.values():
            length = len(list_of_entities)
            if length > 4:
                reaction_with_input_relationship_information[5] = reaction_with_input_relationship_information[5] + 1
            else:
                reaction_with_input_relationship_information[length] = reaction_with_input_relationship_information[
                                                                           length] + 1

        print("reaction with one input relationship: " + str(
            reaction_with_input_relationship_information[1]) + " " + "{:.2%}".format(
            reaction_with_input_relationship_information[1] / total_num_of_reactions))
        print("reaction with two input relationships: " + str(
            reaction_with_input_relationship_information[2]) + " " + "{:.2%}".format(
            reaction_with_input_relationship_information[2] / total_num_of_reactions))
        print("reaction with three input relationships: " + str(
            reaction_with_input_relationship_information[3]) + " " + "{:.2%}".format(
            reaction_with_input_relationship_information[3] / total_num_of_reactions))
        print("reaction with four input relationships: " + str(
            reaction_with_input_relationship_information[4]) + " " + "{:.2%}".format(
            reaction_with_input_relationship_information[4] / total_num_of_reactions))
        print("reaction with more than four input relationships: " + str(
            reaction_with_input_relationship_information[5]) + " " + "{:.2%}".format(
            reaction_with_input_relationship_information[5] / total_num_of_reactions))

        for list_of_entities in self.__reaction_to_list_of_output_entities_dict.values():
            length = len(list_of_entities)
            if length > 4:
                reaction_with_output_relationship_information[5] = reaction_with_output_relationship_information[5] + 1
            else:
                reaction_with_output_relationship_information[length] = reaction_with_output_relationship_information[
                                                                            length] + 1

        # "  " + "{:.2%}".format(float(num_9 / totol_num)))
        print("reaction with one output relationship: " + str(
            reaction_with_output_relationship_information[1]) + " " + "{:.2%}".format(
            reaction_with_output_relationship_information[1] / total_num_of_reactions))
        print("reaction with two output relationships: " + str(
            reaction_with_output_relationship_information[2]) + " " + "{:.2%}".format(
            reaction_with_output_relationship_information[2] / total_num_of_reactions))
        print("reaction with three output relationships: " + str(
            reaction_with_output_relationship_information[3]) + " " + "{:.2%}".format(
            reaction_with_output_relationship_information[3] / total_num_of_reactions))
        print("reaction with four output relationships: " + str(
            reaction_with_output_relationship_information[4]) + " " + "{:.2%}".format(
            reaction_with_output_relationship_information[4] / total_num_of_reactions))
        print("reaction with more output four output relationships: " + str(
            reaction_with_output_relationship_information[5]) + " " + "{:.2%}".format(
            reaction_with_output_relationship_information[5] / total_num_of_reactions))

        total_num_of_entities = len(self.__entities_ids)
        entity_with_relationships_information = [0, 0, 0, 0, 0, 0]

        for list_of_reactions in self.__entity_to_list_of_reactions_dict.values():
            length = len(list_of_reactions)
            if length > 4:
                entity_with_relationships_information[5] = entity_with_relationships_information[5] + 1
            else:
                entity_with_relationships_information[length] = entity_with_relationships_information[
                                                                    length] + 1

        print("entity with one relationship: " + str(entity_with_relationships_information[1]) + " " + "{:.2%}".format(
            entity_with_relationships_information[1] / total_num_of_entities))
        print("entity with two relationships: " + str(entity_with_relationships_information[2]) + " " + "{:.2%}".format(
            entity_with_relationships_information[2] / total_num_of_entities))
        print(
            "entity with three relationships: " + str(entity_with_relationships_information[3]) + " " + "{:.2%}".format(
                entity_with_relationships_information[3] / total_num_of_entities))
        print(
            "entity with four relationships: " + str(entity_with_relationships_information[4]) + " " + "{:.2%}".format(
                entity_with_relationships_information[4] / total_num_of_entities))
        print("entity with more four relationships: " + str(
            entity_with_relationships_information[5]) + " " + "{:.2%}".format(
            entity_with_relationships_information[5] / total_num_of_entities))

        total_num_of_components = len(self.__components_ids)
        entity_with_components_information = [0, 0, 0, 0, 0, 0]

        for list_of_components in self.__entity_to_list_of_components_dict.values():
            length = len(list_of_components)
            if length > 4:
                entity_with_components_information[5] = entity_with_components_information[5] + 1
            else:
                entity_with_components_information[length] = entity_with_components_information[
                                                                 length] + 1
        print("entity with one component: " + str(entity_with_components_information[1]) + " " + "{:.2%}".format(
            entity_with_components_information[1] / total_num_of_components))
        print("entity with two components: " + str(entity_with_components_information[2]) + " " + "{:.2%}".format(
            entity_with_components_information[2] / total_num_of_components))
        print("entity with three components: " + str(entity_with_components_information[3]) + " " + "{:.2%}".format(
            entity_with_components_information[3] / total_num_of_components))
        print("entity with four components: " + str(entity_with_components_information[4]) + " " + "{:.2%}".format(
            entity_with_components_information[4] / total_num_of_components))
        print("entity with more four components: " + str(entity_with_components_information[5]) + " " + "{:.2%}".format(
            entity_with_components_information[5] / total_num_of_components))

        compoent_with_entity_information = [0, 0, 0, 0, 0, 0]

        for list_of_entities in self.__component_to_list_of_entities_dict.values():
            length = len(list_of_entities)
            if length > 4:
                compoent_with_entity_information[5] = compoent_with_entity_information[5] + 1
            else:
                compoent_with_entity_information[length] = compoent_with_entity_information[
                                                               length] + 1

        print("component with one entity: " + str(compoent_with_entity_information[1]) + " " + "{:.2%}".format(
            compoent_with_entity_information[1] / total_num_of_components))
        print("component with two entities: " + str(compoent_with_entity_information[2]) + " " + "{:.2%}".format(
            compoent_with_entity_information[2] / total_num_of_components))
        print("component with three entities: " + str(compoent_with_entity_information[3]) + " " + "{:.2%}".format(
            compoent_with_entity_information[3] / total_num_of_components))
        print("component with four entities: " + str(compoent_with_entity_information[4]) + " " + "{:.2%}".format(
            compoent_with_entity_information[4] / total_num_of_components))
        print("component with more than four entities: " + str(
            compoent_with_entity_information[5]) + " " + "{:.2%}".format(
            compoent_with_entity_information[5] / total_num_of_components))


class PathwayName(Enum):
    all_data = "All_data_in_Reactome"
    autophagy = "Autophagy"
    cell_cell_communication = "Cell-Cell communication"
    cell_cycle = "Cell Cycle"
    cellular_responses_to_stimuli = "Cellular responses to stimuli"
    chromatin_organization = "Chromatin organization"
    circadian_clock = "Circadian Clock"
    developmental_biology = "Developmental Biology"
    digestion_and_absorption = "Digestion and absorption"
    disease = "Disease"
    DNA_repair = "DNA Repair"
    DNA_replication = "DNA Replication"
    drug_ADME = "Drug ADME"
    extracellular_matrix_organization = "Extracellular matrix organization"
    gene_expression_transcription = "Gene expression (Transcription)"
    hemostasis = "Hemostasis"
    immune_system = "Immune System"
    metabolism = "Metabolism"
    metabolism_of_proteins = "Metabolism of proteins"
    metabolism_of_RNA = "Metabolism of RNA"
    muscle_contraction = "Muscle contraction"
    neuronal_system = "Neuronal System"
    organelle_biogenesis_and_maintenance = "Organelle biogenesis and maintenance"
    programmed_cell_death = "Programmed Cell Death"
    Protein_localization = "Protein localization"
    reproduction = "Reproduction"
    sensory_perception = "Sensory Perception"
    signal_transduction = "Signal Transduction"
    transport_of_small_molecules = "Transport of small molecules"
    vesicle_mediated_transport = "Vesicle-mediated transport"


class DataBean:
    def __init__(self, pathway: str, is_raw_dataset: bool = True, is_divided_dataset: bool = False,
                 is_combination_dataset: bool = True, divided_dataset_task: str = None,
                 divided_dataset_type: str = None):

        self.__entity_id_index_of_relationship = 0
        self.__reaction_id_index_of_relationship = 1
        self.__direction_index_of_relationship = 2

        self.__entity_index_of_pair_of_entity_and_component = 0
        self.__component_index_of_pair_of_entity_and_component = 1

        self.__file_processor = FileProcessor()
        self.__file_name_properties = Properties("file_name.properties")
        self.__pathway_name_path_properties = Properties("pathway_name_path.properties")

        self.__reactions: list[str] = list()
        self.__entities: list[str] = list()
        self.__components: list[str] = list()
        self.__relationships: list[list[str]] = list()
        self.__list_of_pair_of_entity_and_component: list[list[str]] = list()

        self.is_raw_dataset = is_raw_dataset
        self.is_divided_dataset = is_divided_dataset
        self.is_combination_dataset = is_combination_dataset

        self.__divided_dataset_task = divided_dataset_task
        self.__divided_dataset_type = divided_dataset_type

        self.__pathway = pathway
        self.__path = ""

        self.__check_data_bean_status()

        self.__init_path()

        self.__init_inner_elements_from_file()

        # Dictionary of statistical information
        self.__input_relationships: list[list[str]] = list()
        self.__output_relationships: list[list[str]] = list()
        self.__regulation_relationships: list[list[str]] = list()

        self.__reaction_to_list_of_entities_dict: dict[str, list[str]] = {}
        self.__reaction_to_list_of_input_entities_dict: dict[str, list[str]] = {}
        self.__reaction_to_list_of_output_entities_dict: dict[str, list[str]] = {}
        self.__reaction_to_list_of_regulation_entities_dict: dict[str, list[str]] = {}

        self.__entity_to_list_of_reactions_dict: dict[str, list[str]] = {}
        self.__entity_to_list_of_input_reactions_dict: dict[str, list[str]] = {}
        self.__entity_to_list_of_output_reactions_dict: dict[str, list[str]] = {}
        self.__entity_to_list_of_regulation_reactions_dict: dict[str, list[str]] = {}

        self.__entity_to_list_of_components_dict: dict[str, list[str]] = {}
        self.__component_to_list_of_entities_dict: dict[str, list[str]] = {}

        self.__initialisation_inner_reaction_to_list_of_entities_and_entity_to_list_of_reactions_dict()
        self.__initialisation_inner_entity_and_component_dict()

    def __check_data_bean_status(self):
        pathway_name_allowance_list: list[str] = []
        for pathway_name_item in PathwayName:
            pathway_name_allowance_list.append(pathway_name_item)

        if self.is_raw_dataset and self.is_divided_dataset:
            raise Exception(
                "Status Conflict! This Data Bean can't be a raw dataset and a divided dataset at the same time")

        if self.is_raw_dataset and self.is_combination_dataset:
            raise Exception(
                "Status Conflict! This Data Bean can't be a raw dataset and a combination dataset at the same time")

        divided_dataset_task_allowance_list = ["attribute prediction dataset", "input link prediction dataset",
                                               "output link prediction dataset"]
        divided_dataset_type_allowance_list = ["test", "train", "validation"]

        if None is not self.__divided_dataset_task and self.__divided_dataset_task not in divided_dataset_task_allowance_list:
            raise Exception(
                "Your divided dataset task is illegal, the allowed input is one of \"attribute prediction dataset\", \"input link prediction dataset\", \"output link prediction dataset\"")

        if None is not self.__divided_dataset_type and self.__divided_dataset_type not in divided_dataset_type_allowance_list:
            raise Exception(
                "Your divided dataset task is illegal, the allowed input is one of \"test\", \"train\", \"validation\"")

    def __init_path(self):
        import re
        pathway_with_no_space = re.sub(r"\s+", '_', self.__pathway)
        self.__path = self.__pathway_name_path_properties.get(pathway_with_no_space)

        # not found the pathway and its corresponding path, it's defined as a combination one
        if "" == self.__path and True is self.is_combination_dataset:
            self.__path = "data/" + self.__pathway + "/"

        if None is not self.__divided_dataset_task and None is not self.__divided_dataset_type:
            self.__path = self.__path + self.__divided_dataset_task + "/" + self.__divided_dataset_type + "/"

    def __init_inner_elements_from_file(self):
        self.__generate_inner_reactions_from_file()
        self.__generate_inner_entities_from_file()
        self.__generate_inner_components_from_file()
        self.__generate_inner_relationships_from_file()
        self.__generate_inner_pair_of_entity_and_component_from_file()

    def __generate_inner_reactions_from_file(self) -> None:
        reactions_ids_file_name = self.__file_name_properties.get("reactions_ids_file_name")
        self.__reactions = self.__file_processor.read_file_via_lines(self.__path, reactions_ids_file_name)

    # data/Neuronal System/divided_dataset_methodA/test/components-mapping.txt
    def __generate_inner_entities_from_file(self) -> None:
        entities_ids_file_name = self.__file_name_properties.get("entities_ids_file_name")
        self.__entities = self.__file_processor.read_file_via_lines(self.__path, entities_ids_file_name)

    def __generate_inner_components_from_file(self) -> None:
        components_ids_file_name = self.__file_name_properties.get("components_ids_file_name")
        self.__components = self.__file_processor.read_file_via_lines(self.__path, components_ids_file_name)

    def __generate_inner_relationships_from_file(self) -> None:
        relationships_ids_file_name = self.__file_name_properties.get("relationships_ids_file_name")

        relationships_string_style = self.__file_processor.read_file_via_lines(self.__path,
                                                                               relationships_ids_file_name)

        for relationship in relationships_string_style:
            # 13,192,-1.0
            # entity_id_index, reaction_id_index, direction
            # line_of_reaction_id_and_entity_id_and_direction
            elements = relationship.split(",")

            entity_index = elements[self.__entity_id_index_of_relationship]

            if int(entity_index) > len(self.__entities):
                print(entity_index)

            entity_id = self.__entities[int(entity_index)]

            reaction_index = elements[self.__reaction_id_index_of_relationship]
            reaction_id = self.__reactions[int(reaction_index)]

            direction = elements[self.__direction_index_of_relationship]

            line_of_reaction_id_and_entity_id_and_direction: list[str] = list()

            line_of_reaction_id_and_entity_id_and_direction.append(entity_id)
            line_of_reaction_id_and_entity_id_and_direction.append(reaction_id)
            line_of_reaction_id_and_entity_id_and_direction.append(direction)

            # entity_id_index, reaction_id_index, direction
            self.__relationships.append(line_of_reaction_id_and_entity_id_and_direction)

        # list of [entity_id, component_id]

    def __generate_inner_pair_of_entity_and_component_from_file(self) -> None:

        entities_components_association_file_name = self.__file_name_properties.get(
            "entities_components_association_file_name")

        # 355,1190,1209
        list_of_entity_components_mappings_with_index_style = self.__file_processor.read_file_via_lines(
            self.__path, entities_components_association_file_name)

        for i in range(len(list_of_entity_components_mappings_with_index_style)):
            entity_id = self.__entities[i]
            components_str = list_of_entity_components_mappings_with_index_style[i]
            list_of_component_index_str_style = components_str.split(",")

            for component_str in list_of_component_index_str_style:
                line_list_of_entity_id_and_component_id: list[str] = list()
                component_index = int(component_str)

                # if component_index == 0:
                #     print("component index = 0 mapping " + str(i))
                # if component_index == 1:
                #     print("component index = 1 mapping " + str(i))
                # if component_index == 2:
                #     print("component index = 2 mapping " + str(i))

                component_id = self.__components[component_index]
                line_list_of_entity_id_and_component_id.append(entity_id)
                line_list_of_entity_id_and_component_id.append(component_id)
                self.__list_of_pair_of_entity_and_component.append(line_list_of_entity_id_and_component_id)

    def __initialisation_inner_reaction_to_list_of_entities_and_entity_to_list_of_reactions_dict(self):
        """ initialise the inner dictionary of reaction to entities and entity to reactions based on different direction
        This method initialise the following inner dictionaries, and will be called by self.__initialisation_set_reactions_entities_and_components_dict(self)
        self.__all_reaction_to_list_of_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_input_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_output_entities_dict: dict[str, list[str]] = {}
        self.__all_reaction_to_list_of_regulation_entities_dict: dict[str, list[str]] = {}

        self.__all_entity_to_list_of_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_input_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_output_reactions_dict: dict[str, list[str]] = {}
        self.__all_entity_to_list_of_regulation_reactions_dict: dict[str, list[str]] = {}
        :return:
        """

        for relationship in self.__relationships:
            entity_id = relationship[self.__entity_id_index_of_relationship]
            reaction_id = relationship[self.__reaction_id_index_of_relationship]
            direction = relationship[self.__direction_index_of_relationship]

            # general reaction to list of entities
            if reaction_id in self.__reaction_to_list_of_entities_dict.keys():
                entities_list = self.__reaction_to_list_of_entities_dict[reaction_id]
                entities_list.append(entity_id)
            else:
                entities_list = list()
                entities_list.append(entity_id)
                self.__reaction_to_list_of_entities_dict[reaction_id] = entities_list

            # general entity to list of reactions dict
            if entity_id in self.__entity_to_list_of_reactions_dict.keys():
                reactions_list = self.__entity_to_list_of_reactions_dict[entity_id]
                reactions_list.append(reaction_id)
            else:
                reactions_list = list()
                reactions_list.append(reaction_id)
                self.__entity_to_list_of_reactions_dict[entity_id] = reactions_list

            # direction = -1, input
            if int(eval(direction)) < 0:
                # add element to input relationships list
                self.__input_relationships.append(relationship)

                # reaction to list of input entities dict
                if reaction_id in self.__reaction_to_list_of_input_entities_dict.keys():
                    entities_list = self.__reaction_to_list_of_input_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    self.__reaction_to_list_of_input_entities_dict[reaction_id] = entities_list

                # entity to list of input reactions dict
                if entity_id in self.__entity_to_list_of_input_reactions_dict.keys():
                    reactions_list = self.__entity_to_list_of_input_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    self.__entity_to_list_of_input_reactions_dict[entity_id] = reactions_list

            # direction = 1, output
            elif int(eval(direction)) > 0:
                # add element to output relationships list
                self.__output_relationships.append(relationship)

                # reaction to list of output entities dict
                if reaction_id in self.__reaction_to_list_of_output_entities_dict.keys():
                    entities_list = self.__reaction_to_list_of_output_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    self.__reaction_to_list_of_output_entities_dict[reaction_id] = entities_list

                # entity to list of output reactions dict
                if entity_id in self.__entity_to_list_of_output_reactions_dict.keys():
                    reactions_list = self.__entity_to_list_of_output_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    self.__entity_to_list_of_output_reactions_dict[entity_id] = reactions_list

            # direction = 0, regulation
            else:
                # add element to output relationships list
                self.__regulation_relationships.append(relationship)

                # reaction to list of regulation entities dict
                if reaction_id in self.__reaction_to_list_of_regulation_entities_dict.keys():
                    entities_list = self.__reaction_to_list_of_regulation_entities_dict[reaction_id]
                    entities_list.append(entity_id)
                else:
                    entities_list = list()
                    entities_list.append(entity_id)
                    self.__reaction_to_list_of_regulation_entities_dict[reaction_id] = entities_list

                # entity to list of regulation reactions dict
                if entity_id in self.__entity_to_list_of_regulation_reactions_dict.keys():
                    reactions_list = self.__entity_to_list_of_regulation_reactions_dict[entity_id]
                    reactions_list.append(reaction_id)
                else:
                    reactions_list = list()
                    reactions_list.append(reaction_id)
                    self.__entity_to_list_of_regulation_reactions_dict[entity_id] = reactions_list

    def __initialisation_inner_entity_and_component_dict(self):
        """
        self.__all_entity_to_list_of_components_dict: dict[str, list[str]] = {}
        self.__all_component_to_list_of_entities_dict: dict[str, list[str]] = {}
        :return:
        """

        for pair_of_entity_and_component in self.__list_of_pair_of_entity_and_component:
            entity_id = pair_of_entity_and_component[self.__entity_index_of_pair_of_entity_and_component]
            component_id = pair_of_entity_and_component[self.__component_index_of_pair_of_entity_and_component]

            # initialise self.__all_entity_to_list_of_components_dict
            if entity_id in self.__entity_to_list_of_components_dict.keys():
                components_list = self.__entity_to_list_of_components_dict[entity_id]
                components_list.append(component_id)
            else:
                components_list = list()
                components_list.append(component_id)
                self.__entity_to_list_of_components_dict[entity_id] = components_list

            # initialise self.__all_component_to_list_of_entities_dict
            if component_id in self.__component_to_list_of_entities_dict.keys():
                entities_list = self.__component_to_list_of_entities_dict[component_id]
                entities_list.append(entity_id)
            else:
                entities_list = list()
                entities_list.append(entity_id)
                self.__component_to_list_of_entities_dict[component_id] = entities_list

    def information(self):

        self.__initialisation_inner_reaction_to_list_of_entities_and_entity_to_list_of_reactions_dict()
        self.__initialisation_inner_entity_and_component_dict()

        print(self.__pathway)
        if None is not self.__divided_dataset_task:
            print(" | " + self.__divided_dataset_task)
        if None is not self.__divided_dataset_type:
            print("   | " + self.__divided_dataset_type + "\n")
        else:
            print(" | " + "raw dataset" + "\n")
        print("num of nodes: " + str(len(self.__entities)) + "\n")
        print("num of edges: " + str(len(self.__reactions)) + "\n")
        print("num of components: " + str(len(self.__components)) + "\n")
        print("num of relationships: " + str(len(self.__relationships)) + "\n")
        print("num of input relationships: " + str(len(self.__input_relationships)) + "\n")
        print("num of output relationships: " + str(len(self.__output_relationships)) + "\n")
        print("num of regulation relationships: " + str(len(self.__regulation_relationships)) + "\n")

        temp_num_dict: dict[int, str] = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                                         7: "seven", 8: "eight", 9: "more than eight"}

        reaction_with_relationship_information = [0 for x in range(0, 10)]
        reaction_with_input_relationship_information = [0 for x in range(0, 10)]
        reaction_with_output_relationship_information = [0 for x in range(0, 10)]
        reaction_with_regulation_relationship_information = [0 for x in range(0, 10)]
        total_num_of_reactions = len(self.__reactions)

        # all the reactions and their entities
        for list_of_entities in self.__reaction_to_list_of_entities_dict.values():
            length = len(list_of_entities)
            if length > 8:
                reaction_with_relationship_information[9] = reaction_with_relationship_information[9] + 1
            else:
                reaction_with_relationship_information[length] = reaction_with_relationship_information[length] + 1

        # i = 0 -> 9
        for i in range(0, 10):
            print(
                "reaction with " + temp_num_dict[i] + " entities: " + str(
                    reaction_with_relationship_information[i]) + " (" + "{:.2%}".format(
                    reaction_with_relationship_information[i] / total_num_of_reactions) + ")")
        print("\n")

        # reactions and their input entities
        for list_of_entities in self.__reaction_to_list_of_input_entities_dict.values():
            length = len(list_of_entities)
            if length > 8:
                reaction_with_input_relationship_information[9] = reaction_with_input_relationship_information[9] + 1
            else:
                reaction_with_input_relationship_information[length] = reaction_with_input_relationship_information[
                                                                           length] + 1

        # i = 0 -> 9
        for i in range(0, 10):
            print("reaction with " + temp_num_dict[i] + " input entities: " + str(
                reaction_with_input_relationship_information[i]) + " (" + "{:.2%}".format(
                reaction_with_input_relationship_information[i] / total_num_of_reactions) + ")")
        print("\n")

        # reactions and their output entities
        for list_of_entities in self.__reaction_to_list_of_output_entities_dict.values():
            length = len(list_of_entities)
            if length > 8:
                reaction_with_output_relationship_information[9] = reaction_with_output_relationship_information[9] + 1
            else:
                reaction_with_output_relationship_information[length] = reaction_with_output_relationship_information[
                                                                            length] + 1

        # i = 0 -> 9
        for i in range(0, 10):
            print("reaction with " + temp_num_dict[i] + " output entities: " + str(
                reaction_with_output_relationship_information[i]) + " (" + "{:.2%}".format(
                reaction_with_output_relationship_information[i] / total_num_of_reactions) + ")")
        print("\n")

        # reactions and their output entities
        for list_of_entities in self.__reaction_to_list_of_regulation_entities_dict.values():
            length = len(list_of_entities)
            if length > 8:
                reaction_with_regulation_relationship_information[9] = \
                    reaction_with_regulation_relationship_information[9] + 1
            else:
                reaction_with_regulation_relationship_information[length] = \
                    reaction_with_regulation_relationship_information[
                        length] + 1

        # i = 0 -> 9
        for i in range(0, 10):
            print("reaction with " + temp_num_dict[i] + " output entities: " + str(
                reaction_with_regulation_relationship_information[i]) + " (" + "{:.2%}".format(
                reaction_with_regulation_relationship_information[i] / total_num_of_reactions) + ")")
        print("\n")

        total_num_of_entities = len(self.__entities)
        entity_with_relationships_information = [0 for x in range(0, 10)]
        entity_with_input_relationships_information = [0 for x in range(0, 10)]
        entity_with_output_relationships_information = [0 for x in range(0, 10)]
        entity_with_regulation_relationships_information = [0 for x in range(0, 10)]

        # entity to list of reactions
        for list_of_reactions in self.__entity_to_list_of_reactions_dict.values():
            length = len(list_of_reactions)
            if length > 8:
                entity_with_relationships_information[9] = entity_with_relationships_information[9] + 1
            else:
                entity_with_relationships_information[length] = entity_with_relationships_information[
                                                                    length] + 1

        for i in range(0, 10):
            print("entity with " + temp_num_dict[i] + " reactions: " + str(
                entity_with_relationships_information[i]) + " (" + "{:.2%}".format(
                entity_with_relationships_information[i] / total_num_of_entities) + ")")
        print("\n")

        # entity to list of input reactions
        for list_of_reactions in self.__entity_to_list_of_input_reactions_dict.values():
            length = len(list_of_reactions)
            if length > 8:
                entity_with_input_relationships_information[9] = entity_with_input_relationships_information[9] + 1
            else:
                entity_with_input_relationships_information[length] = entity_with_input_relationships_information[
                                                                          length] + 1

        for i in range(0, 10):
            print("entity with " + temp_num_dict[i] + " input reactions: " + str(
                entity_with_input_relationships_information[i]) + " (" + "{:.2%}".format(
                entity_with_input_relationships_information[i] / total_num_of_entities) + ")")
        print("\n")

        # entity to list of output reactions
        for list_of_reactions in self.__entity_to_list_of_output_reactions_dict.values():
            length = len(list_of_reactions)
            if length > 8:
                entity_with_output_relationships_information[9] = entity_with_output_relationships_information[9] + 1
            else:
                entity_with_output_relationships_information[length] = entity_with_output_relationships_information[
                                                                           length] + 1

        for i in range(0, 10):
            print("entity with " + temp_num_dict[i] + " output reactions: " + str(
                entity_with_output_relationships_information[i]) + " (" + "{:.2%}".format(
                entity_with_output_relationships_information[i] / total_num_of_entities) + ")")
        print("\n")

        # entity to list of regulation reactions
        for list_of_reactions in self.__entity_to_list_of_regulation_reactions_dict.values():
            length = len(list_of_reactions)
            if length > 8:
                entity_with_regulation_relationships_information[9] = entity_with_regulation_relationships_information[
                                                                          9] + 1
            else:
                entity_with_regulation_relationships_information[length] = \
                    entity_with_regulation_relationships_information[length] + 1

        for i in range(0, 10):
            print("entity with " + temp_num_dict[i] + " output reactions: " + str(
                entity_with_regulation_relationships_information[i]) + " (" + "{:.2%}".format(
                entity_with_regulation_relationships_information[i] / total_num_of_entities) + ")")
        print("\n")

        entity_with_components_information = [0 for x in range(0, 10)]
        component_with_entities_information = [0 for x in range(0, 10)]

        # entity to list of components
        for list_of_components in self.__entity_to_list_of_components_dict.values():
            length = len(list_of_components)
            if length > 8:
                entity_with_components_information[9] = entity_with_components_information[9] + 1
            else:
                entity_with_components_information[length] = entity_with_components_information[
                                                                 length] + 1

        for i in range(0, 10):
            print("entity with " + temp_num_dict[i] + " components: " + str(
                entity_with_components_information[i]) + " (" + "{:.2%}".format(
                entity_with_components_information[i] / total_num_of_entities) + ")")
        print("\n")

        total_num_of_components = len(self.__components)
        # component to list of entities
        for list_of_entities in self.__component_to_list_of_entities_dict.values():
            length = len(list_of_entities)
            if length > 8:
                component_with_entities_information[9] = component_with_entities_information[9] + 1
            else:
                component_with_entities_information[length] = component_with_entities_information[
                                                                  length] + 1

        for i in range(0, 10):
            print("component with " + temp_num_dict[i] + " entities: " + str(
                component_with_entities_information[i]) + " (" + "{:.2%}".format(
                component_with_entities_information[i] / total_num_of_components) + ")")
        print("\n")


if __name__ == '__main__':
    disease_raw = DataBean("Disease", is_raw_dataset=True, is_divided_dataset=False, is_combination_dataset=False)
    disease_train = DataBean("Disease", is_raw_dataset=False, is_divided_dataset=True, is_combination_dataset=False,
                             divided_dataset_task="input link prediction dataset", divided_dataset_type="train")
    disease_validation = DataBean("Disease", is_raw_dataset=False, is_divided_dataset=True,
                                  is_combination_dataset=False, divided_dataset_task="input link prediction dataset",
                                  divided_dataset_type="validation")
    disease_test = DataBean("Disease", is_raw_dataset=False, is_divided_dataset=True, is_combination_dataset=False,
                            divided_dataset_task="input link prediction dataset", divided_dataset_type="test")

    disease_raw.information()
    disease_train.information()
    disease_validation.information()
    disease_test.information()
