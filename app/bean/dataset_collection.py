from typing import Dict
from app.bean.bean_collection import Attribute, Node, Edge, Relationship, PairOfNodeAndAttribute


class Dataset:
    def __init__(self):
        self.attributes_dict: Dict[int, Attribute] = dict()
        self.nodes_dict: Dict[int, Node] = dict()
        self.edges_dict: Dict[int, Edge] = dict()

        self.attributes_list: list[Attribute] = list()
        self.nodes_list: list[Node] = list()
        self.edges_list: list[Edge] = list()

        self.relationships_list: list[Relationship] = list()
        self.pair_of_node_and_attribute_list: list[PairOfNodeAndAttribute] = list()

        self.dataset_message = DataSetTextMessage()
        self.dataset_obs = DataSetObs()

    def add_attribute(self, attribute: Attribute) -> bool:
        if attribute.index not in self.attributes_dict.keys():
            self.attributes_list.append(attribute)
            self.attributes_dict[attribute.index] = attribute

            self.notify_add(attribute)

            return True

        return False

    def add_node(self, node: Node) -> bool:
        if node.index not in self.nodes_dict.keys():
            self.nodes_list.append(node)
            self.nodes_dict[node.index] = node

            self.notify_add(node)

            return True

        return False

    def add_edge(self, edge: Edge) -> bool:
        if edge.index not in self.edges_dict.keys():
            self.edges_list.append(edge)
            self.edges_dict[edge.index] = edge

            self.notify_add(edge)

            return True

        return False

    def add_relationship(self, relationship: Relationship) -> bool:
        self.relationships_list.append(relationship)

        self.notify_add(relationship)

        return True

    def add_pair_of_node_and_attribute(self, pair_of_node_and_attribute: PairOfNodeAndAttribute) -> bool:
        self.pair_of_node_and_attribute_list.append(pair_of_node_and_attribute)

        self.notify_add(pair_of_node_and_attribute)

        return True

    def notify_add(self, obj):
        self.dataset_obs.obs_add_obj(obj)

    def select_attribute_based_on_index(self, index):
        return self.attributes_dict[index]

    def select_node_based_on_index(self, index):
        return self.nodes_dict[index]

    def select_edge_based_on_index(self, index):
        return self.edges_dict[index]

    def get_attribute_list(self):
        return self.attributes_list

    def get_node_list(self):
        return self.nodes_list

    def get_edge_list(self):
        return self.edges_list




class DataSetTextMessage:
    def __init__(self):
        self.pathway_name: str = ""
        # "attribute prediction dataset", "input link prediction dataset", "output link prediction dataset"
        self.task_name: str = ""
        # "train", "validation", "test"
        self.type_name: str = ""

    def initialize(self, **args):
        if "pathway_name" in args.keys():
            self.pathway_name = args["pathway_name"]
        else:
            raise Exception("pathway name is needed!")

        if "task_name" in args.keys():
            self.task_name = args["task_name"]
        else:
            self.task_name = "raw"

        if "type_name" in args.keys():
            self.type_name = args["type_name"]
        else:
            self.type_name = "raw"


class DataSetObs:
    class __PairOfNodeAndComponentObs:
        def __init__(self):
            self.node_to_set_of_attributes_dict: dict[int, set[int]] = dict()
            self.attribute_to_set_of_nodes_dict: dict[int, set[int]] = dict()

        def obs_add_attribute(self, attribute: Attribute):
            attribute_index: int = attribute.index
            if attribute_index not in self.attribute_to_set_of_nodes_dict.keys():
                self.attribute_to_set_of_nodes_dict[attribute_index] = set()

        def obs_add_node(self, node: Node):
            node_index: int = node.index
            if node_index not in self.node_to_set_of_attributes_dict.keys():
                self.node_to_set_of_attributes_dict[node_index] = set()

        def obs_add_pair_of_node_and_attribute(self, pair_of_node_and_attribute: PairOfNodeAndAttribute):
            node_index = pair_of_node_and_attribute.node_index
            attribute_index = pair_of_node_and_attribute.attribute_index

            self.__check_obs_add_or_delete_pair_of_node_and_attribute(pair_of_node_and_attribute)
            self.node_to_set_of_attributes_dict[node_index].add(attribute_index)
            self.attribute_to_set_of_nodes_dict[attribute_index].add(node_index)

        def obs_delete_pair_of_node_and_attribute(self, pair_of_node_and_attribute: PairOfNodeAndAttribute):
            node_index = pair_of_node_and_attribute.node_index
            attribute_index = pair_of_node_and_attribute.attribute_index
            self.node_to_set_of_attributes_dict[node_index].remove(attribute_index)
            self.attribute_to_set_of_nodes_dict[attribute_index].remove(node_index)

        def __check_obs_add_or_delete_pair_of_node_and_attribute(self,
                                                                 pair_of_node_and_attribute: PairOfNodeAndAttribute):
            node_index = pair_of_node_and_attribute.node_index
            attribute_index = pair_of_node_and_attribute.attribute_index
            if node_index not in self.node_to_set_of_attributes_dict.keys():
                raise Exception("The node %d doesn't exist in (node: %d, attribute: %d)", node_index, node_index,
                                attribute_index)
            if attribute_index not in self.attribute_to_set_of_nodes_dict.keys():
                raise Exception("The attribute %d doesn't exist in (node: %d, attribute: %d)", attribute_index,
                                node_index, attribute_index)

            return True

    class __RelationshipObs:
        def __init__(self):
            self.edge_to_set_of_input_nodes_dict: dict[int, set[int]] = dict()
            self.edge_to_set_of_output_nodes_dict: dict[int, set[int]] = dict()
            self.edge_to_set_of_nodes_dict: dict[int, set[int]] = dict()

            self.node_to_set_of_input_edges_dict: dict[int, set[int]] = dict()
            self.node_to_set_of_output_edges_dict: dict[int, set[int]] = dict()
            self.node_to_set_of_edges_dict: dict[int, set[int]] = dict()

            self.edge_to_set_of_regulation_nodes_dict: dict[int, set[int]] = dict()
            self.node_to_set_of_regulation_edges_dict: dict[int, set[int]] = dict()

        def obs_add_node(self, node: Node):
            node_index = node.index
            if node_index not in self.node_to_set_of_input_edges_dict.keys():
                self.node_to_set_of_input_edges_dict[node_index] = set()
            if node_index not in self.node_to_set_of_output_edges_dict.keys():
                self.node_to_set_of_output_edges_dict[node_index] = set()
            if node_index not in self.node_to_set_of_edges_dict.keys():
                self.node_to_set_of_edges_dict[node_index] = set()

            if node_index not in self.node_to_set_of_regulation_edges_dict.keys():
                self.node_to_set_of_regulation_edges_dict[node_index] = set()

        def obs_add_edge(self, edge: Edge):
            edge_index = edge.index
            if edge_index not in self.edge_to_set_of_input_nodes_dict.keys():
                self.edge_to_set_of_input_nodes_dict[edge_index] = set()
            if edge_index not in self.edge_to_set_of_output_nodes_dict.keys():
                self.edge_to_set_of_output_nodes_dict[edge_index] = set()
            if edge_index not in self.edge_to_set_of_nodes_dict.keys():
                self.edge_to_set_of_nodes_dict[edge_index] = set()

            if edge_index not in self.edge_to_set_of_regulation_nodes_dict.keys():
                self.edge_to_set_of_regulation_nodes_dict[edge_index] = set()

        def obs_add_relationship(self, relationship: Relationship):
            node_index = relationship.node_index
            edge_index = relationship.edge_index

            direction = relationship.direction

            self.__check_obs_add_or_delete_relationship(relationship)

            self.edge_to_set_of_nodes_dict[edge_index].add(node_index)
            self.node_to_set_of_edges_dict[node_index].add(edge_index)

            if -1 == direction:
                self.edge_to_set_of_input_nodes_dict[edge_index].add(node_index)
                self.node_to_set_of_input_edges_dict[node_index].add(edge_index)
            elif 1 == direction:
                self.edge_to_set_of_output_nodes_dict[edge_index].add(node_index)
                self.node_to_set_of_output_edges_dict[node_index].add(edge_index)
            elif 0 == direction:
                self.edge_to_set_of_regulation_nodes_dict[edge_index].add(node_index)
                self.node_to_set_of_regulation_edges_dict[node_index].add(edge_index)

        def obs_delete_relationship(self, relationship: Relationship):
            node_index = relationship.node_index
            edge_index = relationship.edge_index

            direction = relationship.direction

            self.__check_obs_add_or_delete_relationship(relationship)

            self.edge_to_set_of_nodes_dict[edge_index].remove(node_index)
            self.node_to_set_of_edges_dict[node_index].remove(edge_index)

            if -1 == direction:
                self.edge_to_set_of_input_nodes_dict[edge_index].remove(node_index)
                self.node_to_set_of_input_edges_dict[node_index].remove(edge_index)
            elif 1 == direction:
                self.edge_to_set_of_output_nodes_dict[edge_index].remove(node_index)
                self.node_to_set_of_output_edges_dict[node_index].remove(edge_index)
            elif 0 == direction:
                self.edge_to_set_of_regulation_nodes_dict[edge_index].remove(node_index)
                self.node_to_set_of_regulation_edges_dict[node_index].remove(edge_index)

        def __check_obs_add_or_delete_relationship(self, relationship: Relationship) -> bool:
            node_index = relationship.node_index
            edge_index = relationship.edge_index

            direction = relationship.direction

            if node_index not in self.node_to_set_of_edges_dict.keys():
                raise Exception("The node %d doesn't exist in (node: %d, edge: %d, direction: %d)", node_index,
                                node_index,
                                edge_index, direction)

            if edge_index not in self.edge_to_set_of_nodes_dict.keys():
                raise Exception("The edge %d doesn't exist in (node: %d, edge: %d, direction: %d)", edge_index,
                                node_index,
                                edge_index, direction)

            if -1 == direction:
                if node_index not in self.node_to_set_of_input_edges_dict.keys():
                    raise Exception("The node %d doesn't exist in (node: %d, edge: %d, direction: %d)", node_index,
                                    node_index,
                                    edge_index, direction)

                if edge_index not in self.edge_to_set_of_input_nodes_dict.keys():
                    raise Exception("The edge %d doesn't exist in (node: %d, edge: %d, direction: %d)", edge_index,
                                    node_index,
                                    edge_index, direction)

            elif 1 == direction:
                if node_index not in self.node_to_set_of_output_edges_dict.keys():
                    raise Exception("The node %d doesn't exist in (node: %d, edge: %d, direction: %d)", node_index,
                                    node_index,
                                    edge_index, direction)

                if edge_index not in self.edge_to_set_of_output_nodes_dict.keys():
                    raise Exception("The edge %d doesn't exist in (node: %d, edge: %d, direction: %d)", edge_index,
                                    node_index,
                                    edge_index, direction)

            elif 0 == direction:
                if node_index not in self.node_to_set_of_regulation_edges_dict.keys():
                    raise Exception("The node %d doesn't exist in (node: %d, edge: %d, direction: %d)", node_index,
                                    node_index,
                                    edge_index, direction)

                if edge_index not in self.edge_to_set_of_regulation_nodes_dict.keys():
                    raise Exception("The edge %d doesn't exist in (node: %d, edge: %d, direction: %d)", edge_index,
                                    node_index,
                                    edge_index, direction)
            return True

    def __init__(self):
        self.pair_of_node_and_component_obs = self.__PairOfNodeAndComponentObs()
        self.relationship_obs = self.__RelationshipObs()

    def obs_add_obj(self, obj):
        if isinstance(obj, Edge):
            self.relationship_obs.obs_add_edge(obj)

        elif isinstance(obj, Node):
            self.relationship_obs.obs_add_node(obj)
            self.pair_of_node_and_component_obs.obs_add_node(obj)

        elif isinstance(obj, Attribute):
            self.pair_of_node_and_component_obs.obs_add_attribute(obj)

        elif isinstance(obj, Relationship):
            self.relationship_obs.obs_add_relationship(obj)

        elif isinstance(obj, PairOfNodeAndAttribute):
            self.pair_of_node_and_component_obs.obs_add_pair_of_node_and_attribute(obj)

    def obs_delete_obj(self, obj):
        if isinstance(obj, Relationship):
            self.relationship_obs.obs_delete_relationship(obj)

        elif isinstance(obj, PairOfNodeAndAttribute):
            self.pair_of_node_and_component_obs.obs_delete_pair_of_node_and_attribute(obj)

    def information_dict(self):
        """
        todo
        :return:
        """
        num_of_attributes: int = len(self.pair_of_node_and_component_obs.attribute_to_set_of_nodes_dict.keys())
        num_of_nodes: int = len(self.pair_of_node_and_component_obs.node_to_set_of_attributes_dict.keys())
        num_of_edges: int = len(self.relationship_obs.edge_to_set_of_nodes_dict.keys())

        num_of_pair_of_entity_and_component: int = 0
        for node, set_of_attributes in self.pair_of_node_and_component_obs.node_to_set_of_attributes_dict.items():
            num_of_pair_of_entity_and_component += len(set_of_attributes)

        num_of_relationships: int = 0
        for edge, set_of_nodes in self.relationship_obs.edge_to_set_of_nodes_dict.items():
            num_of_relationships += len(set_of_nodes)

        num_of_input_relationships: int = 0
        for edge, set_of_input_nodes in self.relationship_obs.edge_to_set_of_input_nodes_dict.items():
            num_of_input_relationships += len(set_of_input_nodes)

        num_of_output_relationships: int = 0
        for edge, set_of_output_nodes in self.relationship_obs.edge_to_set_of_output_nodes_dict.items():
            num_of_output_relationships += len(set_of_output_nodes)

        num_of_regulation_relationships: int = 0
        for edge, set_of_regulation_nodes in self.relationship_obs.edge_to_set_of_regulation_nodes_dict.items():
            num_of_regulation_relationships += len(set_of_regulation_nodes)

        edge_with_dif_num_relationships: list[int] = list()

        edge_with_dif_num_input_relationships_count: list[int] = list()

        edge_with_dif_num_output_relationships_count: list[int] = list()

        edge_with_dif_num_regulation_relationships_count: list[int] = list()
