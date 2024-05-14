from typing import List, Optional

from utils.constant_definition import MessageTextEnum

class ToplevelPathway:
    def __init__(self, index: int, name: str, description: str):
        self.index: int = index
        self.name: str = name
        self.description: str = description
        self.name_url_format: str = name.replace("\x20", "_").lower()


class ToplevelPathwaySelector:
    def __init__(self, list_of_toplevel_pathways: List[ToplevelPathway]):
        self.list_of_toplevel_pathways = list_of_toplevel_pathways

    def select_toplevel_pathway_based_on_name(self, name) -> Optional[ToplevelPathway]:
        for toplevel_pathway in self.list_of_toplevel_pathways:
            if name == toplevel_pathway.name:
                return toplevel_pathway
        return None

    def select_toplevel_pathway_based_on_name_url_format(self, name_url_format) -> Optional[ToplevelPathway]:
        for toplevel_pathway in self.list_of_toplevel_pathways:
            if name_url_format == toplevel_pathway.name_url_format:
                return toplevel_pathway
        return None

class DataBean:
    def __init__(self, index: int, pathway_name: str, stId: str = MessageTextEnum.UNKNOWN.value,
                 name: str = MessageTextEnum.UNKNOWN.value):
        self.index: int = index
        self.pathway_name: str = pathway_name
        self.stId: str = stId
        self.name: str = name
        self.is_masked: bool = False


class Attribute(DataBean):
    def __init__(self, index: int, pathway_name: str, stId: str = MessageTextEnum.UNKNOWN.value,
                 name: str = MessageTextEnum.UNKNOWN.value):
        super().__init__(index, pathway_name, stId, name)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Attribute):
            return (self.index == o.index) and (self.stId == o.stId) and (self.name == o.name) and (
                    self.pathway_name == o.pathway_name)
        return False

    def __hash__(self) -> int:
        return hash(self.index) + hash(self.stId) + hash(self.name) + hash(self.pathway_name)


class Node(DataBean):
    def __init__(self, index: int, pathway_name: str, stId: str = MessageTextEnum.UNKNOWN.value,
                 name: str = MessageTextEnum.UNKNOWN.value):
        super().__init__(index, pathway_name, stId, name)

        self.attributes_list: list[Attribute] = list()

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Node):
            return (self.index == o.index) and (self.stId == o.stId) and (self.name == o.name) and (
                    self.pathway_name == o.pathway_name)
        return False

    def __hash__(self) -> int:
        return hash(self.index) + hash(self.stId) + hash(self.name) + hash(self.pathway_name)

    def add_attribute_to_inner_list(self, attribute: Attribute) -> bool:
        if attribute not in self.attributes_list:
            self.attributes_list.append(attribute)
            return True
        return False


class Edge(DataBean):
    def __init__(self, index: int, pathway_name: str, stId: str = MessageTextEnum.UNKNOWN.value,
                 name: str = MessageTextEnum.UNKNOWN.value):
        super().__init__(index, pathway_name, stId, name)

        self.input_nodes_list: list[Node] = list()
        self.output_nodes_list: list[Node] = list()
        self.regulator_nodes_list: list[Node] = list()

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Edge):
            return (self.index == o.index) and (self.stId == o.stId) and (self.name == o.name) and (
                    self.pathway_name == o.pathway_name)
        return False

    def __hash__(self) -> int:
        return hash(self.index) + hash(self.stId) + hash(self.name) + hash(self.pathway_name)

    def add_node_to_inner_input_list(self, node: Node) -> bool:
        if node not in self.input_nodes_list:
            self.input_nodes_list.append(node)
            return True
        return False

    def add_node_to_inner_output_list(self, node: Node) -> bool:
        if node not in self.output_nodes_list:
            self.output_nodes_list.append(node)
            return True
        return False

    def add_node_to_inner_regulator_list(self, node: Node) -> bool:
        if node not in self.regulator_nodes_list:
            self.regulator_nodes_list.append(node)
            return True
        return False


class Relationship:
    def __init__(self, node_index, edge_index, direction):
        self.node_index: int = node_index
        self.edge_index: int = edge_index
        self.direction: int = direction

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Relationship):
            return (self.node_index == o.node_index) and (self.edge_index == o.edge_index) and (self.direction == o.direction)
        return False

    def __hash__(self) -> int:
        return hash(self.node_index) + hash(self.edge_index) + hash(self.direction)


class PairOfNodeAndAttribute:
    def __init__(self, node_index, attribute_index):
        self.node_index: int = node_index
        self.attribute_index: int = attribute_index

    def __eq__(self, o: object) -> bool:
        if isinstance(o, PairOfNodeAndAttribute):
            return (self.node_index == o.node_index) and (self.attribute_index == o.attribute_index)

    def __hash__(self) -> int:
        return hash(self.node_index) + hash(self.attribute_index)


