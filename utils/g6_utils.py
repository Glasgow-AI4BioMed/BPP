from typing import List
from app.bean.bean_collection import Edge, Node


class NodeG6:
    def __init__(self, index: int, id: str, name: str, node_type: str):
        self.index: int = index
        self.id: str = id
        self.name: str = name
        # self.x: int = x
        # self.y: int = y
        self.node_type = node_type
        self.is_masked = False
        self.is_deleted = False


class EdgeG6:
    def __init__(self, source: str, target: str):
        self.source: str = source
        self.target: str = target


class DataG6:
    def __init__(self):
        self.nodes: list[NodeG6] = list()
        self.edges: list[EdgeG6] = list()


class G6DataFactoryUtil:

    # @staticmethod
    # def convert_g6_data_to_dict(data: DataG6):
    #     node_dict_list: list[dict] = list()
    #     for node in data.nodes:
    #         node_dict: dict = dict()
    #         node_dict['id'] = node.id
    #         node_dict['x'] = node.x
    #         node_dict['y'] = node.y
    #         node_dict_list.append(node_dict)
    #
    #     edge_dict_list: list[dict] = list()
    #     for edge in data.edges:
    #         edge_dict: dict = dict()

    @staticmethod
    def generate_g6_data_from_edge(edge: Edge) -> DataG6:
        data = DataG6()

        # todo
        theme_node_g6 = NodeG6(edge.index, edge.stId, edge.name, 'reaction')
        data.nodes.append(theme_node_g6)

        for input_node in edge.input_nodes_list:
            # todo
            node_g6 = NodeG6(input_node.index, input_node.stId, input_node.name, 'input_node')
            edge_g6 = EdgeG6(node_g6.id, theme_node_g6.id)
            data.nodes.append(node_g6)
            data.edges.append(edge_g6)

        for output_node in edge.output_nodes_list:
            # todo
            node_g6 = NodeG6(output_node.index, output_node.stId, output_node.name, 'output_node')
            edge_g6 = EdgeG6(theme_node_g6.id, node_g6.id)
            data.nodes.append(node_g6)
            data.edges.append(edge_g6)

        return data

    @staticmethod
    def generate_g6_data_from_edge_list(edge_list: List[Edge]) -> List[DataG6]:
        data_list: list[DataG6] = list()
        for edge in edge_list:
            data: DataG6 = G6DataFactoryUtil.generate_g6_data_from_edge(edge)
            data_list.append(data)

        return data_list

    @staticmethod
    def generate_g6_node_from_input_node(node: Node):
        node_g6 = NodeG6(node.index, node.stId, node.name, 'input_node')
        return node_g6

    @staticmethod
    def generate_g6_node_from_output_node(node: Node):
        node_g6 = NodeG6(node.index, node.stId, node.name, 'output_node')
        return node_g6
