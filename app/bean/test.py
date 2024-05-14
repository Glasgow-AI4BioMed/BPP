from flask import url_for

from app.bean.bean_collection import Edge, Node
from utils.constant_definition import ToplevelPathwayNameEnum

def convert_toplevel_name_url_to_toplevel_name(toplevel_name_url: str) -> str:
    toplevel_name_url = toplevel_name_url.replace('_', '\x20')
    return ' '.join(word[0].upper() + word[1:] for word in toplevel_name_url.split())

if __name__ == '__main__':
    pass
    # edge: Edge = Edge(1, DataSetNameEnum.DISEASE.value)
    #
    # nodeA: Node = Node(2, DataSetNameEnum.DISEASE.value)
    #
    # nodeB: Node = Node(1, DataSetNameEnum.DISEASE.value)
    #
    # resA = edge.add_node_to_inner_input_list(nodeA)
    #
    # resB = edge.add_node_to_inner_input_list(nodeB)
    #
    # print(resA)
    #
    # print(resB)
    print('Error: The association[node index={} and attribute_index={}] is not in the nodes & attributes dict'.format(
        1, 2))

    print(convert_toplevel_name_url_to_toplevel_name('disease'))

    print(convert_toplevel_name_url_to_toplevel_name('immune_system'))

