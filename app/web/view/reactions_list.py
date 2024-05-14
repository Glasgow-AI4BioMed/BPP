import json

from flask import render_template, request, jsonify, Response

from app.algorithm.service.model_service import model_service_obj
from app.bean import dataset_dict
from app.bean.bean_collection import Attribute, Node, Edge, ToplevelPathway
from app.bean.dataset_factory import ToplevelPathwayFactory
from app.web import web_bp
from app.web.service.attribute_service import attribute_service_obj
from app.web.service.edge_service import edge_service_obj
from app.web.service.node_service import node_service_obj
from app.web.service.toplevel_pathway_service import toplevel_pathway_service_obj
from app.web.service.page_service import PageNavigation, PageInf
from utils.g6_utils import G6DataFactoryUtil, NodeG6


@web_bp.route('/<toplevel_pathway_name_url_format>/reactions', methods=('GET', 'POST'))
def show_reactions(toplevel_pathway_name_url_format):
    allow_list: list[str] = ["disease", "immune_system", "metabolism", "signal_transduction"]

    all_toplevel_pathways: list[ToplevelPathway] = toplevel_pathway_service_obj.get_all_toplevel_pathways()

    if toplevel_pathway_name_url_format in allow_list:
        page: int = int(request.args.get('page'))

        current_toplevel_pathway = toplevel_pathway_service_obj.get_toplevel_pathway_based_on_name_url_format(
            toplevel_pathway_name_url_format)

        # toplevel_pathway_name = toplevel_pathway_service_obj.convert_toplevel_name_url_to_toplevel_name(
        #     toplevel_pathway_name_url_format)

        # attributes_list = attribute_service_obj.get_attribute_list_from_dataset(toplevel_pathway_name)
        #
        # nodes_list = node_service_obj.get_node_list_from_dataset(toplevel_pathway_name)

        edges_list = edge_service_obj.get_edge_list_from_dataset(current_toplevel_pathway.name)

        edge_page_navigation = PageNavigation(edges_list, page_size=4)

        edges_list_for_view, page_info = edge_page_navigation.page_navigate(page)

        # for edge_for_view in edges_list_for_view:
        #     if isinstance(edge_for_view, Edge):
        #         a = G6DataFactoryUtil.generate_g6_data_from_edge(edge_for_view)

        g6_data_list = G6DataFactoryUtil.generate_g6_data_from_edge_list(edges_list_for_view)

        g6_data_list_json = json.dumps(g6_data_list, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        # g6_data_list_json = jsonify(g6_data_list, default=lambda o: o.__dict__, sort_keys=True, indent=4)

        # print(g6_data_list_json)

        # category_name_show = {"disease": "Disease", "immune_system": "Immune System", "metabolism": "Metabolism",
        #                       "signal_transduction": "Signal Transduction"}
        return render_template('reaction_list.html', current_toplevel_pathway=current_toplevel_pathway,
                               edges_list=edges_list_for_view, page_info=page_info,
                               all_toplevel_pathways=all_toplevel_pathways, g6_data_list_json=
                               g6_data_list_json)


@web_bp.route('/search', methods=('GET', 'POST'))
def search():
    all_toplevel_pathways: list[ToplevelPathway] = toplevel_pathway_service_obj.get_all_toplevel_pathways()

    page: int = 1

    content = request.form.get('content')
    edge_list1 = edge_service_obj.get_edge_from_dataset_based_on_name("Disease", content)
    edge_list2 = edge_service_obj.get_edge_from_dataset_based_on_name("Immune System", content)
    edge_list3 = edge_service_obj.get_edge_from_dataset_based_on_name("Metabolism", content)
    edge_list4 = edge_service_obj.get_edge_from_dataset_based_on_name("Signal Transduction", content)

    edge_list5 = edge_service_obj.get_edge_from_dataset_based_on_stId("Disease", content)
    edge_list6 = edge_service_obj.get_edge_from_dataset_based_on_stId("Immune System", content)
    edge_list7 = edge_service_obj.get_edge_from_dataset_based_on_stId("Metabolism", content)
    edge_list8 = edge_service_obj.get_edge_from_dataset_based_on_stId("Signal Transduction", content)

    all_edge_list: list[Edge] = list()
    all_edge_list.extend(edge_list1)
    all_edge_list.extend(edge_list2)
    all_edge_list.extend(edge_list3)
    all_edge_list.extend(edge_list4)
    all_edge_list.extend(edge_list5)
    all_edge_list.extend(edge_list6)
    all_edge_list.extend(edge_list7)
    all_edge_list.extend(edge_list8)

    edge_page_navigation = PageNavigation(all_edge_list, page_size=4)

    edges_list_for_view, page_info = edge_page_navigation.page_navigate(page)

    for edge_for_view in edges_list_for_view:
        if isinstance(edge_for_view, Edge):
            G6DataFactoryUtil.generate_g6_data_from_edge(edge_for_view)

    g6_data_list = G6DataFactoryUtil.generate_g6_data_from_edge_list(edges_list_for_view)
    g6_data_list_json = json.dumps(g6_data_list, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    return render_template('reaction_list.html', current_toplevel_pathway="Disease",
                           edges_list=edges_list_for_view, page_info=page_info,
                           all_toplevel_pathways=all_toplevel_pathways, g6_data_list_json=g6_data_list_json)


@web_bp.route('/reaction/customize', methods=('GET', 'POST'))
def customize_reaction():
    dataset = request.args.get('dataset')
    all_toplevel_pathways: list[ToplevelPathway] = toplevel_pathway_service_obj.get_all_toplevel_pathways()
    return render_template('reaction_customize.html', all_toplevel_pathways=all_toplevel_pathways, dataset=dataset)


@web_bp.route('/reaction/customize/get_reactions', methods=('GET', 'POST'))
def get_all_reactions_of_toplevel_pathway():
    # todo
    toplevel_pathway_name_url_format = request.form.get('toplevel_pathway')
    current_toplevel_pathway = toplevel_pathway_service_obj.get_toplevel_pathway_based_on_name_url_format(
        toplevel_pathway_name_url_format)

    edge_list: list[Edge] = edge_service_obj.get_edge_list_from_dataset(current_toplevel_pathway.name)
    edge_view_list: list[Edge] = list()
    
    # from pdb import set_trace; set_trace()

    for edge in edge_list:
        edge_view = Edge(index=edge.index, pathway_name=current_toplevel_pathway.name, stId=edge.stId, name=edge.name)
        edge_view_list.append(edge_view)

    edge_view_list_json = json.dumps(edge_view_list, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    
    # from pdb import set_trace; set_trace()

    return Response(edge_view_list_json, mimetype='application/json')


@web_bp.route('/reaction/customize/get_nodes', methods=('GET', 'POST'))
def get_all_nodes_of_toplevel_pathway():
    toplevel_pathway_name_url_format = request.form.get('toplevel_pathway')
    current_toplevel_pathway = toplevel_pathway_service_obj.get_toplevel_pathway_based_on_name_url_format(
        toplevel_pathway_name_url_format)

    node_list: list[Node] = node_service_obj.get_node_list_from_dataset(current_toplevel_pathway.name)
    node_view_list: list[Node] = list()

    for node in node_list:
        node_view = Node(index=node.index, pathway_name=current_toplevel_pathway.name, stId=node.stId, name=node.name)
        node_view_list.append(node_view)

    node_view_list_json = json.dumps(node_view_list, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    return Response(node_view_list_json, mimetype='application/json')


@web_bp.route('/reaction/customize/get_single_reaction', methods=('GET', 'POST'))
def get_reaction_of_toplevel_pathway_based_on_index():
    toplevel_pathway_name_url_format = request.form.get('toplevel_pathway')
    selected_reaction_index = request.form.get('selected_reaction_index')
    selected_reaction_index = int(selected_reaction_index)
    
    # from pdb import set_trace; set_trace()

    current_toplevel_pathway = toplevel_pathway_service_obj.get_toplevel_pathway_based_on_name_url_format(
        toplevel_pathway_name_url_format)

    edge = edge_service_obj.get_edge_from_dataset_based_on_index(current_toplevel_pathway.name, selected_reaction_index)

    g6_data = G6DataFactoryUtil.generate_g6_data_from_edge(edge)

    g6_data_json = json.dumps(g6_data, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    return Response(g6_data_json, mimetype='application/json')

@web_bp.route('/reaction/customize/generate_new_reaction', methods=('GET', 'POST'))
def generate_new_reaction_of_toplevel_pathway():
    toplevel_pathway_name_url_format = request.form.get('toplevel_pathway')

    current_toplevel_pathway = toplevel_pathway_service_obj.get_toplevel_pathway_based_on_name_url_format(
        toplevel_pathway_name_url_format)

    edge = edge_service_obj.generate_new_edge(toplevel_pathway_name=current_toplevel_pathway.name, stId="CUSTOMISED-REACTION", name="Customised Reactions")

    g6_data = G6DataFactoryUtil.generate_g6_data_from_edge(edge)

    g6_data_json = json.dumps(g6_data, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    return Response(g6_data_json, mimetype='application/json')


@web_bp.route('/reaction/customize/get_single_node', methods=('GET', 'POST'))
def get_node_of_toplevel_pathway_based_on_index():
    toplevel_pathway_name_url_format = request.form.get('toplevel_pathway')
    selected_node_index = int(request.form.get('selected_node_index'))
    direction = request.form.get('direction')

    current_toplevel_pathway = toplevel_pathway_service_obj.get_toplevel_pathway_based_on_name_url_format(
        toplevel_pathway_name_url_format)

    node = node_service_obj.get_node_from_dataset_based_on_index(current_toplevel_pathway.name, selected_node_index)
    node_g6 = None
    if direction == "input":
        node_g6 = NodeG6(node.index, node.stId, node.name, 'input_node')
    elif direction == "output":
        node_g6 = NodeG6(node.index, node.stId, node.name, 'output_node')

    node_g6_json = json.dumps(node_g6, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    return Response(node_g6_json, mimetype='application/json')



@web_bp.route('/<toplevel_pathway_name_url_format>/reaction/<int:edge_index>', methods=('GET', 'POST'))
def show_single_reaction(toplevel_pathway_name_url_format: str, edge_index: int):
    allow_list: list[str] = ["disease", "immune_system", "metabolism", "signal_transduction"]

    all_toplevel_pathways: list[ToplevelPathway] = toplevel_pathway_service_obj.get_all_toplevel_pathways()

    node_prediction_list: list[Node] = list()

    prediction_indexes = model_service_obj.predict({})

    dataset = dataset_dict['Disease']

    edge = dataset.select_edge_based_on_index(0)

    edge_prediction = Edge(0, 'disease', 'test_stid', 'name')

    for prediction_index in prediction_indexes:
        node = node_service_obj.get_node_from_dataset_based_on_index('Disease', prediction_index)
        node_prediction_list.append(node)

    if toplevel_pathway_name_url_format in allow_list:
        current_toplevel_pathway = toplevel_pathway_service_obj.get_toplevel_pathway_based_on_name_url_format(
            toplevel_pathway_name_url_format)

        current_edge = edge_service_obj.get_edge_from_dataset_based_on_index(current_toplevel_pathway.name, edge_index)
        # category_name_show = {"disease": "Disease", "immune_system": "Immune System", "metabolism": "Metabolism",
        #                       "signal_transduction": "Signal Transduction"}
        return render_template('reaction_customize.html', current_toplevel_pathway=current_toplevel_pathway,
                               current_edge=current_edge,
                               all_toplevel_pathways=all_toplevel_pathways, node_prediction_list=node_prediction_list)


def test_generate_items():
    a1 = Attribute(0, "disease", "R-COV-9694459", "14-sugar N-glycan unfolded Spike")
    a2 = Attribute(1, "disease", "R-ALL-1217506", "17-AAG")
    a3 = Attribute(2, "disease", "R-ALL-1217507", "17-DMAG")
    a4 = Attribute(3, "disease", "R-HSA-72391", "18S rRNA")
    a5 = Attribute(4, "disease", "R-ALL-880042", "2HG")
    a6 = Attribute(5, "disease", "R-ALL-29406", "2OG")

    n1 = Node(0, "disease", "R-HSA-8936661", "'receiver' RAF")
    n2 = Node(1, "disease", "R-HSA-8936661", "1-LTR form of circular viral DNA")
    n3 = Node(2, "disease", "R-HSA-8936661", "14-3-3 dimer")
    n4 = Node(3, "disease", "R-HSA-8936661", "14-3-3 dimer:N")
    n5 = Node(4, "disease", "R-HSA-8936661", "14-sugar N-glycan unfolded Spike")
    n6 = Node(5, "disease", "R-HSA-8936661", "2-LTR form of circular viral DNA")

    e1 = Edge(0, "disease", "R-HSA-2029466", "(WASPs, WAVE):G-actin:ARP2/3 binds F-actin")
    e2 = Edge(1, "disease", "R-HSA-175117", "1-LTR circle formation")
    e3 = Edge(2, "disease", "R-HSA-167076",
              "2-4 nt.backtracking of Pol II complex on the HIV-1 template leading to elongation pausing")
    e4 = Edge(3, "disease", "R-HSA-175258", "2-LTR formation due to circularization of viral DNA")

    n1.add_attribute_to_inner_list(a1)
    n1.add_attribute_to_inner_list(a2)
    n1.add_attribute_to_inner_list(a3)

    n2.add_attribute_to_inner_list(a2)
    n2.add_attribute_to_inner_list(a4)

    n3.add_attribute_to_inner_list(a1)
    n3.add_attribute_to_inner_list(a3)
    n3.add_attribute_to_inner_list(a5)

    n3.add_attribute_to_inner_list(a5)
    n3.add_attribute_to_inner_list(a6)

    n4.add_attribute_to_inner_list(a3)
    n4.add_attribute_to_inner_list(a4)

    n5.add_attribute_to_inner_list(a1)
    n5.add_attribute_to_inner_list(a2)
    n5.add_attribute_to_inner_list(a5)
    n5.add_attribute_to_inner_list(a6)

    n6.add_attribute_to_inner_list(a3)
    n6.add_attribute_to_inner_list(a6)

    e1.add_node_to_inner_input_list(n1)
    e1.add_node_to_inner_input_list(n2)

    e1.add_node_to_inner_output_list(n3)
    e1.add_node_to_inner_output_list(n4)

    e2.add_node_to_inner_input_list(n2)
    e2.add_node_to_inner_input_list(n6)
    e2.add_node_to_inner_output_list(n3)

    e3.add_node_to_inner_input_list(n4)
    e3.add_node_to_inner_input_list(n5)
    e3.add_node_to_inner_input_list(n6)
    e3.add_node_to_inner_output_list(n2)
    e3.add_node_to_inner_output_list(n3)

    e4.add_node_to_inner_input_list(n1)
    e4.add_node_to_inner_input_list(n4)
    e4.add_node_to_inner_output_list(n6)

    attributes_list: list[Attribute] = list()
    attributes_list.append(a1)
    attributes_list.append(a2)
    attributes_list.append(a3)
    attributes_list.append(a4)
    attributes_list.append(a5)
    attributes_list.append(a6)

    nodes_list: list[Node] = list()
    nodes_list.append(n1)
    nodes_list.append(n2)
    nodes_list.append(n3)
    nodes_list.append(n4)
    nodes_list.append(n5)
    nodes_list.append(n6)

    edges_list: list[Edge] = list()
    edges_list.append(e1)
    edges_list.append(e2)
    edges_list.append(e3)
    edges_list.append(e4)

    return attributes_list, nodes_list, edges_list
