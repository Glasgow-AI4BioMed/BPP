import json

from flask import request, Response

from app.algorithm import algorithm_bp
from app.bean.bean_collection import Node, Attribute
from app.web.service.attribute_service import attribute_service_obj
from app.web.service.node_service import node_service_obj
from app.algorithm.service.model_service import model_service_obj
from app.web.service.toplevel_pathway_service import toplevel_pathway_service_obj
from utils.constant_definition import ToplevelPathwayNameEnum


@algorithm_bp.route('/', methods=('GET', 'POST'))
def predict():
    direction = request.form.get('direction')

    model = request.form.get('model')

    toplevel_pathway_name = request.form.get('toplevel_pathway')

    # check toplevel_pathway_name
    if toplevel_pathway_name not in [ToplevelPathwayNameEnum.DISEASE.value, ToplevelPathwayNameEnum.METABOLISM.value,
                                     ToplevelPathwayNameEnum.IMMUNE_SYSTEM.value,
                                     ToplevelPathwayNameEnum.SIGNAL_TRANSDUCTION.value]:
        toplevel_pathway_name_url_format = toplevel_pathway_name
        current_toplevel_pathway = toplevel_pathway_service_obj.get_toplevel_pathway_based_on_name_url_format(
            toplevel_pathway_name_url_format)
        toplevel_pathway_name = current_toplevel_pathway.name

    edge_index = int(request.form.get('edge_index'))

    valid_nodes = json.loads(request.form.get('valid_nodes'))

    valid_node_index_list: list[int] = list()
    for valid_node_dict in valid_nodes:
        valid_node_index_list.append(valid_node_dict['index'])

    node_prediction_list: list[Node] = list()

    config = dict()
    config['toplevel_pathway'] = toplevel_pathway_name
    if "input" == direction:
        config['task'] = "input link prediction dataset"
    elif "output" == direction:
        config['task'] = "output link prediction dataset"

    config['edge_or_node_task_level'] = 'edge_level_task'

    config['model_name'] = model

    config['input_or_output_direction'] = direction

    config['select_edge_index'] = edge_index

    model_service_obj.clear_current_predict_status_in_session()

    model_service_obj.store_current_predict_status_to_session(config=config, top_k=15,
                                                              list_of_node_index_for_single_edge=valid_node_index_list)

    # prediction_indexes = model_service_obj.run_prediction_model(config, 10)
    prediction_indexes = model_service_obj.run_prediction_model_on_single_edge(config, 15, valid_node_index_list)

    # prediction_indexes = model_service_obj.run_prediction_model_test()

    # dataset = dataset_dict['disease']
    #
    # edge = dataset.select_edge_based_on_index(0)

    for prediction_index in prediction_indexes:
        node = node_service_obj.get_node_from_dataset_based_on_index(toplevel_pathway_name, prediction_index)
        node_prediction_list.append(node)

    node_prediction_list_json: str = json.dumps(node_prediction_list, default=lambda o: o.__dict__, sort_keys=True,
                                                indent=4)

    return Response(json.dumps(node_prediction_list_json), mimetype='application/json')


@algorithm_bp.route('/explain', methods=('GET', 'POST'))
def explain():
    rank: int = int(request.form.get('rank'))
    
    node_name = str(request.form.get('pred_node_name'))
    
    # print("pred_node_name: " + node_name)

    rank_str: str = str(rank + 1)

    toplevel_pathway_name = request.form.get('toplevel_pathway')

    # check toplevel_pathway_name
    if toplevel_pathway_name not in [ToplevelPathwayNameEnum.DISEASE.value, ToplevelPathwayNameEnum.METABOLISM.value,
                                     ToplevelPathwayNameEnum.IMMUNE_SYSTEM.value,
                                     ToplevelPathwayNameEnum.SIGNAL_TRANSDUCTION.value]:
        toplevel_pathway_name_url_format = toplevel_pathway_name
        current_toplevel_pathway = toplevel_pathway_service_obj.get_toplevel_pathway_based_on_name_url_format(
            toplevel_pathway_name_url_format)
        toplevel_pathway_name = current_toplevel_pathway.name

    config, top_k, list_of_node_index_for_single_edge = model_service_obj.load_current_predict_status_from_session()

    explain = None

    try:
        explain = model_service_obj.run_gnn_explainer(config=config, rank=rank,
                                                  list_of_node_index_for_single_edge=list_of_node_index_for_single_edge)
    except Exception as e:
        print("An exception has occurred: ", e)

    print("The GNN explainer is done.")

    explain = model_service_obj.softmax_weight_of_explain(explain)

    print("The softmax operation on GNN explainer is done.")

    explain_res = list()

    enhanced_attributes_dict: dict[int, list[dict]] = dict()
    enhanced_node_dict: dict[int, dict] = dict()

    node_element_list: list = list()

    for item in explain:
        name = item[0]
        weight = item[1]

        elements = name.split('_')

        if 'attr' == elements[0]:
            attribute_index: int = int(elements[2])
            node_index: int = int(elements[1])
            attribute: Attribute = attribute_service_obj.get_attribute_from_dataset_based_on_index(
                toplevel_pathway_name, attribute_index)

            if node_index not in enhanced_attributes_dict.keys():
                enhanced_attributes_dict[node_index] = list()
            enhanced_attributes_dict[node_index].append({'element': attribute, 'weight': weight, 'type': 'attribute'})

            explain_res.append({'element': attribute, 'weight': weight, 'type': 'attribute'})

        elif 'link' == elements[0]:
            node_index: int = int(elements[1])
            node: Node = node_service_obj.get_node_from_dataset_based_on_index(toplevel_pathway_name, node_index)
            enhanced_node_dict[node_index] = {'element': node, 'weight': weight, 'type': 'node'}
    
    for node_index, enhanced_node in enhanced_node_dict.items():
        if len(enhanced_node["element"].attributes_list) == 1 and enhanced_node["element"].name == enhanced_node["element"].attributes_list[0].name:
            attribute_name = enhanced_node["element"].attributes_list[0].name
            if node_index in enhanced_attributes_dict:
                attributes_list = enhanced_attributes_dict[node_index]
                                
                enhanced_attributes_dict[node_index] = [
                    attr for attr in attributes_list if (attr["element"].name != attribute_name) and attr["element"].name != node_name
                ]
    
    for node_index, enhanced_attributes in enhanced_attributes_dict.items():
        enhanced_attributes_dict[node_index] = [
                attr for attr in enhanced_attributes if attr["element"].name != node_name
            ]
        
                    

    list_dict_for_ranking: list[dict] = list()

    for node_index, list_of_attributes_dict in enhanced_attributes_dict.items():
        for attribute_dict in list_of_attributes_dict:
            list_dict_for_ranking.append(attribute_dict)

    for node_index, node_dict in enhanced_node_dict.items():
        list_dict_for_ranking.append(node_dict)

    sorted_data = sorted(list_dict_for_ranking, key=lambda x: x['weight'], reverse=True)

    print("The sort operation on GNN explainer is done.")

    # 为每个字典添加 ranking 键值对
    for index, item in enumerate(sorted_data, start=1):
        item['ranking'] = index

    print("The adding ranking element operation on GNN explainer is done.")

    for node_index, enhanced_node in enhanced_node_dict.items():
        node_element = {'element': enhanced_node['element'], 'weight': enhanced_node['weight'],
                        'ranking': enhanced_node['ranking'],
                        'type': enhanced_node['type'],
                        'children': enhanced_attributes_dict[node_index]}
        node_element_list.append(node_element)

    data = {'explain_res': explain_res, 'rank': rank_str, "node_element_list": node_element_list,
            "sorted_nodes_and_attributes": sorted_data}

    print("The final data of GNN explainer is done.")

    data_json: str = json.dumps(data, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    # print(explain_res_json)

    return Response(json.dumps(data_json), mimetype='application/json')
