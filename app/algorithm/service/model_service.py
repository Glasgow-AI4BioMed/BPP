import random
from typing import List

import numpy as np
import torch
from dhg import Graph, Hypergraph
from dhg.models import GCN, HGNN, HGNNP
from flask import session
from scipy.special import softmax

from app.bean import model_selector
from utils import utils
from utils.constant_definition import ModelPathEnum
from utils.data_loader import DataLoaderLink
from utils.gnn_explainer_utils import HGNNExplainer


class ModelService:

    def run_prediction_model(self, config, top_k: int) -> List[int]:
        # set device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        data_loader = DataLoaderLink(config['toplevel_pathway'], config['task'])

        # the GCN model
        # model_path_root: str = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'app', 'static', 'models')
        # raw_data_path: str = os.path.join('app', 'static', 'data', config.toplevel_pathway, config.task)

        model_path = model_selector.select_model_path(config['edge_or_node_task_level'], config['model_name'],
                                                      config['toplevel_pathway'], config['input_or_output_direction'])

        emb_size = model_selector.select_model_hyper_parameter_emb_size(config['edge_or_node_task_level'], config['model_name'],
                                                      config['toplevel_pathway'], config['input_or_output_direction'])

        model_dict = {ModelPathEnum.GCN_MODEL_NAME.value: GCN, ModelPathEnum.HGNN_MODEL_NAME.value: HGNN, ModelPathEnum.HGNNP_MODEL_NAME.value: HGNNP}

        net_model = model_dict[config['model_name']](
            data_loader["num_features"], emb_size, data_loader["num_features"], use_bn=True
        )

        net_model.load_state_dict(torch.load(model_path, map_location=device))

        net_model.eval()

        # get the total number of nodes of this graph
        num_of_nodes: int = data_loader["num_nodes"]

        train_nodes_features = torch.FloatTensor(data_loader["train_nodes_features"])

        # generate the relationship between hyper edge and nodes
        # ex. [[1,2,3,4], [3,4], [9,7,4]...] where [1,2,3,4] represent a hyper edge
        train_all_hyper_edge_list = data_loader["train_edge_list"]
        train_hyper_edge_list = data_loader["train_masked_edge_list"]

        # train_edge_mask = data_loader["train_edge_mask"]

        # the hyper graph
        hyper_graph = Hypergraph(num_of_nodes, train_all_hyper_edge_list)

        # generate graph based on hyper graph or just use hyper graph
        if config['model_name'] in [ModelPathEnum.GCN_MODEL_NAME.value]:
            graph = Graph.from_hypergraph_clique(hyper_graph, weighted=True)
        else:
            graph = hyper_graph

        # set device
        train_nodes_features = train_nodes_features.to(device)
        graph = graph.to(device)
        # net_model = net_model.to(device)

        outs = self.__prediction_assist(
            net_model,
            train_nodes_features,
            train_hyper_edge_list,
            graph
        )

        selected_edge_index = config['select_edge_index']
        selected_edge_predictions = outs[selected_edge_index]

        top_k_selected_edge_predictions_value, top_k_selected_edge_predictions_index = torch.topk(
            selected_edge_predictions, top_k)
        
        # from pdb import set_trace; set_trace()
        # print(top_k_selected_edge_predictions_value)

        top_k_selected_edge_predictions_index = top_k_selected_edge_predictions_index.cpu().numpy()

        return top_k_selected_edge_predictions_index.tolist()

    @torch.no_grad()
    def __prediction_assist(
            self,
            net_model,
            nodes_features,
            hyper_edge_list: List[List[int]],
            graph,
    ):
        net_model.eval()
        # [[1,2,3],[2,3,4,5]...]
        edges_embeddings = (
            utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
                hyper_edge_list, nodes_features
            )
        )

        # edges_embeddings = edges_embeddings.to(net_model.device)

        nodes_embeddings = net_model(nodes_features, graph)

        outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

        utils.filter_prediction_(outs, hyper_edge_list)

        return outs


    def run_prediction_model_on_single_edge(self, config, top_k: int, list_of_node_index_for_single_edge: List[int]) -> List[int]:
        # set device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        data_loader = DataLoaderLink(config['toplevel_pathway'], config['task'])
        # the GCN model
        # model_path_root: str = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'app', 'static', 'models')
        # raw_data_path: str = os.path.join('app', 'static', 'data', config.toplevel_pathway, config.task)

        model_path = model_selector.select_model_path(config['edge_or_node_task_level'], config['model_name'],
                                                      config['toplevel_pathway'], config['input_or_output_direction'])

        emb_size = model_selector.select_model_hyper_parameter_emb_size(config['edge_or_node_task_level'], config['model_name'],
                                                      config['toplevel_pathway'], config['input_or_output_direction'])

        model_dict = {ModelPathEnum.GCN_MODEL_NAME.value: GCN, ModelPathEnum.HGNN_MODEL_NAME.value: HGNN, ModelPathEnum.HGNNP_MODEL_NAME.value: HGNNP}

        net_model = model_dict[config['model_name']](
            data_loader["num_features"], emb_size, data_loader["num_features"], use_bn=True
        )

        net_model.load_state_dict(torch.load(model_path, map_location=device))

        net_model.eval()

        # get the total number of nodes of this graph
        num_of_nodes: int = data_loader["num_nodes"]

        train_nodes_features = torch.FloatTensor(data_loader["train_nodes_features"])

        # generate the relationship between hyper edge and nodes
        # ex. [[1,2,3,4], [3,4], [9,7,4]...] where [1,2,3,4] represent a hyper edge
        train_all_hyper_edge_list = data_loader["train_edge_list"]
        train_hyper_edge_list = data_loader["train_masked_edge_list"]

        # train_edge_mask = data_loader["train_edge_mask"]

        # the hyper graph
        hyper_graph = Hypergraph(num_of_nodes, train_all_hyper_edge_list)

        # generate graph based on hyper graph or just use hyper graph
        if config['model_name'] in [ModelPathEnum.GCN_MODEL_NAME.value]:
            graph = Graph.from_hypergraph_clique(hyper_graph, weighted=True)
        else:
            graph = hyper_graph

        # set device
        train_nodes_features = train_nodes_features.to(device)
        graph = graph.to(device)
        net_model = net_model.to(device)

        selected_edge_predictions = self.__prediction_assist_on_single_edge(
            net_model,
            train_nodes_features,
            list_of_node_index_for_single_edge,
            graph
        )

        top_k_selected_edge_predictions_value, top_k_selected_edge_predictions_index = torch.topk(
            selected_edge_predictions, top_k)
        
        print(top_k_selected_edge_predictions_value)

        top_k_selected_edge_predictions_index = top_k_selected_edge_predictions_index.cpu().numpy()

        return top_k_selected_edge_predictions_index.tolist()

    @torch.no_grad()
    def __prediction_assist_on_single_edge(
            self,
            net_model,
            nodes_features,
            hyper_edge: List[int],
            graph,
    ):
        net_model.eval()
        # [[1,2,3],[2,3,4,5]...]
        hyper_edge_list = list()
        hyper_edge_list.append(hyper_edge)
        edges_embeddings = (
            utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
                hyper_edge_list, nodes_features
            )
        )

        # edges_embeddings = edges_embeddings.to(net_model.device)

        nodes_embeddings = net_model(nodes_features, graph)

        outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

        utils.filter_prediction_(outs, hyper_edge_list)

        return outs[0]

    def run_prediction_model_test(self, config=None):
        random.seed(2023)
        random_list = list(range(500))
        random.shuffle(random_list)

        return random_list[0:10]


    def run_gnn_explainer(self, config, rank: int, list_of_node_index_for_single_edge: List[int]):
        # set device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print("run_gnn_explainer_entrance")

        data_loader = DataLoaderLink(config['toplevel_pathway'], config['task'])
        # the GCN model
        # model_path_root: str = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'app', 'static', 'models')
        # raw_data_path: str = os.path.join('app', 'static', 'data', config.toplevel_pathway, config.task)

        # print("run_gnn_explainer_check_point1")

        model_path = model_selector.select_model_path(config['edge_or_node_task_level'], config['model_name'],
                                                      config['toplevel_pathway'], config['input_or_output_direction'])

        # print("run_gnn_explainer_check_point2")

        emb_size = model_selector.select_model_hyper_parameter_emb_size(config['edge_or_node_task_level'],
                                                                        config['model_name'],
                                                                        config['toplevel_pathway'],
                                                                        config['input_or_output_direction'])

        # print("run_gnn_explainer_check_point3")

        model_dict = {ModelPathEnum.GCN_MODEL_NAME.value: GCN, ModelPathEnum.HGNN_MODEL_NAME.value: HGNN,
                      ModelPathEnum.HGNNP_MODEL_NAME.value: HGNNP}

        # print("run_gnn_explainer_check_point4")

        net_model = model_dict[config['model_name']](
            data_loader["num_features"], emb_size, data_loader["num_features"], use_bn=True
        )

        # print("run_gnn_explainer_check_point5")

        net_model.load_state_dict(torch.load(model_path, map_location=device))

        # print("run_gnn_explainer_check_point6")

        net_model.eval()

        # print("run_gnn_explainer_check_point7")

        # get the total number of nodes of this graph
        num_of_nodes: int = data_loader["num_nodes"]

        train_nodes_features = torch.FloatTensor(data_loader["train_nodes_features"])

        # generate the relationship between hyper edge and nodes
        # ex. [[1,2,3,4], [3,4], [9,7,4]...] where [1,2,3,4] represent a hyper edge
        train_all_hyper_edge_list = data_loader["train_edge_list"]
        train_hyper_edge_list = data_loader["train_masked_edge_list"]

        # train_edge_mask = data_loader["train_edge_mask"]

        # the hyper graph
        hyper_graph = Hypergraph(num_of_nodes, train_all_hyper_edge_list)

        # print("run_gnn_explainer_check_point8")

        # generate graph based on hyper graph or just use hyper graph
        if config['model_name'] in [ModelPathEnum.GCN_MODEL_NAME.value]:
            graph = Graph.from_hypergraph_clique(hyper_graph, weighted=True)
        else:
            graph = hyper_graph

        # set device
        nodes_features = train_nodes_features.to(device)
        graph = graph.to(device)
        # net_model = net_model.to(device)

        # print("run_gnn_explainer_check_point9")

        with torch.no_grad():
            net_model.eval()
            nodes_embeddings = net_model(nodes_features, graph)

        # print("run_gnn_explainer_check_point10")

        # print(nodes_embeddings)
        # print("type of nodes_features: ", type(nodes_features))
        # print("type of nodes_features: ", type(nodes_embeddings))

        # nodes_attributes = np.array(nodes_features.tolist())
        # nodes_embeddings = np.array(nodes_embeddings.tolist())

        nodes_attributes = nodes_features.numpy()
        nodes_embeddings = nodes_embeddings.numpy()

        print("GNN explainer preparation work is done.")

        gnn_explainer = HGNNExplainer(node_attributes=nodes_attributes, node_embeddings=nodes_embeddings)

        prediction, explain = gnn_explainer.explain_link(list_of_node_index_for_single_edge, rank)
        
        


        # todo
        # print('Predicted link: {}'.format(prediction))
        # print('Explain:')
        # print(explain)
        #
        # print('Done')

        return explain


    def softmax_weight_of_explain(self, explain):
        weights = [item[1] for item in explain]

        names = [item[0] for item in explain]

        weights = softmax(weights)

        # for index, weight in enumerate(weights):
        #     weight = round_float_number(weight)
        #     weights[index] = weight

        explain = list(zip(names, weights))

        return explain





    def store_current_predict_status_to_session(self, config, top_k: int, list_of_node_index_for_single_edge: List[int]):
        session['config'] = config
        session['top_k'] = top_k
        session['list_of_node_index_for_single_edge'] = list_of_node_index_for_single_edge

    def load_current_predict_status_from_session(self):
        config = session.get('config')
        top_k = session.get('top_k')
        list_of_node_index_for_single_edge = session.get('list_of_node_index_for_single_edge')

        return config, top_k, list_of_node_index_for_single_edge

    def clear_current_predict_status_in_session(self) -> None:
        session.pop('config', None)
        session.pop('top_k', None)
        session.pop('list_of_node_index_for_single_edge', None)




# "Immune System",
#             "Metabolism",
#             "Signal Transduction",
#             "Disease",

# 'input link prediction dataset'
# 'output link prediction dataset'
# 'attribute prediction dataset'

model_service_obj = ModelService()

# if __name__ == '__main__':
#     # edge_level_task
#     config = {'toplevel_pathway': 'Disease',
#               'task': 'input link prediction dataset',
#               'edge_or_node_task_level': 'edge_level_task',
#               'model_name': 'gcn',
#               'input_or_output_direction': 'input',
#               'select_edge_index': 2}
#     res = model_service_obj.run_prediction_model(config, 10)
#
#     print(res)

