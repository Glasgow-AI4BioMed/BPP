import os
from typing import Union, Tuple

import numpy as np
import torch
from dhg import Hypergraph, Graph
from dhg.models import GCN, HGNN, HGNNP
from sklearn.metrics import ndcg_score, accuracy_score, top_k_accuracy_score

from case_study.bean.bean_collection import Edge, Node, Relationship
from case_study.bean.data_version import DataWithVersion
from case_study.bean.model_path import ModelSelector
from case_study.service.edge_service import EdgeService
from case_study.service.node_service import NodeService
from case_study.service.primary_secondary_entity_engine import PrimarySecondaryEntityEngine
from case_study.utils import utils
from case_study.utils.constant_definition import ModelPathEnum
from case_study.utils.data_loader import DataLoaderLink, Database
from case_study.utils.utils import read_file_via_lines, encode_edges_features, predict_full


def filter_regulator_relationships(relationships: list[Relationship]):
    relationships_after_filter: list[Relationship] = list()
    for relationship in relationships:
        if relationship.direction != 0:
            relationships_after_filter.append(relationship)

    return relationships_after_filter


class ModelEngine:
    """
        toplevel_pathway_name: "Disease", “Immune System", "Metabolism", "Signal Transduction"
        model: "gcn", "hgnn", "hgnnp"
        direction: "input", "output"
    """

    def __init__(self, data_version_name, task_name: str, toplevel_pathway_name: str, model: str, direction: str):
        self.__data_version_name = data_version_name
        self.__toplevel_pathway_name = toplevel_pathway_name
        self.__model = model
        self.__direction = direction
        self.__model_selector: ModelSelector = ModelSelector()

        self.edge_service = EdgeService()
        self.node_service = NodeService()

        self.raw_data_file_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                                               'multi_version_datasets', self.__data_version_name,
                                               self.__toplevel_pathway_name)
        self.task_file_path = os.path.join(self.raw_data_file_path, task_name)

    def get_num_of_nodes_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)
        node_line_message_list: list[str] = read_file_via_lines(path, "nodes.txt")
        num_of_nodes = len(node_line_message_list)
        return num_of_nodes

    def get_num_of_features_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)
        feature_line_message_list: list[str] = read_file_via_lines(
            path, "components-all.txt"
        )
        num_of_features = len(feature_line_message_list)
        return num_of_features

    def get_num_of_edges_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)
        edge_line_message_list: list[str] = read_file_via_lines(path, "edges.txt")
        num_of_edges = len(edge_line_message_list)
        return num_of_edges

    def get_reactions_with_entities_masked_for_testing(self, data_version: DataWithVersion, threshold_degree: int) -> Tuple[list[Edge], list[Edge]]:

        toplevel_pathway_name = self.__toplevel_pathway_name

        list_of_edges = self.edge_service.get_edge_list_from_dataset(toplevel_pathway_name=toplevel_pathway_name,
                                                                     data_version=data_version)

        degree2nodes_dict = self.node_service.get_degree2node_dict(toplevel_pathway_name, data_version)

        primary_nodes_indexes, secondary_nodes_indexes = PrimarySecondaryEntityEngine().get_primary_secondary_entities(
            degree2nodes_dict, threshold_degree)

        config = dict()
        config['toplevel_pathway'] = self.__toplevel_pathway_name
        if "input" == self.__direction:
            config['task'] = "input link prediction dataset"
        elif "output" == self.__direction:
            config['task'] = "output link prediction dataset"

        config['edge_or_node_task_level'] = 'edge_level_task'

        config['model_name'] = self.__model

        config['input_or_output_direction'] = self.__direction

        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        data_loader = DataLoaderLink(config['toplevel_pathway'], config['task'], data_version_name='data')

        edge_to_list_of_masked_nodes_for_test_dict: dict[int, list[int]] = data_loader[
            "edge_to_list_of_masked_nodes_for_test_dict"]

        edge_to_list_of_masked_nodes_for_test_list: list[int] = [edge for edge in edge_to_list_of_masked_nodes_for_test_dict.keys()]

        reactions_with_primary_entities_masked: list[Edge] = list()
        reactions_with_secondary_entities_masked: list[Edge] = list()

        for edge in list_of_edges:
            if edge.index in edge_to_list_of_masked_nodes_for_test_list:
                if edge_to_list_of_masked_nodes_for_test_dict[edge.index][0] in primary_nodes_indexes:
                    reactions_with_primary_entities_masked.append(edge)
                elif edge_to_list_of_masked_nodes_for_test_dict[edge.index][0] in secondary_nodes_indexes:
                    reactions_with_secondary_entities_masked.append(edge)

        return reactions_with_primary_entities_masked, reactions_with_secondary_entities_masked

    def evaluate_on_list_of_reactions(self, list_of_selected_reaction: list[Edge]):

        config = dict()
        config['toplevel_pathway'] = self.__toplevel_pathway_name
        if "input" == self.__direction:
            config['task'] = "input link prediction dataset"
        elif "output" == self.__direction:
            config['task'] = "output link prediction dataset"

        config['edge_or_node_task_level'] = 'edge_level_task'

        config['model_name'] = self.__model

        config['input_or_output_direction'] = self.__direction

        list_of_selected_edge_index = [selected_reaction.index for selected_reaction in list_of_selected_reaction]

        # config['list_of_selected_edge_index'] = list_of_selected_edge_index
        return self.__evaluate(config=config, list_of_selected_edge_index=list_of_selected_edge_index)

    def __evaluate(self, config, list_of_selected_edge_index: list[int]):

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        data_loader = DataLoaderLink(config['toplevel_pathway'], config['task'], data_version_name='data')

        model_path = self.__model_selector.select_model_path(config['edge_or_node_task_level'], config['model_name'],
                                                             config['toplevel_pathway'],
                                                             config['input_or_output_direction'])

        emb_size = self.__model_selector.select_model_hyper_parameter_emb_size(config['edge_or_node_task_level'],
                                                                               config['model_name'],
                                                                               config['toplevel_pathway'],
                                                                               config['input_or_output_direction'])

        model_dict = {ModelPathEnum.GCN_MODEL_NAME.value: GCN, ModelPathEnum.HGNN_MODEL_NAME.value: HGNN,
                      ModelPathEnum.HGNNP_MODEL_NAME.value: HGNNP}

        net_model = model_dict[config['model_name']](
            data_loader["num_features"], emb_size, data_loader["num_features"], use_bn=True
        )

        # net_model.load_state_dict(torch.load(model_path))
        net_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        net_model.eval()

        # get the total number of nodes of this graph
        num_of_nodes: int = data_loader["num_nodes"]

        train_nodes_features = torch.FloatTensor(data_loader["train_nodes_features"])

        train_all_hyper_edge_list = data_loader["train_edge_list"]

        # train_hyper_edge_list = data_loader["train_masked_edge_list"]

        list_of_edges_with_remaining_nodes: list[list[int]] = [train_all_hyper_edge_list[i] for i in
                                                               list_of_selected_edge_index]

        edge_to_list_of_all_masked_nodes_dict: dict[int, list[int]] = data_loader["edge_to_list_of_all_masked_nodes_dict"]

        list_of_edges_with_masked_nodes: list[list[int]] = list()

        for edge_index, list_of_masked_node_indexes in edge_to_list_of_all_masked_nodes_dict.items():
            if edge_index in list_of_selected_edge_index:
                list_of_edges_with_masked_nodes.append(list_of_masked_node_indexes)

        # num_of_nodes_of_raw_dataset: int = data_loader["num_nodes"]
        #
        # evaluate_labels = torch.FloatTensor(
        #     encode_edges_features(
        #         list_of_edges_with_masked_nodes,
        #         len(list_of_edges_with_masked_nodes),
        #         num_of_nodes_of_raw_dataset,
        #     )
        # )


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

        return self.__evaluate_assist(
            net_model=net_model,
            nodes_features=train_nodes_features,
            hyper_edge_list=list_of_edges_with_remaining_nodes,
            labels=list_of_edges_with_masked_nodes,
            graph=graph,
        )

    def case_study_predict(self, list_of_selected_edge: list[Edge],
                           list_of_target_node_of_all_selected_edge: list[Node]) -> dict[str, float]:

        list_of_selected_edge_index = [edge.index for edge in list_of_selected_edge]

        # edge_index = int(request.form.get('edge_index'))

        list_of_valid_nodes_of_all_selected_edges: list[list[Node]] = [(edge.input_nodes_list + edge.output_nodes_list)
                                                                       for edge in list_of_selected_edge]

        list_of_valid_nodes_indexes_of_all_selected_edges: list[list[int]] = list()
        for index, valid_nodes_of_single_edge in enumerate(list_of_valid_nodes_of_all_selected_edges):
            list_of_valid_nodes_indexes_of_single_selected_edges: list[int] = list()
            for valid_node in valid_nodes_of_single_edge:
                list_of_valid_nodes_indexes_of_single_selected_edges.append(valid_node.index)

            list_of_valid_nodes_indexes_of_all_selected_edges.append(
                list_of_valid_nodes_indexes_of_single_selected_edges)

        list_of_target_node_index_of_all_selected_edges: list[list[int]] = list()

        for target_node_of_single_selected_edge in list_of_target_node_of_all_selected_edge:
            list_of_target_node_index_of_single_selected_edges = list()
            list_of_target_node_index_of_single_selected_edges.append(target_node_of_single_selected_edge.index)
            list_of_target_node_index_of_all_selected_edges.append(list_of_target_node_index_of_single_selected_edges)

        """
        [target_node_of_single_selected_edge.index for
                                                                      target_node_of_single_selected_edge in
                                                                      list_of_target_node_of_all_selected_edge]
        """

        config = dict()
        config['toplevel_pathway'] = self.__toplevel_pathway_name
        if "input" == self.__direction:
            config['task'] = "input link prediction dataset"
        elif "output" == self.__direction:
            config['task'] = "output link prediction dataset"

        config['edge_or_node_task_level'] = 'edge_level_task'

        config['model_name'] = self.__model

        config['input_or_output_direction'] = self.__direction

        # todo
        config['list_of_selected_edge_index'] = list_of_selected_edge_index

        # prediction_indexes = model_service_obj.run_prediction_model(config, 10)
        return self.__case_study_evaluate(config, list_of_valid_nodes_indexes_of_all_selected_edges,
                                          list_of_target_node_index_of_all_selected_edges)

        # prediction_indexes = model_service_obj.run_prediction_model_test()

        # dataset = dataset_dict['disease']
        #
        # edge = dataset.select_edge_based_on_index(0)

    def __case_study_evaluate(self, config, list_of_valid_nodes_indexes_of_all_selected_edges: list[list[int]],
                              list_of_target_node_index_of_all_selected_edges: list[list[int]]) -> dict[str, float]:
        # set device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        data_loader = DataLoaderLink(config['toplevel_pathway'], config['task'], data_version_name='data')

        model_path = self.__model_selector.select_model_path(config['edge_or_node_task_level'], config['model_name'],
                                                             config['toplevel_pathway'],
                                                             config['input_or_output_direction'])

        emb_size = self.__model_selector.select_model_hyper_parameter_emb_size(config['edge_or_node_task_level'],
                                                                               config['model_name'],
                                                                               config['toplevel_pathway'],
                                                                               config['input_or_output_direction'])

        model_dict = {ModelPathEnum.GCN_MODEL_NAME.value: GCN, ModelPathEnum.HGNN_MODEL_NAME.value: HGNN,
                      ModelPathEnum.HGNNP_MODEL_NAME.value: HGNNP}

        net_model = model_dict[config['model_name']](
            data_loader["num_features"], emb_size, data_loader["num_features"], use_bn=True
        )

        # net_model.load_state_dict(torch.load(model_path))
        net_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

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

        return self.__evaluate_assist(net_model=net_model, nodes_features=train_nodes_features,
                                      hyper_edge_list=list_of_valid_nodes_indexes_of_all_selected_edges,
                                      labels=list_of_target_node_index_of_all_selected_edges, graph=graph)

    @torch.no_grad()
    def __evaluate_assist(
            self,
            net_model,
            nodes_features,
            hyper_edge_list: list[list[int]],
            labels: list[list[int]],
            graph,
    ) -> dict[str, float]:
        net_model.eval()
        # [[1,2,3],[2,3,4,5]...]
        edges_embeddings = (
            utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
                hyper_edge_list, nodes_features
            )
        )

        # labels = torch.tensor(labels)

        num_of_nodes_validation = self.get_num_of_nodes_based_on_type_name()

        num_of_features = self.get_num_of_features_based_on_type_name()

        label_onehot = list()
        for label in labels:
            onehot = [0] * num_of_nodes_validation
            for index in label:
                onehot[index] = 1
            label_onehot.append(onehot)
        labels = label_onehot

        labels = torch.FloatTensor(labels)

        # edges_embeddings = edges_embeddings.to(net_model.device)

        nodes_embeddings = net_model(nodes_features, graph)

        outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

        utils.filter_prediction_(outs, hyper_edge_list)

        outs = outs.cpu().numpy()

        labels = labels.cpu().numpy()
        # outs = [[0.1, 0.9, 0.3, 0.9],[0.1, 0.2, 0.3, 0.9]]
        # labels = [[0, 1, 0, 1], [0, 0, 0, 1]]
        # outs, labels = outs[validation_idx], labels[validation_idx]
        cat_labels = labels.argmax(axis=1)
        cat_outs = outs.argmax(axis=1)

        ndcg_res = ndcg_score(labels, outs)
        # ndcg_res_3 = ndcg_score(labels, outs, k=3)
        # ndcg_res_5 = ndcg_score(labels, outs, k=5)
        ndcg_res_10 = ndcg_score(labels, outs, k=10)
        # ndcg_res_15 = ndcg_score(labels, outs, k=15)
        # ndcg_res_20 = ndcg_score(labels, outs, k=20)
        # ndcg_res_25 = ndcg_score(labels, outs, k=25)
        # ndcg_res_30 = ndcg_score(labels, outs, k=30)
        # ndcg_res_50 = ndcg_score(labels, outs, k=50)
        # ndcg_res_100 = ndcg_score(labels, outs, k=100)
        # ndcg_res_200 = ndcg_score(labels, outs, k=200)
        # ndcg_res_500 = ndcg_score(labels, outs, k=500)

        acc_res = accuracy_score(cat_labels, cat_outs)
        acc_res_3 = top_k_accuracy_score(cat_labels, outs, k=3, labels=range(outs.shape[1]))
        acc_res_5 = top_k_accuracy_score(cat_labels, outs, k=5, labels=range(outs.shape[1]))
        acc_res_10 = top_k_accuracy_score(
            cat_labels, outs, k=10, labels=range(outs.shape[1])
        )
        acc_res_15 = top_k_accuracy_score(
            cat_labels, outs, k=15, labels=range(outs.shape[1])
        )
        acc_res_20 = top_k_accuracy_score(
            cat_labels, outs, k=20, labels=range(outs.shape[1])
        )
        acc_res_25 = top_k_accuracy_score(
            cat_labels, outs, k=25, labels=range(outs.shape[1])
        )
        acc_res_30 = top_k_accuracy_score(
            cat_labels, outs, k=30, labels=range(outs.shape[1])
        )
        # acc_res_50 = top_k_accuracy_score(
        #     cat_labels, outs, k=50, labels=range(outs.shape[1])
        # )
        # acc_res_100 = top_k_accuracy_score(
        #     cat_labels, outs, k=100, labels=range(outs.shape[1])
        # )
        # acc_res_200 = top_k_accuracy_score(
        #     cat_labels, outs, k=200, labels=range(outs.shape[1])
        # )
        # acc_res_500 = top_k_accuracy_score(
        #     cat_labels, outs, k=500, labels=range(outs.shape[1])
        # )

        # print("\033[1;32m" + "The ndcg is: " + "{:.5f}".format(ndcg_res) + "\033[0m")
        # print("\033[1;32m" + "The ndcg_3 is: " + "{:.5f}".format(ndcg_res_3) + "\033[0m")
        # print("\033[1;32m" + "The ndcg_5 is: " + "{:.5f}".format(ndcg_res_5) + "\033[0m")
        # print("\033[1;32m" + "The ndcg_10 is: " + "{:.5f}".format(ndcg_res_10) + "\033[0m")
        # print("\033[1;32m" + "The ndcg_15 is: " + "{:.5f}".format(ndcg_res_15) + "\033[0m")
        # print("\033[1;32m" + "The ndcg_20 is: " + "{:.5f}".format(ndcg_res_20) + "\033[0m")
        # print("\033[1;32m" + "The ndcg_25 is: " + "{:.5f}".format(ndcg_res_25) + "\033[0m")
        # print("\033[1;32m" + "The ndcg_30 is: " + "{:.5f}".format(ndcg_res_30) + "\033[0m")
        # print("\033[1;32m" + "The ndcg_50 is: " + "{:.5f}".format(ndcg_res_50) + "\033[0m")
        # print("\033[1;32m" + "The ndcg_100 is: " + "{:.5f}".format(ndcg_res_100) + "\033[0m")
        # print("\033[1;32m" + "The ndcg_200 is: " + "{:.5f}".format(ndcg_res_200) + "\033[0m")
        # print("\033[1;32m" + "The ndcg_500 is: " + "{:.5f}".format(ndcg_res_500) + "\033[0m")
        # print(
        #     "\033[1;32m" + "The test accuracy is: " + "{:.5f}".format(acc_res) + "\033[0m"
        # )
        # print(
        #     "\033[1;32m" + "The test accuracy_3 is: " + "{:.5f}".format(acc_res_3) + "\033[0m"
        # )
        # print(
        #     "\033[1;32m" + "The test accuracy_5 is: " + "{:.5f}".format(acc_res_5) + "\033[0m"
        # )
        # print(
        #     "\033[1;32m" + "The test accuracy_10 is: " + "{:.5f}".format(acc_res_10) + "\033[0m"
        # )
        # print(
        #     "\033[1;32m" + "The test accuracy_15 is: " + "{:.5f}".format(acc_res_15) + "\033[0m"
        # )
        # print(
        #     "\033[1;32m" + "The test accuracy_20 is: " + "{:.5f}".format(acc_res_20) + "\033[0m"
        # )
        # print(
        #     "\033[1;32m" + "The test accuracy_25 is: " + "{:.5f}".format(acc_res_25) + "\033[0m"
        # )
        # print(
        #     "\033[1;32m" + "The test accuracy_30 is: " + "{:.5f}".format(acc_res_30) + "\033[0m"
        # )
        # print(
        #     "\033[1;32m" + "The test accuracy_50 is: " + "{:.5f}".format(acc_res_50) + "\033[0m"
        # )
        # print(
        #     "\033[1;32m" + "The test accuracy_100 is: " + "{:.5f}".format(acc_res_100) + "\033[0m"
        # )
        # print(
        #     "\033[1;32m" + "The test accuracy_200 is: " + "{:.5f}".format(acc_res_200) + "\033[0m"
        # )
        # print(
        #     "\033[1;32m" + "The test accuracy_500 is: " + "{:.5f}".format(acc_res_500) + "\033[0m"
        # )
        return {
            "test_ndcg": ndcg_res,
            # "test_ndcg_3": ndcg_res_3,
            # "test_ndcg_5": ndcg_res_5,
            "test_ndcg_10": ndcg_res_10,
            # "test_ndcg_15": ndcg_res_15,
            # "test_ndcg_20": ndcg_res_20,
            # "test_ndcg_25": ndcg_res_25,
            # "test_ndcg_30": ndcg_res_30,
            # "test_ndcg_50": ndcg_res_50,
            # "test_ndcg_100": ndcg_res_100,
            # "test_ndcg_200": ndcg_res_200,
            # "test_ndcg_500": ndcg_res_500,
            "test_acc": acc_res,
            "test_acc_3": acc_res_3,
            "test_acc_5": acc_res_5,
            "test_acc_10": acc_res_10,
            "test_acc_15": acc_res_15,
            "test_acc_20": acc_res_20,
            "test_acc_25": acc_res_25,
            "test_acc_30": acc_res_30,
            # "test_acc_50": acc_res_50,
            # "test_acc_100": acc_res_100,
            # "test_acc_200": acc_res_200,
            # "test_acc_500": acc_res_500,
        }

# class MF_train:
#     """MF_train Class."""
#
#     def __init__(self, args):
#         """Initialize MF_train Class."""
#         self.config = args
#         self.data = Database(args["dataset"], args["task"])
#         self.train_set = self.data.train
#         self.test_set = self.data.test
#         self.valid_set = self.data.valid
#         self.n_entity = max(list(self.train_set["entity"])) + 1
#         self.n_reaction = max(list(self.train_set["reaction"])) + 1
#         self.config["n_entity"] = self.n_entity
#         self.config["n_reaction"] = self.n_reaction
#         self.best_model = None
#
#     def train(self):
#         """Train the model."""
#
#         global valid_result
#
#         self.engine = MFEngine(self.config)
#         self.model_save_dir = os.path.join(
#             self.config["model_save_dir"], self.config["save_name"]
#         )
#         best_valid_performance = 0
#         best_epoch = 0
#         epoch_bar = range(self.config["max_epoch"])
#         for epoch in epoch_bar:
#             print("Epoch", epoch)
#             loss = self.engine.train_an_epoch(train_loader, epoch_id=epoch)
#             """evaluate model on validation and test sets"""
#             validation_set = self.valid_set
#             n_samples = len(validation_set)
#             predictions = predict_full(validation_set, self.engine)
#             valid_result = self.validation(predictions, n_samples)
#             test_result = self.test(self.engine)
#             epoch_log = {
#                 "loss": loss,
#                 "epoch": epoch,
#             }
#             if valid_result["valid_ndcg"] > best_valid_performance:
#                 best_valid_performance = valid_result["valid_ndcg"]
#                 best_epoch = epoch
#                 self.best_model = self.engine
#                 self.best_model.save_checkpoint(self.model_save_dir) # one can use resume_checkpoint(model_dir) to resume/load a checkpoint
#             print("valid_ndcg", valid_result["valid_ndcg"])
#             print("valid_acc", valid_result["valid_acc"])
#             print("loss")
#             epoch_log.update(valid_result)
#             epoch_log.update(test_result)
#             wandb.log(epoch_log)
#
#         print(
#             "BEST ndcg performenace on validation set is %f"
#             % valid_result["valid_ndcg"]
#         )
#         print(
#             "BEST acc performenace on validation set is %f" % valid_result["valid_ndcg"]
#         )
#         print("BEST performance happens at epoch", best_epoch)
#         return best_valid_performance
#
#     def test(self, model=None):
#         """Evaluate the performance for the testing sets based on the best performing model."""
#         if model is None:
#             model = self.best_model
#         test_set = self.test_set
#         predictions = predict_full(test_set, model)
#         n_samples = len(test_set)
#         test_result = self.evaluate(predictions, n_samples, "test")
#         return test_result
#
#     def validation(self, predictions, n_samples):
#         return self.evaluate(predictions, n_samples, "validation")
#
#     def evaluate(self, predictions, n_samples, evaluate_type: str):
#         if evaluate_type not in ["test", "validation"]:
#             raise Exception('Please choose "test" or "validation" type')
#         predictions = predictions.reshape(
#             n_samples, int(predictions.shape[0] / n_samples)
#         )
#         ground_truth = np.zeros(int(predictions.shape[1]))
#         ground_truth[0] = 1
#         new = []
#         for i in range(n_samples):
#             new.append(list(ground_truth))
#
#         ground_truth = np.array(new)
#         cat_labels = ground_truth.argmax(axis=1)
#         cat_outs = predictions.argmax(axis=1)
#
#         ndcg_res = metrics.ndcg_score(ground_truth, predictions)
#         ndcg_res_3 = metrics.ndcg_score(ground_truth, predictions, k=3)
#         ndcg_res_5 = metrics.ndcg_score(ground_truth, predictions, k=5)
#         ndcg_res_10 = metrics.ndcg_score(ground_truth, predictions, k=10)
#         ndcg_res_15 = metrics.ndcg_score(ground_truth, predictions, k=15)
#
#         acc_res = metrics.accuracy_score(cat_labels, cat_outs)
#         acc_res_3 = metrics.top_k_accuracy_score(
#             cat_labels, predictions, k=3, labels=range(predictions.shape[1])
#         )
#         acc_res_5 = metrics.top_k_accuracy_score(
#             cat_labels, predictions, k=5, labels=range(predictions.shape[1])
#         )
#         acc_res_10 = metrics.top_k_accuracy_score(
#             cat_labels, predictions, k=10, labels=range(predictions.shape[1])
#         )
#         acc_res_15 = metrics.top_k_accuracy_score(
#             cat_labels, predictions, k=15, labels=range(predictions.shape[1])
#         )
#
#         if "test" == evaluate_type:
#             ndcg: str = "test_ndcg"
#             ndcg_3: str = "test_ndcg_3"
#             ndcg_5: str = "test_ndcg_5"
#             ndcg_10: str = "test_ndcg_10"
#             ndcg_15: str = "test_ndcg_15"
#             acc: str = "test_acc"
#             acc_3: str = "test_acc_3"
#             acc_5: str = "test_acc_5"
#             acc_10: str = "test_acc_10"
#             acc_15: str = "test_acc_15"
#         elif "validation" == evaluate_type:
#             ndcg: str = "valid_ndcg"
#             ndcg_3: str = "valid_ndcg_3"
#             ndcg_5: str = "valid_ndcg_5"
#             ndcg_10: str = "valid_ndcg_10"
#             ndcg_15: str = "valid_ndcg_15"
#             acc: str = "valid_acc"
#             acc_3: str = "valid_acc_3"
#             acc_5: str = "valid_acc_5"
#             acc_10: str = "valid_acc_10"
#             acc_15: str = "valid_acc_15"
#         else:
#             raise Exception('Please choose "test" or "validation" type')
#
#         print(
#             "\033[1;32m"
#             + "The"
#             + evaluate_type
#             + "ndcg is: "
#             + "{:.5f}".format(ndcg_res)
#             + "\033[0m"
#         )
#         print(
#             "\033[1;32m"
#             + "The"
#             + evaluate_type
#             + "accuracy is: "
#             + "{:.5f}".format(acc_res)
#             + "\033[0m"
#         )
#         return {
#             ndcg: ndcg_res,
#             ndcg_3: ndcg_res_3,
#             ndcg_5: ndcg_res_5,
#             ndcg_10: ndcg_res_10,
#             ndcg_15: ndcg_res_15,
#             acc: acc_res,
#             acc_3: acc_res_3,
#             acc_5: acc_res_5,
#             acc_10: acc_res_10,
#             acc_15: acc_res_15,
#         }
