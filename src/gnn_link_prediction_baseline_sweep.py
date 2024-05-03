import copy
import pprint
import time
import os
from typing import List

import torch
import torch.nn.functional as F
import torch.optim as optim
from dhg import Graph, Hypergraph
from dhg.models import GCN, HGNN, HGNNP
from sklearn.metrics import accuracy_score, ndcg_score, top_k_accuracy_score

import utils
import wandb
from data_loader import DataLoaderLink

learning_rate = 0.01
weight_decay = 5e-4
project_name = "gnn_link_prediction_sweep_2023_Jan"

# set device
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

def ensureDir(dir_path):
    """Ensure a dir exist, otherwise create the path.
    Args:
        dir_path (str): the target dir.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def train(
    net_model: torch.nn.Module,
    nodes_features: torch.Tensor,
    train_hyper_edge_list: List[List[int]],
    graph: Graph,
    labels: torch.Tensor,
    optimizer: optim.Adam,
    epoch: int,
):
    net_model.train()

    st = time.time()
    optimizer.zero_grad()

    edges_embeddings = (
        utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
            train_hyper_edge_list, nodes_features
        )
    )

    edges_embeddings = edges_embeddings.to(net_model.device)
    # edges_embeddings = edges_embeddings[train_idx]

    nodes_embeddings = net_model(nodes_features, graph)

    nodes_embeddings = nodes_embeddings.to(net_model.device)

    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

    loss = F.cross_entropy(outs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def validation(
    net_model,
    nodes_features,
    validation_hyper_edge_list: List[List[int]],
    graph,
    labels,
):
    net_model.eval()

    edges_embeddings = (
        utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
            validation_hyper_edge_list, nodes_features
        )
    )

    edges_embeddings = edges_embeddings.to(net_model.device)
    nodes_embeddings = net_model(nodes_features, graph)

    # torch.backends.cudnn.enabled = False
    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

    utils.filter_prediction_(outs, validation_hyper_edge_list)
    outs = outs.cpu().numpy()

    labels = labels.cpu().numpy()
    # outs = [[0.1, 0.9, 0.3, 0.9],[0.1, 0.2, 0.3, 0.9]]
    # labels = [[0, 1, 0, 1], [0, 0, 0, 1]]
    # outs, labels = outs[validation_idx], labels[validation_idx]
    cat_labels = labels.argmax(axis=1)
    cat_outs = outs.argmax(axis=1)

    ndcg_res = ndcg_score(labels, outs)
    ndcg_res_3 = ndcg_score(labels, outs, k=3)
    ndcg_res_5 = ndcg_score(labels, outs, k=5)
    ndcg_res_10 = ndcg_score(labels, outs, k=10)
    ndcg_res_15 = ndcg_score(labels, outs, k=15)

    acc_res = accuracy_score(cat_labels, cat_outs)
    acc_res_3 = top_k_accuracy_score(cat_labels, outs, k=3, labels=range(outs.shape[1]))
    acc_res_5 = top_k_accuracy_score(cat_labels, outs, k=5, labels=range(outs.shape[1]))
    acc_res_10 = top_k_accuracy_score(
        cat_labels, outs, k=10, labels=range(outs.shape[1])
    )
    acc_res_15 = top_k_accuracy_score(
        cat_labels, outs, k=15, labels=range(outs.shape[1])
    )

    print(
        "\033[1;32m"
        + "The validation ndcg is: "
        + "{:.5f}".format(ndcg_res)
        + "\033[0m"
    )
    print(
        "\033[1;32m"
        + "The validation accuracy is: "
        + "{:.5f}".format(acc_res)
        + "\033[0m"
    )
    return {
        "valid_ndcg": ndcg_res,
        "valid_ndcg_3": ndcg_res_3,
        "valid_ndcg_5": ndcg_res_5,
        "valid_ndcg_10": ndcg_res_10,
        "valid_ndcg_15": ndcg_res_15,
        "valid_acc": acc_res,
        "valid_acc_3": acc_res_3,
        "valid_acc_5": acc_res_5,
        "valid_acc_10": acc_res_10,
        "valid_acc_15": acc_res_15,
    }


@torch.no_grad()
def test(
    net_model,
    nodes_features,
    test_hyper_edge_list: List[List[int]],
    graph,
    labels,
):
    net_model.eval()
    # [[1,2,3],[2,3,4,5]...]
    edges_embeddings = (
        utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
            test_hyper_edge_list, nodes_features
        )
    )

    edges_embeddings = edges_embeddings.to(net_model.device)

    nodes_embeddings = net_model(nodes_features, graph)

    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

    utils.filter_prediction_(outs, test_hyper_edge_list)
    outs = outs.cpu().numpy()

    labels = labels.cpu().numpy()
    # outs = [[0.1, 0.9, 0.3, 0.9],[0.1, 0.2, 0.3, 0.9]]
    # labels = [[0, 1, 0, 1], [0, 0, 0, 1]]
    # outs, labels = outs[validation_idx], labels[validation_idx]
    cat_labels = labels.argmax(axis=1)
    cat_outs = outs.argmax(axis=1)

    ndcg_res = ndcg_score(labels, outs)
    ndcg_res_3 = ndcg_score(labels, outs, k=3)
    ndcg_res_5 = ndcg_score(labels, outs, k=5)
    ndcg_res_10 = ndcg_score(labels, outs, k=10)
    ndcg_res_15 = ndcg_score(labels, outs, k=15)

    acc_res = accuracy_score(cat_labels, cat_outs)
    acc_res_3 = top_k_accuracy_score(cat_labels, outs, k=3, labels=range(outs.shape[1]))
    acc_res_5 = top_k_accuracy_score(cat_labels, outs, k=5, labels=range(outs.shape[1]))
    acc_res_10 = top_k_accuracy_score(
        cat_labels, outs, k=10, labels=range(outs.shape[1])
    )
    acc_res_15 = top_k_accuracy_score(
        cat_labels, outs, k=15, labels=range(outs.shape[1])
    )

    print("\033[1;32m" + "The test ndcg is: " + "{:.5f}".format(ndcg_res) + "\033[0m")
    print(
        "\033[1;32m" + "The test accuracy is: " + "{:.5f}".format(acc_res) + "\033[0m"
    )
    return {
        "test_ndcg": ndcg_res,
        "test_ndcg_3": ndcg_res_3,
        "test_ndcg_5": ndcg_res_5,
        "test_ndcg_10": ndcg_res_10,
        "test_ndcg_15": ndcg_res_15,
        "test_acc": acc_res,
        "test_acc_3": acc_res_3,
        "test_acc_5": acc_res_5,
        "test_acc_10": acc_res_10,
        "test_acc_15": acc_res_15,
    }


def main(config=None):
    with wandb.init(project=project_name):
        if config is not None:
            wandb.config.update(config)
        config = wandb.config

        # # set device
        # device = (
        #     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # )
        # initialize the data_loader
        # data_loader = DataLoaderLink("Disease", "input link prediction dataset")

        data_loader = DataLoaderLink(config.dataset, config.task)

        # get the total number of nodes of this graph
        num_of_nodes: int = data_loader["num_nodes"]

        # get the raw, train,val,test nodes features
        train_nodes_features = torch.FloatTensor(data_loader["train_nodes_features"])

        # generate the relationship between hyper edge and nodes
        # ex. [[1,2,3,4], [3,4], [9,7,4]...] where [1,2,3,4] represent a hyper edge
        train_all_hyper_edge_list = data_loader["train_edge_list"]
        train_hyper_edge_list = data_loader["train_masked_edge_list"]
        validation_hyper_edge_list = data_loader["validation_edge_list"]
        test_hyper_edge_list = data_loader["test_edge_list"]

        # get train, validation, test mask to track the nodes
        train_edge_mask = data_loader["train_edge_mask"]
        val_edge_mask = data_loader["val_edge_mask"]
        test_edge_mask = data_loader["test_edge_mask"]

        train_labels = data_loader["train_labels"]
        test_labels = data_loader["test_labels"]
        validation_labels = data_loader["validation_labels"]

        # to device
        # train_all_hyper_edge_list = train_all_hyper_edge_list.to(device)

        # the train hyper graph
        hyper_graph_train = Hypergraph(
            num_of_nodes, copy.deepcopy(train_all_hyper_edge_list)
        )

        # the train hyper graph
        hyper_graph_validation = Hypergraph(
            num_of_nodes, copy.deepcopy(train_all_hyper_edge_list)
        )

        # the train hyper graph
        hyper_graph_test = Hypergraph(
            num_of_nodes, copy.deepcopy(train_all_hyper_edge_list)
        )

        if config.model_name == "GCN":
            # generate train graph based on hyper graph
            graph_train = Graph.from_hypergraph_clique(hyper_graph_train, weighted=True)

            # generate train graph based on hyper graph
            graph_validation = Graph.from_hypergraph_clique(
                hyper_graph_validation, weighted=True
            )

            # generate train graph based on hyper graph
            graph_test = Graph.from_hypergraph_clique(hyper_graph_test, weighted=True)
        else:
            graph_train = hyper_graph_train
            graph_validation = hyper_graph_validation
            graph_test = hyper_graph_test

        # the GCN model
        if config.model_name == "GCN":
            net_model = GCN(
                data_loader["num_features"],
                config.emb_dim,
                data_loader["num_features"],
                use_bn=True,
                drop_rate=config.drop_out,
            )
        elif config.model_name == "HGNN":
            # the HGNN model
            net_model = HGNN(
                data_loader["num_features"],
                config.emb_dim,
                data_loader["num_features"],
                use_bn=True,
                drop_rate=config.drop_out,
            )
        elif config.model_name == "HGNNP":
            # the HGNNP model
            net_model = HGNNP(
                data_loader["num_features"],
                config.emb_dim,
                data_loader["num_features"],
                use_bn=True,
                drop_rate=config.drop_out,
            )
        else:
            raise Exception("Sorry, no model_name has been recognized.")

        model_save_dir = f"../save_model_ckp/{config.model_name}_{config.dataset}_{config.task}_{config.learning_rate}.bin"
        ensureDir("../save_model_ckp")
        net_model.device = device
        # set the optimizer
        optimizer = optim.Adam(
            net_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # set the device
        train_nodes_features, train_labels, test_labels, validation_labels = (
            train_nodes_features.to(device),
            train_labels.to(device),
            test_labels.to(device),
            validation_labels.to(device),
        )

        graph_train = graph_train.to(device)
        graph_validation = graph_validation.to(device)
        graph_test = graph_test.to(device)
        net_model = net_model.to(device)

        print(f"{config.model_name} Baseline")

        # start to train
        for epoch in range(200):
            # train
            # call the train method
            loss = train(
                net_model,
                train_nodes_features,
                train_hyper_edge_list,
                graph_train,
                train_labels,
                optimizer,
                epoch,
            )
            epoch_log = {
                "loss": loss,
                "epoch": epoch,
            }
            best_valid_ndcg = 0
            if epoch % 1 == 0:
                with torch.no_grad():
                    # validation(net_model, validation_nodes_features, validation_hyper_edge_list, graph_validation, labels, val_edge_mask)
                    valid_result = validation(
                        net_model,
                        train_nodes_features,
                        validation_hyper_edge_list,
                        graph_validation,
                        validation_labels,
                    )
                    if best_valid_ndcg<valid_result['valid_ndcg']:
                        best_valid_ndcg = valid_result['valid_ndcg']
                        torch.save(net_model.state_dict(), model_save_dir)
                    
                    # valid_ndcg, valid_acc = (
                    #     valid_result["valid_ndcg"],
                    #     valid_result["valid_acc"],
                    # )
                    test_result = test(
                        net_model,
                        train_nodes_features,
                        test_hyper_edge_list,
                        graph_test,
                        test_labels,
                    )
                    # test_ndcg, test_acc = (
                    #     test_result["test_ndcg"],
                    #     test_result["test_acc"],
                    # )
                    epoch_log.update(valid_result)
                    epoch_log.update(test_result)
                    wandb.log(epoch_log)


def sweep():
    print("Please input model name. Options: GCN, HGNN, HGNNP")
    model_name = input()
    print(f"start tunning {model_name}")
    for task in ["output link prediction dataset", "input link prediction dataset"]:
        for dataset in [
            "Immune System",
            "Metabolism",
            "Signal Transduction",
            "Disease",
        ]:
            sweep_config = {"method": "grid"}
            metric = {"name": "valid_ndcg", "goal": "maximize"}
            sweep_config["metric"] = metric
            parameters_dict = {
                "learning_rate": {"values": [0.01, 0.05, 0.005]},
                "emb_dim": {"values": [256]},
                "drop_out": {"values": [0.5]},
                "weight_decay": {"values": [5e-4]},
                "model_name": {"values": [model_name]},
                "task": {"values": [task]},
                "dataset": {"values": [dataset]},
            }
            sweep_config["parameters"] = parameters_dict
            pprint.pprint(sweep_config)
            sweep_id = wandb.sweep(sweep_config, project=f"{task}_sweep_2023_May")
            wandb.agent(sweep_id, main)


print("Are you going to run it as a sweep program? Y/N")
answer = input()
if answer.lower() == "y":
    sweep()
else:
    config = {
        "learning_rate": 0.05,
        "emb_dim": 128,
        "drop_out": 0.5,
        "weight_decay": 5e-4,
        "model_name": "HGNN",
        "task": "output link prediction dataset",
        "dataset": "Disease",
    }
    main(config)
