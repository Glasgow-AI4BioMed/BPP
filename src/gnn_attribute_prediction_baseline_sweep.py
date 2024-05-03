import copy
import pprint
import time
from typing import List

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from dhg import Graph, Hypergraph
from dhg.models import GCN, HGNNP, HGNN
from sklearn.metrics import accuracy_score, ndcg_score, top_k_accuracy_score

from data_loader import DataLoaderAttribute

import utils


learning_rate = 0.01
weight_decay = 5e-4
project_name = "gnn_attribute_prediction_sweep_2023_Jan"


def train(
    net_model: torch.nn.Module,
    nodes_features: torch.Tensor,
    graph: Graph,
    labels: torch.Tensor,
    train_idx: List[bool],
    optimizer: optim.Adam,
    epoch: int,
):
    net_model.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net_model(nodes_features, graph)

    outs = outs[train_idx]
    loss = F.cross_entropy(outs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def validation(
    net_model, nodes_attributes, nodes_features, graph, labels, validation_idx
):
    net_model.eval()
    outs = net_model(nodes_features, graph)

    outs = outs[validation_idx]

    utils.filter_prediction_(outs, nodes_attributes)

    outs = outs.cpu().numpy()

    labels = labels.cpu().numpy()

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
def test(net_model, nodes_attributes, nodes_features, graph, labels, test_idx):
    net_model.eval()
    outs = net_model(nodes_features, graph)

    outs = outs[test_idx]

    utils.filter_prediction_(outs, nodes_attributes)

    outs = outs.cpu().numpy()

    labels = labels.cpu().numpy()

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
        print(config)
        # set device
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # initialize the data_loader
        # data_loader = DataLoaderAttribute("Disease", "attribute prediction dataset")
        data_loader = DataLoaderAttribute(config.dataset, config.task)

        # get the labels - the original nodes features
        # labels = torch.FloatTensor(data_loader["raw_nodes_features"])
        train_labels = data_loader["train_labels"]
        validation_labels = data_loader["validation_labels"]
        test_labels = data_loader["test_labels"]

        validation_nodes_attributes = data_loader["validation_nodes_components"]
        test_nodes_attributes = data_loader["test_nodes_components"]

        # get the train,val,test nodes features
        train_nodes_features = torch.FloatTensor(data_loader["train_nodes_features"])
        validation_nodes_features = torch.FloatTensor(
            data_loader["validation_nodes_features"]
        )
        test_nodes_features = torch.FloatTensor(data_loader["test_nodes_features"])

        # get train, validation, test mask to track the nodes
        train_mask = data_loader["train_node_mask"]
        val_mask = data_loader["val_node_mask"]
        test_mask = data_loader["test_node_mask"]

        # get the total number of nodes of this graph
        num_of_nodes: int = data_loader["num_nodes"]

        # generate the relationship between hyper edge and nodes
        # ex. [[1,2,3,4], [3,4], [9,7,4]...] where [1,2,3,4] represent a hyper edge
        hyper_edge_list = data_loader["edge_list"]

        # the hyper graph
        hyper_graph_train = Hypergraph(num_of_nodes, copy.deepcopy(hyper_edge_list))

        # generate graph based on hyper graph
        if config.model_name == "GCN":
            graph_train = Graph.from_hypergraph_clique(hyper_graph_train, weighted=True)
        else:
            graph_train = hyper_graph_train

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

        # set the optimizer
        optimizer = optim.Adam(
            net_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # set the device
        train_nodes_features, validation_nodes_features, test_nodes_features = (
            train_nodes_features.to(device),
            validation_nodes_features.to(device),
            test_nodes_features.to(device),
        )
        train_labels = train_labels.to(device)
        validation_labels = validation_labels.to(device)
        test_labels = test_labels.to(device)

        net_model = net_model.to(device)

        graph_train = graph_train.to(device)
        net_model = net_model.to(device)

        print(f"{config.model_name} Baseline")

        # start to train
        for epoch in range(200):
            # train
            # call the train method
            loss = train(
                net_model,
                train_nodes_features,
                graph_train,
                train_labels,
                train_mask,
                optimizer,
                epoch,
            )
            epoch_log = {
                "loss": loss,
                "epoch": epoch,
            }
            if epoch % 1 == 0:
                with torch.no_grad():
                    valid_result = validation(
                        net_model,
                        validation_nodes_attributes,
                        validation_nodes_features,
                        graph_train,
                        validation_labels,
                        val_mask,
                    )
                    test_result = test(
                        net_model,
                        test_nodes_attributes,
                        test_nodes_features,
                        graph_train,
                        test_labels,
                        test_mask,
                    )
                    epoch_log.update(valid_result)
                    epoch_log.update(test_result)
                    wandb.log(epoch_log)


def sweep():
    print("Please input model name. Options: GCN, HGNN, HGNNP")
    model_name = input()
    print(f"start tunning {model_name}")
    for task in ["attribute prediction dataset"]:
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
                "emb_dim": {"values": [64, 128, 256]},
                "drop_out": {"values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
                "weight_decay": {"values": [5e-4]},
                "model_name": {"values": [model_name]},
                "task": {"values": [task]},
                "dataset": {"values": [dataset]},
            }
            sweep_config["parameters"] = parameters_dict
            pprint.pprint(sweep_config)
            sweep_id = wandb.sweep(sweep_config, project=f"{task}_sweep_2023_Jan")
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
        "task": "attribute prediction dataset",
        "dataset": "Disease",
    }
    main(config)
