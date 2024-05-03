import os
import platform
import random
import time
from functools import wraps

import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn.functional as F
from numpy import ndarray
from scipy.sparse import csr_matrix

# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset


def read_file_via_lines(path: str, file_name: str) -> list[str]:
    # root_path: str = get_root_path_of_project("PathwayGNN")
    url: str = os.path.join(path, file_name)
    # url: str = os.path.join(path, file_name)
    res_list: list[str] = []

    try:
        file_handler = open(url, "r")
        while True:
            # Get next line from file
            line = file_handler.readline()
            line = line.replace("\r", "").replace("\n", "").replace("\t", "")

            # If the line is empty then the end of file reached
            if not line:
                break
            res_list.append(line)
    except Exception as e:
        print(e)
        print("we can't find the " + url + ", please make sure that the file exists")
    finally:
        return res_list


def get_sys_platform():
    sys_platform = platform.platform()
    if "Windows" in sys_platform:
        sys_platform_return = "windows"
    elif "macOS" in sys_platform:
        sys_platform_return = "macos"
    elif "linux" in sys_platform or "Linux" in sys_platform:
        sys_platform_return = "linux"
    else:
        sys_platform_return = "other"
    return sys_platform_return


def get_root_path_of_project(project_name: str):
    """
    This method is to get the root path of the project
    ex. when project name is PathwayGNN, root path is "E:\Python_Project\HGNN_On_Reactome\PathwayGNN"
    :param project_name: the name of the project
    :return:
    """
    cur_path: str = os.path.abspath(os.path.dirname(__file__))
    if "windows" == get_sys_platform():
        root_path: str = cur_path[
            : cur_path.find(project_name + "\\") + len(project_name + "\\")
        ]
    elif "macos" == get_sys_platform() or "linux" == get_sys_platform():
        root_path: str = cur_path[
            : cur_path.find(project_name + "/") + len(project_name + "/")
        ]
    else:
        raise Exception(
            "We can't support other system platform! Please use windows or macos"
        )
    return root_path


def normalize_sparse_matrix(mat):
    """Row-normalize sparse matrix"""
    # sum(1) 是计算每一行的和
    # 会得到一个（2708,1）的矩阵
    rowsum: ndarray = np.array(mat.sum(1))

    # 把这玩意儿取倒，然后拍平
    r_inv = np.power(rowsum, -1).flatten()

    # 在计算倒数的时候存在一个问题，如果原来的值为0，则其倒数为无穷大，因此需要对r_inv中无穷大的值进行修正，更改为0
    r_inv[np.isinf(r_inv)] = 0.0

    # np.diag() 应该也可以
    # 这里就是生成 对角矩阵
    r_mat_inv = sp.diags(r_inv)

    # 点乘,得到归一化后的结果
    # 注意是 归一化矩阵 点乘 原矩阵，别搞错了!!
    mat = r_mat_inv.dot(mat)
    return mat


def get_normalized_features_in_tensor(features) -> torch.Tensor:
    features_mat: csr_matrix = csr_matrix(features, dtype=np.float32)
    features_mat: csr_matrix = normalize_sparse_matrix(features_mat)
    features: torch.Tensor = torch.FloatTensor(np.array(features_mat.todense()))
    return features


def encode_node_features(
    components_mapping_list: list[list[int]],
    num_of_nodes: int,
    num_of_feature_dimension: int,
) -> list[list[int]]:
    row = []
    column = []
    val = []
    for line_index in range(num_of_nodes):
        features_of_one_entity = components_mapping_list[line_index]
        for feature_index in features_of_one_entity:
            row.append(line_index)
            column.append(feature_index)
            val.append(1)

    component_csc_mat = csr_matrix(
        (val, (row, column)), shape=(num_of_nodes, num_of_feature_dimension)
    )
    nodes_features: list[list[int]] = component_csc_mat.toarray().tolist()

    return nodes_features


def decode_node_features(node_features: list[int]):
    attributes_of_single_node: list[int] = list()
    for index, value in enumerate(node_features):
        if value > 0:
            attributes_of_single_node.append(index)
    return attributes_of_single_node


def decode_multi_nodes_features(multi_nodes_features: list[list[int]]):
    attributes_of_multi_nodes: list[list[int]] = list()
    for node_features in multi_nodes_features:
        attributes_of_single_node: list[int] = decode_node_features(node_features)
        attributes_of_multi_nodes.append(attributes_of_single_node)
    return attributes_of_multi_nodes


def encode_edges_features(
    edge_to_nodes_mapping_list: list[list[int]], num_of_edges: int, num_of_nodes: int
):
    row = []
    column = []
    val = []
    for line_index in range(num_of_edges):
        nodes_of_one_reaction = edge_to_nodes_mapping_list[line_index]
        for node in nodes_of_one_reaction:
            row.append(line_index)
            column.append(node)
            val.append(1)

    component_csc_mat = csr_matrix(
        (val, (row, column)), shape=(num_of_edges, num_of_nodes)
    )
    edges_features: list[list[int]] = component_csc_mat.toarray().tolist()

    return edges_features


def read_out_to_generate_single_hyper_edge_embedding(
    list_of_nodes_for_single_hyper_edge: list[int], nodes_features: torch.Tensor
) -> torch.Tensor:
    """
    Generate edge embedding for single edge based on its surrounding nodes
    :param list_of_nodes_for_single_hyper_edge: the nodes for a single hyper graph to slice the nodes_features tensor. ex. [0,1,4,5...]
    :param nodes_features: all the nodes features shape n*m, n is the number of nodes, m is the dimension of attributes.
    :return: after readout(mean method), we get an edge embedding.
    """
    nodes_features_for_single_hyper_edge = nodes_features[
        list_of_nodes_for_single_hyper_edge
    ]
    edge_embedding: torch.Tensor = torch.mean(
        nodes_features_for_single_hyper_edge, dim=0
    )
    # edge_embedding = torch.sigmoid(edge_embedding)

    return edge_embedding


def read_out_to_generate_multi_hyper_edges_embeddings_from_edge_dict(
    edge_to_nodes_dict: dict[int, list[int]], nodes_features: torch.Tensor
):
    """
    :param edge_to_nodes_dict: the key is the index of the edge and the value is a list of surrounding nodes. ex. {0:[1,2,3], 1:[2,4,5]}
    :param nodes_features: all the nodes features shape n*m, n is the number of nodes, m is the dimension of attributes.
    :return: after readout(mean method), we get multi edges embeddings.
    """
    multi_hyper_edges_embeddings_list: list[list[float]] = list()
    for edge, list_of_nodes in edge_to_nodes_dict.items():
        edge_embedding = read_out_to_generate_single_hyper_edge_embedding(
            list_of_nodes, nodes_features
        )
        multi_hyper_edges_embeddings_list.append(edge_embedding.tolist())

    multi_hyper_edges_embeddings_tensor = torch.Tensor(
        multi_hyper_edges_embeddings_list
    )

    return multi_hyper_edges_embeddings_tensor


def read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
    edge_to_nodes_list: list[list[int]], nodes_features: torch.Tensor
):
    """
    :param edge_to_nodes_list: a list of surrounding nodes of multi edges. ex. [[1,2,3], [2,4,5].....]
    :param nodes_features: all the nodes features shape n*m, n is the number of nodes, m is the dimension of attributes.
    :return: after readout(mean method), we get multi edges embeddings.
    """
    multi_hyper_edges_embeddings_list: list[list[float]] = list()
    for list_of_nodes in edge_to_nodes_list:
        edge_embedding = read_out_to_generate_single_hyper_edge_embedding(
            list_of_nodes, nodes_features
        )
        multi_hyper_edges_embeddings_list.append(edge_embedding.tolist())

    multi_hyper_edges_embeddings_tensor = torch.Tensor(
        multi_hyper_edges_embeddings_list
    )

    return multi_hyper_edges_embeddings_tensor


def filter_prediction_(prediction: torch.Tensor, filter_indexes_list: list[list[int]]):
    if prediction.shape[0] != len(filter_indexes_list):
        raise Exception(
            "Error! The prediction and filter_indexes_list not match in dimension"
        )

    for i in range(prediction.shape[0]):
        filter_indexes = filter_indexes_list[i]
        prediction[i][filter_indexes] = 0


class ModelEngineMF(object):
    def __init__(self, config):
        """Initialize ModelEngine Class."""
        self.config = config  # model configuration, should be a dic
        self.set_device()
        self.set_optimizer()
        self.model.to(self.device)
        print(self.model)
        # self.writer = SummaryWriter(log_dir=config["run_dir"])  # tensorboard writer

    def set_optimizer(self):
        """Set optimizer in the model."""
        if self.config["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config["lr"],
            )
        elif self.config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["lr"],
            )
        elif self.config["optimizer"] == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.config["lr"],
            )

    def set_device(self):
        """Set device."""
        self.device = torch.device(self.config["device_str"])
        self.model.device = self.device
        print("Setting device for torch_engine", self.device)

    def train_single_batch(self, batch_data, ratings=None):
        """Train the model in a single batch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.optimizer.zero_grad()
        ratings_pred = self.model.forward(batch_data)
        loss = self.model.loss(ratings_pred.view(-1), ratings)
        loss.backward()
        self.model.optimizer.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        """Train the model in one epoch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        for batch_id, batch_data in enumerate(train_loader):
            assert isinstance(batch_data, torch.LongTensor)
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, total_loss))
        # self.writer.add_scalar("model/loss", total_loss, epoch_id)
        return total_loss

    def save_checkpoint(self, model_dir):
        """Save checkpoint."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        torch.save(self.model.state_dict(), model_dir)

    # to do
    def resume_checkpoint(self, model_dir, model=None):
        """Resume model with checkpoint."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        print("loading model from:", model_dir)
        state_dict = torch.load(
            model_dir, map_location=self.device
        )  # ensure all storage are on gpu
        if model is None:
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            return self.model
        else:
            model.load_state_dict(state_dict)
            model.to(self.device)
            return model

    def bpr_loss(self, pos_scores, neg_scores):
        """Bayesian Personalised Ranking (BPR) pairwise loss function.
        Note that the sizes of pos_scores and neg_scores should be equal.
        Args:
            pos_scores (tensor): Tensor containing predictions for known positive items.
            neg_scores (tensor): Tensor containing predictions for sampled negative items.
        Returns:
            loss.
        """
        maxi = F.logsigmoid(pos_scores - neg_scores)
        loss = -torch.mean(maxi)
        return loss

    def bce_loss(self, scores, ratings):
        """Binary Cross-Entropy (BCE) pointwise loss, also known as log loss or logistic loss.
        Args:
            scores (tensor): Tensor containing predictions for both positive and negative items.
            ratings (tensor): Tensor containing ratings for both positive and negative items.
        Returns:
            loss.
        """
        # Calculate Binary Cross Entropy loss
        criterion = torch.nn.BCELoss()
        loss = criterion(scores, ratings)
        return loss


def timeit(method):
    """Generate decorator for tracking the execution time for the specific method.
    Args:
        method: The method need to timeit.
    To use:
        @timeit
        def method(self):
            pass
    Returns:
        None
    """

    @wraps(method)
    def wrapper(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print(
                "Execute [{}] method costing {:2.2f} ms".format(
                    method.__name__, (te - ts) * 1000
                )
            )
        return result

    return wrapper


class RatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset."""

    def __init__(self, entity_tensor, reaction_tensor, type_tensor):
        """Init UserItemRatingDataset Class.
        Args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair.
        """
        self.entity_tensor = entity_tensor
        self.reaction_tensor = reaction_tensor
        self.type_tensor = type_tensor

    def __getitem__(self, index):
        """Get an item from dataset."""
        return (
            self.entity_tensor[index],
            self.reaction_tensor[index],
            self.type_tensor[index],
        )

    def __len__(self):
        """Get the size of the dataset."""
        return self.entity_tensor.size(0)


def instance_bce_loader(data, batch_size, device, num_negative):
    """Instance a train DataLoader that have rating."""
    """entity is user, reaction is item, type is rating, """
    entity, reaction, type = [], [], []
    entity_pool = list(data["entity"].unique())
    reaction_pool = list(data["reaction"].unique())
    n_entity = len(entity_pool)
    n_reactio = len(reaction_pool)
    entity_id_pool = [i for i in range(n_entity)]
    reaction_id_pool = [i for i in range(n_reactio)]
    interact_status = (
        data.groupby("entity")["reaction"]
        .apply(set)
        .reset_index()
        .rename(columns={"reaction": "observed_reaction"})
    )
    interact_status["unobserved_reaction"] = interact_status["observed_reaction"].apply(
        lambda x: set(reaction_id_pool) - x
    )
    train_ratings = pd.merge(
        data,
        interact_status[["entity", "unobserved_reaction"]],
        on="entity",
    )
    train_ratings["unobserved_reaction"] = train_ratings["unobserved_reaction"].apply(
        lambda x: random.sample(list(x), num_negative)
    )
    for _, row in train_ratings.iterrows():
        entity.append(int(row["entity"]))
        reaction.append(int(row["reaction"]))
        type.append(float(row["type"]))
        for i in range(num_negative):
            entity.append(int(row["entity"]))
            reaction.append(int(row["unobserved_reaction"][i]))
            type.append(float(0))  # negative samples get 0 rating
    dataset = RatingDataset(
        entity_tensor=torch.LongTensor(entity).to(device),
        reaction_tensor=torch.LongTensor(reaction).to(device),
        type_tensor=torch.FloatTensor(type).to(device),
    )
    print(f"Making RatingDataset of length {len(dataset)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class PairwiseNegativeDataset(Dataset):
    """Wrapper, convert <user, pos_item, neg_item> Tensor into Pytorch Dataset."""

    def __init__(self, user_tensor, pos_item_tensor, neg_item_tensor):
        """Init PairwiseNegativeDataset Class.
        Args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair.
        """
        self.user_tensor = user_tensor
        self.pos_item_tensor = pos_item_tensor
        self.neg_item_tensor = neg_item_tensor

    def __getitem__(self, index):
        """Get an item from the dataset."""
        return (
            self.user_tensor[index],
            self.pos_item_tensor[index],
            self.neg_item_tensor[index],
        )

    def __len__(self):
        """Get the size of the dataset."""
        return self.user_tensor.size(0)


def instance_bpr_loader(data, batch_size, device, n_entity, n_reaction):
    """Instance a pairwise Data_loader for training.
    Sample ONE negative items for each user-item pare, and shuffle them with positive items.
    A batch of data in this DataLoader is suitable for a binary cross-entropy loss.
    # todo implement the item popularity-biased sampling
    """
    entity, pos_reaction, neg_reaction = [], [], []
    entity_id_pool = [i for i in range(n_entity)]
    reaction_id_pool = [i for i in range(n_reaction)]

    interact_status = (
        data.groupby("entity")["reaction"]
        .apply(set)
        .reset_index()
        .rename(columns={"reaction": "observed_reaction"})
    )
    interact_status["unobserved_reaction"] = interact_status["observed_reaction"].apply(
        lambda x: set(reaction_id_pool) - x
    )
    train_ratings = pd.merge(
        data,
        interact_status[["entity", "unobserved_reaction"]],
        on="entity",
    )
    train_ratings["unobserved_reaction"] = train_ratings["unobserved_reaction"].apply(
        lambda x: random.sample(list(x), 1)[0]
    )
    for _, row in train_ratings.iterrows():
        entity.append(row["entity"])
        pos_reaction.append(row["reaction"])
        neg_reaction.append(row["unobserved_reaction"])

    dataset = PairwiseNegativeDataset(
        user_tensor=torch.LongTensor(entity).to(device),
        pos_item_tensor=torch.LongTensor(pos_reaction).to(device),
        neg_item_tensor=torch.LongTensor(neg_reaction).to(device),
    )
    print(f"Making PairwiseNegativeDataset of length {len(dataset)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def predict_full(data_df, engine, batch_eval=True):
    """Make prediction for a trained model.
    Args:
        data_df (DataFrame): A dataset to be evaluated.
        model: A trained model.
        batch_eval (Boolean): A signal to indicate if the model is evaluated in batches.
    Returns:
        array: predicted scores.
    """
    entity_ids = data_df["entity"].to_numpy()
    entity_set = set(entity_ids)
    reaction_ids = data_df["reaction"].to_numpy()
    reaction_set = set(reaction_ids)
    full_entity = []
    full_reac = []
    for i in range(len(entity_ids)):
        full_entity.append(entity_ids[i])
        full_reac.append(reaction_ids[i])
        full_entity.extend(list(entity_set - set([entity_ids[i]])))
        full_reac.extend([reaction_ids[i]] * (len(entity_set) - 1))

    entity_ids = np.array(full_entity)
    reaction_ids = np.array(full_reac)
    batch_size = 50
    if batch_eval:
        n_batch = len(entity_ids) // batch_size + 1
        predictions = np.array([])
        for idx in range(n_batch):
            start_idx = idx * batch_size
            end_idx = min((idx + 1) * batch_size, len(entity_ids))
            sub_entity_ids = entity_ids[start_idx:end_idx]
            sub_reaction_ids = reaction_ids[start_idx:end_idx]
            sub_predictions = np.array(
                engine.__model.case_study_predict(sub_entity_ids, sub_reaction_ids)
                .flatten()
                .to(torch.device("cpu"))
                .detach()
                .numpy()
            )
            predictions = np.append(predictions, sub_predictions)
    else:
        predictions = np.array(
            engine.__model.case_study_predict(entity_ids, reaction_ids)
            .flatten()
            .to(torch.device("cpu"))
            .detach()
            .numpy()
        )

    return predictions
