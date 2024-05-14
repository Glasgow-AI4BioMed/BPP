import sys

import torch
import torch.nn as nn
from torch.nn import Parameter

sys.path.append("../src/")
from utils import ModelEngine, timeit


class MF(torch.nn.Module):
    """A pytorch Module for Matrix Factorization."""

    def __init__(self, config):
        """Initialize MF Class."""
        super(MF, self).__init__()
        self.config = config
        self.device = self.config["device_str"]
        self.stddev = self.config["stddev"] if "stddev" in self.config else 0.1
        self.n_entity = self.config["n_entity"]
        self.n_reaction = self.config["n_reaction"]
        self.emb_dim = self.config["emb_dim"]
        self.entity_emb = nn.Embedding(self.n_entity, self.emb_dim)
        self.reaction_emb = nn.Embedding(self.n_reaction, self.emb_dim)
        self.entity_bias = nn.Embedding(self.n_entity, 1)
        self.reaction_bias = nn.Embedding(self.n_entity, 1)
        self.global_bias = Parameter(torch.zeros(1))
        self.entity_bias.weight.data.fill_(0.0)
        self.reaction_bias.weight.data.fill_(0.0)
        self.global_bias.data.fill_(0.0)
        nn.init.normal_(self.entity_emb.weight, 0, self.stddev)
        nn.init.normal_(self.reaction_emb.weight, 0, self.stddev)

    def forward(self, batch_data):
        """Train the model.
        Args:
            batch_data: tuple consists of (users, pos_items, neg_items), which must be LongTensor.
        """
        entity, reaction = batch_data
        e_emb = self.entity_emb(entity)
        e_bias = self.entity_bias(entity)
        r_emb = self.reaction_emb(reaction)
        r_bias = self.reaction_bias(reaction)
        scores = torch.sigmoid(
            torch.sum(torch.mul(e_emb, r_emb).squeeze(), dim=1)
            + e_bias.squeeze()
            + r_bias.squeeze()
            + self.global_bias
        )
        regularizer = (
            (e_emb**2).sum()
            + (r_emb**2).sum()
            + (e_bias**2).sum()
            + (r_bias**2).sum()
        ) / e_emb.size()[0]
        return scores, regularizer

    def predict(self, entity, reaction):
        """Predict result with the model.
        Args:
            users (int, or list of int):  user id(s).
            items (int, or list of int):  item id(s).
        Return:
            scores (int, or list of int): predicted scores of these user-item pairs.
        """
        entity_t = torch.LongTensor(entity).to(self.device)
        reaction_t = torch.LongTensor(reaction).to(self.device)
        with torch.no_grad():
            scores, _ = self.forward((entity_t, reaction_t))
        return scores


class MFAttr(torch.nn.Module):
    """A pytorch Module for Matrix Factorization."""

    def __init__(self, config):
        """Initialize MF Class."""
        super(MFAttr, self).__init__()
        self.config = config
        self.device = self.config["device_str"]
        self.stddev = self.config["stddev"] if "stddev" in self.config else 0.1
        
        self.n_attribute = self.config["n_attribute"]
        self.n_entity = self.config["n_entity"]
        self.emb_dim = self.config["emb_dim"]
        
        self.attribute_emb = nn.Embedding(self.n_attribute, self.emb_dim)
        self.entity_emb = nn.Embedding(self.n_entity, self.emb_dim)
                
        self.attribute_bias = nn.Embedding(self.n_attribute, 1)
        self.entity_bias = nn.Embedding(self.n_entity, 1)
        
        # from pdb import set_trace; set_trace()
        
        self.global_bias = Parameter(torch.zeros(1))
        
        self.attribute_bias.weight.data.fill_(0.0)
        self.entity_bias.weight.data.fill_(0.0)
        
        self.global_bias.data.fill_(0.0)
        nn.init.normal_(self.attribute_emb.weight, 0, self.stddev)
        nn.init.normal_(self.entity_emb.weight, 0, self.stddev)
        

    def forward(self, batch_data):
        """Train the model.
        Args:
            batch_data: tuple consists of (users, pos_items, neg_items), which must be LongTensor.
        """
        attribute, entity = batch_data
        a_emb = self.attribute_emb(attribute)
        a_bias = self.attribute_bias(attribute)
        
        e_emb = self.entity_emb(entity)
        e_bias = self.entity_bias(entity)
        
        # from pdb import set_trace; set_trace()
        
        try:
            scores = torch.sigmoid(
                torch.sum(torch.mul(a_emb, e_emb).squeeze(1), dim=1)
                + a_bias.squeeze()
                + e_bias.squeeze()
                + self.global_bias
            )
        except Exception as e:
            from pdb import set_trace; set_trace()

        regularizer = (
            (a_emb**2).sum()
            + (e_emb**2).sum()
            + (a_bias**2).sum()
            + (e_bias**2).sum()
        ) / a_emb.size()[0]
        return scores, regularizer

    def predict(self, attribute, entity):
        """Predict result with the model.
        Args:
            users (int, or list of int):  user id(s).
            items (int, or list of int):  item id(s).
        Return:
            scores (int, or list of int): predicted scores of these user-item pairs.
        """
        attribute_t = torch.LongTensor(attribute).to(self.device)
        entity_t = torch.LongTensor(entity).to(self.device)
        with torch.no_grad():
            scores, _ = self.forward((attribute_t, entity_t))
        return scores





class MFEngine(ModelEngine):
    """MFEngine Class."""

    def __init__(self, config):
        """Initialize MFEngine Class."""
        self.config = config
        # print_dict_as_table(config["model"], tag="MF model config")
        if "task" in config.keys() and "attr" in config["task"].lower():
            self.model = MFAttr(config) 
        else:
            self.model = MF(config)
        self.reg = 0.01
        # self.loss = torch.nn.HingeEmbeddingLoss()
        self.batch_size = config["batch_size"]
        super(MFEngine, self).__init__(config)
        self.model.to(self.device)
        # self.loss = (
        #     self.config["model"]["loss"] if "loss" in self.config["model"] else "bpr"
        # )

    def train_single_batch(self, batch_data):
        """Train a single batch.
        Args:
            batch_data (list): batch users, positive items and negative items.
        Return:
            loss (float): batch loss.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        entity, pos_reaction, neg_reaction = batch_data
        pos_scores, pos_regularizer = self.model.forward((entity, pos_reaction))
        neg_scores, neg_regularizer = self.model.forward((entity, neg_reaction))
        loss = self.bpr_loss(pos_scores, neg_scores)
        regularizer = pos_regularizer + neg_regularizer

        batch_loss = loss + self.reg * regularizer
        batch_loss.backward()
        self.optimizer.step()
        return loss.item(), regularizer.item()

    @timeit
    def train_an_epoch(self, train_loader, epoch_id):
        """Train a epoch, generate batch_data from data_loader, and call train_single_batch.
        Args:
            train_loader (DataLoader):
            epoch_id (int): the number of epoch.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0.0
        regularizer = 0.0
        
        # from pdb import set_trace; set_trace()
        
        for batch_data in train_loader:
            loss, reg = self.train_single_batch(batch_data)
            total_loss += loss
            regularizer += reg
        print(f"[Training Epoch {epoch_id}], Loss {loss}, Regularizer {regularizer}")
        # self.writer.add_scalar("model/loss", total_loss, epoch_id)
        # self.writer.add_scalar("model/regularizer", regularizer, epoch_id)
        return loss
