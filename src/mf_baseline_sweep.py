import os
import pprint
import sys

import torch
import wandb

sys.path.append("../src/")
import numpy as np
from sklearn import metrics

from data_loader import Database
from matrix_factorisation import MFEngine
from utils import instance_bpr_loader, predict_full

model_name = "MF"
project_name = "pathway_link_predict_MF"


class MF_train:
    """MF_train Class."""

    def __init__(self, args):
        """Initialize MF_train Class."""
        self.config = args
        self.data = Database(args["dataset"], args["task"])
        self.train_set = self.data.train
        self.test_set = self.data.test
        self.valid_set = self.data.valid
        self.n_entity = max(list(self.train_set["entity"])) + 1
        self.n_reaction = max(list(self.train_set["reaction"])) + 1
        self.config["n_entity"] = self.n_entity
        self.config["n_reaction"] = self.n_reaction
        self.best_model = None

    def train(self):
        """Train the model."""

        global valid_result
        train_loader = instance_bpr_loader(
            data=self.train_set,
            batch_size=self.config["batch_size"],
            device=self.config["device_str"],
            n_entity=self.n_entity,
            n_reaction=self.n_reaction,
        )

        self.engine = MFEngine(self.config)
        self.model_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["save_name"]
        )
        best_valid_performance = 0
        best_epoch = 0
        epoch_bar = range(self.config["max_epoch"])
        for epoch in epoch_bar:
            print("Epoch", epoch)
            loss = self.engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on validation and test sets"""
            validation_set = self.valid_set
            n_samples = len(validation_set)
            predictions = predict_full(validation_set, self.engine)
            valid_result = self.validation(predictions, n_samples)
            test_result = self.test(self.engine)
            epoch_log = {
                "loss": loss,
                "epoch": epoch,
            }
            if valid_result["valid_ndcg"] > best_valid_performance:
                best_valid_performance = valid_result["valid_ndcg"]
                best_epoch = epoch
                self.best_model = self.engine
                self.best_model.save_checkpoint(self.model_save_dir) # one can use resume_checkpoint(model_dir) to resume/load a checkpoint
            print("valid_ndcg", valid_result["valid_ndcg"])
            print("valid_acc", valid_result["valid_acc"])
            print("loss")
            epoch_log.update(valid_result)
            epoch_log.update(test_result)
            wandb.log(epoch_log)

        print(
            "BEST ndcg performenace on validation set is %f"
            % valid_result["valid_ndcg"]
        )
        print(
            "BEST acc performenace on validation set is %f" % valid_result["valid_ndcg"]
        )
        print("BEST performance happens at epoch", best_epoch)
        return best_valid_performance

    def test(self, model=None):
        """Evaluate the performance for the testing sets based on the best performing model."""
        if model is None:
            model = self.best_model
        test_set = self.test_set
        predictions = predict_full(test_set, model)
        n_samples = len(test_set)
        test_result = self.evaluate(predictions, n_samples, "test")
        return test_result

    def validation(self, predictions, n_samples):
        return self.evaluate(predictions, n_samples, "validation")

    def evaluate(self, predictions, n_samples, evaluate_type: str):
        if evaluate_type not in ["test", "validation"]:
            raise Exception('Please choose "test" or "validation" type')
        predictions = predictions.reshape(
            n_samples, int(predictions.shape[0] / n_samples)
        )
        ground_truth = np.zeros(int(predictions.shape[1]))
        ground_truth[0] = 1
        new = []
        for i in range(n_samples):
            new.append(list(ground_truth))

        ground_truth = np.array(new)
        cat_labels = ground_truth.argmax(axis=1)
        cat_outs = predictions.argmax(axis=1)

        ndcg_res = metrics.ndcg_score(ground_truth, predictions)
        ndcg_res_3 = metrics.ndcg_score(ground_truth, predictions, k=3)
        ndcg_res_5 = metrics.ndcg_score(ground_truth, predictions, k=5)
        ndcg_res_10 = metrics.ndcg_score(ground_truth, predictions, k=10)
        ndcg_res_15 = metrics.ndcg_score(ground_truth, predictions, k=15)

        acc_res = metrics.accuracy_score(cat_labels, cat_outs)
        acc_res_3 = metrics.top_k_accuracy_score(
            cat_labels, predictions, k=3, labels=range(predictions.shape[1])
        )
        acc_res_5 = metrics.top_k_accuracy_score(
            cat_labels, predictions, k=5, labels=range(predictions.shape[1])
        )
        acc_res_10 = metrics.top_k_accuracy_score(
            cat_labels, predictions, k=10, labels=range(predictions.shape[1])
        )
        acc_res_15 = metrics.top_k_accuracy_score(
            cat_labels, predictions, k=15, labels=range(predictions.shape[1])
        )

        if "test" == evaluate_type:
            ndcg: str = "test_ndcg"
            ndcg_3: str = "test_ndcg_3"
            ndcg_5: str = "test_ndcg_5"
            ndcg_10: str = "test_ndcg_10"
            ndcg_15: str = "test_ndcg_15"
            acc: str = "test_acc"
            acc_3: str = "test_acc_3"
            acc_5: str = "test_acc_5"
            acc_10: str = "test_acc_10"
            acc_15: str = "test_acc_15"
        elif "validation" == evaluate_type:
            ndcg: str = "valid_ndcg"
            ndcg_3: str = "valid_ndcg_3"
            ndcg_5: str = "valid_ndcg_5"
            ndcg_10: str = "valid_ndcg_10"
            ndcg_15: str = "valid_ndcg_15"
            acc: str = "valid_acc"
            acc_3: str = "valid_acc_3"
            acc_5: str = "valid_acc_5"
            acc_10: str = "valid_acc_10"
            acc_15: str = "valid_acc_15"
        else:
            raise Exception('Please choose "test" or "validation" type')

        print(
            "\033[1;32m"
            + "The"
            + evaluate_type
            + "ndcg is: "
            + "{:.5f}".format(ndcg_res)
            + "\033[0m"
        )
        print(
            "\033[1;32m"
            + "The"
            + evaluate_type
            + "accuracy is: "
            + "{:.5f}".format(acc_res)
            + "\033[0m"
        )
        return {
            ndcg: ndcg_res,
            ndcg_3: ndcg_res_3,
            ndcg_5: ndcg_res_5,
            ndcg_10: ndcg_res_10,
            ndcg_15: ndcg_res_15,
            acc: acc_res,
            acc_3: acc_res_3,
            acc_5: acc_res_5,
            acc_10: acc_res_10,
            acc_15: acc_res_15,
        }


def main(config=None):
    with wandb.init(project=project_name):
        if config is not None:
            wandb.config.update(config)
        config = wandb.config
        print(config)
        args = {
            "batch_size": config.batch_size,
            "lr": config.learning_rate,
            "emb_dim": config.emb_dim,
            "dataset": config.dataset,
            "task": config.task,
        }
        wandb.config.update(args)
        args["device_str"] = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        args["model_save_dir"] = "../save_model_ckp"
        args["optimizer"] = "adam"
        args["save_name"] = f"mf_{config.dataset}_{config.task}_{config.learning_rate}_{config.batch_size}.bin"
        args["max_epoch"] = 100
        MF_disease = MF_train(args)
        MF_disease.train()
        MF_disease.test()


def sweep():
    # model_name = "MF"
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
                "learning_rate": {"values": [0.05, 0.01, 0.005]},
                "emb_dim": {"values": [256]},
                "batch_size": {"values": [64, 128, 256]},
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
        "batch_size": 128,
        "model_name": "MF",
        "task": "output link prediction dataset",
        "dataset": "Disease",
    }
    main(config)
