
import copy
import json
from hydra import compose, initialize
import torch
import torch.nn.functional as F
import mlflow
import optuna
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.optim import Adam

from loss import Wasserstein_1d_Loss, Wasserstein_2_1d_Loss
from model import UCCDRNModel, DRNOnlyModel
from dataset import MnistDataset, MnistEncodedDataset
from omegaconf import DictConfig
from utils import get_or_create_experiment, parse_experiment_runs_to_optuna_study
torch.autograd.set_detect_anomaly(True)


# set random seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_model_and_optimizer(args, model_cfg, device):
    model = UCCDRNModel(model_cfg).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    return model, optimizer


def init_dataloader(args):
    train_dataset_len = args.train_num_steps * args.batch_size
    train_dataset = MnistDataset(
        mode="train",
        num_instances=args.num_instances,
        num_samples_per_class=args.num_samples_per_class,
        digit_arr=list(range(0, 10)),
        ucc_start=args.ucc_start,
        ucc_end=args.ucc_end,
        length=train_dataset_len,
    )
    val_dataset_len = args.val_num_steps * args.batch_size
    val_dataset = MnistDataset(
        mode="val",
        num_instances=args.num_instances,
        num_samples_per_class=args.num_samples_per_class,
        digit_arr=list(range(0, 10)),
        ucc_start=args.ucc_start,
        ucc_end=args.ucc_end,
        length=val_dataset_len,
    )
    # create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return train_loader, val_loader


def evaluate(model, val_loader, device):
    wass_criterion = Wasserstein_2_1d_Loss()
    model.eval()
    val_wass_loss_list = []
    val_ae_loss_list = []
    val_ucc_loss_list = []
    val_acc_list = []
    with torch.no_grad():
        for batch_samples, batch_labels in val_loader:
            batch_samples = batch_samples.to(device)
            batch_labels = batch_labels.to(device)

            ucc_logits, reconstruction = model(
                batch_samples, return_reconstruction=True)

            w_loss = wass_criterion(
                ucc_logits, batch_labels, model.num_classes)
            val_wass_loss_list.append(w_loss.item())

            ucc_loss = F.cross_entropy(ucc_logits, batch_labels)
            val_ucc_loss_list.append(ucc_loss.item())

            ae_loss = F.mse_loss(batch_samples, reconstruction)
            val_ae_loss_list.append(ae_loss.item())

            # acculate accuracy
            # _, batch_labels = torch.max(batch_labels, dim=1)

            _, ucc_predicts = torch.max(ucc_logits, dim=1)
            acc = torch.sum(
                ucc_predicts == batch_labels).item() / len(batch_labels)
            val_acc_list.append(acc)
    return {
        "eval_ucc_wass_loss": np.round(np.mean(val_wass_loss_list), 5),
        "eval_ae_loss": np.round(np.mean(val_ae_loss_list), 5),
        "eval_ucc_loss": np.round(np.mean(val_ucc_loss_list), 5),
        "eval_ucc_acc": np.round(np.mean(val_acc_list), 5)
    }


def train(args, model, optimizer, lr_scheduler, train_loader, val_loader, device):
    step = 0
    best_eval_acc = 0
    wass_criterion = Wasserstein_2_1d_Loss()
    for batch_samples, batch_labels in train_loader:
        batch_samples = batch_samples.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()

        ucc_logits, reconstruction = model(
            batch_samples, return_reconstruction=True)
        ae_loss = F.mse_loss(reconstruction, batch_samples)
        w_loss = wass_criterion(ucc_logits, batch_labels, model.num_classes)
        # ucc_loss = model.compute_loss(
        #     output=ucc_logits,
        #     labels=batch_labels,
        # )
        loss = (1-model.alpha)*w_loss + model.alpha*ae_loss

        loss.backward()
        optimizer.step()

        step += 1

        if step % 10 == 0:
            with torch.no_grad():
                grad_log = {name: torch.mean(param.grad).cpu().item(
                ) for name, param in model.named_parameters() if isinstance(param.grad, torch.Tensor)}
                mlflow.log_metrics(grad_log, step=step)
                _, pred = torch.max(ucc_logits, dim=1)
                ucc_loss = F.cross_entropy(
                    ucc_logits,
                    batch_labels,
                )
                # w_loss = wass_criterion(ucc_logits, batch_labels, model.num_classes)
                accuracy = torch.sum(
                    pred.flatten() == batch_labels.flatten())/len(batch_labels)
            metric_dict = {"train_ae_loss": ae_loss.detach().item(),
                           "train_ucc_loss": ucc_loss.detach().item(),
                           "train_ucc_wass_loss": w_loss.detach().item(),
                           "train_ucc_acc": float(accuracy)}
            print(f"Step {step}: {metric_dict}")
            mlflow.log_metrics(metric_dict, step=step)

        if step % args.save_interval == 0:
            eval_metric_dict = evaluate(model, val_loader, device)
            eval_acc = eval_metric_dict["eval_ucc_acc"]
            print(
                f"step: {step}," + ",".join([f"{key}: {value}"for key, value in eval_metric_dict.items()]))
            mlflow.log_metrics(eval_metric_dict, step=step)
            eval_acc = eval_metric_dict["eval_ucc_acc"]
            # early stop
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                mlflow.log_metric("best_eval_acc", best_eval_acc)
                mlflow.pytorch.log_model(model, artifact_path="best_model")
                torch.save(optimizer, "optimizer.pt")
                mlflow.log_artifact("optimizer.pt", "optimizer.pt")
            if step==400000:
                break
    print("Training finished!!!")
    return best_eval_acc

# def objective(trial:optuna.Trial):
#      with mlflow.start_run(nested=True):
#             # cfg = OmegaConf.load("../configs/train_drn.yaml")
#         with initialize(version_base=None, config_path="../configs"):
#             cfg = compose(config_name="train_drn")
#         with open("params.json", "r") as file:
#             params_config = json.loads(file.read())
#         params_config["num_bins"]["value"]=4
#         params_config["hidden_q"]["value"]=4
#         params_config["num_layers"]["value"]=1
#         for key, value in params_config.items():
#             if "value" in value:
#                 v = value["value"]
#             else:
#                 if value["type"]=="int":
#                     v = trial.suggest_int(key, value["range"][0], value["range"][1])
#                 else:
#                     v = trial.suggest_float(key, value["range"][0], value["range"][1])
#             for a in value["aliases"]:
#                 exec(f"cfg.{a} = {v}")

#         print(cfg)
#         mlflow.log_dict(dict(OmegaConf.to_object(cfg)), "config.yaml")
#         params = trial.params
#         for key, value in params.items():
#             mlflow.log_param(key=key, value=value)

#         args = cfg.args
#         device = torch.device("cuda" if torch.cuda.is_available() else "mps")
#         model, optimizer = init_model_and_optimizer(args, cfg, device)
#         train_loader, val_loader = init_dataloader(args)
#         best_acc = train(args, model, optimizer, None,
#                         train_loader, val_loader, device)
#         return best_acc


if __name__ == "__main__":
    mlflow.set_tracking_uri("mlruns")
    run_name = "ucc-drn-2-wasserstein-loss"
    experiment_id = get_or_create_experiment(experiment_name=run_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    cfg_name = "train_drn"
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=cfg_name)

    with mlflow.start_run(nested=True):
        mlflow.log_dict(dict(OmegaConf.to_object(cfg)), "config.yaml")
        # model = torch.load(f"mlruns/189454739472380536/b76e52db991c4b90a51eb9b8da9fc6ab/artifacts/best_model.pth/data/model.pth")
        args = cfg.args
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        print(cfg.model.drn)
        model, optimizer = init_model_and_optimizer(args, cfg, device)
        train_loader, val_loader = init_dataloader(args)
        mlflow.pytorch.log_model(model, "init_model")
        mlflow.log_params({"learning_rate": args.learning_rate})
        mlflow.log_params({"num_layers": cfg.model.drn.num_layers})
        mlflow.log_params({"num_nodes": cfg.model.drn.num_nodes})
        best_acc = train(args, model, optimizer, None,
                         train_loader, val_loader, device)
