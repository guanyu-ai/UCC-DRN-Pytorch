import os
import ast
import json
import hydra
from hydra import compose, initialize
import torch
import torch.nn as nn
import torch.nn.functional as F

import mlflow
import optuna
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import DRNOnlyModel, UCCModel
from dataset import MnistEncodedDataset
from omegaconf import DictConfig
from utils import get_or_create_experiment, parse_experiment_runs_to_optuna_study
# from optimizers import SGLD

torch.autograd.set_detect_anomaly(True)


# set random seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_parent_model(device):
    model_path = "outputs\\2024-03-01\\14-44-08"
    ucc_cfg = OmegaConf.load(os.path.join(model_path, ".hydra\\config.yaml"))
    model = UCCModel(ucc_cfg)
    state_dict = torch.load(os.path.join(model_path, "mnist_ucc_best.pth"), weights_only=False)["model_state_dict"]
    model.load_state_dict(state_dict)
    parent_model = model.to(device)
    parent_model.eval()
    return parent_model


def init_model_and_optimizer(args, model_cfg, device):
    model = DRNOnlyModel(model_cfg).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    return model, optimizer


def init_dataloader(args):
    train_dataset_len = args.train_num_steps * args.batch_size
    train_dataset = MnistEncodedDataset(
        mode="train",
        num_instances=args.num_instances,
        num_samples_per_class=args.num_samples_per_class,
        digit_arr=list(range(args.ucc_end-args.ucc_start+1)),
        ucc_start=args.ucc_start,
        ucc_end=args.ucc_end,
        length=train_dataset_len,
    )
    val_dataset_len = args.val_num_steps * args.batch_size
    val_dataset = MnistEncodedDataset(
        mode="val",
        num_instances=args.num_instances,
        num_samples_per_class=args.num_samples_per_class,
        digit_arr=list(range(args.ucc_end-args.ucc_start+1)),
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


def evaluate(model, parent_model, val_loader, device)-> Tuple[np.float32, np.float32]:
    T=2
    model.eval()
    val_loss_list = []
    val_acc_list = []
    with torch.no_grad():
        for batch_samples, batch_labels in val_loader:
            batch_samples = batch_samples.to(device)
            batch_labels = batch_labels.to(device)
            ucc_logits = model(batch_samples)
            parent_logits = parent_model.ucc_classifier(parent_model.kde(batch_samples, parent_model.num_nodes, parent_model.sigma))
            ucc_loss = F.cross_entropy(ucc_logits, batch_labels)
            
            soft_targets = nn.functional.softmax(parent_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(ucc_logits / T, dim=-1)
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
            loss = 0.75*ucc_loss + 0.25*soft_targets_loss
            # acculate accuracy
            _, ucc_predicts = torch.max(ucc_logits, dim=1)
            acc = torch.sum(ucc_predicts == batch_labels).item() / len(batch_labels)
            val_acc_list.append(acc)
            val_loss_list.append(loss.item())
    return np.mean(val_loss_list), np.mean(val_acc_list)


def train(args, model, parent_model, optimizer, lr_scheduler, train_loader, val_loader, device):
    # distillation temperature
    T=2
    # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    parent_model.eval()
    model.train()
    step = 0
    best_eval_acc = 0
    patience = 100
    for batch_samples, batch_labels in tqdm(train_loader):
        batch_samples = batch_samples.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()

        # if model.alpha==1:
        #     ucc_logits = model(batch_samples, batch_labels)
        #     loss:torch.Tensor = model.compute_loss(
        #         labels=batch_labels,
        #         output=ucc_logits
        #     )
        # else:
        # original loss
        ucc_logits = model(batch_samples)
        ucc_loss = model.compute_loss(
            outputs=ucc_logits,
            labels=batch_labels,
        )
        # distillation loss
        with torch.no_grad():
            parent_logits = parent_model.ucc_classifier(parent_model.kde(batch_samples, parent_model.num_nodes, parent_model.sigma))
            
        soft_targets = nn.functional.softmax(parent_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(ucc_logits / T, dim=-1)
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
        loss = 0.75*ucc_loss + 0.25*soft_targets_loss


        loss.backward()
        optimizer.step()
        
        step += 1

        if step % 10 == 0:
            with torch.no_grad():
                _, pred = torch.max(ucc_logits, dim=1)
                accuracy = torch.sum(pred.flatten()==batch_labels.flatten())/len(batch_labels)
            mlflow.log_metrics({
                "train_ucc_loss": ucc_loss.detach().item(), 
                "train_distilation_loss": soft_targets_loss.detach().item(), 
                "total_loss":loss.detach().item(), 
                "train_ucc_acc": float(accuracy)}, step=step)

        if step % args.save_interval == 0:
            eval_loss, eval_acc = evaluate(model, parent_model, val_loader, device)
            print(f"step: {step}, eval loss: {eval_loss}, eval acc: {eval_acc}")
            mlflow.log_metrics({"eval_ucc_loss": loss.item(), "eval_ucc_acc": float(eval_acc)}, step=step)
            # early stop
            if eval_acc > best_eval_acc:
                patience=10
                best_eval_acc = eval_acc
                # save model
                # save_path = os.path.join(output_dir, f"{args.model_name}_best.pth")
                # put eval loss and acc in model state dict
                # save_dict = {
                #     "model_state_dict": model.state_dict(),
                #     "optimizer_state_dict": optimizer.state_dict(),
                #     "eval_loss": eval_loss,
                #     "eval_acc": eval_acc,
                #     "step": step,
                # }
                mlflow.pytorch.log_model(
                    model,
                    "best_model.pth"
                )
                # maybe save optimizer state dict as well
                # torch.save(save_dict, save_path)
            else:
                patience-=1
            if patience<=0:
                break
            model.train()
    print("Training finished!!!")
    return best_eval_acc


def objective(trial:optuna.Trial):
    with mlflow.start_run(nested=True):
        # cfg = OmegaConf.load("../configs/train_drn.yaml")
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="train_drn")
        with open("params.json", "r") as file:
            params_config = json.loads(file.read())
        default_params = {}

        for key, value in params_config.items():
            if value["type"]=="int":
                v = trial.suggest_int(key, value["range"][0], value["range"][1])
            else:
                v = trial.suggest_float(key, value["range"][0], value["range"][1])
            for a in value["aliases"]:
                exec(f"cfg.{a} = {v}")

        mlflow.log_dict(dict(OmegaConf.to_object(cfg)), "config.yaml")
        params = trial.params
        for key, value in params.items():
            mlflow.log_param(key=key, value=value)
        args = cfg.args
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        model, optimizer = init_model_and_optimizer(args, cfg, device)
        parent_model = load_parent_model(device)
        train_loader, val_loader = init_dataloader(args)
        print(cfg)
        best_acc = train(args, model, parent_model, optimizer, None, train_loader, val_loader, device)
    return best_acc


if __name__ == "__main__":
    mlflow.set_tracking_uri("mlruns")
    experiment_id = get_or_create_experiment(experiment_name="ucc-drn-distil")
    mlflow.set_experiment(experiment_id=experiment_id)

    study = parse_experiment_runs_to_optuna_study(
        experiment_name="ucc-drn-distil",
        study_name="ucc-drn-distilr")
    study.optimize(func=objective, n_trials=30, show_progress_bar=True)
