# using optimization to find the optimal mean and variance for normal initialization
from copy import deepcopy
from hydra import compose, initialize
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import optuna
import numpy as np
from tqdm import tqdm
from typing import Tuple
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

from model import UCCDRNModel
from dataset import MnistDataset
from utils import get_or_create_experiment, parse_experiment_runs_to_optuna_study
torch.autograd.set_detect_anomaly(True)


    
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_model_and_optimizer(args, model_cfg, device):
    model = UCCDRNModel(model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001)
    return model, optimizer

def load_model_and_optimizer(experiment_id, run_id):
    model = torch.load(f"mlruns/{experiment_id}/{run_id}/artifacts/best_model/data/model.pth", weights_only=False)
    old_optimizer = torch.load(f"mlruns/{experiment_id}/{run_id}/artifacts/best_optimizer.pth/best_optimizer.pth", weights_only=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(old_optimizer.state_dict())
    with open(f"mlruns/{experiment_id}/{run_id}/metrics/eval_ucc_acc") as file:
        lines = file.readlines()
        step = int(lines[-1].split(" ")[-1])
    return model, optimizer, step

def init_dataloader(args):
    train_dataset_len = args.train_num_steps * args.batch_size
    train_dataset = MnistDataset(
        mode="train",
        num_instances=args.num_instances,
        num_samples_per_class=args.num_samples_per_class,
        digit_arr=list(range(0,10)),
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
    # create dataloader
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

def evaluate(model, val_loader, lr_scheduler, device):
    model.eval()
    val_ae_loss_list = []
    val_ucc_loss_list = []
    val_acc_list = []
    with torch.no_grad():
        for batch_samples, batch_labels in val_loader:
            batch_samples = batch_samples.to(device)
            batch_labels = batch_labels.to(device)

            batch_size, num_instances, num_channel, patch_size, _ = batch_samples.shape
            x = batch_samples.view(-1, num_channel,
                                    batch_samples.shape[-2], batch_samples.shape[-1])
            features = model.encoder(x)
            reconstruction = model.decoder(features)
            reconstruction = reconstruction.view(batch_size, num_instances,
                                1, patch_size, patch_size)


            ucc_logits, reconstruction = model(batch_samples, return_reconstruction=True)

            ucc_loss = F.cross_entropy(ucc_logits, batch_labels)
            val_ucc_loss_list.append(ucc_loss.item())

            ae_loss = F.mse_loss(batch_samples, reconstruction)
            val_ae_loss_list.append(ae_loss.item())

            # acculate accuracy
            _, ucc_predicts = torch.max(ucc_logits, dim=1)
            acc = torch.sum(ucc_predicts == batch_labels).item() / len(batch_labels)
            val_acc_list.append(acc)
        lr_scheduler.step(np.mean(val_ucc_loss_list))
        print(f"learning rate:{lr_scheduler.get_last_lr()}")
    return {
                "eval_ae_loss": np.round(np.mean(val_ae_loss_list), 5),
                "eval_ucc_loss": np.round(np.mean(val_ucc_loss_list), 5),
                "eval_ucc_acc": np.round(np.mean(val_acc_list), 5)
            }

def train(args, model, optimizer, lr_scheduler, train_loader, val_loader, device, step=0):
    
    print("training")
    model.train()
    best_eval_acc = 0
    for batch_samples, batch_labels in train_loader:
        batch_samples = batch_samples.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        ucc_logits, reconstruction = model(batch_samples, return_reconstruction=True)
        ucc_loss, ae_loss, loss = model.compute_loss(batch_samples, batch_labels, ucc_logits, reconstruction, return_losses=True)

        loss.backward()

        optimizer.step()

        step += 1

        if step % 10 == 0:
            with torch.no_grad():
                metric_dict = {}
                grad_log = {name: torch.mean(param.grad).cpu().item(
                ) for name, param in model.named_parameters() if isinstance(param.grad, torch.Tensor)}
                mlflow.log_metrics(grad_log, step=step)
                metric_dict["train_ae_loss"] = np.round(ae_loss.detach().item(), 5)
                _, pred = torch.max(ucc_logits, dim=1)
                accuracy = torch.sum(pred.flatten() == batch_labels.flatten())/len(batch_labels)
                metric_dict["train_ucc_loss"] = np.round(ucc_loss.detach().item(), 5)
                metric_dict["train_ucc_acc"] = np.round(float(accuracy), 5)
                metric_dict["loss"] = np.round(float(loss), 5)
            mlflow.log_metrics(metric_dict, step=step)

        if step % args.save_interval == 0:
            eval_metric_dict = evaluate(
                model,
                val_loader,
                lr_scheduler,
                device)
            print(f"step: {step}," + ",".join([f"{key}: {value}"for key, value in eval_metric_dict.items()]))
            mlflow.log_metrics(eval_metric_dict, step=step)
            # early stop
            eval_acc = eval_metric_dict["eval_ucc_acc"]
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                mlflow.log_metric("best_eval_acc", best_eval_acc)
                mlflow.pytorch.log_model(
                    model,
                    "best_model"
                )
                torch.save(optimizer, "best_optimizer.pth")
                mlflow.log_artifact(
                    "best_optimizer.pth",
                    "best_optimizer.pth"
                )
            if step == 200000:
                break
            model.train()

    print("Training finished!!!")
    return best_eval_acc

if __name__ == "__main__":
    mlflow.set_tracking_uri("mlruns")
    experiment_name = "ucc-drn-goated-init"
    experiment_id = get_or_create_experiment(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    run_id = "48ebf24973744164844891aa7584ff54"
    cfg_name = "train_drn"
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=cfg_name)
    # device = torch.device("mps")
    # model = torch.load(f"mlruns/189454739472380536/b76e52db991c4b90a51eb9b8da9fc6ab/artifacts/best_model/data/model.pth", map_location=device, weights_only=False)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.008)
    with mlflow.start_run(nested=True, run_name=f"{run_id}_resume"):
        mlflow.log_dict(dict(OmegaConf.to_object(cfg)), "config.yaml")
        args = cfg.args
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        # model = torch.load(f"mlruns/189454739472380536/b76e52db991c4b90a51eb9b8da9fc6ab/artifacts/best_model.pth/data/model.pth")
        model, optimizer, step = load_model_and_optimizer(experiment_id=experiment_id, run_id=run_id)
        
        train_loader, val_loader = init_dataloader(args)
        # mlflow.pytorch.log_model(model, "init_model")
        lr_scheduler = ReduceLROnPlateau(optimizer, "min",factor=0.5, threshold=0.01, patience=12, min_lr=0.00001)
        best_acc = train(args, model, optimizer, lr_scheduler,
                        train_loader, val_loader, device, step)
