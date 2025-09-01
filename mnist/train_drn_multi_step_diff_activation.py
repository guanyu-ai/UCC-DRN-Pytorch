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
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from model import UCCDRNModel
from dataset import MnistDataset
from utils import get_or_create_experiment, parse_experiment_runs_to_optuna_study
torch.autograd.set_detect_anomaly(True)


# Concern was vanishing gradients at the encoder model.
# Using Hardtanh(min=0, max=1)
# For this I altered the activation of the encoder so that we dont have non-linearities at the activation after encoder.
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_model_and_optimizer(args, model_cfg, device):
    model = UCCDRNModel(model_cfg).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    param_group_names = [name for name, _ in model.named_parameters()]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, amsgrad=True)
    # for name, param_group in zip(param_group_names, optimizer.param_groups):
    #     if "encoder" in name:
    #         param_group['lr'] = args.learning_rate*args.lr_multiplier
    #     print(f'    {name}: {param_group["lr"]}')
    return model, optimizer


def init_dataloader(args):
    # assert args.dataset in [
    #     "mnist",
    #     "camelyon",
    # ], "Mode should be either mnist or camelyon"
    # if args.dataset == "mnist":
    train_dataset_len = args.train_num_steps * args.batch_size
    train_dataset = MnistDataset(
        mode="train",
        num_instances=args.num_instances,
        num_samples_per_class=args.num_samples_per_class,
        digit_arr=list(range(args.ucc_end-args.ucc_start+1)),
        ucc_start=args.ucc_start,
        ucc_end=args.ucc_end,
        length=train_dataset_len,
    )
    val_dataset_len = args.val_num_steps * args.batch_size
    val_dataset = MnistDataset(
        mode="val",
        num_instances=args.num_instances,
        num_samples_per_class=args.num_samples_per_class,
        digit_arr=list(range(args.ucc_end-args.ucc_start+1)),
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


def evaluate(model, val_loader, device, ae_mode, clf_mode) -> dict:
    model.eval()
    val_ae_loss_list = []
    val_ucc_loss_list = []
    val_acc_list = []
    rec_criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_samples, batch_labels in val_loader:
            batch_samples = batch_samples.to(device)
            batch_labels = batch_labels.to(device)
            if ae_mode:
                batch_size, num_instances, num_channel, patch_size, _ = batch_samples.shape
                x = batch_samples.view(-1, num_channel,
                                       batch_samples.shape[-2], batch_samples.shape[-1])
                features = model.encoder(x)
                reconstruction = model.decoder(features)
                reconstruction = reconstruction.view(batch_size, num_instances,
                                    1, patch_size, patch_size)
                ae_loss = rec_criterion(batch_samples, reconstruction)
                val_ae_loss_list.append(ae_loss.item())

            if clf_mode:
                ucc_logits, _ = model(batch_samples)
                ucc_val_loss = F.cross_entropy(ucc_logits, batch_labels)
                # acculate accuracy
                _, ucc_predicts = torch.max(ucc_logits, dim=1)
                acc = torch.sum(
                    ucc_predicts == batch_labels).item() / len(batch_labels)
                val_acc_list.append(acc)
                val_ucc_loss_list.append(ucc_val_loss.item())

        if ae_mode and clf_mode:
            return {
                "eval_ae_loss": np.round(np.mean(val_ae_loss_list), 5),
                "eval_ucc_loss": np.round(np.mean(val_ucc_loss_list), 5),
                "eval_ucc_acc": np.round(np.mean(val_acc_list), 5)
            }
        elif ae_mode:
            return {
                "eval_ae_loss": np.round(np.mean(val_ae_loss_list), 5),
            }
        elif clf_mode:
            return {
                "eval_ucc_loss": np.round(np.mean(val_ucc_loss_list), 5),
                "eval_ucc_acc": np.round(np.mean(val_acc_list), 5)
            }


def train(args, model, optimizer, lr_scheduler, train_loader, val_loader, device):
    print("training")
    # mlflow.pytorch.log_model(model, "init_model")
    # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model.train()
    step = 0
    best_eval_acc = 0
    patience = 2
    ae_steps = 500

    rec_criterion = nn.MSELoss()
    if step == 0:
        mlflow.pytorch.log_model(
            model,
            "best_model.pth"
        )
    for batch_samples, batch_labels in train_loader:
        batch_samples = batch_samples.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()

        if ae_steps > 0:
            batch_size, num_instances, num_channel, patch_size, _ = batch_samples.shape
            x = batch_samples.view(-1, num_channel,
                                   batch_samples.shape[-2], batch_samples.shape[-1])
            feature = model.encoder(x)

            reconstruction = model.decoder(feature)
            reconstruction = reconstruction.view(
                batch_size, num_instances, 1, patch_size, patch_size)
            loss = rec_criterion(batch_samples, reconstruction)
            ae_loss = loss
            ae_steps -= 1

        if ae_steps == 0:
            ucc_logits, reconstruction = model(batch_samples, batch_labels)
            loss: torch.Tensor = F.cross_entropy(
                ucc_logits,
                batch_labels
            )
            ae_loss = F.mse_loss(batch_samples, reconstruction)

        loss.backward()

        optimizer.step()

        step += 1

        if step % 10 == 0:
            with torch.no_grad():
                metric_dict = {}
                grad_log = {name: torch.mean(param.grad).cpu().item(
                ) for name, param in model.named_parameters() if isinstance(param.grad, torch.Tensor)}
                mlflow.log_metrics(grad_log, step=step)
                metric_dict["train_ae_loss"] = np.round(
                    ae_loss.detach().item(), 5)
                if ae_steps == 0:
                    _, pred = torch.max(ucc_logits, dim=1)
                    accuracy = torch.sum(
                        pred.flatten() == batch_labels.flatten())/len(batch_labels)
                    metric_dict["train_ucc_loss"] = np.round(
                        loss.detach().item(), 5)
                    metric_dict["train_ucc_acc"] = np.round(float(accuracy), 5)

            mlflow.log_metrics(metric_dict, step=step)

        if step % args.save_interval == 0:

            eval_metric_dict = evaluate(
                model,
                val_loader,
                device,
                ae_mode=True,
                clf_mode=(ae_steps == 0))

            print(
                f"step: {step}," + ",".join([f"{key}: {value}"for key, value in eval_metric_dict.items()]))
            mlflow.log_metrics(eval_metric_dict, step=step)
            # early stop
            if ae_steps == 0:
                eval_acc = eval_metric_dict["eval_ucc_acc"]
                if eval_acc > best_eval_acc:
                    patience = 2
                    best_eval_acc = eval_acc
                    mlflow.pytorch.log_model(
                        model,
                        "best_model.pth"
                    )
                else:
                    patience -= 1

            if patience <= 0:
                break
            if step == 10000:
                break
            model.train()

    print("Training finished!!!")
    return best_eval_acc

def objective(trial: optuna.Trial, cfg_name):
    print(cfg_name)
    with mlflow.start_run(nested=True):
        # cfg = OmegaConf.load("../configs/train_drn.yaml")
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name=cfg_name)
        # with open("params.json", "r") as file:
        #     params_config = json.loads(file.read())

        defaults = {
            # "init_method": {
            #     "type": "categorical",
            #     "range": ["uniform", "normal", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"],
            #     "aliases": [
            #         "model.drn.init_method",
            #     ]
            # },
            # "num_bins": {
            #     "type": "int",
            #     "value": 10,
            #     "range": [5, 100],
            #     "aliases": [
            #         "model.drn.num_bins",
            #         "args.num_bins",
            #         "model.kde_model.num_bins"
            #     ]
            # },
            "lr": {
                "type": "float",
                "value": 0.005,
                "range": [0.008, 0.08],
                "aliases": ["args.learning_rate"]
            },
            "hidden_q": {
                "type": "int",
                "value": 100,
                "range": [4, 100],
                "aliases": ["model.drn.hidden_q"]
            },
            "num_layers": {
                "type": "int",
                "value": 2,
                "range": [1, 10],
                "aliases": ["model.drn.num_layers"]
            },
            "num_nodes": {
                "type": "int",
                "value": 9,
                "range": [1, 10],
                "aliases": ["model.drn.num_nodes"]
            }
        }
        for key, value in defaults.items():
            if "value" in value:
                v = value["value"]
            else:
                if value["type"] == "int":
                    v = trial.suggest_int(
                        key, value["range"][0], value["range"][1])
                elif value["type"] == "categorical":
                    v = trial.suggest_categorical(key, value["range"])
                else:
                    v = trial.suggest_float(
                        key, value["range"][0], value["range"][1])
            for a in value["aliases"]:
                exec(f"cfg.{a} = '{v}'") if isinstance(
                    v, str) else exec(f"cfg.{a} = {v}")

        print(cfg)
        mlflow.log_dict(dict(OmegaConf.to_object(cfg)), "config.yaml")

        params = trial.params
        for key, value in params.items():
            mlflow.log_param(key=key, value=value)

        args = cfg.args
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        model, optimizer = init_model_and_optimizer(args, cfg, device)
        train_loader, val_loader = init_dataloader(args)
        mlflow.pytorch.log_model(model, "init_model")
        best_acc = train(args, model, optimizer, None,
                         train_loader, val_loader, device)
    return best_acc

if __name__=="__main__":
    print("run this once")
    mlflow.set_tracking_uri("mlruns")
    experiment_name = "ucc-drn-multi-step-hardtanh"
    experiment_id = get_or_create_experiment(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    cfg_name = "train_drn_multi_step"
    study = parse_experiment_runs_to_optuna_study(
        experiment_name=experiment_name,
        study_name=experiment_name,
        # cfg_name=cfg_name,
        # params_file="params-init.json"
    )
    study.optimize(lambda trial: objective(trial, cfg_name), n_trials=100, show_progress_bar=True)
