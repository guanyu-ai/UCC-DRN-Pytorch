
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
from torch.optim import AdamW

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
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, amsgrad=True)
    return model, optimizer

def init_dataloader(args):
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


def evaluate(model, val_loader, device)-> Tuple[np.float32, np.float32]:
    model.eval()
    val_loss_list = []
    val_acc_list = []
    with torch.no_grad():
        for batch_samples, batch_labels in val_loader:
            batch_samples = batch_samples.to(device)
            batch_labels = batch_labels.to(device)
            ucc_logits = model(batch_samples)
            ucc_loss = F.cross_entropy(ucc_logits, batch_labels)

            # acculate accuracy
            _, ucc_predicts = torch.max(ucc_logits, dim=1)
            acc = torch.sum(ucc_predicts == batch_labels).item() / len(batch_labels)
            val_acc_list.append(acc)
            val_loss_list.append(ucc_loss.item())
    return np.mean(val_loss_list), np.mean(val_acc_list)


def train(args, model, optimizer, lr_scheduler, train_loader, val_loader, device):
    step = 0
    best_eval_acc = 0
    patience = 2
    for batch_samples, batch_labels in train_loader:
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
            output=ucc_logits,
            labels=batch_labels,
        )

        ucc_loss.backward()
        optimizer.step()

        step += 1

        if step % 10 == 0:
            with torch.no_grad():
                grad_log = {name: torch.mean(param.grad).cpu().item() for name, param in model.named_parameters()}
                mlflow.log_metrics(grad_log, step=step)
                _, pred = torch.max(ucc_logits, dim=1)
                accuracy = torch.sum(pred.flatten()==batch_labels.flatten())/len(batch_labels)
            mlflow.log_metrics({
                "train_ucc_loss": ucc_loss.detach().item(),
                "train_ucc_acc": float(accuracy)}, step=step)

        if step % args.save_interval == 0:
            eval_loss, eval_acc = evaluate(model, val_loader, device)
            print(f"step: {step}, eval loss: {eval_loss}, eval acc: {eval_acc}")
            mlflow.log_metrics({"eval_ucc_loss": eval_loss.item(), "eval_ucc_acc": float(eval_acc)}, step=step)
            # early stop
            if eval_acc > best_eval_acc:
                patience=20
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
            if patience<=0  or step>5000:
                break
    print("Training finished!!!")
    return best_eval_acc

def objective(trial:optuna.Trial):
     with mlflow.start_run(nested=True):
            # cfg = OmegaConf.load("../configs/train_drn.yaml")
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="train_drn")
        with open("params.json", "r") as file:
            params_config = json.loads(file.read())
        params_config["num_bins"]["value"]=4
        params_config["hidden_q"]["value"]=4
        params_config["num_layers"]["value"]=1
        for key, value in params_config.items():
            if "value" in value:
                v = value["value"]
            else:
                if value["type"]=="int":
                    v = trial.suggest_int(key, value["range"][0], value["range"][1])
                else:
                    v = trial.suggest_float(key, value["range"][0], value["range"][1])
            for a in value["aliases"]:
                exec(f"cfg.{a} = {v}")

        print(cfg)
        mlflow.log_dict(dict(OmegaConf.to_object(cfg)), "config.yaml")
        params = trial.params
        for key, value in params.items():
            mlflow.log_param(key=key, value=value)

        args = cfg.args
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        model, optimizer = init_model_and_optimizer(args, cfg, device)
        train_loader, val_loader = init_dataloader(args)
        best_acc = train(args, model, optimizer, None,
                        train_loader, val_loader, device)
        return best_acc

if __name__ =="__main__":
    mlflow.set_tracking_uri("mlruns")
    run_name = "ucc-drn-bin-4-layer-1-local"
    experiment_id = get_or_create_experiment(experiment_name=run_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    study = parse_experiment_runs_to_optuna_study(
        experiment_name=run_name,
        study_name=run_name
        )
    study.optimize(func=objective, n_trials=30, show_progress_bar=True)
