
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from model_ori import UCCModel
from dataset import MnistDataset
from hydra import initialize, compose
from utils import get_or_create_experiment
from omegaconf import OmegaConf
torch.autograd.set_detect_anomaly(True)

def KL_div(y, y_hat):
    # checked
    y = torch.clamp(y, 1e-15, 1.0)
    y_hat = torch.clamp(y_hat, 1e-15, 1.0)
    return torch.sum(y * torch.log(y/y_hat), dim=1)


def Ldjs(P, Q):
    # checked
    M = (P+Q)/2.0
    l = 0.5*KL_div(P, M)+0.5*KL_div(Q, M)
    return torch.mean(l)/torch.log(torch.tensor(2.0))


# set random seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_model_and_optimizer(args, model_cfg, device):
    model = UCCModel(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
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
    # else:
    #     train_dataset_len = args.train_num_steps * args.batch_size
    #     train_dataset = CamelyonUCCDataset(
    #         mode="train",
    #         num_instances=args.num_instances,
    #         data_augment=args.data_augment,
    #         patch_size=args.patch_size,
    #         dataset_len=train_dataset_len,
    #     )
    #     val_dataset_len = args.val_num_steps * args.batch_size
    #     val_dataset = CamelyonUCCDataset(
    #         mode="val",
    #         num_instances=args.num_instances,
    #         data_augment=args.data_augment,
    #         patch_size=args.patch_size,
    #         dataset_len=val_dataset_len,
    #     )
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
    val_jsd_loss_list = []
    val_ae_loss_list = []
    val_ucc_loss_list = []
    val_acc_list = []
    with torch.no_grad():
        for batch_samples, batch_labels in val_loader:
            batch_samples = batch_samples.to(device)
            batch_labels = batch_labels.to(device)
            batch_labels_one_hot = F.one_hot(batch_labels, num_classes=model.num_classes)

            ucc_logits, reconstruction = model(
                batch_samples, return_reconstruction=True)

            jsd_loss = Ldjs(ucc_logits, batch_labels_one_hot)
            val_jsd_loss_list.append(jsd_loss.item())

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
        "eval_ucc_djs_loss": np.round(np.mean(val_jsd_loss_list), 5),
        "eval_ae_loss": np.round(np.mean(val_ae_loss_list), 5),
        "eval_ucc_loss": np.round(np.mean(val_ucc_loss_list), 5),
        "eval_ucc_acc": np.round(np.mean(val_acc_list), 5)
    }


def train(args, model, optimizer, lr_scheduler, train_loader, val_loader, device):
    model.train()
    step = 0
    best_eval_acc = 0
    best_eval_loss = 2
    for batch_samples, batch_labels in train_loader:
        batch_samples = batch_samples.to(device)
        batch_labels = batch_labels.to(device)
        batch_labels_one_hot = F.one_hot(batch_labels, 4)
        optimizer.zero_grad()
        ucc_logits, reconstruction = model(batch_samples, return_reconstruction=True)
        ae_loss = F.mse_loss(batch_samples, reconstruction)
        djs_loss = Ldjs(ucc_logits, batch_labels_one_hot)
        # ucc_loss, ae_loss, loss = model.compute_loss(batch_samples, batch_labels, ucc_logits, reconstruction, return_losses=True)
        loss = (1-model.alpha)*ae_loss+model.alpha*djs_loss
        
        loss.backward()
        optimizer.step()
        step += 1

        if step % 10 == 0:
            with torch.no_grad():
                ucc_loss = F.cross_entropy(ucc_logits, batch_labels)
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
            print(f"Step {step}:", metric_dict)
            # switch to train mode
        if step % args.save_interval == 0:
            eval_metric_dict = evaluate(
                model,
                val_loader,
                lr_scheduler,
                device)
            print(f"Step {step}:" + ",".join([f"{key}: {value}"for key, value in eval_metric_dict.items()]))
            mlflow.log_metrics(eval_metric_dict, step=step)
            # early stop
            eval_acc = eval_metric_dict["eval_ucc_acc"]
            eval_loss = eval_metric_dict["eval_ae_loss"]
            if eval_acc > best_eval_acc or eval_loss <best_eval_loss:
                if eval_loss <best_eval_loss:
                    best_eval_loss = eval_loss
                    
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                mlflow.log_metric("best_eval_acc", best_eval_acc, step=step)
                mlflow.log_metric("best_eval_loss", best_eval_loss, step=step)
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


if __name__ == "__main__":
    mlflow.set_tracking_uri("mlruns")
    run_name = "ucc-ori-jsd"
    experiment_id = get_or_create_experiment(experiment_name=run_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    cfg_name = "train"
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=cfg_name)
    with mlflow.start_run(nested=True):
        mlflow.log_dict(dict(OmegaConf.to_object(cfg)), "config.yaml")
        args = cfg.args
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        model, optimizer = init_model_and_optimizer(args, cfg, device)
        train_loader, val_loader = init_dataloader(args)
        mlflow.pytorch.log_model(model, "init_model")
        best_acc = train(args, model, optimizer, None,
                        train_loader, val_loader, device)
