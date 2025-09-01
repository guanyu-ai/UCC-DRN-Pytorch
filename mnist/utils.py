import os
import json
from typing import List
import mlflow
import optuna
import datetime
import numpy as np
import h5py

from omegaconf import OmegaConf
from hydra import initialize, compose
from model_ori import UCCModel
import torch


name_map = {
    'encoder.0.weight': '/model_1/conv2d_1/kernel:0',
    'encoder.0.bias': '/model_1/conv2d_1/bias:0',
    'encoder.1.blocks.0.conv1.weight': '/model_1/conv2d_2/kernel:0',
    'encoder.1.blocks.0.conv1.bias': '/model_1/conv2d_2/bias:0',
    'encoder.1.blocks.0.conv2.weight': '/model_1/conv2d_3/kernel:0',
    'encoder.1.blocks.0.conv2.bias': '/model_1/conv2d_3/bias:0',
    'encoder.2.blocks.0.conv1.weight': '/model_1/conv2d_4/kernel:0',
    'encoder.2.blocks.0.conv1.bias': '/model_1/conv2d_4/bias:0',
    'encoder.2.blocks.0.conv2.weight': '/model_1/conv2d_5/kernel:0',
    'encoder.2.blocks.0.conv2.bias': '/model_1/conv2d_5/bias:0',
    'encoder.2.blocks.0.skip_conv.weight': '/model_1/conv2d_6/kernel:0',
    'encoder.2.blocks.0.skip_conv.bias': '/model_1/conv2d_6/bias:0',
    'encoder.3.blocks.0.conv1.weight': '/model_1/conv2d_7/kernel:0',
    'encoder.3.blocks.0.conv1.bias': '/model_1/conv2d_7/bias:0',
    'encoder.3.blocks.0.conv2.weight': '/model_1/conv2d_8/kernel:0',
    'encoder.3.blocks.0.conv2.bias': '/model_1/conv2d_8/bias:0',
    'encoder.3.blocks.0.skip_conv.weight': '/model_1/conv2d_9/kernel:0',
    'encoder.3.blocks.0.skip_conv.bias': '/model_1/conv2d_9/bias:0',
    'encoder.6.weight': '/model_1/fc_sigmoid/kernel:0',
    'ucc_classifier.0.weight': '/fc_relu1/fc_relu1/kernel:0',
    'ucc_classifier.0.bias': '/fc_relu1/fc_relu1/bias:0',
    'ucc_classifier.2.weight': '/fc_relu2/fc_relu2/kernel:0',
    'ucc_classifier.2.bias': '/fc_relu2/fc_relu2/bias:0',
    'ucc_classifier.4.weight': '/fc_softmax/fc_softmax/kernel:0',
    'ucc_classifier.4.bias': '/fc_softmax/fc_softmax/bias:0'
    }

def setup(cfg, trial):
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H-%M-%S")
    save_path = os.path.join('outputs', date, time)
    config_path = os.path.join(save_path, '.hydra')
    os.makedirs(config_path, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(config_path, 'config.yaml'))
    trial.set_user_attr('path', save_path)
    return save_path

def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def parse_experiment_runs_to_optuna_study(experiment_name:str, study_name:str):
    study = optuna.create_study(study_name=study_name, direction="maximize")
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train_drn")
    with open("params.json", "r") as file:
        params_config:dict = json.loads(file.read())
    params = {}
    for key, value in params_config.items():
        value_= eval(str(f"cfg.{value['aliases'][0]}"))
        if isinstance(value_, int):
            distribution = optuna.distributions.IntDistribution(*value["range"])
        elif isinstance(value_, float):
            distribution = optuna.distributions.FloatDistribution(*value["range"])
        else:
            distribution = None
        params[key] = {
            "value": value_,
            "distribution": distribution
        }


    if experiment := mlflow.get_experiment_by_name(experiment_name):
        experiment_id = experiment.experiment_id
        runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="list")
        trials = []

        run_ids = [run.info.run_id for run in runs]
        run_data = [run.data for run in runs]
        for run_id, run_data in zip(run_ids, run_data):
            with mlflow.start_run(run_id) as inner_run:
                run_params = run_data.params
                not_included = set(params.keys()) - set(run_params.keys())
                for key in not_included:
                    mlflow.log_param(key=key, value=params[key]["value"])
                if "best_eval_ucc_acc" not in run_data.metrics:
                    mlflow.log_metric(key = "best_eval_ucc_acc", value = run_data.metrics["eval_ucc_acc"])
                    value = run_data.metrics["eval_ucc_acc"]
                else:
                    value = run_data.metrics["best_eval_ucc_acc"]

            run = mlflow.get_run(run_id)
            current_params = run.data.params
            # mlflow.end_run()
            distributions:dict[str, optuna.distributions.BaseDistribution] = {k: v["distribution"] for k,v in params.items()}
            # for value in current_params.values():
                # if isinstance(value, int):
                    # distributions.append(optuna.distributions.IntDistribution())

            trials.append(optuna.create_trial(
                value=value,
                params=current_params,
                distributions=distributions
            ))
        study.add_trials(trials)
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
        study = optuna.create_study(study_name=study_name, direction="maximize")
    return study


def parse_experiment_runs_to_optuna_study_2(experiment_name: str, study_name:str, config_dict:dict):
    study = optuna.create_study(study_name=study_name, direction="maximize")
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train_drn")
    params = {}
    for key, value in config_dict.items():
        value_= eval(str(f"cfg.{value['aliases'][0]}"))
        if isinstance(value_, int):
            distribution = optuna.distributions.IntDistribution(*value["range"])
        elif isinstance(value_, float):
            distribution = optuna.distributions.FloatDistribution(*value["range"])
        else:
            distribution = None
        params[key] = {
            "value": value_,
            "distribution": distribution
        }


    if experiment := mlflow.get_experiment_by_name(experiment_name):
        experiment_id = experiment.experiment_id
        runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="list")
        trials = []

        run_ids = [run.info.run_id for run in runs]
        run_data = [run.data for run in runs]
        for run_id, run_data in zip(run_ids, run_data):
            with mlflow.start_run(run_id) as inner_run:
                run_params = run_data.params
                not_included = set(params.keys()) - set(run_params.keys())
                for key in not_included:
                    mlflow.log_param(key=key, value=params[key]["value"])
                if "best_eval_ucc_acc" not in run_data.metrics:
                    mlflow.log_metric(key = "best_eval_ucc_acc", value = run_data.metrics["eval_ucc_acc"])
                    value = run_data.metrics["eval_ucc_acc"]
                else:
                    value = run_data.metrics["best_eval_ucc_acc"]

            run = mlflow.get_run(run_id)
            current_params = run.data.params
            # mlflow.end_run()
            distributions:dict[str, optuna.distributions.BaseDistribution] = {k: v["distribution"] for k,v in params.items()}
            # for value in current_params.values():
                # if isinstance(value, int):
                    # distributions.append(optuna.distributions.IntDistribution())

            trials.append(optuna.create_trial(
                value=value,
                params=current_params,
                distributions=distributions
            ))
        study.add_trials(trials)
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
        study = optuna.create_study(study_name=study_name, direction="maximize")
    return study


def get_ori_model_weights():
    model_weights = {}
    keys = []
    with h5py.File("model_weights__2019_09_05__18_43_15__0123456789__128000.h5", 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                # if "model_1" in key:
                model_weights[f[key].name] = f[key][()]
    return model_weights


def load_original_paper_model():
    cfg_name = "train"
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=cfg_name)
    model_weights = get_ori_model_weights()
    model = UCCModel(cfg=cfg)
    state_dict = {
        name: torch.tensor(
            np.transpose(model_weights[value], (3,2,0,1)) 
            if len(model_weights[value].shape)==4 
            else (np.transpose(model_weights[value], (1,0)) if len(model_weights[value].shape)==2 
                else model_weights[value])
            ) 
        for name, value in name_map.items()
    }

    model.load_state_dict(state_dict, strict=False)
    return model