import os
import json
from typing import List
import mlflow
import optuna
import datetime
from omegaconf import OmegaConf
from hydra import initialize, compose

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


def parse_experiment_runs_to_optuna_study(experiment_name:str, study_name:str, cfg_name:str, params_file:str="params.json", experiments=False):
    study = optuna.create_study(study_name=study_name, direction="maximize")
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=cfg_name)
    if experiments:
        cfg.model = cfg.experiments
    with open(params_file, "r") as file:
        params_config:dict = json.loads(file.read())
    params = {}
    for key, value in params_config.items():
        value_= eval(str(f"cfg.{value['aliases'][0]}"))
        if "value" not in value:
            if isinstance(value_, int):
                distribution = optuna.distributions.IntDistribution(*value["range"])
            elif isinstance(value_, float):
                distribution = optuna.distributions.FloatDistribution(*value["range"])
            elif isinstance(value_, str):
                distribution = optuna.distributions.CategoricalDistribution(value["range"])
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
