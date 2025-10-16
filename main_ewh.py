import os
import json
import random

import requests
import pandas as pd
from loguru import logger
from datetime import datetime

import settings
from ewh.ewh_flex import ewh_optimization, build_varBackpack
from ewh.ewh_flex.ewh_power_functions import real_ewh_load_estimator, create_usage_dataset


def get_ewh_usage():
    """
    Returns one random electric water heater (EWH) usage profile.
    Each profile is a list of heating events with start time and duration (minutes).

    The profiles reflect realistic household usage patterns, including
    morning showers, meal-related water heating, and evening routines.
    """

    profiles = [
        # Profile 1 — typical family (morning showers, lunch, evening use)
        [
            {"start": "07/12/2022 06:30", "duration": 20},
            {"start": "07/12/2022 12:30", "duration": 10},
            {"start": "07/12/2022 19:00", "duration": 25},
        ],

        # Profile 2 — single occupant, moderate evening use
        [
            {"start": "07/12/2022 07:15", "duration": 10},
            {"start": "07/12/2022 18:45", "duration": 15},
        ],

        # Profile 3 — family with children (multiple showers, laundry, dishes)
        [
            {"start": "07/12/2022 06:15", "duration": 25},
            {"start": "07/12/2022 07:30", "duration": 15},
            {"start": "07/12/2022 13:00", "duration": 10},
            {"start": "07/12/2022 19:30", "duration": 20},
            {"start": "07/12/2022 21:00", "duration": 10},
        ],

        # Profile 4 — retired couple (late morning, lunch, early evening)
        [
            {"start": "07/12/2022 08:30", "duration": 15},
            {"start": "07/12/2022 12:15", "duration": 10},
            {"start": "07/12/2022 17:30", "duration": 15},
        ],

        # Profile 5 — shift worker (late wake-up, late-night use)
        [
            {"start": "07/12/2022 10:00", "duration": 10},
            {"start": "07/12/2022 15:30", "duration": 10},
            {"start": "07/12/2022 23:15", "duration": 20},
        ],

        # Profile 6 — energy-conscious single (short and few events)
        [
            {"start": "07/12/2022 07:00", "duration": 5},
            {"start": "07/12/2022 20:00", "duration": 10},
        ],

        # Profile 7 — large household (clustered morning & evening peaks)
        [
            {"start": "07/12/2022 06:00", "duration": 20},
            {"start": "07/12/2022 06:45", "duration": 15},
            {"start": "07/12/2022 07:30", "duration": 15},
            {"start": "07/12/2022 19:00", "duration": 25},
            {"start": "07/12/2022 20:00", "duration": 15},
        ],

        # Profile 8 — student apartment (irregular hours, few uses)
        [
            {"start": "07/12/2022 09:30", "duration": 10},
            {"start": "07/12/2022 14:30", "duration": 5},
            {"start": "07/12/2022 22:00", "duration": 15},
        ],

        # Profile 9 — family with early risers and evening laundry
        [
            {"start": "07/12/2022 05:45", "duration": 20},
            {"start": "07/12/2022 12:00", "duration": 10},
            {"start": "07/12/2022 18:30", "duration": 20},
            {"start": "07/12/2022 21:00", "duration": 15},
        ],

        # Profile 10 — light user (compact morning + evening routine)
        [
            {"start": "07/12/2022 07:15", "duration": 10},
            {"start": "07/12/2022 19:30", "duration": 15},
        ],
    ]

    return {'ewh_usage': random.choice(profiles)}


def get_backpack(paramsInput, dataset):
    backpack = build_varBackpack(paramsInput, dataset)
    return backpack


def get_ewh_load(usage, backpack):
    ewh_usage = create_usage_dataset(usage)
    return real_ewh_load_estimator(ewh_usage, backpack)


def get_ewh_mean_profile(load):
    return load.groupby(load.index.hour).mean()


def optimize_ewh_load(paramsInput, dataset, resample='no', solver='CBC', dir_path=None):

    # -- Run optimization --
    opt_result = ewh_optimization(paramsInput, dataset, resample=resample, optSolver=solver)
    logger.info(f"Optimization result: \n{opt_result['opt_diagrams']}")

    # -- Make results serializable --
    opt_result_serializable = opt_result.copy()
    opt_result_serializable['opt_diagrams'] = {k: v.tolist() for k, v in opt_result['opt_diagrams'].items()}
    opt_result_serializable['ranking'] = {k: v.tolist() for k, v in opt_result['ranking'].items()}
    opt_result_serializable['opt_diagrams']['timestamp'] = [x.strftime('%Y-%m-%d %H:%M:%S') for x in opt_result_serializable['opt_diagrams']['timestamp']]
    opt_result_serializable['ranking']['duration'] = [str(x) for x in opt_result_serializable['ranking']['duration']]

    # -- Avoid NaNs in JSON output --
    opt_result_serializable['opt_diagrams']['binAux'] = [int(x) if pd.notna(x) else None for x in opt_result_serializable['opt_diagrams']['binAux']]

    return opt_result_serializable


def get_clustering_task_result(id):
    """ Fetch clustering task result from API.
    Args:
        id (str): Task ID.
    Returns:
        dict: Clustering results if task is completed, else None.
    """

    try:
        resp = requests.get(f"{settings.API_BASE_URL}/clustering_result/{id}")
        resp.raise_for_status()
        task_data = resp.json()
        logger.debug(f"Task data retrieved: {task_data}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error accessing task result: {e}")
        raise e

    if task_data and task_data['status'] == 'completed':
        inputs = task_data['inputs']
        results = task_data['outputs']
    else:
        inputs = None
        results = None
    return inputs, results


def run_ewh_optimization(inputs, dir_path=None):

    # --- Get clustering outputs ---
    task_id = inputs.get('task_id')
    clustering_inputs, clustering_outputs = get_clustering_task_result(task_id)
    if clustering_outputs is None:
        raise ValueError(f"No results found for the given task ID: {task_id}")
    else:
        logger.debug(f"Results found for task ID: {task_id}")
    clustering_results = clustering_outputs.get('clusters')

    # --- Preprocess ---
    nr_clusters = len(clustering_results)
    params_input = clustering_results['0']['backpack'][0]
    params_input['load_diagram_exists'] = 1

    # --- Run optimization analysis for each cluster ---
    out = {}
    for i in range(nr_clusters):
        # -- Prepare dataset for optimization --
        mean_load_ = clustering_results[str(i)]['mean_ewh']
        today = datetime.now().date()
        timestamps = pd.date_range(start=datetime.combine(today, datetime.min.time()), periods=24, freq='H')
        dataset_ = pd.DataFrame(mean_load_, columns=['load'], dtype=float)
        dataset_['timestamp'] = timestamps
        dataset_ = dataset_[['timestamp', 'load']]

        # -- Optimize EWH load --
        opt_output = optimize_ewh_load(params_input, dataset_, resample='1h', solver='CBC', dir_path=dir_path)
        out[f'cluster_{i}'] = {
            'ewh_specs': params_input,
            'mean_ewh_load': mean_load_,
            'opt_output': opt_output
        }
        logger.info(f"Optimization completed for cluster {i+1}/{nr_clusters}")

    # --- Save clustering results to JSON ---
    json.dump(out, open(f"{dir_path}/opt_results.json", 'w'), indent=4)
    logger.info(f"Optimization results saved to {dir_path}/opt_results.json")

    return out


stdInput = {
  "user": "sample_user",
  "datetime_start": "2022-12-07T00:00:00.000Z",
  "datetime_end": "2022-12-13T23:59:00.000Z",
  "load_diagram_exists": 0,
  "ewh_specs": {
    "ewh_capacity": 100,
    "ewh_power": 1800,
    "ewh_max_temp": 80,
    "user_comf_temp": 40,
    "tariff": 1,
    "price_simple": 0.119585,
    "price_dual_day": 0.149118,
    "price_dual_night": 0.070511,
    "tariff_simple": 0.3604,
    "tariff_dual": 0.4285
  }
}

if __name__ == "__main__":

    # Input data
    input = {'task_id': 'd132140d-37ab-47a4-9358-c6475e1ed07d'}

    # Prepare output folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory_name = f"results/ewh_{timestamp}"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    logger.add(f"{directory_name}/out.log", level="DEBUG", colorize=False, backtrace=True, diagnose=True)
    json.dump(input, open(f"{directory_name}/inputs.json", 'w'), indent=4)

    # Run optimization
    opt_output = run_ewh_optimization(input, dir_path=directory_name)
    logger.success("EWH optimization completed successfully.")
