import os
import json
import random
import requests
import matplotlib
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import settings
from ewh.ewh_flex.ewh_opt_functions import resample_data
from ewh.ewh_flex import ewh_optimization, build_varBackpack
from ewh.ewh_flex.ewh_power_functions import real_ewh_load_estimator, create_usage_dataset

matplotlib.rcParams.update({'font.size': 14, 'font.family': 'serif',
                            'font.serif': 'cmr10', 'mathtext.fontset': 'cm',
                            'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})


def get_ewh_usage(days=1):
    """
    Generate realistic electric water heater (EWH) usage profiles
    for a household of 1–5 people over a given number of days,
    ensuring no overlapping events.
    Returns:
      [{"start": "DD/MM/YYYY HH:MM", "duration": X}, ...]
    """

    # Randomly select one household size, with weights, for the entire period
    household_size = random.choices([1, 2, 3, 4, 5],
                                    weights=[0.25, 0.25, 0.25, 0.15, 0.10])[0]

    # Day periods windows
    morning_window = (6, 10)    # 06:00–10:00
    midday_window = (11, 14)    # 11:00–14:00
    evening_window = (18, 24)   # 18:00–00:00

    def random_time_in_window(start_hour, end_hour):
        hour = random.randint(start_hour, end_hour - 1)
        minute = random.randint(0, 59)
        return f"{hour:02d}:{minute:02d}"

    all_events = []

    for day_offset in range(days):
        date_obj = datetime.now() + timedelta(days=day_offset)
        date_str = date_obj.strftime("%d/%m/%Y")
        events = []

        ### Bath events ###
        num_baths = 0
        for _ in range(household_size):
            num_baths += 1
            if random.random() < 0.1:  # 10% chance of a second bath
                num_baths += 1

        morning_baths = random.randint(1, min(num_baths, household_size))
        evening_baths = num_baths - morning_baths

        for _ in range(morning_baths):
            t = random_time_in_window(*morning_window)
            events.append({"start": f"{date_str} {t}", "duration": random.randint(3, 15)})

        for _ in range(evening_baths):
            t = random_time_in_window(*evening_window)
            events.append({"start": f"{date_str} {t}", "duration": random.randint(3, 15)})

        ### Cooking/cleaning events ###
        num_cooking = 1
        if random.random() < 0.3:  # 30% chance of 2 cooking events
            num_cooking = 2

        for _ in range(num_cooking):
            window = random.choice([morning_window, midday_window, evening_window])
            t = random_time_in_window(*window)
            events.append({"start": f"{date_str} {t}", "duration": random.randint(2, 10)})

        ### Add slight jitter (~10 min) ###
        ### The goal is to prevent all events to ###
        ### start in static XXh00 timestamps     ###
        def jitter(event):
            t = datetime.strptime(event["start"], "%d/%m/%Y %H:%M")
            jitter = timedelta(minutes=random.randint(-10, 10))
            shifted = t + jitter
            return {"start": shifted.strftime("%d/%m/%Y %H:%M"), "duration": event["duration"]}

        events = [jitter(e) for e in events]

        ### Sort and ensure no overlaps ###
        events.sort(key=lambda e: datetime.strptime(e["start"], "%d/%m/%Y %H:%M"))

        non_overlap_events = []
        last_end = None

        for e in events:
            start_time = datetime.strptime(e["start"], "%d/%m/%Y %H:%M")
            end_time = start_time + timedelta(minutes=e["duration"])

            if last_end and start_time < last_end:
                # Shift start time forward to the last event end
                start_time = last_end
                end_time = start_time + timedelta(minutes=e["duration"])

            last_end = end_time
            non_overlap_events.append({
                "start": start_time.strftime("%d/%m/%Y %H:%M"),
                "duration": e["duration"]
            })

        all_events.extend(non_overlap_events)

    ### Sort all events (ensuring cross-day safety) ###
    all_events.sort(key=lambda e: datetime.strptime(e["start"], "%d/%m/%Y %H:%M"))

    return all_events


def get_backpack(paramsInput, dataset):
    backpack = build_varBackpack(paramsInput, dataset)
    return backpack


def get_ewh_load(usage, backpack):
    ewh_usage = create_usage_dataset(usage)
    load = real_ewh_load_estimator(ewh_usage, backpack)
    df = resample_data(pd.DataFrame({'timestamp': ewh_usage['timestamp'], 'load': load.values, 'delta_use': ewh_usage['delta_use']}), resolution='1h')
    return df


def get_ewh_mean_profile(load):
    load.set_index('timestamp', inplace=True)
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
        mean_load_ = clustering_results[str(i)]['mean_ewh_load']
        mean_usage_ = clustering_results[str(i)]['mean_ewh_usage']
        today = datetime.now().date()
        timestamps = pd.date_range(start=datetime.combine(today, datetime.min.time()), periods=24, freq='H')
        dataset_ = pd.DataFrame(mean_load_, columns=['load'], dtype=float)
        dataset_['timestamp'] = timestamps
        dataset_['delta_use'] = mean_usage_
        dataset_ = dataset_[['timestamp', 'load', 'delta_use']]
        logger.debug(f"Dataset for cluster {i}:\n{dataset_.head()}")

        # -- Optimize EWH load --
        opt_output = optimize_ewh_load(params_input, dataset_, resample='1h', solver='CBC', dir_path=dir_path)
        out[f'cluster_{i}'] = {
            'ewh_specs': params_input,
            'mean_ewh_load': mean_load_,
            'opt_output': opt_output
        }
        logger.info(f"Optimization completed for cluster {i+1}/{nr_clusters}")

    # Plot original vs optimized load for each cluster
    colors = plt.cm.get_cmap('tab10', nr_clusters)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i in range(nr_clusters):
        cluster_data = out[f'cluster_{i}']
        cluster_load = [x / 1000 for x in cluster_data['mean_ewh_load']]
        timestamps = pd.date_range(start=datetime.combine(today, datetime.min.time()), periods=24, freq='H')

        # Original mean load
        ax.plot(timestamps, cluster_load, label=f'Cluster {i}', linestyle='--', color=colors(i))

        # Optimized load
        opt_load = cluster_data['opt_output']['opt_diagrams']['optimized_load']
        opt_load = [float(x) / 1000 for x in opt_load]
        ax.plot(timestamps, opt_load, label=f'Cluster {i} (Opt.)', linestyle='-', color=colors(i), linewidth=2)
    ax.set_xlabel('Hour')
    ax.set_ylabel('EWH Load (kWh)')
    ax.set_facecolor('#FBF6EF')
    ax.set_ylim(0, None)
    ax.set_xlim([timestamps[0], timestamps[-1]])
    ax.set_xticks(timestamps)
    ax.set_xticklabels([int(ts.strftime('%H')) for ts in timestamps])
    ax.legend(loc='upper left', frameon=False, ncols=nr_clusters, bbox_to_anchor=(0, 1.5))
    plt.tight_layout()
    plt.savefig(f"{dir_path}/ewh_optimization_comparison.pdf", bbox_inches="tight")

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
    "tariff": 0,
    "price_simple": 0,
    "price_dual_day": 0,
    "price_dual_night": 0,
    "tariff_simple": 0,
    "tariff_dual": 0
  }
}

if __name__ == "__main__":

    # Input data
    input = {'task_id': 'd040212f-e9a3-4623-a42a-af5cd88f04c7'}

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
