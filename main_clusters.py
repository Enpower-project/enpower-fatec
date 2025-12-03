import os
import json
import requests
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.cm as cm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

import settings
from main_ewh import get_ewh_usage, get_ewh_load, get_ewh_mean_profile, get_backpack, stdInput
from processing.DataPreprocessor import DataPreprocessor


def get_consumers_data(inputs, dir_name):
    """ Fetch consumer data using DataPreprocessor.
    Args:
        inputs (dict): Dictionary containing 'consumers', 'partner', and 'city'.
        dir_name (str): Directory name for saving outputs.
    Returns:
        pd.DataFrame: Aggregated consumer data.
    """

    consumer_ids = inputs.get('consumers')
    partner = inputs.get('partner')
    city = inputs.get('city')

    preprocessor = DataPreprocessor(partner, city, consumer_ids, dir_name=dir_name)
    agg_data, _ = preprocessor.process(get_raw=True)
    return agg_data


def get_task_results(id):
    """ Fetch task results from API.
    Args:
        id (str): Task ID.
    Returns:
        tuple: (inputs, results) if task is completed, else (None, None).
    """

    try:
        resp = requests.get(f"{settings.API_BASE_URL}/result/{id}")
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


def clustering_process(df, range_n_clusters, dir_name, use_ewh=False):
    """ Perform clustering on the given DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing consumer data.
        range_n_clusters (range): Range of cluster numbers to evaluate.
        dir_name (str): Directory name for saving outputs.
    Returns:
        dict: Clustering results including optimal clusters and silhouette scores.
    """

    logger.debug(f"Running clustering on {df.shape[0]} consumers with {df.shape[1]-1} features.")
    X = df.drop(columns=['house'], axis=1).values.copy()
    sc = MinMaxScaler()
    X = sc.fit_transform(X)

    logger.debug(f"Evaluating silhouette scores for k in {list(range_n_clusters)}")
    silhouette_scores = []
    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
        logger.debug(f"k={k}, Silhouette Score={score:.4f}")
    best_k = range_n_clusters[np.argmax(silhouette_scores)]
    logger.debug(f"Optimal number of clusters: {best_k}")

    # Final clustering
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = final_kmeans.fit_predict(X)

    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(range_n_clusters, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(f"{dir_name}/silhouette_scores.png")
    plt.close()

    # --- Prepare JSON output ---
    clusters_output = {
        'optimal_clusters': int(best_k),
        'silhouette_scores': {int(k): round(float(s), 2) for k, s in zip(range_n_clusters, silhouette_scores)},
        'clusters': {}
    }
    for cluster in range(best_k):
        cluster_members = df[df['cluster'] == cluster]['house'].tolist()
        mean_consumption = df[df['cluster'] == cluster].drop(columns=['house', 'cluster']).iloc[:, :24].mean().tolist()
        mean_elasticity = df[df['cluster'] == cluster].drop(columns=['house', 'cluster']).iloc[:, 24:48].mean().tolist()
        clusters_output['clusters'][str(cluster)] = {
            'members': cluster_members,
            'mean_consumption': mean_consumption,
            'mean_elasticity': mean_elasticity
        }
        if use_ewh:
            mean_ewh_load = df[df['cluster'] == cluster].drop(columns=['house', 'cluster']).iloc[:, -48:-24].mean().tolist()
            mean_ewh_usage = df[df['cluster'] == cluster].drop(columns=['house', 'cluster']).iloc[:, -24:].mean().tolist()
            clusters_output['clusters'][str(cluster)]['mean_ewh_load'] = mean_ewh_load
            clusters_output['clusters'][str(cluster)]['mean_ewh_usage'] = mean_ewh_usage

    # --- Figure layout ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 7))

    profiles = ["consumption", "elasticity", "ewh_load", "ewh_usage"]
    captions = ["Consumption (kWh)", "Elasticity", "EWH Load (kWh)", "EWH Usage"]
    timestamps = list(range(24))  # 0–23 hours
    tick_positions = list(range(0, 24, 2))  # every 2 hours
    tick_labels = [h for h in tick_positions]

    colors = cm.get_cmap('tab10', best_k)

    # --- Helper to extract slices ---
    def extract_profile(cluster_data, feature):
        if feature == "consumption":
            return cluster_data[:, :24]
        elif feature == "elasticity" and cluster_data.shape[1] >= 48:
            return cluster_data[:, 24:48]
        elif feature == "ewh_load" and cluster_data.shape[1] >= 72:
            return cluster_data[:, -48:-24] / 1000  # Wh → kWh
        elif feature == "ewh_usage" and cluster_data.shape[1] >= 96:
            return cluster_data[:, -24:]
        return None

    # --- Plot all 4 subplots ---
    for i, feature in enumerate(profiles):
        ax = axs[i // 2, i % 2]

        for cluster in range(best_k):
            color = colors(cluster)
            cluster_data = df[df['cluster'] == cluster].drop(columns=['house', 'cluster']).values

            # Extract feature slice
            prof_matrix = extract_profile(cluster_data, feature)
            if prof_matrix is None:
                continue

            # Plot all individual curves
            for row in prof_matrix:
                ax.plot(timestamps, row, color=color, alpha=0.15)

            # Plot mean curve
            mean_profile = np.mean(prof_matrix, axis=0)
            ax.plot(timestamps, mean_profile, color=color, linewidth=2)

        # Style
        ax.set_xlabel("Hour")
        ax.set_xlim([timestamps[0], timestamps[-1]])
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        if feature != "elasticity":
            ax.set_ylim(bottom=0)

        ax.set_facecolor("#FBF6EF")
        ax.set_ylabel(captions[i])

    # --- Global legend (once) ---
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color=colors(c), linewidth=3, label=f"Cluster {c}")
        for c in range(best_k)
    ]
    fig.legend(handles=legend_handles,
               loc='upper center',
               ncol=best_k,
               frameon=False,
               fontsize=16,
               bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{dir_name}/cluster_profiles.pdf")
    plt.close(fig)

    return clusters_output


def run_clustering(task_id, dir_name, min_clusters=2, max_clusters=11, use_ewh=True):
    """ Main function to run clustering analysis.
    Args:
        task_id (str): Task ID to fetch elasticity results.
        dir_name (str): Directory name for saving outputs.
        min_clusters (int): Minimum number of clusters to evaluate.
        max_clusters (int): Maximum number of clusters to evaluate.
    Returns:
        dict: Clustering results.
    """

    # --- Get elasticity outputs ---
    elasticity_inputs, elasticity_results = get_task_results(task_id)
    if elasticity_results is None:
        raise ValueError(f"No results found for the given task ID: {task_id}")
    else:
        logger.debug(f"Results found for task ID: {task_id}")
    elasticity_results = elasticity_results.get('elasticity')

    # --- Assert number of consumers is greater than min_clusters ---
    if len(elasticity_results) < min_clusters:
        raise ValueError(f"Number of consumers ({len(elasticity_results)}) is less than min_clusters ({min_clusters}).")

    # --- Get consumer data ---
    consumers_data = get_consumers_data(elasticity_inputs, dir_name)

    # --- Merge elasticity-consumption data ---
    flat_data = {}
    for mac_id, values in elasticity_results.items():
        flat_values = [v[0] for v in values]
        flat_data[mac_id] = flat_values
    pivoted_df = pd.DataFrame.from_dict(flat_data, orient='index')
    pivoted_df.columns = [f"e_{i}" for i in range(pivoted_df.shape[1])]
    pivoted_df.reset_index(inplace=True)
    pivoted_df.rename(columns={'index': 'house'}, inplace=True)
    df_long = consumers_data.reset_index().melt(
        id_vars=consumers_data.index.name or 'index',
        var_name='house',
        value_name='energy_consumption'
    )
    df_long = df_long.rename(columns={consumers_data.index.name or 'index': 'DateTime'})
    df_long['DateTime'] = pd.to_datetime(df_long['DateTime'])
    df_long['energy_consumption'] = pd.to_numeric(df_long['energy_consumption'], errors='coerce')
    df_long['energy_consumption'] = df_long['energy_consumption'].dropna()
    df_long.reset_index(drop=True, inplace=True)
    df_long.dropna(inplace=True)
    df_long['DateTime'] = pd.to_datetime(df_long['DateTime'])
    df_long['hour'] = df_long['DateTime'].dt.hour
    df_long['half_hour'] = df_long['DateTime'].dt.round('30min').dt.time
    mean_consumption = df_long.groupby(['house', 'half_hour'])['energy_consumption'].mean().reset_index()
    pivoted_df_1 = mean_consumption.pivot(index='house', columns='half_hour', values='energy_consumption').reset_index()
    pivoted_df = pd.merge(pivoted_df_1, pivoted_df, on='house', how='inner')

    # --- Merge with EWH profile ---
    backpack = None
    if use_ewh:
        logger.debug("Generating EWH profiles for each consumer.")
        profile_cols_load = [f'ewh_load_{i}' for i in range(24)]
        profile_cols_usage = [f'ewh_usage_{i}' for i in range(24)]
        usage = [get_ewh_usage(days=7) for _ in range(len(pivoted_df))]
        backpack = [get_backpack(stdInput, usage_) for usage_ in usage]
        loads_and_usages = [get_ewh_load(usage_, backpack_) for usage_, backpack_ in zip(usage, backpack)]
        profiles_load = [get_ewh_mean_profile(x[['timestamp', 'load']]).values.flatten().tolist() for x in loads_and_usages]
        profiles_usage = [get_ewh_mean_profile(x[['timestamp', 'delta_use']]).values.flatten().tolist() for x in loads_and_usages]
        pivoted_df[profile_cols_load] = pd.DataFrame(profiles_load, index=pivoted_df.index)
        pivoted_df[profile_cols_usage] = pd.DataFrame(profiles_usage, index=pivoted_df.index)

    # --- Run cluster analysis ---
    range_n_clusters = range(min_clusters, max_clusters)
    clusters = clustering_process(pivoted_df, range_n_clusters, dir_name, use_ewh=use_ewh)

    # --- Additional outputs for optimization ---
    if use_ewh and backpack:
        for cluster_id, cluster_info in clusters['clusters'].items():
            member_backpacks = [backpack[pivoted_df[pivoted_df['house'] == member].index[0]] for member in cluster_info['members']]
            clusters['clusters'][cluster_id]['backpack'] = member_backpacks

    # --- Save clustering results to JSON ---
    with open(f"{dir_name}/clusters.json", 'w') as f:
        json.dump(clusters, f, indent=4)
    logger.debug(f"Clustering results saved to {dir_name}/clusters.json")

    return clusters


if __name__ == "__main__":

    # Input data
    inputs = {
        'task_id': "cc231d53-3a36-43b8-b5bb-bd9363231609",
        'min_clusters': 2,
        'max_clusters': 8
    }

    # Prepare output folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory_name = f"results/cluster_{timestamp}"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    logger.add(f"{directory_name}/out.log", level="DEBUG", colorize=False, backtrace=True, diagnose=True)
    json.dump(inputs, open(f"{directory_name}/inputs.json", 'w'), indent=4)

    # Process data
    task_id = inputs.get('task_id')
    min_clusters = inputs.get('min_clusters', 2)
    max_clusters = inputs.get('max_clusters', 11)

    # Run analysis
    clustering_results = run_clustering(task_id, directory_name, min_clusters=min_clusters, max_clusters=max_clusters)
    logger.success("Clustering analysis completed successfully.")
