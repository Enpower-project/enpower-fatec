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


def clustering_process(df, range_n_clusters, dir_name):
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

    # --- Plot cluster consumption profiles ---
    colors = cm.get_cmap('tab10', best_k)
    plt.figure(figsize=(10, 6))
    for cluster in range(best_k):
        cluster_data = df[df['cluster'] == cluster].drop(columns=['house', 'cluster']).values
        if cluster_data.shape[1] >= 24:
            color = colors(cluster)
            for row in cluster_data[:, :24]:
                plt.plot(row, color=color, alpha=0.2)
            plt.plot(np.mean(cluster_data[:, :24], axis=0), color=color, linewidth=2, label=f'Cluster {cluster}')
    plt.xlabel('Hour')
    plt.ylabel('Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dir_name}/cluster_consumption.png")
    plt.close()

    # --- Plot cluster elasticity profiles ---
    plt.figure(figsize=(10, 6))
    for cluster in range(best_k):
        cluster_data = df[df['cluster'] == cluster].drop(columns=['house', 'cluster']).values
        if cluster_data.shape[1] >= 24:
            color = colors(cluster)
            for row in cluster_data[:, -24:]:
                plt.plot(row, color=color, alpha=0.2)
            plt.plot(np.mean(cluster_data[:, -24:], axis=0), color=color, linewidth=2, label=f'Cluster {cluster}')
    plt.xlabel('Hour')
    plt.ylabel('Elasticity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dir_name}/cluster_elasticity.png")
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
        mean_elasticity = df[df['cluster'] == cluster].drop(columns=['house', 'cluster']).iloc[:, -24:].mean().tolist()
        clusters_output['clusters'][str(cluster)] = {
            'members': cluster_members,
            'mean_consumption': mean_consumption,
            'mean_elasticity': mean_elasticity
        }

    # --- Save clustering results to JSON ---
    with open(f"{dir_name}/clusters.json", 'w') as f:
        json.dump(clusters_output, f, indent=4)
    logger.debug(f"Clustering results saved to {dir_name}/clusters.json")

    return clusters_output


def run_clustering(task_id, dir_name, min_clusters=2, max_clusters=11):
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

    # --- Run cluster analysis ---
    range_n_clusters = range(min_clusters, max_clusters)
    clusters = clustering_process(pivoted_df, range_n_clusters, dir_name)

    return clusters


if __name__ == "__main__":

    # Input data
    inputs = {
        'task_id': "e9811572-aa97-4987-aea1-765c4fe6fe29",
        'min_clusters': 2,
        'max_clusters': 11
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
