from processing.DataPreprocessor import DataPreprocessor
from causality.ElasticityAnalysis import ElasticityModel

import os
import json
from loguru import logger
from datetime import datetime


def run_consumers(partner, city, consumers, aggregated, plot=1, dir_name=''):
    assert type(partner) == str, "The partner must be a string."
    assert type(city) == str, "The city must be a string."
    assert type(consumers) == list, "The consumers must be a list."

    logger.debug(f"Processing consumers: {consumers}")
    out = {}
    consumer_elasticities = []
    for i, consumer in enumerate(consumers):
        logger.debug(f"Running for consumer: {consumer} ({i + 1}/{len(consumers)})")
        preprocessor = DataPreprocessor(partner, city, consumer, dir_name=dir_name)
        preprocessor.process()
        elasticity_model = ElasticityModel(preprocessor, model_y_type='XGBoost', model_t_type='XGBoost',
                                           cv=3, mc_iters=2, dir_name=dir_name)
        elasticity_model.process(plot=plot)
        consumer_elasticities.append(elasticity_model.elasticity[0])
        out.update({consumer: consumer_elasticities[i].tolist()})
    if len(consumers) > 1:
        ranks = ElasticityModel.ranking(consumers, consumer_elasticities, plot=plot, dir_name=dir_name)
        out.update({'ranking': {'order': ranks.index.to_list(), 'magnitude': ranks.values.tolist()}})
    json.dump(out, open(f"{dir_name}/out.json", 'w'), indent=4)


if __name__ == "__main__":
    # Input data
    inputs = {'partner': 'coopernico', 'consumers': ['test', 'test1', 'test2'], 'city': 'Porto', 'aggregated': 0}

    # Prepare output folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory_name = f"results/exec_{timestamp}"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    logger.add(f"{directory_name}/out.log", level="DEBUG", colorize=False, backtrace=True, diagnose=True)
    json.dump(inputs, open(f"{directory_name}/inputs.json", 'w'), indent=4)

    # Process data
    city = inputs['city']
    partner = inputs['partner']
    consumers = inputs['consumers']
    aggregated = inputs['aggregated']

    # Run analysis
    run_consumers(partner, city, consumers, aggregated, plot=1, dir_name=directory_name)
    logger.success(f"Execution completed successfully. Results available in {directory_name}.")
