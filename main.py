from processing.DataPreprocessor import DataPreprocessor
# from causality.ElasticityAnalysis import ElasticityModel
from causality.ElasticityAnalysisDoWhy import ElasticityModel

import os
import json
import numpy as np
from loguru import logger
from datetime import datetime


def run_consumers(partner, city, consumers, aggregated, plot=1, dir_name=''):
    assert type(partner) == str, "The partner must be a string."
    assert type(city) == str, "The city must be a string."
    assert type(consumers) == list, "The consumers must be a list."

    out = {'elasticity': {}, 'ranking': None}
    consumer_elasticities = []
    if not aggregated:
        logger.debug(f"Processing consumers: {consumers}")
        for i, consumer in enumerate(consumers):
            logger.debug(f"Running for consumer: {consumer} ({i + 1}/{len(consumers)})")
            preprocessor = DataPreprocessor(partner, city, consumer, dir_name=dir_name)
            preprocessor.process()
            elasticity_model = ElasticityModel(preprocessor, model_y_type='XGBoost', model_t_type='XGBoost',
                                               cv=5, mc_iters=5, dir_name=dir_name)
            elasticity_model.process(plot=plot)
            if elasticity_model.elasticity.shape == (24, 1):
                consumer_elasticities.append(elasticity_model.elasticity)
            else:
                consumer_elasticities.append(elasticity_model.elasticity[0])
            out.get('elasticity').update({consumer: consumer_elasticities[i].tolist()})
        if len(consumers) > 1:
            ranks = ElasticityModel.ranking(consumers, consumer_elasticities, plot=plot, dir_name=dir_name)
            out.update({'ranking': {'order': ranks.index.to_list(), 'magnitude': ranks.values.tolist()}})
        json.dump(out, open(f"{dir_name}/out.json", 'w'), indent=4)
    else:
        logger.debug(f"Processing aggregated consumers: {consumers}")
        preprocessor = DataPreprocessor(partner, city, consumers, dir_name=dir_name)
        preprocessor.process(diagnose=True)
        elasticity_model = ElasticityModel(preprocessor, model_y_type='XGBoost', model_t_type='XGBoost',
                                           cv=5, mc_iters=5, dir_name=dir_name)
        elasticity_model.process(plot=plot)
        if elasticity_model.elasticity.shape == (24, 1):
            consumer_elasticities = elasticity_model.elasticity
        else:
            consumer_elasticities = elasticity_model.elasticity[0]
        out.update({'elasticity': consumer_elasticities.tolist()})
        json.dump(out, open(f"{dir_name}/out.json", 'w'), indent=4)

    return out


if __name__ == "__main__":
    # Input data
    consumers = ['D0026', 'D0027', 'D0044', 'D0070', 'D0073', 'D0075', 'D0092', 'D0097', 'D0112', 'D0130',
                 'D0153', 'D0155', 'D0173', 'D0176', 'D0181', 'D0213', 'D0252', 'D0253', 'D0263', 'D0268',
                 'D0270', 'D0275', 'D0285', 'D0290', 'D0319', 'D0323', 'D0328', 'D0337', 'D0340', 'D0341',
                 'D0346', 'D0359', 'D0403', 'D0421', 'D0424', 'D0431', 'D0451', 'D0466', 'D0478', 'D0486',
                 'D0492', 'D0500', 'D0512', 'D0514', 'D0516', 'D0540', 'D0546', 'D0556', 'D0557', 'D0566',
                 'D0575', 'D0611', 'D0628', 'D0630', 'D0641', 'D0644', 'D0666', 'D0700', 'D0737', 'D0742',
                 'D0744', 'D0754', 'D0760', 'D0767', 'D0780', 'D0801', 'D0804', 'D0806', 'D0813', 'D0814',
                 'D0822', 'D0825', 'D0881', 'D0887', 'D0910', 'D0938', 'D0946', 'D0955', 'D1010', 'D1017',
                 'D1018']
    n_test = 10
    consumers = np.random.choice(consumers, n_test, replace=False).tolist()
    inputs = {'partner': 'testing',
              'consumers': consumers,
              'city': 'London', 'aggregated': 0}

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
