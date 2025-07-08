from processing.DataPreprocessor import DataPreprocessor
from causality.ElasticityAnalysis import ElasticityModel

import os
import json
import numpy as np
from loguru import logger
from datetime import datetime


def run_consumers(partner, city, consumers, aggregated, plot=1, dir_name=''):
    assert type(partner) == str, "The partner must be a string."
    assert type(city) == str, "The city must be a string."
    assert type(consumers) == list, "The consumers must be a list."

    out = {'elasticity': {}, 'ranking': None, 'pval': {}}
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
            out.get('pval').update({consumer: elasticity_model.model.summary().tables[0].data[1][-3]})
        if len(consumers) > 1:
            ranks = ElasticityModel.ranking(consumers, consumer_elasticities, plot=plot, dir_name=dir_name)
            df_results = ranks.copy()
            df_results['pval'] = df_results.index.map(out.get('pval'))
            df_results.to_csv(f"{dir_name}/results.csv", index=True)
            logger.debug(f'\n{df_results}')
            out.update({'ranking': {'order': ranks['Magnitude'].index.to_list(), 'magnitude': ranks.values.tolist()}})
        json.dump(out, open(f"{dir_name}/out.json", 'w'), indent=4)
    else:
        logger.debug(f"Processing aggregated consumers: {consumers}")
        preprocessor = DataPreprocessor(partner, city, consumers, dir_name=dir_name)
        preprocessor.process()
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
    consumers = ['MAC000005', 'MAC000014', 'MAC000015', 'MAC000017', 'MAC000031', 'MAC000044', 'MAC000046', 'MAC000048',
                 'MAC000051', 'MAC000052', 'MAC000064', 'MAC000065', 'MAC000071', 'MAC000075', 'MAC000076', 'MAC000082',
                 'MAC000088', 'MAC000106', 'MAC000107', 'MAC000109', 'MAC000124', 'MAC000139', 'MAC000141', 'MAC000146',
                 'MAC000147', 'MAC000158', 'MAC000165', 'MAC000170', 'MAC000173', 'MAC000186', 'MAC000187', 'MAC000193',
                 'MAC000194', 'MAC000195', 'MAC000198', 'MAC000219', 'MAC000236', 'MAC000247', 'MAC000257', 'MAC000259',
                 'MAC000261', 'MAC000265', 'MAC000266', 'MAC000278', 'MAC000286', 'MAC000288', 'MAC000290', 'MAC000292',
                 'MAC000293', 'MAC000298', 'MAC000303', 'MAC000328', 'MAC000336', 'MAC000339', 'MAC000342', 'MAC000344',
                 'MAC000352', 'MAC000354', 'MAC000366', 'MAC000369', 'MAC000383', 'MAC000384', 'MAC000392', 'MAC000395',
                 'MAC000401', 'MAC000402', 'MAC000406', 'MAC000408', 'MAC000418', 'MAC000421', 'MAC000433', 'MAC000439',
                 'MAC000447', 'MAC000451', 'MAC000456', 'MAC000461', 'MAC000464', 'MAC000465', 'MAC000467', 'MAC000470',
                 'MAC000475', 'MAC000478', 'MAC000487', 'MAC000494', 'MAC000497', 'MAC000500', 'MAC000512', 'MAC000521',
                 'MAC000523', 'MAC000525', 'MAC000529', 'MAC000537', 'MAC000544', 'MAC000549', 'MAC000561', 'MAC000565',
                 'MAC000582', 'MAC000583', 'MAC000584', 'MAC000586', 'MAC000587', 'MAC000590', 'MAC000593', 'MAC000595',
                 'MAC000598', 'MAC000605', 'MAC000609', 'MAC000611', 'MAC000617', 'MAC000620', 'MAC000628', 'MAC000633',
                 'MAC000635', 'MAC000640', 'MAC000641', 'MAC000653', 'MAC000665', 'MAC000666', 'MAC000672', 'MAC000673',
                 'MAC000680', 'MAC000681', 'MAC000683', 'MAC000687', 'MAC000688', 'MAC000704', 'MAC000708', 'MAC000709',
                 'MAC000710', 'MAC000716', 'MAC000719', 'MAC000722', 'MAC000725', 'MAC000728', 'MAC000738', 'MAC000739',
                 'MAC000751', 'MAC000754', 'MAC000755', 'MAC000759', 'MAC000766', 'MAC000772', 'MAC000775', 'MAC000785',
                 'MAC000789', 'MAC000791', 'MAC000797', 'MAC000801', 'MAC000803', 'MAC000805', 'MAC000810', 'MAC000814',
                 'MAC000817', 'MAC000819', 'MAC000821', 'MAC000822', 'MAC000828', 'MAC000830', 'MAC000834', 'MAC000841',
                 'MAC000845', 'MAC000847', 'MAC000851', 'MAC000857', 'MAC000859', 'MAC000868', 'MAC000877', 'MAC000878',
                 'MAC000879', 'MAC000884', 'MAC000887', 'MAC000895', 'MAC000898', 'MAC000901', 'MAC000910', 'MAC000913',
                 'MAC000924', 'MAC000931', 'MAC000932', 'MAC000935', 'MAC000937', 'MAC000938', 'MAC000939', 'MAC000942',
                 'MAC000943', 'MAC000944', 'MAC000945', 'MAC000946', 'MAC000961', 'MAC000965', 'MAC000976', 'MAC000982',
                 'MAC000985', 'MAC000991', 'MAC000993', 'MAC000994', 'MAC000999', 'MAC001004', 'MAC001012', 'MAC001013']
    remove = ['MAC000146', 'MAC000408', 'MAC000879', 'MAC000932', 'MAC000194', 'MAC000418', 'MAC000433',
              'MAC000672', 'MAC000666', 'MAC000945']
    consumers = list(set(consumers) - set(remove))
    n_test = len(consumers)
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
