from processing.DataPreprocessor import DataPreprocessor
from causality.ElasticityAnalysis import ElasticityModel

if __name__ == "__main__":
    # Input data
    partner = 'coopernico'
    consumer = 'test'
    city = 'Porto'

    # Data processing
    preprocessor = DataPreprocessor(partner, city, consumer)
    preprocessor.process()

    # Model training
    elasticity_model_rf = ElasticityModel(preprocessor, model_y_type='XGBoost', model_t_type='XGBoost',
                                          cv=3, mc_iters=5)
    elasticity_model_rf.process(plot=1)
