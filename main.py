from loguru import logger

from processing.DataPreprocessor import DataPreprocessor
from elasticity.ElasticityAnalysis import ElasticityModel

if __name__ == "__main__":
    # Input data
    partner = 'coopernico'
    consumer = 'test'
    city = 'Porto'

    # Data processing
    preprocessor = DataPreprocessor(partner, city, consumer)
    preprocessor.process()

    # Model training
    elasticity_model_rf = ElasticityModel(preprocessor, model_y_type='RandomForest', model_t_type='RandomForest')
    elasticity_model_rf.process(plot=1)
