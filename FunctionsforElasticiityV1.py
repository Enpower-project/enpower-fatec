import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
from econml.dml import LinearDML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from numba.core.errors import NumbaDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

np.random.seed(42)

class DataPreprocessor:
    def __init__(self, file_path, days_lag=1):
        self.file_path = file_path
        self.days_lag = days_lag
        self.data = None
        self.n_periods = None
        self.n_panels = None
        self.list_hours = None
        self.target_y = 'log_consumption'
        self.target_t = 'log_price'
        self.target_y_col = 'consumption (MWh)'
        self.target_t_col = 'price (EUR/MWh)'
        self.hourly_confounders = ['temperature', 'precipitation', 'consumption_lag']
        self.daily_confounders = ['holiday', 'month', 'day', 'year', 'weekend']
        self.effect_shifters = ['weekday']

    def read_data(self):
        logger.debug(f"Reading input data...")
        self.data = pd.read_csv(self.file_path, index_col='datetime')
        self.data.index = pd.to_datetime(self.data.index, format='%Y-%m-%d %H:%M:%S')
        
        inferred_freq = pd.infer_freq(self.data.index)
        if inferred_freq != 'H':
            logger.debug(f"Data frequency is {inferred_freq}, resampling to hourly intervals...")
            self.data = self.data.resample('H').sum()
        
        self.n_periods = len(self.data.index.hour.unique())
        self.list_hours = self.data.index.hour.unique().tolist()
        self.n_panels = self.n_panels = int(self.data.index.shape[0] / self.n_periods) - 1

    def process_data(self):
        logger.debug(f"Processing input data...")
        self.data['consumption_lag'] = self.data[self.target_y_col].shift(self.n_periods * self.days_lag)
        self.data = self.data[self.n_periods * self.days_lag:]
        self.n_panels = self.n_panels - self.days_lag

        data_filter = self.data.groupby(pd.Grouper(freq='1D')).count()
        complete_days = data_filter[(data_filter[self.target_y_col] == self.n_periods) &
                                    (data_filter[self.target_t_col] == self.n_periods) &
                                    (data_filter['temperature'] == self.n_periods)].index.to_list()
        self.data = self.data.loc[np.isin(self.data.index.date, [x.date() for x in complete_days])]
        data_daily = self.data.resample('D').mean()
        data_daily = data_daily.loc[np.isin(data_daily.index.date, [x.date() for x in complete_days])]

        self.data[self.target_y] = self.data[self.target_y_col].apply(lambda x: (np.log(x + 1)))
        Y = self.data[[self.target_y]].values

        Y = Y.reshape((self.n_panels, self.n_periods))

        self.data[self.target_t] = self.data[self.target_t_col].apply(lambda x: np.log(x))
        T = self.data[[self.target_t]].values

        T = T.reshape((self.n_panels, self.n_periods))

        logger.debug(f"Processing covariates...")
        W = self.data[self.hourly_confounders].values

        W = W.reshape((self.n_panels, self.n_periods * len(self.hourly_confounders)))

        W_aux = pd.DataFrame(W)
        for confounder in self.daily_confounders:
            W_aux[confounder] = data_daily[confounder].values
            W = W_aux.values

            W = StandardScaler().fit_transform(W)

        # Separate by effect shifters
        X = self.data[self.effect_shifters].values
        X = X.reshape((self.n_panels, self.n_periods))

        return Y, T, W, X, self.n_periods, self.list_hours

class ElasticityModel:
    def __init__(self, Y, T, W, X, n_periods, list_hours, daily_confounders, model_y_type='RandomForest',
                 model_t_type='RandomForest', cv=5, random_state=42, mc_iters=3):
        self.Y = Y
        self.T = T
        self.W = W
        self.X = X
        self.n_periods = n_periods
        self.list_hours = list_hours
        self.daily_confounders = daily_confounders
        self.cv = cv
        self.random_state = random_state
        self.mc_iters = mc_iters

        # Store model type parameters
        self.model_y_type = model_y_type
        self.model_t_type = model_t_type

        # Initialize model_y
        self.model_y = None

    def set_model_y(self):
        if self.model_y_type == 'RandomForest':
            self.model_y = RandomForestRegressor()
        elif self.model_y_type == 'LinearRegression':
            self.model_y = LinearRegression()
        elif self.model_y_type == 'XGBoost':
            self.model_y = XGBRegressor()
        else:
            raise ValueError(f"Unsupported model_y_type: {self.model_y_type}")

    def create_model(self):
        logger.debug(
            f"Training DML model with cv={self.cv}, random_state={self.random_state}, mc_iters={self.mc_iters}...")

        # Set model_y using the new method
        self.set_model_y()

        if self.model_t_type == 'RandomForest':
            model_t = RandomForestRegressor()
        elif self.model_t_type == 'LinearRegression':
            model_t = LinearRegression()
        elif self.model_t_type == 'XGBoost':
            model_t = XGBRegressor()
        else:
            raise ValueError(f"Unsupported model_t_type: {self.model_t_type}")

        # Initialize LinearDML model
        self.model = LinearDML(
            model_y=self.model_y,
            model_t=model_t,
            cv=self.cv,
            random_state=self.random_state,
            mc_iters=self.mc_iters
        )
        self.model.fit(self.Y, self.T, W=self.W, X=self.X, cache_values=True, inference='auto')

    def evaluate_model(self):
        try:
            logger.debug(f"Model scores:")
            logger.debug(f"Model Y (R2): {np.round(self.model.nuisance_scores_y, 3)}")
            logger.debug(f"Model T (R2): {np.round(self.model.nuisance_scores_t, 3)}")
        except Exception as e:
            logger.debug(f'Scores for nuisance models are unavailable: {e}')
        logger.debug(f"Final model (MSE): {round(self.model.score_, 3)}")

if __name__ == "__main__":
    file_path = "coopernico_data.csv"
    preprocessor = DataPreprocessor(file_path)
    preprocessor.read_data()
    Y, T, W, X, n_periods, list_hours = preprocessor.process_data()

    # Example with RandomForest models
    elasticity_model_rf = ElasticityModel(Y, T, W, X, n_periods, list_hours, preprocessor.daily_confounders, model_y_type='RandomForest', model_t_type='RandomForest')
    elasticity_model_rf.create_model()
    elasticity_model_rf.evaluate_model()