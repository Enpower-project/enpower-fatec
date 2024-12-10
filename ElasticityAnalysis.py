from settings import coordinates
from processing.process_partners import process_data_partners

import holidays
import requests
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from econml.dml import LinearDML
from geopy.geocoders import Nominatim
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from numba.core.errors import NumbaDeprecationWarning
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

np.random.seed(42)


class DataPreprocessor:
    def __init__(self, partner_, city_, days_lag_=None):
        self.partner = partner_
        self.city = city_
        self.days_lag = days_lag_
        self.data = None
        self.n_periods = None
        self.n_panels = None
        self.list_hours = None
        self.weather = None
        self.start = None
        self.end = None
        self.vif_results = None
        self.feature_importance_results = None
        self.low_loading_vars = None
        self.target_y_col = 'consumption (kWh)'
        self.target_t_col = 'price (EUR/kWh)'
        self.target_y = 'log_consumption'
        self.target_t = 'log_price'
        self.hourly_confounders = []
        self.daily_confounders = ['month', 'day', 'weekend', 'weekday', 'temperature', 'precipitation', 'consumption_lag_1', 'consumption_lag_7']
        self.effect_shifters = []

        if self.city not in coordinates:
            geolocator = Nominatim(user_agent="enpower-fatec")
            location = geolocator.geocode(self.city)
            self.latitude, self.longitude = location.latitude, location.longitude
            self.country = location.address.split(', ')[1]
        else:
            self.latitude, self.longitude = coordinates[self.city]['latitude'], coordinates[self.city]['longitude']
            self.country = coordinates[self.city]['country']

    def read_data(self):
        logger.debug(f"Reading input data...")
        self.data = process_data_partners(self.partner, None)
        self.data.index = pd.to_datetime(self.data.index, format='%Y-%m-%d %H:%M:%S')
        
        inferred_freq = pd.infer_freq(self.data.index)
        if inferred_freq != 'H':
            logger.debug(f"Data frequency is {inferred_freq}, resampling to hourly intervals...")
            self.data = self.data.resample('h').sum()
        
        self.n_periods = len(self.data.index.hour.unique())
        self.list_hours = self.data.index.hour.unique().tolist()
        self.n_panels = self.n_panels = int(self.data.index.shape[0] / self.n_periods)
        self.start = self.data.index.min()
        self.end = self.data.index.max()

    def calendar_data(self):

        self.data['day'] = self.data.index.day
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['hour'] = self.data.index.hour
        self.data['season'] = (self.data.index.month - 1) // 3
        self.data['weekday'] = self.data.index.weekday
        self.data['weekend'] = self.data['weekday'].apply(lambda x: 1 if x >= 5 else 0)
        self.data['period'] = (self.data.index.hour % 24 + 4) // 4
        if self.country == 'Portugal':
            holidays_ = holidays.Portugal(subdiv=13)
            holidays_.get('2025')
            holidays_.get('2024')
            holidays_.get('2023')
            holidays_ = [pd.to_datetime(x).strftime('%Y-%m-%d') for x in holidays_]
            self.data['datetime'] = self.data.index
            self.data['holiday'] = self.data['datetime'].apply(
                lambda x: 1 if pd.to_datetime(x).strftime('%Y-%m-%d') in holidays_ else 0)
            self.data.drop(columns=['datetime'], inplace=True)
        elif self.country == 'Spain':
            holidays_ = holidays.Spain(subdiv=13)
            holidays_.get('2025')
            holidays_.get('2024')
            holidays_.get('2023')
            holidays_ = [pd.to_datetime(x).strftime('%Y-%m-%d') for x in holidays_]
            self.data['datetime'] = self.data.index
            self.data['holiday'] = self.data['datetime'].apply(
                lambda x: 1 if pd.to_datetime(x).strftime('%Y-%m-%d') in holidays_ else 0)
            self.data.drop(columns=['datetime'], inplace=True)
        else:
            pass

    def weather_data(self):

        url = (f"https://archive-api.open-meteo.com/v1/era5?latitude="
               f"{self.latitude}&longitude={self.longitude}&"
               f"start_date={self.start.strftime('%Y-%m-%d')}&end_date={self.end.strftime('%Y-%m-%d')}"
               f"&hourly=temperature_2m,wind_speed_10m,precipitation")
        response = requests.get(url)
        if response.status_code == 200:
            self.weather = response.json()
            logger.debug("Weather data fetched successfully.")
            weather_df = pd.DataFrame(index=self.data.index,
                                      data={'temperature': self.weather['hourly']['temperature_2m'],
                                            'wind_speed': self.weather['hourly']['wind_speed_10m'],
                                            'precipitation': self.weather['hourly']['precipitation']})
            self.data = pd.concat([self.data, weather_df], axis=1)
        else:
            logger.error("Failed when fetching Weather data.")

    def calculate_vif(self):
        """Calculate Variance Inflation Factor (VIF) for confounder columns."""
        logger.debug("Calculating VIF for confounders...")
        confounders_df = self.data[self.hourly_confounders + self.daily_confounders].copy()
        confounders_df.dropna(inplace=True)

        vif_data = pd.DataFrame()
        vif_data["Variable"] = confounders_df.columns
        vif_data["VIF"] = [variance_inflation_factor(confounders_df.values, i) 
                           for i in range(confounders_df.shape[1])]

        logger.debug(f"VIF Data:\n{vif_data}")
        return vif_data

    def calculate_feature_importance(self):
        """Calculate feature importance using a RandomForest model on confounders."""
        logger.debug("Calculating feature importance using RandomForest...")

        # Prepare data for RandomForest model
        X_features = self.data[self.hourly_confounders + self.daily_confounders].copy()
        y_target = self.data[self.target_y].copy()
        
        # Drop any rows with NaN values
        X_features.dropna(inplace=True)
        y_target = y_target.loc[X_features.index]

        # Initialize and train the RandomForest model
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_features, y_target)

        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_features.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        logger.debug(f"Feature Importance:\n{feature_importance}")
        return feature_importance

    def perform_factor_analysis(self):
        logger.debug("Performing Factor Analysis for variable reduction...")
        confounders_df = self.data[self.hourly_confounders + self.daily_confounders].dropna()
        scaler = StandardScaler()
        confounders_scaled = scaler.fit_transform(confounders_df)
        fa = FactorAnalysis(n_components=5, random_state=42)
        fa.fit(confounders_scaled)
        factor_loadings = pd.DataFrame(fa.components_, columns=confounders_df.columns).T
        low_loading_vars = factor_loadings[factor_loadings.abs().max(axis=1) < 0.4].index.tolist()

        logger.debug(f"Factor Loadings:\n{low_loading_vars}")
        return low_loading_vars

    def process_data(self):
        logger.debug(f"Processing input data...")

        # Shift consumption data by the specified lag
        if self.days_lag:
            for lag in self.days_lag:
                self.data[f'consumption_lag_{lag}'] = self.data[self.target_y_col].shift(self.n_periods * lag)
            self.data = self.data[self.n_periods * max(self.days_lag):]
            self.n_panels = self.n_panels - max(self.days_lag)

        # Remove very-high-consumption days
        threshold = 20
        data_filter = self.data.groupby(pd.Grouper(freq='1D')).mean()
        high_consumption_days = data_filter[data_filter[self.target_y_col] > threshold].index.to_list()
        self.data = self.data.loc[~np.isin(self.data.index.date, [x.date() for x in high_consumption_days])]
        self.n_panels = int(self.data.index.shape[0] / self.n_periods)

        # Remove holidays
        holidays = self.data[self.data['holiday'] == 1].index.strftime('%Y-%m-%d').unique().to_list()
        self.data = self.data.loc[~np.isin(self.data.index.date, [pd.to_datetime(x).date() for x in holidays])]
        self.n_panels = int(self.data.index.shape[0] / self.n_periods)

        # Filter out incomplete days
        data_filter = self.data.groupby(pd.Grouper(freq='1D')).count()
        complete_days = data_filter[(data_filter[self.target_y_col] == self.n_periods) &
                                    (data_filter[self.target_t_col] == self.n_periods) &
                                    (data_filter['temperature'] == self.n_periods)].index.to_list()
        self.data = self.data.loc[np.isin(self.data.index.date, [x.date() for x in complete_days])]
        self.n_panels = int(self.data.index.shape[0] / self.n_periods)
        data_daily = self.data.resample('D').mean()
        data_daily = data_daily.loc[np.isin(data_daily.index.date, [x.date() for x in complete_days])]

        if self.data.empty:
            raise ValueError("No data available after filtering.")

        # Apply log transformation to target_y_col
        if self.data[self.data[self.target_y_col] == 0].shape[0] > 0:
            self.data[self.target_y] = self.data[self.target_y_col].apply(lambda x: (np.log(x + 0.001)))
        else:
            self.data[self.target_y] = self.data[self.target_y_col].apply(lambda x: (np.log(x)))
        Y = self.data[self.target_y].values
        Y = Y.reshape((self.n_panels, self.n_periods))

        # Apply log transformation to target_t_col
        if self.data[self.data[self.target_t_col] == 0].shape[0] > 0:
            self.data[self.target_t] = self.data[self.target_t_col].apply(lambda x: np.log(x + 0.001))
        else:
            self.data[self.target_t] = self.data[self.target_t_col].apply(lambda x: np.log(x))
        T = self.data[[self.target_t]].values
        T = T.reshape((self.n_panels, self.n_periods))

        logger.debug(f"Processing covariates...")

        # Process hourly and daily confounders
        W = self.data[self.hourly_confounders].values
        W = W.reshape((self.n_panels, self.n_periods * len(self.hourly_confounders)))
        W_aux = pd.DataFrame(W)
        for confounder in self.daily_confounders:
            W_aux[confounder] = data_daily[confounder].values
        W = W_aux.values
        W = StandardScaler().fit_transform(W)

        # Separate by effect shifters
        if self.effect_shifters:
            X = self.data[self.effect_shifters].values
            X = X.reshape((self.n_panels, self.n_periods))
        else:
            X = None

        # Calculate VIF for confounders
        self.vif_results = self.calculate_vif()

        # Calculate feature importance using RandomForest
        self.feature_importance_results = self.calculate_feature_importance()

        # Perform Factor Analysis for variable reduction
        self.low_loading_vars = self.perform_factor_analysis()

        # Return all processed data
        return Y, T, W, X, self.n_periods, self.list_hours


class ElasticityModel:
    def __init__(self, y, t, w, x, n_periods_, list_hours_, model_y_type='RandomForest',
                 model_t_type='RandomForest', cv=5, random_state=42, mc_iters=3):
        self.Y = y
        self.T = t
        self.W = w
        self.X = x
        self.n_periods = n_periods_
        self.list_hours = list_hours_
        self.cv = cv
        self.random_state = random_state
        self.mc_iters = mc_iters

        # Store model type parameters
        self.model_y_type = model_y_type
        self.model_t_type = model_t_type

        # Initialize model
        self.model = None
        self.model_y = None
        self.model_t = None
        self.elasticity = None
        self.elasticity_transposed = None

    def set_model_y(self):
        if self.model_y_type == 'RandomForest':
            self.model_y = RandomForestRegressor()
        elif self.model_y_type == 'LinearRegression':
            self.model_y = LinearRegression()
        elif self.model_y_type == 'XGBoost':
            self.model_y = XGBRegressor()
        else:
            raise ValueError(f"Unsupported model_y_type: {self.model_y_type}")

    def set_model_t(self):
        if self.model_t_type == 'RandomForest':
            self.model_t = RandomForestRegressor()
        elif self.model_t_type == 'LinearRegression':
            self.model_t = LinearRegression()
        elif self.model_t_type == 'XGBoost':
            self.model_t = XGBRegressor()
        else:
            raise ValueError(f"Unsupported model_t_type: {self.model_t_type}")

    def create_model(self):
        logger.debug(
            f"Training DML model with cv={self.cv}, random_state={self.random_state}, mc_iters={self.mc_iters}...")

        # Set model_y using the new method
        self.set_model_y()
        self.set_model_t()

        # Initialize LinearDML model
        self.model = LinearDML(
            model_y=self.model_y,
            model_t=self.model_t,
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

    def plot_elasticities(self):
        logger.debug(f"Calculating Price Elasticity Matrix...")
        self.elasticity = self.model.const_marginal_effect(X=self.X)
        self.elasticity_transposed = np.transpose(self.elasticity[0]).reshape(1, self.n_periods, self.n_periods)

        fig, ax = plt.subplots(1, 1)
        vmin = min(np.min(self.elasticity_transposed[0]), np.min(self.elasticity_transposed))
        vmax = max(np.max(self.elasticity_transposed[0]), np.max(self.elasticity_transposed))
        sns.heatmap(self.elasticity_transposed[0], annot=True, fmt=".2f", center=0, cmap='vlag', cbar=True, ax=ax,
                    annot_kws={"size": 6}, vmin=vmin, vmax=vmax)
        ax.set(xlabel='Hour', ylabel='Hour', title='Elasticity')
        ax.set_xticklabels(self.list_hours)
        ax.set_yticklabels(self.list_hours)
        plt.show()


if __name__ == "__main__":
    # Input data
    partner = 'coopernico'
    city = 'Porto'

    if partner not in ['coopernico', 'ceve', 'cuerva']:
        raise ValueError(f"Partner '{partner}' not found.")

    # Data processing
    preprocessor = DataPreprocessor(partner, city, days_lag_=[1, 7])
    preprocessor.read_data()
    preprocessor.calendar_data()
    preprocessor.weather_data()
    Y, T, W, X, n_periods, list_hours = preprocessor.process_data()

    # Model training
    elasticity_model_rf = ElasticityModel(Y, T, W, X, n_periods, list_hours,
                                          model_y_type='RandomForest', model_t_type='RandomForest')
    elasticity_model_rf.create_model()
    elasticity_model_rf.evaluate_model()
    elasticity_model_rf.plot_elasticities()

    logger.debug(f"End of script.")
