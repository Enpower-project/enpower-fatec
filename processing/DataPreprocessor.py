import settings
from processing.PartnersProcessor import PartnersProcessor

import holidays
import requests
import warnings
import numpy as np
import pandas as pd
from loguru import logger
from geopy.geocoders import Nominatim
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from numba.core.errors import NumbaDeprecationWarning
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)


class DataPreprocessor:
    def __init__(self, partner_, city_, consumer_, days_lag_=[1, 7]):
        if partner_ not in settings.partners:
            raise ValueError(f"Partner '{partner_}' not found.")
        else:
            self.partner = partner_
        self.city = city_
        self.consumer = consumer_
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
        self.daily_confounders = ['day', 'weekday', 'temperature', 'precipitation',
                                  'consumption_lag_1', 'consumption_lag_7']
        self.effect_shifters = []
        self.Y = None
        self.T = None
        self.W = None
        self.X = None

        if self.city not in settings.coordinates:
            geolocator = Nominatim(user_agent="enpower-fatec")
            location = geolocator.geocode(self.city)
            self.latitude, self.longitude = location.latitude, location.longitude
            self.country = location.address.split(', ')[1]
        else:
            self.latitude = settings.coordinates[self.city]['latitude']
            self.longitude = settings.coordinates[self.city]['longitude']
            self.country = settings.coordinates[self.city]['country']

    def __repr__(self):
        return f"DataPreprocessor(partner={self.partner}, city={self.city}, days_lag={self.days_lag})"

    def process(self):
        self.read_data()
        self.calendar_data()
        self.weather_data()
        return self.process_data()

    def read_data(self):
        logger.debug(f"Reading input data for partner '{self.partner}' and consumer '{self.consumer}'...")
        partner_processing = PartnersProcessor(self.partner, self.consumer)
        self.data = partner_processing.process()
        self.data.index = pd.to_datetime(self.data.index, format='%Y-%m-%d %H:%M:%S')

        inferred_freq = pd.infer_freq(self.data.index)
        if inferred_freq != 'h':
            logger.debug(f"Data frequency is {inferred_freq}, resampling to hourly intervals...")
            self.data = self.data.resample('h').sum()

        self.n_periods = len(self.data.index.hour.unique())
        self.list_hours = self.data.index.hour.unique().tolist()
        self.n_panels = self.n_panels = int(self.data.index.shape[0] / self.n_periods)
        self.start = self.data.index.min()
        self.end = self.data.index.max()

    def calendar_data(self):
        logger.debug(f"Processing calendar data...")
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
        logger.debug(f"Fetching weather data...")

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
            raise ValueError(f"Failed to fetch weather data. Status code: {response.status_code}")

    def calculate_vif(self):
        logger.debug("Calculating VIF for confounders...")

        confounders_df = self.data[self.hourly_confounders + self.daily_confounders].copy()
        confounders_df.dropna(inplace=True)

        vif_data = pd.DataFrame()
        vif_data["Variable"] = confounders_df.columns
        vif_data["VIF"] = [variance_inflation_factor(confounders_df.values, i)
                           for i in range(confounders_df.shape[1])]
        logger.debug(f"{vif_data}")
        return vif_data

    def calculate_feature_importance(self):
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

        logger.debug(f"{feature_importance}")
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

        logger.debug(f"{low_loading_vars}")
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
        threshold = 20  # kWh
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
        self.Y = Y

        # Apply log transformation to target_t_col
        if self.data[self.data[self.target_t_col] == 0].shape[0] > 0:
            self.data[self.target_t] = self.data[self.target_t_col].apply(lambda x: np.log(x + 0.001))
        else:
            self.data[self.target_t] = self.data[self.target_t_col].apply(lambda x: np.log(x))
        T = self.data[[self.target_t]].values
        T = T.reshape((self.n_panels, self.n_periods))
        self.T = T

        # Process hourly and daily confounders
        logger.debug(f"Processing covariates...")
        W = self.data[self.hourly_confounders].values
        W = W.reshape((self.n_panels, self.n_periods * len(self.hourly_confounders)))
        W_aux = pd.DataFrame(W)
        for confounder in self.daily_confounders:
            W_aux[confounder] = data_daily[confounder].values
        W = W_aux.values
        W = StandardScaler().fit_transform(W)
        self.W = W

        # Separate by effect shifters
        if self.effect_shifters:
            X = self.data[self.effect_shifters].values
            X = X.reshape((self.n_panels, self.n_periods))
        else:
            X = None
        self.X = X

        # Feature diagnosis
        self.vif_results = self.calculate_vif()
        self.feature_importance_results = self.calculate_feature_importance()
        self.low_loading_vars = self.perform_factor_analysis()

