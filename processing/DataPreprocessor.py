from processing.DataDiagnosis import Diagnosis
from processing.PartnersProcessor import PartnersProcessor

import settings
import holidays
import requests
import warnings
import numpy as np
import pandas as pd
from loguru import logger
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)


class DataPreprocessor:
    def __init__(self, partner_, city_, consumers_, days_lag_=[1, 7], dir_name=''):
        if partner_ not in settings.partners:
            raise ValueError(f"Partner '{partner_}' not found.")
        else:
            self.partner = partner_
        self.city = city_
        self.consumers = consumers_
        self.dir_name = dir_name
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
        self.common_causes = ['temperature', 'weekday',
                              'consumption_lag_1', 'consumption_lag_7', 'consumption_lag_1h']
        self.modifiers = ['hour']
        self.instruments = []
        self.hourly_confounders = []
        self.daily_confounders = ['day', 'weekday', 'temperature', 'precipitation',
                                  'consumption_lag_1', 'consumption_lag_7']
        self.effect_shifters = []
        self.Y = None
        self.T = None
        self.W = None
        self.X = None

        self.data_original = None
        self.data_original_ = None
        self.data_scaled = None

        if self.city not in settings.coordinates:
            geolocator = Nominatim(user_agent="enpower-fatec")
            location = geolocator.geocode(self.city)
            self.latitude, self.longitude = location.latitude, location.longitude
            self.country = location.address.split(', ')[1]
        else:
            self.latitude = settings.coordinates[self.city]['latitude']
            self.longitude = settings.coordinates[self.city]['longitude']
            self.country = settings.coordinates[self.city]['country']

        if self.country not in settings.available_countries:
            raise ValueError(f"City '{self.city}' not supported. Available countries: {settings.available_countries}")

    def __repr__(self):
        return f"DataPreprocessor(partner={self.partner}, city={self.city}, days_lag={self.days_lag})"

    def process(self, diagnose=True):
        if type(self.consumers) == list:
            consumers_dict = {}
            for consumer in self.consumers:
                data_consumer = self.read_data(consumer)
                consumers_dict.update({consumer: data_consumer})
            # Filter by common index
            min_timestamp = min([consumers_dict[consumer].index.min() for consumer in self.consumers])
            max_timestamp = max([consumers_dict[consumer].index.max() for consumer in self.consumers])
            aggregated_data = pd.DataFrame(index=pd.date_range(start=min_timestamp, end=max_timestamp, freq='h'))
            for consumer in self.consumers:
                data_consumer = consumers_dict[consumer]
                data_consumer = data_consumer.reindex(aggregated_data.index)
                aggregated_data[consumer] = data_consumer['consumption (kWh)'].copy()
            # Sum consumption of all consumers
            aggregated = pd.DataFrame(aggregated_data.sum(axis=1), columns=['consumption (kWh)'])
            aggregated['price (EUR/kWh)'] = data_consumer['price (EUR/kWh)'].copy()
            self.data = aggregated

        else:
            self.read_data(self.consumers)
        self.calendar_data()
        self.weather_data()  # todo: consumer-specific weather data when running aggregated
        return self.process_data(diagnose=diagnose)

    def read_data(self, consumer):
        logger.debug(f"Reading input data for partner '{self.partner}' and consumer '{consumer}'...")
        partner_processing = PartnersProcessor(self.partner, consumer)
        self.data = partner_processing.process()
        self.data.index = pd.to_datetime(self.data.index, format='%Y-%m-%d %H:%M:%S')

        inferred_freq = pd.infer_freq(self.data.index)
        if inferred_freq != 'h':
            logger.debug(f"Data frequency is {inferred_freq}, resampling to hourly intervals...")
            self.data = self.data.resample('h').sum()

        self.n_periods = len(self.data.index.hour.unique())
        self.list_hours = self.data.index.hour.unique().sort_values().to_list()
        self.n_panels = self.n_panels = int(self.data.index.shape[0] / self.n_periods)
        self.start = self.data.index.min()
        self.end = self.data.index.max()

        return self.data

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
            holidays_ = holidays.Portugal()
        elif self.country == 'Spain':
            holidays_ = holidays.Spain()
        elif self.country == 'England':
            holidays_ = holidays.UK(subdiv='ENG')
        for year in list(range(2013, 2026))[::-1]:
            holidays_.get(str(year))
        holidays_ = [pd.to_datetime(x).strftime('%Y-%m-%d') for x in holidays_]
        self.data['datetime'] = self.data.index
        self.data['holiday'] = self.data['datetime'].apply(
            lambda x: 1 if pd.to_datetime(x).strftime('%Y-%m-%d') in holidays_ else 0)
        self.data.drop(columns=['datetime'], inplace=True)

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
            weather_df = pd.DataFrame(index=self.weather['hourly']['time'],
                                      data={'temperature': self.weather['hourly']['temperature_2m'],
                                            'wind_speed': self.weather['hourly']['wind_speed_10m'],
                                            'precipitation': self.weather['hourly']['precipitation']})
            weather_df.index = pd.to_datetime(weather_df.index, format='%Y-%m-%dT%H:%M')
            weather_df = weather_df[(weather_df.index >= self.start) & (weather_df.index <= self.end)]
            self.data = pd.concat([self.data, weather_df], axis=1)
        else:
            raise ValueError(f"Failed to fetch weather data. Status code: {response.status_code}")

    def process_data(self, min_days=180, diagnose=True):
        logger.debug(f"Processing input data...")

        # Shift consumption data by the specified lag
        if self.days_lag:
            for lag in self.days_lag:
                self.data[f'consumption_lag_{lag}'] = self.data[self.target_y_col].shift(self.n_periods * lag)
            self.data[f'consumption_lag_1h'] = self.data[self.target_y_col].shift(1)
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
        elif self.n_panels < min_days:
            raise ValueError("Insufficient data available after filtering.")

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

        # Original and scaled df for DoWhy
        self.data_original = self.data[self.common_causes + self.modifiers + self.instruments + [self.target_y_col, self.target_t_col]].copy()
        if self.data_original[self.data_original[self.target_t_col] == 0].shape[0] > 0:
            self.data_original[self.target_t] = self.data_original[self.target_t_col].apply(lambda x: np.log(x + 0.001))
        else:
            self.data_original[self.target_t] = self.data_original[self.target_t_col].apply(lambda x: np.log(x))
        if self.data_original[self.data_original[self.target_y_col] == 0].shape[0] > 0:
            self.data_original[self.target_y] = self.data_original[self.target_y_col].apply(lambda x: (np.log(x + 0.001)))
        else:
            self.data_original[self.target_y] = self.data_original[self.target_y_col].apply(lambda x: (np.log(x)))
        scaling_drop = [self.target_y_col, self.target_t_col]
        self.data_original_ = self.data_original.drop(columns=scaling_drop)
        data_columns = self.data_original_.columns
        self.data_scaled = StandardScaler().fit_transform(self.data_original_)
        self.data_scaled = pd.DataFrame(self.data_scaled, columns=data_columns, index=self.data_original.index)
        self.data_scaled[self.target_y_col] = self.data_original [self.target_y_col]
        self.data_scaled[self.target_t_col] = self.data_original[self.target_t_col]

        # Feature diagnosis
        if diagnose:
            logger.debug("Performing data diagnosis...")
            diagnosis = Diagnosis(self.data, self.target_y, self.hourly_confounders, self.daily_confounders)
            diagnosis.execute()
