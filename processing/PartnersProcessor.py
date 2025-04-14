import numpy as np
import pandas as pd
from os import listdir


class PartnersProcessor:
    def __init__(self, partner, consumers):
        self.partner = partner
        self.consumers = consumers

    def __repr__(self):
        return f"PartnersProcessor(partner={self.partner}, consumers={self.consumers})"

    def process(self):
        data = None
        if self.partner == 'coopernico':
            data = PartnersProcessor.coopernico(self.partner, self.consumers)
        elif self.partner == 'ceve':
            data = PartnersProcessor.ceve(self.partner, self.consumers)
        elif self.partner == 'cuerva':
            data = PartnersProcessor.cuerva(self.partner, self.consumers)
        elif self.partner == 'testing':
            data = PartnersProcessor.testing(self.partner, self.consumers)
        elif self.partner == 'epri':
            data = PartnersProcessor.epri(self.partner, self.consumers)
        return data

    @staticmethod
    def testing(partner, consumer):
        try:
            data_price = pd.read_pickle(f'data/testing/testing_data.pkl')
            data_price = data_price['Price']

            data = pd.read_csv(f'data/testing/lcl_dataset.csv')
        except FileNotFoundError:
            raise FileNotFoundError(f"Path not found for consumer {consumer} and partner {partner}")

        data = data[data['house'] == consumer][['DateTime', 'energy_consumption']]
        data.drop_duplicates(['DateTime'], keep='last', inplace=True)
        data.set_index('DateTime', inplace=True)
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
        data.index = data.index.tz_localize('UTC')
        data = data.reindex(pd.date_range(start=data.index.min(), end=data.index.max(), freq='30T'))

        data_price.index = pd.to_datetime(data_price.index, format='%Y-%m-%d %H:%M:%S')
        data_price.index = data_price.index.tz_localize('UTC')
        data_price = data_price.reindex(pd.date_range(start=data_price.index.min(), end=data_price.index.max(), freq='30T'))

        data = pd.merge(data, data_price, how='outer', left_index=True, right_index=True)
        data.columns = ['consumption (kWh)', 'price (EUR/kWh)']
        data = data.resample('h').mean()
        data = data[(data.index >= '2013-01-01') & (data.index <= data_price.index.max())]

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(data.index, data['price (EUR/kWh)'], label='Price (EUR/kWh)')
        # ax.set_xlabel('Datetime')
        # ax.set_ylabel('Price (EUR/kWh)')
        # ax2 = ax.twinx()
        # ax2.plot(data.index, data['consumption (kWh)'], label='Consumption (kWh)', color='orange')
        # ax2.set_ylabel('Consumption (kWh)')
        # ax.legend(loc='upper left')
        # ax2.legend(loc='upper right')
        # plt.tight_layout()
        # plt.savefig(f'data/testing/consumption/{partner}_{consumer}_timeseries.png')
        # plt.close(fig)

        return data

    @staticmethod
    def cuerva(partner, consumer):
        file = f'data/cuerva/Consumption User {consumer}'
        try:
            # Read data from CSV file
            data = pd.read_csv(f'{file}.csv', sep=';')
        except FileNotFoundError:
            raise FileNotFoundError(f"Path not found for consumer {consumer} and partner {partner}")

        data['hour'] = data['FH'].apply(lambda x: x.split(' ')[1].split(':')[0].zfill(2))
        data['date'] = data['FH'].apply(lambda x: x.split(' ')[0])
        try:
            data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
        except:
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
        data['datetime'] = data['date'].astype(str) + ' ' + data['hour'].astype(str) + ':' + '00' + ':00'
        data.index = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
        data.drop(columns=['FH', 'date', 'hour', 'datetime'], inplace=True)

        # Include prices
        try:
            # Read data from XLSX file
            data_prices = pd.read_excel(f'data/{partner}/regulated market energy price 2023.xlt.xls', skiprows=1,
                                        index_col='datetime')
        except FileNotFoundError:
            raise FileNotFoundError(f"Path not found for consumer {consumer} and partner {partner}")
        data_prices.index = pd.to_datetime(data_prices.index, format='ISO8601', utc=True)
        data_prices['datetime'] = data_prices.index.strftime('%Y-%m-%d %H:%M:%S')
        data_prices.index = pd.to_datetime(data_prices['datetime'], format='%Y-%m-%d %H:%M:%S')
        data_prices.drop(columns=['datetime'], inplace=True)
        merged_data = pd.merge(data, data_prices, how='outer', left_index=True, right_index=True)

        cons_col = 'Energy imported from the grid (Wh)'
        price_col = 'value (€/MWh)'
        df = pd.DataFrame(data={'price (EUR/kWh)': merged_data[price_col].astype(np.float64) / 1000,
                                'consumption (kWh)': merged_data[cons_col].astype(np.float64) / 1000
                                })
        df = df.resample('h').sum()

        # Sort data
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def ceve(partner, consumer):
        file = f'data/ceve/P{consumer}'
        try:
            # Read data from XLSX file
            data = pd.read_excel(f'{file}.xlsx')
        except FileNotFoundError:
            raise FileNotFoundError(f"Path not found for consumer {consumer} and partner {partner}")

        data.set_index('reading_time', inplace=True)
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')

        cons_col = 'Consumption [kWh]'
        df = pd.DataFrame(data={'price (EUR/kWh)': 0,
                                'consumption (kWh)': data[cons_col].astype(np.float64)
                                })
        df = df.resample('h').sum()

        # Generate tariff
        def tariff_hourly(x):
            if 8 <= x < 22:
                return 0.1967
            else:
                return 0.1072
        df['price (EUR/kWh)'] = df.index.hour.map(tariff_hourly)

        # Sort data
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def coopernico(partner, consumer):
        # Get all CSV files
        path = f'data/coopernico/{consumer}'
        try:
            filenames = listdir(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Path {path} not found "
                                    f"for consumer {consumer} and partner {partner}")
        filenames = [filename for filename in filenames if filename.endswith('.csv')]

        df_all = pd.DataFrame()
        for file in filenames:
            # Preprocess CSV file
            data = pd.read_csv(f'{path}/{file}', skiprows=3,
                               encoding='latin-1', sep=';', decimal=',', skipfooter=10, engine='python')
            data['hour'] = data['Hora PT'].apply(lambda x: x.split('h')[0].zfill(2))
            data['minute'] = data['Minutos'].apply(lambda x: x.split(' - ')[0].zfill(2))
            try:
                data['date'] = pd.to_datetime(data['Data PT'], format='%d/%m/%Y')
            except:
                data['date'] = pd.to_datetime(data['Data PT'], format='%Y-%m-%d')
            data = data[data['hour'] != '24']
            data['datetime'] = data['date'].astype(str) + 'T' + data['hour'].astype(str) + ':' + data['minute'].astype(
                str) + ':00Z'

            # Create new dataframe
            if 'Preço final de aquisição (EUR/MWh)' in data.columns:
                price_col = 'Preço final de aquisição (EUR/MWh)'
            elif 'Preço final sem ajuste OMIE (EUR/MWh)' in data.columns:
                price_col = 'Preço final sem ajuste OMIE (EUR/MWh)'
            elif 'Preço final sem ajuste OMIE (EUR/MWh) = Fórmula indexada que está no cabeçalho' in data.columns:
                price_col = 'Preço final sem ajuste OMIE (EUR/MWh) = Fórmula indexada que está no cabeçalho'
            else:
                raise KeyError

            if 'Curva carga perfilada (MWh)' in data.columns:
                cons_col = 'Curva carga perfilada (MWh)'
            elif 'Curva carga (MWh)' in data.columns:
                cons_col = 'Curva carga (MWh)'
            elif 'ORD=Curva carga (MWh) OU PERFIL=Total de consumo ORD sem curvas no tipo hora (MWh)' in data.columns:
                cons_col = 'ORD=Curva carga (MWh) OU PERFIL=Total de consumo ORD sem curvas no tipo hora (MWh)'
            else:
                raise KeyError

            df = pd.DataFrame(data={'price (EUR/MWh)': data[price_col].astype(np.float64),
                                    'consumption (MWh)': data[cons_col].astype(np.float64)})

            df.index = data['datetime']
            df.index = pd.to_datetime(df.index, format='%Y-%m-%dT%H:%M:%SZ')

            df_consumption = df['consumption (MWh)'].copy()
            df = df.resample('h').mean()
            df['consumption (MWh)'] = df_consumption.resample('h').sum()

            # Assure kWh and EUR/kWh
            df['price (EUR/kWh)'] = df['price (EUR/MWh)'] / 1000
            df['consumption (kWh)'] = df['consumption (MWh)'] * 1000
            df.drop(columns=['price (EUR/MWh)', 'consumption (MWh)'], inplace=True)

            # Concatenate files
            df_all = pd.concat([df_all, df])

        # Sort data
        df_all.sort_index(inplace=True)

        return df_all

    @staticmethod
    def epri(partner, consumer):
        # -- Price data
        path = f'data/epri/price'
        try:
            filenames = listdir(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Path {path} not found for consumer {consumer} and partner {partner}")
        filenames = [filename for filename in filenames if filename.endswith('.csv')]
        df_all = pd.DataFrame()
        for file in filenames:
            # Preprocess CSV file
            data_ = pd.read_csv(f'{path}/{file}')
            data_.columns = ['ESB_id', 'timestamp', 'elec_price', 'unit']

            # Concatenate files
            df_all = pd.concat([df_all, data_])

        # timestamp as datetime and sort by datetime and esb_id
        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
        df_all = df_all.sort_values(by=['timestamp', 'ESB_id'])
        price = df_all[df_all['ESB_id'] == consumer]
        if price.empty:
            raise ValueError(f"Price data for consumer {consumer} not found in partner {partner}.")
        price.drop_duplicates(['timestamp', 'ESB_id'], keep='last', inplace=True)
        price = price[['timestamp', 'elec_price']]
        price.set_index('timestamp', inplace=True)

        # -- Reindex between minimum and maximum timestamps
        price = price.reindex(pd.date_range(start=price.index.min(), end=price.index.max(), freq='h'))

        # -- Mapping consumers
        path = f'data/epri/Greencom_AS230Meter_Mapping.csv'
        mapping = pd.read_csv(path)
        a230_id = mapping[mapping['GreencomSiteID'] == consumer]['AS230_ID'].unique()
        if not a230_id:
            raise ValueError(f"Consumer {consumer} not mapped in partner {partner}.")
        elif a230_id.tolist()[0] == np.nan:
            raise ValueError(f"Consumer {consumer} not mapped in partner {partner} (nan value).")
        else:
            a230_id = a230_id.tolist()[0]

        # -- Consumption data
        path = f'data/epri/consumption'
        try:
            filenames = listdir(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Path {path} not found for consumer {consumer} and partner {partner}")
        filenames = [filename for filename in filenames if filename.endswith('.csv')]
        df_all = pd.DataFrame()
        for file in filenames:
            # Preprocess CSV file
            data_ = pd.read_csv(f'{path}/{file}', header=None)
            data_.columns = ['meter_id', 'measurement', 'timestamp', 'dst', 'kWh', '0']
            data_ = data_[(data_['meter_id'] == a230_id) & (data_['measurement'] == 'Active-power-import')]
            df_all = pd.concat([df_all, data_])

        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
        df_all = df_all.sort_values(by=['timestamp'])
        df_all.drop_duplicates(['timestamp'], keep='last', inplace=True)
        df_all.set_index('timestamp', inplace=True)
        df_all.index = df_all.index.tz_localize('UTC')
        df_all = df_all[['kWh']]
        if df_all.empty:
            raise ValueError(f"Consumption data for consumer {consumer} not found in partner {partner}.")
        df_all = df_all.reindex(pd.date_range(start=df_all.index.min(), end=df_all.index.max(), freq='15T'))
        consumption = df_all.resample('h').mean()
        consumption = consumption['kWh'].copy()

        df_combined = pd.merge(price, consumption, how='outer', left_index=True, right_index=True)
        df_combined.dropna(inplace=True)
        data = pd.DataFrame(data={'price (EUR/kWh)': df_combined['elec_price'].astype(np.float64),
                                  'consumption (kWh)': df_combined['kWh'].astype(np.float64)
                                  })

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(data.index, data['price (EUR/kWh)'], label='Price (EUR/kWh)')
        # ax.set_xlabel('Datetime')
        # ax.set_ylabel('Price (EUR/kWh)')
        # ax2 = ax.twinx()
        # ax2.plot(data.index, data['consumption (kWh)'], label='Consumption (kWh)', color='orange')
        # ax2.set_ylabel('Consumption (kWh)')
        # ax.legend(loc='upper left')
        # ax2.legend(loc='upper right')
        # plt.tight_layout()
        # plt.savefig(f'data/epri/consumption/{partner}_{consumer}_timeseries.png')
        # plt.show()

        return data
