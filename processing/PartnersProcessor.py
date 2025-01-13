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
            data_prices = pd.read_excel(f'data/{partner}/regulated market energy price 2023.xlt.xls', skiprows=1, index_col='datetime')
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

