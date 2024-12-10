import numpy as np
import pandas as pd


class PartnersProcessor:
    def __init__(self, partner, consumer):
        self.partner = partner
        self.consumer = consumer

    def __repr__(self):
        return f"PartnersProcessor(partner={self.partner}, consumer={self.consumer})"

    def process(self):
        data = None
        if self.partner == 'coopernico':
            data = PartnersProcessor.coopernico(self.partner, self.consumer)
        return data

    @staticmethod
    def coopernico(partner, consumer):
        # Get all CSV files
        from os import listdir
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

            if 'Preço aquisição OMIE (EUR/MWh)' in data.columns:
                market_col = 'Preço aquisição OMIE (EUR/MWh)'
            elif 'Preço OMIE marginal PT (EUR/MWh)' in data.columns:
                market_col = 'Preço OMIE marginal PT (EUR/MWh)'
            else:
                raise KeyError

            df = pd.DataFrame(data={'price (EUR/MWh)': data[price_col].astype(np.float64),
                                    'consumption (MWh)': data[cons_col].astype(np.float64),
                                    'market price (EUR/MWh)': data[market_col]})

            df.index = data['datetime']
            df.index = pd.to_datetime(df.index, format='%Y-%m-%dT%H:%M:%SZ')

            df_consumption = df['consumption (MWh)'].copy()
            df = df.resample('h').mean()
            df['consumption (MWh)'] = df_consumption.resample('h').sum()

            # Assure kWh and EUR/kWh
            df['price (EUR/kWh)'] = df['price (EUR/MWh)'] / 1000
            df['market price (EUR/kWh)'] = df['market price (EUR/MWh)'] / 1000
            df['consumption (kWh)'] = df['consumption (MWh)'] * 1000
            df.drop(columns=['price (EUR/MWh)', 'market price (EUR/MWh)', 'consumption (MWh)'], inplace=True)

            # Concatenate files
            df_all = pd.concat([df_all, df])

        return df_all

